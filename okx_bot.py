"""OKX 4H 趨勢機器人 (通道突破 + ATR 停損 + 日線 SMA200 多空濾網)。

規則 (與 trend.py 回測相同):
- 4H 收盤 > 前 55 根最高價 → 做多;  < 前 55 根最低價 → 做空 (依 STRATEGY_MODE)
- 初始停損: 進場價 ∓ 2 x ATR(20), 下單時直接附帶在 OKX 上
- 出場: 多單收盤 < 前 20 根最低價 / 空單收盤 > 前 20 根最高價 → 市價平倉
- 倉位: 每筆風險 = 帳戶權益 x RISK_PCT, 由停損距離回推張數; 逐倉, 槓桿只是保證金設定
- CPI / NFP / FOMC 前後 4 小時不開新倉

環境變數 (金鑰只放在你自己的電腦, 不要寫進程式或 commit):
  OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE   必填
  OKX_DEMO=1            1 = 模擬盤 (預設), 0 = 實盤
  DRY_RUN=0             1 = 只算訊號、不下單
  SYMBOLS=BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,XRP-USDT-SWAP,ADA-USDT-SWAP
  RISK_PCT=0.01         每筆風險 (帳戶權益比例)
  LEVERAGE=3            逐倉槓桿 (保證金設定)
  MAX_EXPOSURE=2.0      所有部位名目總額上限 (帳戶權益倍數)
  STRATEGY_MODE=regime  regime = 多頭只做多/空頭只做空; long_only = 只做多

用法:
  python okx_bot.py --once     跑一次 (檢查訊號 / 下單) 後結束
  python okx_bot.py            常駐, 每根 4H K棒收盤後執行
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from events import blackout_mask, event_times_utc

BASE_URL = "https://www.okx.com"
HERE = os.path.dirname(os.path.abspath(__file__))

ENTRY_N, EXIT_N, ATR_N, ATR_MULT, SMA_N = 55, 20, 20, 2.0, 200

log = logging.getLogger("okx_bot")


# ───────────────────────── 設定 ─────────────────────────
def env(name, default=None, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


class Config:
    def __init__(self):
        self.key = env("OKX_API_KEY")
        self.secret = env("OKX_API_SECRET")
        self.passphrase = env("OKX_API_PASSPHRASE")
        self.demo = env("OKX_DEMO", "1") != "0"
        self.dry_run = env("DRY_RUN", "0") == "1"
        self.symbols = [s.strip() for s in env("SYMBOLS", "BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP,XRP-USDT-SWAP,ADA-USDT-SWAP").split(",") if s.strip()]
        self.risk_pct = env("RISK_PCT", 0.01, float)
        self.leverage = env("LEVERAGE", 3, int)
        self.max_exposure = env("MAX_EXPOSURE", 2.0, float)
        self.mode = env("STRATEGY_MODE", "regime")
        if self.mode not in ("regime", "long_only"):
            raise SystemExit("STRATEGY_MODE 只能是 regime 或 long_only")
        if not self.dry_run and not (self.key and self.secret and self.passphrase):
            raise SystemExit("缺少 OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE 環境變數 (或設定 DRY_RUN=1)")
        if self.risk_pct > 0.03:
            raise SystemExit("RISK_PCT 超過 3%, 為了安全拒絕執行")


# ───────────────────────── OKX REST ─────────────────────────
class OKX:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _request(self, method, path, params=None, body=None, private=False):
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        data = json.dumps(body) if body is not None else ""
        headers = {"Content-Type": "application/json", "User-Agent": "smc-trend-bot/1.0"}
        if private:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
            msg = ts + method + path + query + data
            sign = base64.b64encode(hmac.new(self.cfg.secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()
            headers.update({"OK-ACCESS-KEY": self.cfg.key, "OK-ACCESS-SIGN": sign,
                            "OK-ACCESS-TIMESTAMP": ts, "OK-ACCESS-PASSPHRASE": self.cfg.passphrase})
            if self.cfg.demo:
                headers["x-simulated-trading"] = "1"
        req = urllib.request.Request(BASE_URL + path + query, data=data.encode() if data else None,
                                     method=method, headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    res = json.load(r)
                break
            except urllib.error.HTTPError as e:
                res = json.loads(e.read() or b"{}")
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:  # 網路錯誤
                res = {"code": "net", "msg": str(e)}
            time.sleep(2 ** attempt)
        if res.get("code") != "0":
            raise RuntimeError(f"OKX {method} {path} 失敗: {res.get('code')} {res.get('msg')} {res.get('data')}")
        return res["data"]

    def get(self, path, params=None, private=False):
        return self._request("GET", path, params, None, private)

    def post(self, path, body):
        return self._request("POST", path, None, body, True)

    # 公開資料
    def candles_4h(self, inst, bars=1500) -> pd.DataFrame:
        rows, after = {}, ""
        while len(rows) < bars:
            params = {"instId": inst, "bar": "4H", "limit": 300 if not after else 100}
            if after:
                params["after"] = after
            # 第一頁用最新K線端點 (剛收盤的K棒會立即出現), 更早的用歷史端點
            path = "/api/v5/market/candles" if not after else "/api/v5/market/history-candles"
            batch = self.get(path, params)
            if not batch:
                break
            for k in batch:
                if k[8] == "1":  # 只用已收盤K棒
                    rows[int(k[0])] = [float(x) for x in k[1:5]]
            after = batch[-1][0]
            time.sleep(0.12)
        df = pd.DataFrame.from_dict(rows, orient="index", columns=["open", "high", "low", "close"]).sort_index()
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
        return df

    def instrument(self, inst):
        return self.get("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst})[0]


# ───────────────────────── 訊號 ─────────────────────────
def compute_signal(df: pd.DataFrame, mode: str) -> dict:
    """用最後一根已收盤 4H K棒計算訊號。"""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean()
    hh = h.rolling(ENTRY_N).max().shift(1)
    ll = l.rolling(ENTRY_N).min().shift(1)
    xh = h.rolling(EXIT_N).max().shift(1)
    xl = l.rolling(EXIT_N).min().shift(1)
    daily = c.resample("1D").last().dropna()
    sma = daily.rolling(SMA_N).mean()
    # 最後一根4H所屬日之前的「已收盤日K」
    last_t = df.index[-1]
    prev_days = daily[daily.index < last_t.floor("1D")]
    if len(prev_days) < SMA_N:
        raise RuntimeError("日線資料不足 200 天, 無法判斷多空")
    bear = bool(prev_days.iloc[-1] < sma[prev_days.index[-1]])
    close = float(c.iloc[-1])
    long_ok = not (mode == "regime" and bear)
    short_ok = mode == "regime" and bear
    entry = 0
    if close > hh.iloc[-1] and long_ok:
        entry = 1
    elif close < ll.iloc[-1] and short_ok:
        entry = -1
    return dict(time=last_t, close=close, atr=float(atr.iloc[-1]), bear=bear, entry=entry,
                exit_long=close < xl.iloc[-1], exit_short=close > xh.iloc[-1],
                hh=float(hh.iloc[-1]), ll=float(ll.iloc[-1]), xh=float(xh.iloc[-1]), xl=float(xl.iloc[-1]))


def in_blackout(now: pd.Timestamp) -> bool:
    idx = pd.DatetimeIndex([now.floor("15min")])
    return bool(blackout_mask(idx, 4.0, pd.Timedelta(minutes=15))[0])


# ───────────────────────── 機器人 ─────────────────────────
class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.okx = OKX(cfg)
        self.pos_mode = None

    def account_ready(self):
        if self.cfg.dry_run:
            return
        conf = self.okx.get("/api/v5/account/config", private=True)[0]
        self.pos_mode = conf.get("posMode")  # net_mode / long_short_mode
        log.info("帳戶模式: %s, 持倉模式: %s, %s", conf.get("acctLv"), self.pos_mode, "模擬盤" if self.cfg.demo else "實盤")

    def equity(self) -> float:
        d = self.okx.get("/api/v5/account/balance", {"ccy": "USDT"}, private=True)[0]
        det = next((x for x in d.get("details", []) if x.get("ccy") == "USDT"), None)
        return float(det["eq"]) if det else float(d["totalEq"])

    def positions(self) -> dict:
        out = {}
        for p in self.okx.get("/api/v5/account/positions", {"instType": "SWAP"}, private=True):
            sz = float(p["pos"] or 0)
            if sz == 0:
                continue
            side = p.get("posSide", "net")
            d = 1 if (side == "long" or (side == "net" and sz > 0)) else -1
            out[p["instId"]] = dict(dir=d, sz=abs(sz), pos_side=side, notional=abs(float(p.get("notionalUsd") or 0)))
        return out

    def _pos_side(self, d):
        return ("long" if d == 1 else "short") if self.pos_mode == "long_short_mode" else None

    def close_position(self, inst, pos):
        body = {"instId": inst, "mgnMode": "isolated", "autoCxl": True}
        if pos["pos_side"] in ("long", "short"):
            body["posSide"] = pos["pos_side"]
        self.okx.post("/api/v5/trade/close-position", body)
        # 取消殘留的停損條件單
        algos = self.okx.get("/api/v5/trade/orders-algo-pending", {"instType": "SWAP", "instId": inst, "ordType": "conditional"}, private=True)
        if algos:
            self.okx.post("/api/v5/trade/cancel-algos", [{"algoId": a["algoId"], "instId": inst} for a in algos])
        log.info("%s 已平倉", inst)

    def open_position(self, inst, sig, equity, exposure_now):
        d = sig["entry"]
        ins = self.okx.instrument(inst)
        ct_val, lot, min_sz = float(ins["ctVal"]), float(ins["lotSz"]), float(ins["minSz"])
        tick = float(ins["tickSz"])
        price = sig["close"]
        stop_dist = ATR_MULT * sig["atr"]
        sl = price - d * stop_dist
        risk_usdt = equity * self.cfg.risk_pct
        contracts = risk_usdt / (stop_dist * ct_val)
        # 名目總額上限
        room = max(self.cfg.max_exposure * equity - exposure_now, 0)
        contracts = min(contracts, room / (price * ct_val))
        contracts = math.floor(contracts / lot) * lot
        if contracts < min_sz:
            log.warning("%s 計算張數 %.4f 低於最小下單量 %s (資金不足或曝險已滿), 略過", inst, contracts, min_sz)
            return
        notional = contracts * ct_val * price
        sl_px = round(round(sl / tick) * tick, 10)
        log.info("%s 開%s: %s 張 (名目 %.0f USDT), 參考價 %.6g, 停損 %.6g, 風險 %.1f USDT",
                 inst, "多" if d == 1 else "空", contracts, notional, price, sl_px, contracts * ct_val * stop_dist)
        if self.cfg.dry_run:
            return
        lev = {"instId": inst, "lever": str(self.cfg.leverage), "mgnMode": "isolated"}
        if self._pos_side(d):
            lev["posSide"] = self._pos_side(d)
        self.okx.post("/api/v5/account/set-leverage", lev)
        order = {"instId": inst, "tdMode": "isolated", "side": "buy" if d == 1 else "sell", "ordType": "market",
                 "sz": f"{contracts:g}",
                 "attachAlgoOrds": [{"slTriggerPx": f"{sl_px:g}", "slOrdPx": "-1", "slTriggerPxType": "last"}]}
        if self._pos_side(d):
            order["posSide"] = self._pos_side(d)
        res = self.okx.post("/api/v5/trade/order", order)
        log.info("%s 下單成功 ordId=%s", inst, res[0].get("ordId"))

    def run_once(self):
        now = pd.Timestamp.now(tz="UTC")
        if max(event_times_utc()) < now:
            log.warning("events.py 的重大事件日期已過期, 請更新 CPI / NFP / FOMC 日期")
        blocked = in_blackout(now)
        if blocked:
            log.info("重大事件前後 4 小時: 本輪不開新倉")
        positions = {} if self.cfg.dry_run else self.positions()
        equity = 0.0 if self.cfg.dry_run else self.equity()
        exposure = sum(p["notional"] for p in positions.values())
        if not self.cfg.dry_run:
            log.info("帳戶權益 %.2f USDT, 持倉 %d 個, 名目曝險 %.0f USDT", equity, len(positions), exposure)
        for inst in self.cfg.symbols:
            try:
                expected = now.floor("4h") - pd.Timedelta(hours=4)  # 最後一根應已收盤的4H K棒
                for _ in range(5):
                    df = self.okx.candles_4h(inst)
                    if df.index[-1] >= expected:
                        break
                    log.info("%s 最新K棒尚未更新, 30 秒後重試", inst)
                    time.sleep(30)
                else:
                    log.error("%s K線資料過舊 (最後 %s), 本輪略過", inst, df.index[-1])
                    continue
                sig = compute_signal(df, self.cfg.mode)
                log.info("%s %s 收 %.6g | 日線%s | 突破線 %.6g / %.6g | 出場線 %.6g / %.6g | 訊號 %s",
                         inst, sig["time"].strftime("%m-%d %H:%M"), sig["close"], "空頭" if sig["bear"] else "多頭",
                         sig["hh"], sig["ll"], sig["xl"], sig["xh"], {1: "做多", -1: "做空", 0: "無"}[sig["entry"]])
                pos = positions.get(inst)
                if pos:
                    if (pos["dir"] == 1 and sig["exit_long"]) or (pos["dir"] == -1 and sig["exit_short"]):
                        if not self.cfg.dry_run:
                            self.close_position(inst, pos)
                            exposure -= pos["notional"]
                        else:
                            log.info("%s [DRY_RUN] 觸發出場", inst)
                    continue
                if sig["entry"] != 0 and not blocked:
                    if self.cfg.dry_run:
                        log.info("%s [DRY_RUN] 觸發%s進場, 停損 %.6g", inst, "多" if sig["entry"] == 1 else "空",
                                 sig["close"] - sig["entry"] * ATR_MULT * sig["atr"])
                    else:
                        self.open_position(inst, sig, equity, exposure)
                        exposure = sum(p["notional"] for p in self.positions().values())
            except Exception:
                log.exception("%s 處理失敗", inst)


def sleep_until_next_bar():
    now = time.time()
    nxt = (math.floor(now / 14400) + 1) * 14400 + 30  # 4H 收盤後 30 秒
    log.info("下次執行: %s UTC", datetime.fromtimestamp(nxt, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(nxt - now)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="執行一次後結束")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(HERE, "logs", "okx_bot.log"), encoding="utf-8")])
    cfg = Config()
    log.info("啟動: %s%s, 幣種 %s, 每筆風險 %.1f%%, 逐倉 %dx, 模式 %s", "模擬盤" if cfg.demo else "*** 實盤 ***",
             " (DRY_RUN 不下單)" if cfg.dry_run else "", ",".join(cfg.symbols), cfg.risk_pct * 100, cfg.leverage, cfg.mode)
    bot = Bot(cfg)
    bot.account_ready()
    if args.once:
        bot.run_once()
        return
    while True:
        bot.run_once()
        sleep_until_next_bar()


if __name__ == "__main__":
    main()

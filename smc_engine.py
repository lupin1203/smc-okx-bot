"""SMC 結構引擎 + 回測 (與 tradingview/smc_auto_structure.pine 同一套規則)。

規則摘要
- 主結構 swing: 左右各 swing_len 根確認的 pivot high / low
- 收盤突破 swing high/low: 方向翻轉 = CHoCH, 同方向 = BOS
- OB: 突破那段走勢起點 (最近的內部 pivot, 否則區間極值) 附近最後一根反向K棒
- 進場: 價格回到 OB (多: 碰到 OB 上緣 / 空: 碰到 OB 下緣) 以限價成交
- 停損: OB 另一側 (+緩衝); 停利: rr 倍風險
- 高週期過濾: 只做與高週期結構同方向的單 (只用已收盤的高週期K棒)

與 TradingView 指標唯一刻意的差異 (為了回測真實):
- 指標在「同一根K棒碰到 OB 又收盤穿過 OB」時不畫進場框 (先判定失效);
  真實掛單在盤中就會成交, 所以這裡先判斷成交, 再判斷 OB 失效。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Params:
    swing_len: int = 5
    int_len: int = 2
    rr: float = 2.0
    sl_buf_pct: float = 0.0
    use_htf: bool = True
    htf_swing: int = 5
    only_choch: bool = True
    max_zones: int = 5
    # "limit": 碰到 OB 就限價成交; "confirm": 碰到後等K棒收盤回到 OB 外才市價進場
    entry_mode: str = "limit"
    # 流動性掃蕩: CHoCH 的走勢起點必須先刺破前一個 swing 低(高)點, 並在 sweep_bars 根內收回
    require_sweep: bool = False
    sweep_bars: int = 3
    # 重大事件: "off" / "no_entry" (前後 N 小時不進場) / "flat" (再加上事件前平倉)
    news_mode: str = "off"
    # 成本 (OKX 永續一般等級): 限價 maker 0.02%, 停損市價 taker 0.05% + 滑價
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    sl_slippage: float = 0.0002


@dataclass
class Trade:
    dir: int
    entry_i: int
    entry: float
    sl: float
    tp: float
    exit_i: int = -1
    exit: float = np.nan
    r_gross: float = np.nan
    r_net: float = np.nan
    entry_fee: float = 0.0002


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("time")[["open", "high", "low", "close"]].astype(float)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def pivots(h: np.ndarray, l: np.ndarray, n: int):
    """回傳在第 i 根被確認的 pivot (中心在 i-n)。與 ta.pivothigh/low 相同: 中心為左右 n 根中的極值。"""
    size = len(h)
    ph = np.full(size, np.nan)
    pl = np.full(size, np.nan)
    w = 2 * n + 1
    if size < w:
        return ph, pl
    from numpy.lib.stride_tricks import sliding_window_view as swv
    hw = swv(h, w)
    lw = swv(l, w)
    ch = hw[:, n]
    cl = lw[:, n]
    is_ph = (ch >= hw.max(axis=1)) & (ch > hw[:, :n].max(axis=1))
    is_pl = (cl <= lw.min(axis=1)) & (cl < lw[:, :n].min(axis=1))
    idx = np.arange(w - 1, size)
    ph[idx[is_ph]] = ch[is_ph]
    pl[idx[is_pl]] = cl[is_pl]
    return ph, pl


def htf_trend(df_htf: pd.DataFrame, swing: int) -> pd.Series:
    """高週期結構方向 (每根高週期K棒收盤時的值)。"""
    h, l, c = (df_htf[k].to_numpy() for k in ("high", "low", "close"))
    ph, pl = pivots(h, l, swing)
    t, sh, sl = 0, np.nan, np.nan
    out = np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if not np.isnan(ph[i]):
            sh = ph[i]
        if not np.isnan(pl[i]):
            sl = pl[i]
        if not np.isnan(sh) and c[i] > sh:
            t, sh = 1, np.nan
        if not np.isnan(sl) and c[i] < sl:
            t, sl = -1, np.nan
        out[i] = t
    return pd.Series(out, index=df_htf.index)


def map_htf(df: pd.DataFrame, htf_rule: str, swing: int) -> np.ndarray:
    """對每根低週期K棒, 取「上一根已收盤高週期K棒」的方向 (= Pine 的 expr[1] + lookahead_on)。"""
    df_htf = resample(df, htf_rule)
    tr = htf_trend(df_htf, swing).shift(1).fillna(0).astype(int).to_numpy()
    pos = df_htf.index.searchsorted(df.index, side="right") - 1  # 每根K棒所屬的高週期K棒
    return np.where(pos >= 0, tr[np.clip(pos, 0, None)], 0)


def backtest(df: pd.DataFrame, htf_dir: np.ndarray, p: Params, blackout: np.ndarray | None = None) -> list[Trade]:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(c)
    ph, pl = pivots(h, l, p.swing_len)
    iph, ipl = pivots(h, l, p.int_len)

    sH = sL = np.nan
    sH_bar = sL_bar = -1
    sH_used = sL_used = True
    iH_bar = iL_bar = -1
    trend = 0
    zones: list[list] = []  # [top, bot, dir, active, traded, born, armed, extreme]
    trades: list[Trade] = []
    pos: Trade | None = None
    swl: list[tuple[int, float]] = []  # (確認K棒, 價位) 主結構 swing low 歷史
    swh: list[tuple[int, float]] = []
    if blackout is None or p.news_mode == "off":
        blackout = np.zeros(n, dtype=bool)

    def allow(d):
        return ((not p.use_htf) or htf_dir[i] == d) and not blackout[i]

    def swept(i, bull, broken_bar):
        """走勢起點是否掃過前一個 swing 點的流動性 (刺破後在 sweep_bars 根內收盤收回)。"""
        lo = max(broken_bar, 0)
        if bull:
            j = lo + int(np.argmin(l[lo:i + 1]))
            hist = swl
        else:
            j = lo + int(np.argmax(h[lo:i + 1]))
            hist = swh
        lvl = next((v for cb, v in reversed(hist) if cb < j), None)
        if lvl is None:
            return False
        if bull and not l[j] < lvl or not bull and not h[j] > lvl:
            return False
        end = min(j + p.sweep_bars, i + 1)
        return bool(np.any(c[j:end] > lvl)) if bull else bool(np.any(c[j:end] < lvl))

    def find_ob(i, bull, broken_bar):
        span = min(i - broken_bar, 500)
        ib = iL_bar if bull else iH_bar
        org = 0
        if ib > broken_bar and ib < i:
            org = i - ib
        if org == 0:
            for k in range(span + 1):
                if (l[i - k] < l[i - org]) if bull else (h[i - k] > h[i - org]):
                    org = k
        ob = org
        for k in range(org, min(org + 5, i) + 1):
            j = i - k
            if (c[j] < o[j]) if bull else (c[j] > o[j]):
                ob = k
                break
        return ob

    for i in range(n):
        # 1) swing 點
        if not np.isnan(ph[i]):
            sH, sH_bar, sH_used = ph[i], i - p.swing_len, False
            swh.append((i, ph[i]))
        if not np.isnan(pl[i]):
            sL, sL_bar, sL_used = pl[i], i - p.swing_len, False
            swl.append((i, pl[i]))
        if not np.isnan(iph[i]):
            iH_bar = i - p.int_len
        if not np.isnan(ipl[i]):
            iL_bar = i - p.int_len

        # 2a) 事件前平倉: 進入禁止交易時段的第一根K棒, 以開盤價市價平倉
        if p.news_mode == "flat" and pos is not None and blackout[i] and i > 0 and not blackout[i - 1]:
            _close(pos, i, o[i], False, p)
            trades.append(pos)
            pos = None

        # 2) 持倉出場 (同一根同時碰到 SL/TP → 保守算 SL)
        if pos is not None and i > pos.entry_i:
            hit_sl = l[i] <= pos.sl if pos.dir == 1 else h[i] >= pos.sl
            hit_tp = h[i] >= pos.tp if pos.dir == 1 else l[i] <= pos.tp
            if hit_sl:
                # 跳空穿過停損 → 以開盤價成交
                px = min(o[i], pos.sl) if pos.dir == 1 else max(o[i], pos.sl)
                _close(pos, i, px, False, p)
                trades.append(pos)
                pos = None
            elif hit_tp:
                _close(pos, i, pos.tp, True, p)
                trades.append(pos)
                pos = None

        # 3b) 確認進場: 已碰過 OB 的區塊, 收盤回到 OB 外 → 收盤價市價進場
        if p.entry_mode == "confirm":
            for z in zones:
                top, bot, d, active, traded, born, armed, ext = z
                if not active or traded or i <= born:
                    continue
                touched = (l[i] <= top) if d == 1 else (h[i] >= bot)
                if not armed and touched:
                    z[6], z[7] = True, (l[i] if d == 1 else h[i])
                elif armed:
                    z[7] = min(ext, l[i]) if d == 1 else max(ext, h[i])
                if not z[6]:
                    continue
                if d == 1 and c[i] <= bot or d == -1 and c[i] >= top:
                    continue  # 收盤穿過 → 下面第 4 步會判定失效
                confirmed = (c[i] > top) if d == 1 else (c[i] < bot)
                if not confirmed:
                    continue
                z[4] = True
                if pos is not None or not allow(d):
                    continue
                entry = c[i]
                if d == 1:
                    sl = min(bot, z[7]) * (1 - p.sl_buf_pct / 100)
                    tp = entry + p.rr * (entry - sl)
                else:
                    sl = max(top, z[7]) * (1 + p.sl_buf_pct / 100)
                    tp = entry - p.rr * (sl - entry)
                pos = Trade(d, i, entry, sl, tp, entry_fee=p.taker_fee + p.sl_slippage)

        # 3) 限價單成交: 價格回到仍有效的 OB
        for z in (zones if p.entry_mode == "limit" else []):
            top, bot, d, active, traded, born = z[:6]
            if not active or traded or i <= born:
                continue
            tapped = (l[i] <= top) if d == 1 else (h[i] >= bot)
            if not tapped:
                continue
            z[4] = True  # 每個 OB 只用一次
            if pos is not None or not allow(d):
                continue
            if d == 1:
                entry = min(top, o[i])  # 開盤就低於限價 → 以開盤價成交
                sl = bot * (1 - p.sl_buf_pct / 100)
                if entry <= sl:
                    continue
                tp = entry + p.rr * (entry - sl)
            else:
                entry = max(bot, o[i])
                sl = top * (1 + p.sl_buf_pct / 100)
                if entry >= sl:
                    continue
                tp = entry - p.rr * (sl - entry)
            pos = Trade(d, i, entry, sl, tp)
            # 成交當根就碰到停損 → 算停損 (保守)
            if (l[i] <= sl) if d == 1 else (h[i] >= sl):
                _close(pos, i, sl, False, p)
                trades.append(pos)
                pos = None

        # 4) OB 失效 (收盤穿過)
        for z in zones:
            if z[3] and ((c[i] < z[1]) if z[2] == 1 else (c[i] > z[0])):
                z[3] = False

        # 5) 結構突破 → CHoCH / BOS → 新 OB
        if not sH_used and c[i] > sH:
            is_choch = trend <= 0
            ok_sweep = (not p.require_sweep) or swept(i, True, sH_bar)
            if (is_choch or not p.only_choch) and ((not p.use_htf) or htf_dir[i] == 1) and ok_sweep:
                k = find_ob(i, True, sH_bar)
                zones.append([h[i - k], l[i - k], 1, True, False, i, False, np.nan])
            sH_used, trend = True, 1
        if not sL_used and c[i] < sL:
            is_choch = trend >= 0
            ok_sweep = (not p.require_sweep) or swept(i, False, sL_bar)
            if (is_choch or not p.only_choch) and ((not p.use_htf) or htf_dir[i] == -1) and ok_sweep:
                k = find_ob(i, False, sL_bar)
                zones.append([h[i - k], l[i - k], -1, True, False, i, False, np.nan])
            sL_used, trend = True, -1
        if len(zones) > p.max_zones:
            del zones[: len(zones) - p.max_zones]
    return trades


def _close(t: Trade, i: int, px: float, is_tp: bool, p: Params):
    t.exit_i, t.exit = i, px
    risk = abs(t.entry - t.sl)
    move = (px - t.entry) * t.dir
    t.r_gross = move / risk
    exit_cost = p.maker_fee if is_tp else (p.taker_fee + p.sl_slippage)
    cost = t.entry * t.entry_fee + px * exit_cost
    t.r_net = (move - cost) / risk


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return dict(trades=0, win_rate=np.nan, avg_r=np.nan, total_r=0.0, pf=np.nan, max_dd_r=0.0, avg_risk_pct=np.nan)
    r = np.array([t.r_net for t in trades])
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.concatenate([[0], eq])) - np.concatenate([[0], eq])).max()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return dict(
        trades=len(r),
        win_rate=float((r > 0).mean()),
        avg_r=float(r.mean()),
        avg_r_gross=float(np.mean([t.r_gross for t in trades])),
        total_r=float(r.sum()),
        pf=float(gains / losses) if losses > 0 else np.inf,
        max_dd_r=float(dd),
        avg_risk_pct=float(np.mean([abs(t.entry - t.sl) / t.entry for t in trades]) * 100),
    )

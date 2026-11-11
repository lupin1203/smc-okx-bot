"""單元測試 (不連網、不需要 API Key)：python -m pytest -q"""
import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

import okx_bot
from events import blackout_mask, event_times_utc
from smc_engine import pivots
from trend import TrendParams, backtest_trend


def make_4h(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c}, index=idx)


def uptrend(n=1500, breakout=True):
    c = np.linspace(100, 200, n)
    if breakout:
        c[-1] = c[-2] * 1.05
    return make_4h(c)


# ───────── 訊號 ─────────
def test_signal_long_breakout_in_bull_market():
    sig = okx_bot.compute_signal(uptrend(), "regime")
    assert sig["bear"] is False
    assert sig["entry"] == 1
    assert sig["atr"] > 0


def test_signal_no_short_in_long_only_mode():
    c = np.linspace(200, 100, 1500)
    c[-1] = c[-2] * 0.95   # 最後一根跌破前 55 根最低價
    df = make_4h(c)
    assert okx_bot.compute_signal(df, "regime")["entry"] == -1
    assert okx_bot.compute_signal(df, "long_only")["entry"] == 0


def test_signal_needs_200_daily_bars():
    with pytest.raises(RuntimeError):
        okx_bot.compute_signal(uptrend(n=600), "regime")


def test_signal_uses_only_prior_bars_for_channel():
    """突破線只能用「前」55 根, 不含當根: 最後一根創新高時 hh 一定小於收盤。"""
    sig = okx_bot.compute_signal(uptrend(), "regime")
    assert sig["hh"] < sig["close"]


# ───────── 事件 ─────────
def test_blackout_covers_event_window_only():
    ev = event_times_utc()[0]
    idx = pd.DatetimeIndex([ev - pd.Timedelta(hours=6), ev - pd.Timedelta(hours=1),
                            ev + pd.Timedelta(hours=3), ev + pd.Timedelta(hours=5)])
    assert list(blackout_mask(idx, 4.0, pd.Timedelta(minutes=15))) == [False, True, True, False]


def test_calendar_days_left():
    last = pd.Timestamp(min(max(okx_bot.CPI), max(okx_bot.NFP), max(okx_bot.FOMC)), tz="UTC")
    assert okx_bot.calendar_days_left(last - pd.Timedelta(days=10)) == 10
    assert okx_bot.calendar_days_left(last + pd.Timedelta(days=3)) < 0


# ───────── 下單安全 ─────────
class FakeCfg:
    key = secret = passphrase = "x"
    demo, dry_run, leverage, risk_pct, max_exposure = True, False, 3, 0.01, 2.0


def _http_error(code, body=b""):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))


def test_post_is_not_retried_on_timeout(monkeypatch):
    """下單逾時時請求可能已經成交, 重送會變成重複下單。"""
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.get_method())
        raise TimeoutError("timed out")

    monkeypatch.setattr(okx_bot.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(okx_bot.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        okx_bot.OKX(FakeCfg()).post("/api/v5/trade/order", {"instId": "BTC-USDT-SWAP"})
    assert calls == ["POST"]


def test_post_is_not_retried_on_5xx_but_is_on_429(monkeypatch):
    monkeypatch.setattr(okx_bot.time, "sleep", lambda s: None)
    calls = []

    def always_502(req, timeout):
        calls.append(1)
        raise _http_error(502, b"<html>bad gateway</html>")   # 不是 JSON 也不能讓程式當掉

    monkeypatch.setattr(okx_bot.urllib.request, "urlopen", always_502)
    with pytest.raises(RuntimeError, match="http502"):
        okx_bot.OKX(FakeCfg()).post("/api/v5/trade/order", {})
    assert len(calls) == 1

    calls.clear()

    def limited_then_ok(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429, json.dumps({"code": "50011", "msg": "rate limit"}).encode())
        return io.BytesIO(json.dumps({"code": "0", "data": [{"ordId": "1"}]}).encode())

    monkeypatch.setattr(okx_bot.urllib.request, "urlopen", limited_then_ok)
    assert okx_bot.OKX(FakeCfg()).post("/api/v5/trade/order", {})[0]["ordId"] == "1"
    assert len(calls) == 3


def test_get_is_retried_on_network_error(monkeypatch):
    monkeypatch.setattr(okx_bot.time, "sleep", lambda s: None)
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("connection reset")
        return io.BytesIO(json.dumps({"code": "0", "data": []}).encode())

    monkeypatch.setattr(okx_bot.urllib.request, "urlopen", flaky)
    assert okx_bot.OKX(FakeCfg()).get("/api/v5/market/candles") == []
    assert len(calls) == 3


class RecordingOKX:
    def __init__(self):
        self.posts = []

    def instrument(self, inst):
        return {"ctVal": "0.01", "lotSz": "0.01", "minSz": "0.01", "tickSz": "0.1"}

    def post(self, path, body):
        self.posts.append((path, body))
        return [{"ordId": "1"}]


def _bot(leverage=3):
    cfg = FakeCfg()
    cfg.leverage = leverage
    bot = okx_bot.Bot.__new__(okx_bot.Bot)
    bot.cfg, bot.okx, bot.pos_mode = cfg, RecordingOKX(), "net_mode"
    return bot


def test_open_position_sizes_by_risk_and_attaches_stop():
    bot = _bot()
    bot.open_position("BTC-USDT-SWAP", {"entry": 1, "close": 50000.0, "atr": 500.0}, equity=10000.0, exposure_now=0.0)
    path, order = bot.okx.posts[-1]
    assert path == "/api/v5/trade/order"
    # 風險 100 USDT ÷ (停損距離 1000 × 每張 0.01 BTC) = 10 張
    assert float(order["sz"]) == pytest.approx(10.0)
    assert order["side"] == "buy"
    assert float(order["attachAlgoOrds"][0]["slTriggerPx"]) == pytest.approx(49000.0)


def test_open_position_respects_exposure_cap():
    bot = _bot()
    bot.open_position("BTC-USDT-SWAP", {"entry": -1, "close": 50000.0, "atr": 50.0}, equity=10000.0, exposure_now=19000.0)
    order = bot.okx.posts[-1][1]
    assert float(order["sz"]) * 0.01 * 50000.0 <= 1000.0 + 1e-6   # 剩餘額度 = 2 × 10000 − 19000
    assert order["side"] == "sell"
    assert float(order["attachAlgoOrds"][0]["slTriggerPx"]) == pytest.approx(50100.0)


def test_open_position_skips_when_stop_is_beyond_liquidation():
    bot = _bot(leverage=10)   # 強平約 10%; 停損 2 × ATR = 8% → 超過 70% 門檻
    bot.open_position("BTC-USDT-SWAP", {"entry": 1, "close": 50000.0, "atr": 2000.0}, equity=10000.0, exposure_now=0.0)
    assert bot.okx.posts == []


def test_open_position_skips_on_nan_atr():
    bot = _bot()
    bot.open_position("BTC-USDT-SWAP", {"entry": 1, "close": 50000.0, "atr": float("nan")}, equity=10000.0, exposure_now=0.0)
    assert bot.okx.posts == []


# ───────── 回測引擎 ─────────
def test_trend_backtest_enters_next_open_and_exits_on_stop():
    c = np.r_[np.full(30, 100.0), 110.0, 111.0, 90.0, 90.0]
    df = make_4h(c)
    trades = backtest_trend(df, TrendParams(entry_n=20, exit_n=10, atr_mult=2.0))
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_i == 31 and t.entry == pytest.approx(111.0)   # 第 30 根收盤突破 → 第 31 根開盤進場
    assert t.exit_i == 32 and t.r_net < 0                         # 第 32 根跌破停損


def test_pivots_are_confirmed_n_bars_later():
    h = np.array([1, 2, 5, 2, 1, 1, 1], dtype=float)
    l = h - 0.5
    ph, _ = pivots(h, l, 2)
    assert ph[4] == 5 and np.isnan(ph[:4]).all()   # 高點在第 2 根, 右邊 2 根走完 (第 4 根) 才確認

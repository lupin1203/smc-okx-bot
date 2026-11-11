"""跑 SMC 回測: 預設參數 + 參數掃描 + 樣本外驗證。

用法: python run_backtest.py
前一年 = 樣本內 (選參數), 後一年 = 樣本外 (驗證選出的參數)。
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from events import blackout_mask
from smc_engine import Params, backtest, load_csv, map_htf, resample, stats

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "ADA-USDT-SWAP"]
SETUPS = [("15min", "1h"), ("1h", "4h")]  # (圖表週期, 高週期)

GRID = dict(
    swing_len=[5, 8, 10],
    int_len=[2, 3],
    rr=[1.5, 2.0, 3.0],
    use_htf=[True, False],
    only_choch=[True],
    entry_mode=["limit", "confirm"],
    require_sweep=[False, True],
    news_mode=["off", "no_entry", "flat"],
)

_cache = {}


def get_data(sym, tf, htf):
    key = (sym, tf, htf)
    if key not in _cache:
        df = load_csv(os.path.join(HERE, "data", f"{sym}_15m.csv"))
        if tf != "15min":
            df = resample(df, tf)
        _cache[key] = df
    return _cache[key]


def run_one(args):
    sym, tf, htf, kw = args
    df = get_data(sym, tf, htf)
    p = Params(**kw)
    htf_dir = map_htf(df, htf, p.htf_swing)
    trades = backtest(df, htf_dir, p, blackout_mask(df.index, 4.0))
    split = df.index[0] + (df.index[-1] - df.index[0]) / 2
    rows = []
    for name, sel in (("all", lambda t: True),
                      ("in", lambda t: df.index[t.entry_i] < split),
                      ("out", lambda t: df.index[t.entry_i] >= split)):
        s = stats([t for t in trades if sel(t)])
        rows.append(dict(symbol=sym.split("-")[0], tf=tf, htf=htf, period=name, **kw, **s))
    return rows


def main():
    jobs = []
    for sym, (tf, htf) in itertools.product(SYMBOLS, SETUPS):
        for combo in itertools.product(*GRID.values()):
            jobs.append((sym, tf, htf, dict(zip(GRID.keys(), combo))))
    with ProcessPoolExecutor() as ex:
        rows = [r for res in ex.map(run_one, jobs, chunksize=4) for r in res]
    res = pd.DataFrame(rows)
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    res.to_csv(os.path.join(out, "grid_results.csv"), index=False)
    print("saved", len(res), "rows")


if __name__ == "__main__":
    main()

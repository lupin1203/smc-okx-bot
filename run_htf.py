"""4H / 日線回測: SMC 規則 + 趨勢策略。結果 → results/htf_results.csv"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from smc_engine import Params, backtest, load_csv, map_htf, resample, stats
from trend import TrendParams, backtest_trend, equity_stats

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "ADA"]
SETUPS = [("4h", "1D"), ("1D", "W-MON")]


def load(sym, tf):
    df = load_csv(os.path.join(HERE, "data", f"{sym}-USDT-SWAP_4H.csv"))
    return df if tf == "4h" else resample(df, tf)


def rows_for(df, trades, base):
    split = df.index[0] + (df.index[-1] - df.index[0]) / 2
    out = []
    for name, sel in (("all", lambda t: True), ("in", lambda t: df.index[t.entry_i] < split),
                      ("out", lambda t: df.index[t.entry_i] >= split)):
        sub = [t for t in trades if sel(t)]
        cagr, mdd = equity_stats(df, sub) if name == "all" else (None, None)
        out.append(dict(**base, period=name, **stats(sub), cagr_1pct=cagr, mdd_1pct=mdd))
    return out


def job(args):
    kind, sym, tf, htf, kw = args
    df = load(sym, tf)
    base = dict(strategy=kind, symbol=sym, tf=tf, **kw)
    if kind == "smc":
        p = Params(**kw)
        trades = backtest(df, map_htf(df, htf, p.htf_swing), p)
    else:
        trades = backtest_trend(df, TrendParams(**kw))
    return rows_for(df, trades, base)


def main():
    jobs = []
    smc_grid = dict(swing_len=[3, 5, 8], int_len=[2, 3], rr=[2.0, 3.0], use_htf=[True, False],
                    entry_mode=["limit", "confirm"], require_sweep=[False, True])
    trend_grid = dict(entry_n=[20, 55], exit_n=[10, 20], atr_mult=[2.0, 3.0], both=[False, True])
    for sym, (tf, htf) in itertools.product(SYMBOLS, SETUPS):
        for combo in itertools.product(*smc_grid.values()):
            jobs.append(("smc", sym, tf, htf, dict(zip(smc_grid, combo))))
        for combo in itertools.product(*trend_grid.values()):
            jobs.append(("trend", sym, tf, htf, dict(zip(trend_grid, combo))))
    with ProcessPoolExecutor() as ex:
        rows = [r for res in ex.map(job, jobs, chunksize=4) for r in res]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "htf_results.csv"), index=False)
    # 買進持有對照
    for sym in SYMBOLS:
        df = load(sym, "1D")
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        bh = (df.close.iloc[-1] / df.open.iloc[0]) ** (1 / yrs) - 1
        mdd = (1 - df.close / df.close.cummax()).max()
        print(f"buy&hold {sym}: {df.index[0].date()}~ CAGR {bh:.1%}  MDD {mdd:.1%}")
    print("saved", len(rows))


if __name__ == "__main__":
    main()

"""槓桿測試: 固定百分比停損 + 全倉名目 N 倍 (逐倉, 保證金 = 該幣分配到的資金)。

資金平均分成 5 份 (每個幣一份, 不再平衡), 每筆用「該份資金 x 槓桿」的名目倉位。
- 每筆盈虧 = 槓桿 x 價格報酬 (已含手續費、滑價、資金費率)
- 爆倉: 虧損達到 (1 - 維持保證金率 x 槓桿) → 該份資金歸零
用法: python leverage_test.py [停損%]
"""
import sys

import numpy as np

from run_htf import SYMBOLS, load
from trend import TrendParams, backtest_trend

MMR = 0.005  # OKX 主流幣維持保證金率約 0.4~0.5%


def run(stop_pct, entry_n, exit_n, regime, levers):
    p = TrendParams(entry_n=entry_n, exit_n=exit_n, regime=regime, fixed_stop_pct=stop_pct)
    events, rets = [], []
    for s in SYMBOLS:
        df = load(s, "4h")
        for t in backtest_trend(df, p):
            ret = t.r_net * abs(t.entry - t.sl0) / t.entry  # 價格報酬 (含成本)
            events.append((df.index[t.exit_i], s, ret))
            rets.append(ret)
    events.sort()
    rets = np.array(rets)
    years = (events[-1][0] - events[0][0]).days / 365.25
    print(f"\n=== 4H 突破{entry_n}/出場{exit_n} {'多頭做多/空頭做空' if regime == 'regime' else '只做多'} 停損{stop_pct}% ===")
    print(f"交易 {len(rets)} 筆, 勝率 {np.mean(rets > 0):.0%}, 停損單平均虧 {-rets[rets < 0].mean():.2%}, "
          f"最大單筆虧損 {-rets.min():.2%} (跳空), 每年 {len(rets) / years:.0f} 筆")
    print(f"{'槓桿':>4} {'總資產':>9} {'年化':>6} {'組合最大回撤':>8} {'爆倉/歸零幣':>8} {'每次停損虧':>8}")
    for L in levers:
        eq = {s: 1.0 for s in SYMBOLS}
        peak, mdd, liq = 1.0, 0.0, 0
        for _, s, ret in events:
            if eq[s] <= 0:
                continue
            pnl = L * ret
            if pnl <= -(1 - MMR * L):
                pnl = -1.0
                liq += 1
            eq[s] *= 1 + pnl
            if eq[s] < 1e-4:
                eq[s] = 0.0
            tot = sum(eq.values()) / len(eq)
            peak = max(peak, tot)
            mdd = max(mdd, 1 - tot / peak)
        tot = sum(eq.values()) / len(eq)
        dead = sum(v == 0 for v in eq.values())
        cagr = tot ** (1 / years) - 1 if tot > 0 else -1
        print(f"{L:>3}x {tot:>8.1f}倍 {cagr:>6.0%} {mdd:>10.0%} {f'{liq}次/{dead}個':>10} {L * stop_pct:>8.0f}%")


if __name__ == "__main__":
    stop = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    for n, m in [(55, 20), (20, 10)]:
        for regime in ["regime", "none"]:
            run(stop, n, m, regime, [1, 3, 5, 10, 15])

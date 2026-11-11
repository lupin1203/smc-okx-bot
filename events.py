"""美國重大經濟事件 (CPI / 非農 NFP / FOMC 利率決議) 與禁止交易時段。

日期來源:
- CPI、NFP: BLS 官方發布紀錄 bls.gov/bls/news-release/cpi.htm、empsit.htm
  (2025/10 因政府關門取消的 CPI 與 NFP 不列入; 2026/11/10 之後為預定日期)
- FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm (會議最後一天 = 決議公布日)
時間: CPI、NFP 美東 08:30; FOMC 美東 14:00 (自動處理夏令時間)
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

CPI = """2024-11-13 2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11
2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10
2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14 2026-11-10 2026-12-10""".split()

NFP = """2024-11-01 2024-12-06 2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06
2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16 2026-01-09 2026-02-11 2026-03-06 2026-04-03
2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02 2026-11-06 2026-12-04""".split()

FOMC = """2024-11-07 2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17
2025-10-29 2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16
2026-10-28 2026-12-09""".split()


def event_times_utc() -> list[pd.Timestamp]:
    out = []
    for dates, hh, mm in ((CPI, 8, 30), (NFP, 8, 30), (FOMC, 14, 0)):
        for d in dates:
            y, m, dd = map(int, d.split("-"))
            out.append(pd.Timestamp(datetime(y, m, dd, hh, mm, tzinfo=ET)).tz_convert("UTC"))
    return sorted(out)


def blackout_mask(index: pd.DatetimeIndex, hours: float = 4.0, bar: pd.Timedelta | None = None) -> np.ndarray:
    """K棒時間區間 [開盤, 收盤) 只要和 [事件-hours, 事件+hours] 重疊就標記為禁止交易。"""
    if bar is None:
        bar = index[1] - index[0]
    index = index.as_unit("ns")
    starts = index.asi8
    ends = (index + bar).asi8
    mask = np.zeros(len(index), dtype=bool)
    w = pd.Timedelta(hours=hours)
    for t in event_times_utc():
        lo, hi = (t - w).value, (t + w).value
        mask |= (ends > lo) & (starts < hi)
    return mask

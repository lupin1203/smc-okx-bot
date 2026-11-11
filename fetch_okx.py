"""從 OKX 公開 API 下載歷史 K 線 (不需要 API Key)。

用法: python fetch_okx.py BTC-USDT-SWAP 15m 730
輸出: data/<instId>_<bar>.csv  (ts 為 UTC 開盤時間)
"""
import csv
import os
import sys
import time
import json
import urllib.request

BASE = "https://www.okx.com/api/v5/market/history-candles"


def fetch(inst: str, bar: str, days: int, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{inst}_{bar}.csv")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows = {}
    after = ""
    while True:
        url = f"{BASE}?instId={inst}&bar={bar}&limit=100" + (f"&after={after}" if after else "")
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "smc-backtest/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.load(r)
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("OKX request failed: " + url)
        if data.get("code") != "0":
            if data.get("code") == "50011":  # rate limit
                time.sleep(1)
                continue
            raise RuntimeError(data)
        batch = data["data"]
        if not batch:
            break
        for k in batch:
            if k[8] == "1":  # 只要已收盤的K棒
                rows[int(k[0])] = k[:6]
        oldest = int(batch[-1][0])
        if oldest <= start_ms:
            break
        after = str(oldest)
        time.sleep(0.12)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "vol"])
        for ts in sorted(rows):
            if ts >= start_ms:
                w.writerow(rows[ts])
    return path


if __name__ == "__main__":
    inst, bar, days = sys.argv[1], sys.argv[2], int(sys.argv[3])
    p = fetch(inst, bar, days, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    print("saved", p)

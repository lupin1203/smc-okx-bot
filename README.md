# SMC 回測 (OKX 永續)

與 `tradingview/smc_auto_structure.pine` 同一套規則的 Python 版本與回測。

```bash
python fetch_okx.py BTC-USDT-SWAP 15m 730   # 下載 2 年 15m K線到 data/ (公開 API, 不需 Key)
python fetch_okx.py ETH-USDT-SWAP 15m 730
python run_backtest.py                      # 參數掃描 → results/grid_results.csv
```

- `smc_engine.py`：結構偵測 (swing / CHoCH / OB)、高週期過濾、逐K回測
- 成本：限價進場與停利 maker 0.02%，停損 taker 0.05% + 0.02% 滑價
- 同一根K棒同時碰到停損與停利 → 算停損；成交當根就碰到停損 → 算停損
- 前一年為樣本內、後一年為樣本外 (`period` 欄位: all / in / out)

## OKX 自動交易機器人 (`okx_bot.py`)

4H 趨勢策略：突破 55 根高/低點進場、2×ATR 停損 (下單時直接掛在 OKX)、跌破/突破 20 根出場、
日線 SMA200 判斷多空、CPI/NFP/FOMC 前後 4 小時不開新倉。**預設連 OKX 模擬盤。**

### 1. 在 OKX 建立 API Key
- 先在 **模擬交易** 頁面建立模擬盤 API Key (實盤要另外建)
- 權限只勾 **讀取 + 交易**，**不要勾提幣**；建議綁定你電腦的 IP

### 2. 設定環境變數 (金鑰只放在自己電腦，不要貼到聊天或 commit)
Windows (PowerShell，設定後要重開視窗):
```powershell
setx OKX_API_KEY "你的API Key"
setx OKX_API_SECRET "你的Secret Key"
setx OKX_API_PASSPHRASE "你的Passphrase"
setx OKX_DEMO "1"
```
macOS / Linux (寫進 ~/.zshrc 或 ~/.bashrc):
```bash
export OKX_API_KEY="你的API Key"
export OKX_API_SECRET="你的Secret Key"
export OKX_API_PASSPHRASE="你的Passphrase"
export OKX_DEMO=1
```
其他可選: `RISK_PCT` (預設 0.01)、`LEVERAGE` (3)、`MAX_EXPOSURE` (2.0)、`STRATEGY_MODE` (regime / long_only)、`SYMBOLS`、`DRY_RUN=1` (只算訊號不下單)

### 3. 執行
```bash
pip install pandas numpy
python okx_bot.py --once   # 跑一次看訊號與帳戶狀態
python okx_bot.py          # 常駐，每根 4H K棒收盤後 30 秒執行
```
紀錄寫在 `logs/okx_bot.log`。模擬盤跑 1–2 個月確認正常後，才把 `OKX_DEMO` 改成 `0` 並換成實盤 Key。

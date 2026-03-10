# CAR Data Pull — Implementation Plan

## Background

Your dissertation needs acquirer **Cumulative Abnormal Returns (CAR)** using a market-model event study. The plan in [CAR-plan.txt](file:///c:/Users/u2512658/hardjoshi-ma/CAR-plan.txt) is methodologically sound — OLS on an estimation window `[-200, -20]`, then abnormal returns in the event window `[-5, +5]`. The Python script [pull_car_data.py](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py) implements this data-pull step but has several issues that need fixing before it can run.

---

## User Review Required

> [!IMPORTANT]
> **Bloomberg terminal access is required.** The script uses `xbbg` (Bloomberg Python API) which **only works when a Bloomberg Terminal is running** on the same machine. Please confirm:
> 1. Will you run this script on a machine with Bloomberg Terminal active?
> 2. Is Python + `xbbg` installed on that Bloomberg machine, or do you need install instructions?

> [!IMPORTANT]
> **Python is not installed on this machine.** The `python` command is not available. If you plan to run the script here, you'll need to install Python first (e.g. via the Microsoft Store or python.org). Alternatively, you may intend to run it on a different machine (Bloomberg workstation).

> [!WARNING]
> **No `MA_Deals.xlsx` file exists.** The script expects this file, but it doesn't exist. Your actual deal data is in [combined_financial_text.csv](file:///c:/Users/u2512658/hardjoshi-ma/combined_financial_text.csv) (4,999 deals) and [ma_export_33205636_175852.csv](file:///c:/Users/u2512658/hardjoshi-ma/ma_export_33205636_175852.csv) (5,000 deals). I will update the script to read from one of these.

---

## Issues Found in Current Script

### 1. Wrong input file & column names
- Script expects `MA_Deals.xlsx` with columns: `Deal_ID`, `Acquirer_Ticker`, `Announcement_Date`
- Actual data has: `Announce Date`, `Acquirer Ticker`, `deal_key` (format: `"TICKER|DATE"`)
- **Fix**: Read from [combined_financial_text.csv](file:///c:/Users/u2512658/hardjoshi-ma/combined_financial_text.csv), extract acquirer ticker and announcement date

### 2. `Rel_Day` uses calendar days instead of trading days
- Line 83: `combined["Rel_Day"] = (combined.index - closest_day).days` counts **calendar** days
- This means weekends/holidays shift windows — e.g. event window may capture 7–8 trading days instead of 11
- **Fix**: Assign `Rel_Day` using integer index position (trading-day rank) relative to Day 0

### 3. [add_trading_days()](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py#16-21) approximation is unreliable
- Multiplying by 1.4 to guess calendar-day offset from trading-day offset is imprecise
- Over 200 days, the error can be 5–10 trading days
- **Fix**: Over-pull by a wider margin (e.g. `× 1.5 + 30`), then trim precisely by `Rel_Day`

### 4. No `Deal_ID` generation
- Deals in the CSV lack a numeric `Deal_ID`
- **Fix**: Use the existing `deal_key` column (`"TICKER|DATE"`) as the deal identifier, or auto-assign an integer

### 5. Day 0 convention not robust
- Current code uses `get_indexer(..., method="nearest")` which may snap to the wrong day for holiday announcements
- **Fix**: Use first trading day on/after `announce_date` (forward-fill logic)

---

## Proposed Changes

### Data Preparation (deals master)

#### [MODIFY] [pull_car_data.py](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py)

Complete rewrite of the script to:

1. **Read [combined_financial_text.csv](file:///c:/Users/u2512658/hardjoshi-ma/combined_financial_text.csv)** → extract `Acquirer Ticker`, `Announce Date`, and `deal_key`
2. **Build `deals_master.csv`** with columns: `deal_id`, `deal_key`, `acq_ticker_bbg`, `announce_date`, `benchmark`
3. **For each deal**, pull Bloomberg prices for acquirer + SPX Index over `[-200, +5]` trading day window (over-pulled via calendar approximation, trimmed precisely after)
4. **Compute log returns**, assign `rel_day` in **trading-day** increments, flag `EST`/`EVENT`/`GAP` windows
5. **Output tidy long-format table** to `timeseries_long.csv` with schema:
   - `deal_id`, `security_role`, `security`, `trading_date`, `px_last`, `ret_1d`, `ann_date`, `rel_day`, `window_flag`

Key changes:
- Fix `Rel_Day` to use trading-day rank (not calendar days)
- Fix Day 0 to use first trading day ≥ announce date
- Fix over-pull margin for date boundaries
- Add quality checks (min 120 estimation-window days, aligned dates)
- Add error handling and progress logging

#### [NEW] [deals_master.csv](file:///c:/Users/u2512658/hardjoshi-ma/deals_master.csv)

Extracted deal list with just the columns needed for CAR computation.

#### [NEW] [timeseries_long.csv](file:///c:/Users/u2512658/hardjoshi-ma/timeseries_long.csv)

Long/tidy price + return data for all deals, both acquirer and benchmark.

---

## Verification Plan

### Automated Checks (built into script)
- Print count of deals successfully pulled vs skipped
- Validate each deal has ≥ 120 estimation-window observations
- Validate event window has exactly 11 rows (days -5 to +5)
- Check no NaN returns in event window

### Manual Verification
1. After running: open `deals_master.csv` — confirm deal count matches expected
2. Open `timeseries_long.csv` — spot-check 2–3 deals:
   - `rel_day` should go from -200 to +5 (with possible gaps for estimation data)
   - Event window (`rel_day` -5 to +5) should have exactly 11 rows per deal per security
   - Returns should be reasonable (absolute value < 0.20 for most days)
3. Cross-check one deal's acquirer price against Bloomberg manually

> [!NOTE]
> The actual CAR calculation (OLS regression + abnormal return summation) is **Step 5** in [CAR-plan.txt](file:///c:/Users/u2512658/hardjoshi-ma/CAR-plan.txt) and will be done **after** this data pull step. This script only produces the inputs for that step.

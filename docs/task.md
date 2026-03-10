# CAR Data Pull —## Phase 1: Establish yfinance baseline data (2,153 deals)
- [x] Fix [pull_car_data.py](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py) to read incorrect input file
- [x] Adopt rigorous trading-day calculation for [rel_day](file:///c:/Users/u2512658/hardjoshi-ma/retry_failed_tickers.py#50-60) and Event Windows
- [x] Circumvent manual calendar date counting by matching first trading day
- [x] Download baseline price data via yfinance where possible

## Phase 2: Pull Delisted Tickers from Bloomberg Excel Add-In (2,357 deals)
- [x] Map the remaining non-yfinance tickers to a Bloomberg formatted request list
- [x] Write Python script [generate_bbg_excel.py](file:///c:/Users/u2512658/hardjoshi-ma/generate_bbg_excel.py) to generate native BDH formula requests inside an .xlsx file
- [x] Execute native BDH formula refresh within the Excel environment
- [x] Extract refreshed Bloomberg prices using [merge_bbg_data.py](file:///c:/Users/u2512658/hardjoshi-ma/merge_bbg_data.py)
- [x] Standardize multi-source date formats (Bloomberg YYYY-MM-DD vs yfinance DD/MM/YYYY) via [fix_dates.py](file:///c:/Users/u2512658/hardjoshi-ma/fix_dates.py)
- [x] Verify data integrity across merged sources with [verify_sources.py](file:///c:/Users/u2512658/hardjoshi-ma/verify_sources.py)

## Phase 3: CAR Calculation
- [x] Run OLS Market Model (alpha, beta) on the Estimation window [-200, -20]
- [x] Filter out datasets with insufficient representation (< 120 estimation days)
- [x] Apply model to Abnormal Returns (AR) calculation in Event Window [-5, +5]
- [x] Sum AR into Cumulative Abnormal Returns (CAR)
- [x] Merge results back directly into standard [combined_financial_text.csv](file:///c:/Users/u2512658/hardjoshi-ma/combined_financial_text.csv) output schema
- [x] *Total computed successful deals: 4,509 (90% dataset coverage)*PI with blpapi for Python 3.12 or earlier).

# CAR Data Pull -- Walkthrough

## Phase 1: yfinance (Complete)
Rewrote [pull_car_data.py](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py) to reliably pull stock data for actively traded tickers.
- **Deals with data**: 2,153

---

## Phase 2: Bloomberg Excel BDH (Complete)
Used Excel BDH formulas via [generate_bbg_excel.py](file:///c:/Users/u2512658/hardjoshi-ma/generate_bbg_excel.py) to retrieve data for delisted/historical tickers blocked by yfinance limits.
Fixed mixed date formatting ([fix_dates.py](file:///c:/Users/u2512658/hardjoshi-ma/fix_dates.py)) and rigorously verified zero column-spillover via benchmark mapping ([verify_sources.py](file:///c:/Users/u2512658/hardjoshi-ma/verify_sources.py)). 
- **Bloomberg deals recovered**: 2,357
- **Total Combined Deals**: 4,510 (90.2% coverage)
- **timeseries_long.csv**: **1,726,932 rows** containing normalized Phase 1 & Phase 2 data. Now also includes `expected_return` and `abnormal_return` metrics specifically for the 11-day Event Window of each Acquirer.

---

## Phase 3: CAR Computation (Complete)

Built [compute_car.py](file:///c:/Users/u2512658/hardjoshi-ma/compute_car.py) strictly following the [CAR-plan.txt](file:///c:/Users/u2512658/hardjoshi-ma/CAR-plan.txt) methodology:
1. **Pivoted data** by relative trading day ([rel_day](file:///c:/Users/u2512658/hardjoshi-ma/pull_car_data.py#68-87)) to align Acquirer and Benchmark prices exactly.
2. **OLS Regression**: Calculated `alpha_hat` and `beta_hat` parameters modeled on the Estimation Window (`Days -200 to -20`). Minimum 120 valid trading days enforced.
3. **Abnormal Return (AR)**: Measured expected baseline return `ER = alpha + beta(Rm)` against the actual return in the Event Window (`Days -5 to +5`).
4. **Cumulative Abnormal Return (CAR)**: Summed the ARs over the 11-day Event Window.

### Output Files Generated
1. **[car_results.csv](file:///c:/Users/u2512658/hardjoshi-ma/car_results.csv)**: Just the mathematical result table mapping `deal_id` to its parameters (`alpha_hat`, `beta_hat`, `car_m5_p5`, estimation day count).
2. **[final_car_dataset.csv](file:///c:/Users/u2512658/hardjoshi-ma/final_car_dataset.csv)**: The **FINAL Deliverable**. It matches your original 4,999 row dataset [combined_financial_text.csv](file:///c:/Users/u2512658/hardjoshi-ma/combined_financial_text.csv) but natively appends the new CAR columns to the end.

### CAR Diagnostic Statistics
* **Count**: Data available for 4,509 deals 
* **Mean CAR**: -0.87% *(Acquirer typically drops slightly upon M&A announcement, this is highly realistic)*
* **Mean Betas**: Realistic spread ranging from ~0.5 to 1.5 across diverse equities.

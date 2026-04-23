import pandas as pd
import numpy as np
from scipy import stats
import sys

print("="*60)
print("  COMPUTING CUMULATIVE ABNORMAL RETURNS (CAR)")
print("="*60)

# Load data
print("Loading timeseries_long.csv...")
ts = pd.read_csv("timeseries_long.csv")
print(f"Loaded {len(ts):,} rows.")

# We need to process each deal
results = []
ar_records = []
bad_deals = 0

# Pivot data so ACQUIRER and BENCHMARK returns are side-by-side per day per deal
print("Pivoting data for OLS Regression...")
# Filter to only the columns we need to save memory
ts_sub = ts[["deal_id", "security_role", "rel_day", "ret_1d", "window_flag", "deal_key", "security"]].copy()

# Fast pivot using pivot_table or just unstack
# Since we know each deal has matched dates, rel_day is a uniquely identifying index within a deal
# Let's pivot: index=['deal_id', 'rel_day', 'window_flag', 'deal_key'], columns='security_role', values='ret_1d'
df_piv = ts_sub.pivot(index=['deal_id', 'deal_key', 'rel_day', 'window_flag'], 
                      columns='security_role', 
                      values='ret_1d').reset_index()

# Note: security name for benchmark
bench_sec = ts[ts["security_role"]=="BENCHMARK"]["security"].iloc[0]

# Pre-group data by deal_id to speed up processing
print("Running OLS regressions and computing CAR...")
grouped = df_piv.groupby('deal_id')

for deal_id, group in grouped:
    # Get deal key
    deal_key = group['deal_key'].iloc[0]
    
    # Estimation Window
    est = group[group['window_flag'] == 'EST']
    
    # Event Window
    evt = group[group['window_flag'] == 'EVENT']
    
    n_est = len(est)
    n_evt = len(evt)
    
    # Must have minimum EST data (e.g., 120 days) and some EVENT data
    if n_est < 120 or n_evt == 0:
        bad_deals += 1
        continue
        
    # Drop NaNs just in case
    est_clean = est.dropna(subset=['ACQUIRER', 'BENCHMARK'])
    if len(est_clean) < 120:
        bad_deals += 1
        continue
        
    # 1) OLS Regression: R_acquirer = alpha + beta * R_benchmark
    R_m = est_clean['BENCHMARK'].values
    R_i = est_clean['ACQUIRER'].values
    
    slope, intercept, _, _, _ = stats.linregress(R_m, R_i)
    beta_hat = slope
    alpha_hat = intercept
    
    # 2) Compute Abnormal Returns (AR) in the Event Window
    evt_clean = evt.dropna(subset=['ACQUIRER', 'BENCHMARK'])
    if len(evt_clean) < 1:
        bad_deals += 1
        continue
        
    R_m_evt = evt_clean['BENCHMARK'].values
    R_i_evt = evt_clean['ACQUIRER'].values
    rel_days_evt = evt_clean['rel_day'].values
    
    # Expected Return E(R_i) = alpha + beta * R_m
    ER_i = alpha_hat + beta_hat * R_m_evt
    
    # Abnormal Return AR = Actual - Expected
    AR = R_i_evt - ER_i
    
    # Store daily AR for writing back to timeseries
    for rd, er_val, ar_val in zip(rel_days_evt, ER_i, AR):
        ar_records.append({
            'deal_id': deal_id,
            'rel_day': rd,
            'security_role': 'ACQUIRER',
            'expected_return': er_val,
            'abnormal_return': ar_val
        })
    
    # 3) Cumulative Abnormal Return (CAR)
    CAR = np.sum(AR)
    
    results.append({
        'deal_id': deal_id,
        'deal_key': deal_key,
        'alpha_hat': alpha_hat,
        'beta_hat': beta_hat,
        'car_m5_p5': CAR,
        'n_est_days': len(est_clean),
        'n_event_days': len(evt_clean),
        'benchmark_security': bench_sec
    })
    
res_df = pd.DataFrame(results)

print(f"\nSuccessfully computed CAR for {len(res_df):,} deals.")
if bad_deals > 0:
    print(f"Skipped {bad_deals} deals due to insufficient data post-pivot.")

print("Saving daily Expected/Abnormal Returns back to timeseries_long.csv...")
ar_df = pd.DataFrame(ar_records)
ts = ts.merge(ar_df, on=['deal_id', 'rel_day', 'security_role'], how='left')
ts.to_csv("timeseries_long.csv", index=False)

print("Saving to car_results.csv...")
res_df.to_csv("car_results.csv", index=False)

# We know that 'deals_master.csv' was built simply by resetting the index of 'combined_financial_text.csv'.
# Therefore, deal_id perfectly corresponds to the row index in the original file.
print("\nMerging with combined_financial_text.csv...")
original_ds = pd.read_csv("combined_financial_text.csv")
pre_len = len(original_ds)

# Join on row index (which equals deal_id in deals_master)
res_df.set_index('deal_id', inplace=True)
original_ds.index.name = 'deal_id'

# Join alpha, beta, CAR, n_est, n_evt based on deal_id index mapping
final_ds = original_ds.join(res_df[['alpha_hat', 'beta_hat', 'car_m5_p5', 'n_est_days', 'n_event_days']])

post_len = len(final_ds)
final_ds.to_csv("final_car_dataset.csv", index=False)

print(f"Saved final dataset to final_car_dataset.csv")
print(f"Original dataset rows: {pre_len:,}")
print(f"Final dataset rows:    {post_len:,}")
print(f"Deals with CAR data:   {final_ds['car_m5_p5'].notna().sum():,}")
print("\nDone!")

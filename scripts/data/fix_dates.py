"""
fix_dates.py -- Standardize all dates in timeseries_long.csv to ISO format
==========================================================================
The yfinance portion wrote dates as DD/MM/YYYY and Bloomberg as YYYY-MM-DD.
This script re-runs pull_car_data.py (yfinance part only) and merge_bbg_data.py
with consistent date formatting, then verifies the result.
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("  Fixing date formats in timeseries_long.csv")
print("=" * 70)

# Step 1: Read the raw CSV without parsing dates
ts = pd.read_csv("timeseries_long.csv")
dm = pd.read_csv("deals_master.csv")
dm["announce_date"] = pd.to_datetime(dm["announce_date"])

print(f"Total rows: {len(ts)}")
print(f"Unique deals: {ts['deal_id'].nunique()}")

# Step 2: Identify which rows are yfinance (DD/MM/YYYY) vs Bloomberg (YYYY-MM-DD)
# yfinance rows have trading_date like "08/01/2016" (contains /)
# Bloomberg rows have trading_date like "1999-03-26 00:00:00" (contains -)
sample_dates = ts["trading_date"].head(1).values[0]
print(f"\nSample yfinance date: {sample_dates}")
sample_bbg = ts["trading_date"].iloc[-1]
print(f"Sample Bloomberg date: {sample_bbg}")

# Parse dates properly based on format
# yfinance dates: DD/MM/YYYY -- must use dayfirst=True
# Bloomberg dates: YYYY-MM-DD HH:MM:SS -- standard ISO

# Detect format by checking if "/" is in the string
is_slash = ts["trading_date"].str.contains("/", na=False)
print(f"\nRows with / (yfinance): {is_slash.sum()}")
print(f"Rows with - (Bloomberg): (~is_slash).sum() = {(~is_slash).sum()}")

# Parse yfinance dates with dayfirst=True
yf_mask = is_slash
bbg_mask = ~is_slash

# Parse each group
ts.loc[yf_mask, "trading_date_parsed"] = pd.to_datetime(
    ts.loc[yf_mask, "trading_date"], dayfirst=True
)
ts.loc[bbg_mask, "trading_date_parsed"] = pd.to_datetime(
    ts.loc[bbg_mask, "trading_date"]
)

# Same for ann_date
is_slash_ann = ts["ann_date"].str.contains("/", na=False)
ts.loc[is_slash_ann, "ann_date_parsed"] = pd.to_datetime(
    ts.loc[is_slash_ann, "ann_date"], dayfirst=True
)
ts.loc[~is_slash_ann, "ann_date_parsed"] = pd.to_datetime(
    ts.loc[~is_slash_ann, "ann_date"]
)

# Replace the columns
ts["trading_date"] = ts["trading_date_parsed"]
ts["ann_date"] = ts["ann_date_parsed"]
ts = ts.drop(columns=["trading_date_parsed", "ann_date_parsed"])

# Step 3: Verify announce dates match deals_master
ts_ann = ts.groupby("deal_id")["ann_date"].first().reset_index()
merged = ts_ann.merge(dm[["deal_id", "announce_date"]], on="deal_id")
mismatched = merged[merged["ann_date"].dt.date != merged["announce_date"].dt.date]
print(f"\nDate mismatches after fix: {len(mismatched)}")
if len(mismatched) > 0:
    print("First 5 mismatches:")
    print(mismatched.head())

# Step 4: For mismatched deals, we need to recompute rel_day and window_flag
# because the ann_date stored in the timeseries was wrong
if len(mismatched) > 0:
    print(f"\nRecomputing rel_day and window_flag for {len(mismatched)} deals with wrong ann_date...")
    
    EST_START, EST_END = -200, -20
    EVT_START, EVT_END = -5, 5
    MIN_EST_OBS = 120
    
    fixed = 0
    dropped = 0
    
    for _, row in mismatched.iterrows():
        did = row["deal_id"]
        correct_ann = row["announce_date"]
        
        # Get this deal's data
        deal_mask = ts["deal_id"] == did
        deal_ts = ts[deal_mask].copy()
        
        # Fix ann_date
        ts.loc[deal_mask, "ann_date"] = correct_ann
        
        # Recompute rel_day for ACQUIRER
        for role in ["ACQUIRER", "BENCHMARK"]:
            role_mask = deal_mask & (ts["security_role"] == role)
            role_ts = ts[role_mask].copy().sort_values("trading_date")
            
            candidates = role_ts[role_ts["trading_date"] >= correct_ann]
            if len(candidates) == 0:
                # Day 0 not found -- drop this deal
                ts = ts[~deal_mask]
                dropped += 1
                break
            
            day0 = candidates["trading_date"].iloc[0]
            day0_pos = role_ts.index.get_loc(role_ts[role_ts["trading_date"] == day0].index[0])
            new_rel_day = np.arange(len(role_ts)) - day0_pos
            
            ts.loc[role_ts.index, "rel_day"] = new_rel_day
            
            # Recompute window_flag
            def flag(rd):
                if EST_START <= rd <= EST_END: return "EST"
                elif EVT_START <= rd <= EVT_END: return "EVENT"
                else: return "GAP"
            
            ts.loc[role_ts.index, "window_flag"] = [flag(r) for r in new_rel_day]
        else:
            fixed += 1
    
    print(f"  Fixed: {fixed}, Dropped: {dropped}")
    
    # Remove GAP rows
    before = len(ts)
    ts = ts[ts["window_flag"].isin(["EST", "EVENT"])]
    print(f"  Removed {before - len(ts)} GAP rows")

# Step 5: Format dates as ISO strings for clean CSV output
ts["trading_date"] = ts["trading_date"].dt.strftime("%Y-%m-%d")
ts["ann_date"] = ts["ann_date"].dt.strftime("%Y-%m-%d")

# Step 6: Save
ts.to_csv("timeseries_long.csv", index=False)
print(f"\n-> Saved timeseries_long.csv")
print(f"   Rows: {len(ts)}")
print(f"   Deals: {ts['deal_id'].nunique()}")

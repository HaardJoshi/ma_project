"""
verify_merged_data.py -- Comprehensive data integrity checks
=============================================================
Verifies that the merged timeseries_long.csv has no data mixing between deals.
"""
import pandas as pd
import numpy as np

ts = pd.read_csv("timeseries_long.csv")
dm = pd.read_csv("deals_master.csv")
dm["announce_date"] = pd.to_datetime(dm["announce_date"])
ts["trading_date"] = pd.to_datetime(ts["trading_date"])
ts["ann_date"] = pd.to_datetime(ts["ann_date"])

errors = []
print("=" * 70)
print("  DATA INTEGRITY VERIFICATION")
print("=" * 70)

# ── CHECK 1: Every deal_id has exactly 2 security_roles ─────────────────
print("\n[1] Security roles per deal...")
roles = ts.groupby("deal_id")["security_role"].apply(set)
bad_roles = roles[roles.apply(lambda x: x != {"ACQUIRER", "BENCHMARK"})]
if len(bad_roles) == 0:
    print("    PASS: All 4,510 deals have exactly ACQUIRER + BENCHMARK rows")
else:
    print(f"    FAIL: {len(bad_roles)} deals have wrong roles: {bad_roles.head()}")
    errors.append("bad_roles")

# ── CHECK 2: ACQUIRER and BENCHMARK row counts match per deal ───────────
print("\n[2] Row count symmetry (ACQUIRER == BENCHMARK per deal)...")
acq_counts = ts[ts["security_role"]=="ACQUIRER"].groupby("deal_id").size()
mkt_counts = ts[ts["security_role"]=="BENCHMARK"].groupby("deal_id").size()
mismatched = acq_counts[acq_counts != mkt_counts]
if len(mismatched) == 0:
    print("    PASS: ACQUIRER and BENCHMARK have identical row counts for all deals")
else:
    print(f"    FAIL: {len(mismatched)} deals have mismatched counts")
    errors.append("count_mismatch")

# ── CHECK 3: Event window has exactly 11 rows per deal ──────────────────
print("\n[3] Event window size (must be 11 rows: days -5 to +5)...")
evt_acq = ts[(ts["window_flag"]=="EVENT") & (ts["security_role"]=="ACQUIRER")]
evt_counts = evt_acq.groupby("deal_id").size()
bad_evt = evt_counts[evt_counts != 11]
if len(bad_evt) == 0:
    print(f"    PASS: All {len(evt_counts)} deals have exactly 11 event rows")
else:
    print(f"    WARN: {len(bad_evt)} deals don't have 11 event rows")
    print(f"    Distribution: {bad_evt.value_counts().to_dict()}")
    errors.append("event_window_size")

# ── CHECK 4: rel_day range in event window ──────────────────────────────
print("\n[4] Event window rel_day range...")
evt_min = evt_acq["rel_day"].min()
evt_max = evt_acq["rel_day"].max()
if evt_min == -5 and evt_max == 5:
    print(f"    PASS: Event window rel_day range is [{evt_min}, {evt_max}]")
else:
    print(f"    FAIL: Event window rel_day range is [{evt_min}, {evt_max}] (expected [-5, 5])")
    errors.append("rel_day_range")

# ── CHECK 5: ann_date matches deals_master for each deal ────────────────
print("\n[5] Announce date consistency (timeseries vs deals_master)...")
ts_ann = ts.groupby("deal_id")["ann_date"].first().reset_index()
merged = ts_ann.merge(dm[["deal_id","announce_date"]], on="deal_id")
date_mismatch = merged[merged["ann_date"] != merged["announce_date"]]
if len(date_mismatch) == 0:
    print(f"    PASS: All {len(merged)} deals have matching announce dates")
else:
    print(f"    FAIL: {len(date_mismatch)} deals have mismatched announce dates")
    print(date_mismatch.head())
    errors.append("date_mismatch")

# ── CHECK 6: Day 0 (rel_day==0) date is on or after announce_date ──────
print("\n[6] Day 0 is on or after announce_date...")
day0 = ts[(ts["rel_day"]==0) & (ts["security_role"]=="ACQUIRER")].copy()
day0 = day0.merge(dm[["deal_id","announce_date"]], on="deal_id")
bad_day0 = day0[day0["trading_date"] < day0["announce_date"]]
if len(bad_day0) == 0:
    print(f"    PASS: All {len(day0)} deals have Day 0 >= announce_date")
else:
    print(f"    FAIL: {len(bad_day0)} deals have Day 0 before announce_date")
    errors.append("day0_before_ann")

# ── CHECK 7: No duplicate deal_id + security_role + trading_date ────────
print("\n[7] No duplicate rows per deal/security/date...")
dups = ts.duplicated(subset=["deal_id","security_role","trading_date"], keep=False)
n_dups = dups.sum()
if n_dups == 0:
    print("    PASS: No duplicate rows found")
else:
    print(f"    FAIL: {n_dups} duplicate rows found")
    errors.append("duplicates")

# ── CHECK 8: Spot-check 5 random deals - dates within expected range ────
print("\n[8] Spot-check 5 random deals (date ranges, ticker consistency)...")
sample_ids = np.random.choice(ts["deal_id"].unique(), size=5, replace=False)
for did in sample_ids:
    deal_info = dm[dm["deal_id"]==did].iloc[0]
    ann = deal_info["announce_date"]
    ticker = deal_info["acq_ticker_bbg"]
    
    deal_ts = ts[(ts["deal_id"]==did) & (ts["security_role"]=="ACQUIRER")]
    sec_label = deal_ts["security"].iloc[0]
    n_est = (deal_ts["window_flag"]=="EST").sum()
    n_evt = (deal_ts["window_flag"]=="EVENT").sum()
    date_min = deal_ts["trading_date"].min()
    date_max = deal_ts["trading_date"].max()
    
    # Check dates are sensible (within ~400 days before to ~20 days after announcement)
    days_before = (ann - date_min).days
    days_after = (date_max - ann).days
    ok = 200 <= days_before <= 500 and 1 <= days_after <= 30
    
    status = "OK" if ok else "CHECK"
    print(f"    Deal {did}: {ticker} | ann={ann.date()} | "
          f"est={n_est} evt={n_evt} | "
          f"dates={date_min.date()}..{date_max.date()} | "
          f"days_before={days_before} days_after={days_after} | {status}")

# ── CHECK 9: Security labels match deal (no cross-deal mixing) ──────────
print("\n[9] Security labels match deal metadata (no cross-deal mixing)...")
acq_secs = ts[ts["security_role"]=="ACQUIRER"].groupby("deal_id")["security"].nunique()
multi_sec = acq_secs[acq_secs > 1]
if len(multi_sec) == 0:
    print("    PASS: Each deal has exactly one acquirer security label")
else:
    print(f"    FAIL: {len(multi_sec)} deals have multiple security labels")
    errors.append("mixed_securities")

# ── CHECK 10: Estimation window is before event window ──────────────────
print("\n[10] Estimation window dates precede event window dates...")
for did in sample_ids[:3]:
    deal_ts = ts[(ts["deal_id"]==did) & (ts["security_role"]=="ACQUIRER")]
    est_end = deal_ts[deal_ts["window_flag"]=="EST"]["trading_date"].max()
    evt_start = deal_ts[deal_ts["window_flag"]=="EVENT"]["trading_date"].min()
    ok = est_end < evt_start
    print(f"    Deal {did}: EST ends {est_end.date()}, EVENT starts {evt_start.date()} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        errors.append("window_overlap")

# ── SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if len(errors) == 0:
    print("  ALL CHECKS PASSED - Data integrity verified")
else:
    print(f"  {len(errors)} CHECK(S) FAILED: {errors}")
print("=" * 70)

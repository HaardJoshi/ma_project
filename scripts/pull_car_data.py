"""
pull_car_data.py  —  CAR data-pull pipeline
============================================
Reads the deal list from combined_financial_text.csv, downloads acquirer and
S&P 500 daily closing prices via yfinance, computes log returns, assigns
trading-day relative days, and writes:
    1. deals_master.csv   – one row per deal
    2. timeseries_long.csv – long/tidy price+return table for all deals

Methodological notes
--------------------
- Estimation window:  rel_day in [-200, -20]   (for OLS market model)
- Event window:       rel_day in [ -5,  +5]    (for CAR calculation)
- Day 0 = first trading day >= announcement date  (forward-fill rule)
- rel_day is counted in *trading days*, not calendar days
- Returns are log returns:  R_t = ln(P_t / P_{t-1})
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import time
import sys
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEAL_FILE       = "combined_financial_text.csv"
OUTPUT_DEALS    = "deals_master.csv"
OUTPUT_TS       = "timeseries_long.csv"
BENCHMARK_YF    = "^GSPC"                    # S&P 500 on Yahoo Finance
BENCHMARK_LABEL = "SPX Index"                # label in output

EST_START  = -200   # estimation window start (trading days)
EST_END    =  -20   # estimation window end
EVT_START  =   -5   # event window start
EVT_END    =   +5   # event window end

CALENDAR_PAD_BEFORE = 380   # calendar days before announce to over-pull
CALENDAR_PAD_AFTER  =  20   # calendar days after announce to over-pull
MIN_EST_OBS         = 120   # minimum estimation-window observations

YF_BATCH_SIZE       =  50   # tickers per yfinance download batch
YF_SLEEP            =   1   # seconds between batches (rate-limit courtesy)


# ── HELPERS ─────────────────────────────────────────────────────────────────
def bbg_to_yf(bbg_ticker: str) -> str:
    """Convert Bloomberg-style ticker ('AAPL US') to Yahoo Finance ticker ('AAPL').
    
    Handles special cases:
      - 'BRK/A US' -> 'BRK-A'  (Berkshire Hathaway class A)
      - 'BRK/B US' -> 'BRK-B'
      - 'BF/A US'  -> 'BF-A'   (Brown-Forman)
      - 'BF/B US'  -> 'BF-B'
    """
    t = str(bbg_ticker).strip()
    # Strip Bloomberg exchange suffixes
    for suffix in (" US Equity", " US", " Equity"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    # Bloomberg uses '/' for share classes; Yahoo uses '-'
    t = t.replace("/", "-")
    return t


def assign_rel_day(price_df: pd.DataFrame, announce_date: pd.Timestamp) -> pd.DataFrame:
    """Assign trading-day relative index (rel_day) to a price DataFrame.

    Day 0 = first trading day >= announce_date.
    rel_day is an integer offset counted in trading days (positions), not calendar days.
    """
    df = price_df.copy()
    df = df.sort_index()

    # Day 0: first trading day on or after the announcement date
    candidates = df.index[df.index >= announce_date]
    if len(candidates) == 0:
        return pd.DataFrame()  # announce date is after all available data
    day0 = candidates[0]
    day0_pos = df.index.get_loc(day0)

    # rel_day = position - day0_position  (in trading-day steps)
    df["rel_day"] = np.arange(len(df)) - day0_pos
    return df


def flag_window(rel_day: int) -> str:
    if EST_START <= rel_day <= EST_END:
        return "EST"
    elif EVT_START <= rel_day <= EVT_END:
        return "EVENT"
    else:
        return "GAP"


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CAR Data-Pull Pipeline")
    print("=" * 70)

    # ── 1. Load deals ───────────────────────────────────────────────────
    raw = pd.read_csv(DEAL_FILE)
    raw["Announce Date"] = pd.to_datetime(raw["Announce Date"])
    print(f"\nLoaded {len(raw)} deals from {DEAL_FILE}")

    # Build deals master with only columns needed for CAR
    deals = raw[["deal_key", "Acquirer Ticker", "Announce Date",
                  "Acquirer Name", "Target Name"]].copy()
    deals = deals.rename(columns={
        "deal_key": "deal_key",
        "Acquirer Ticker": "acq_ticker_bbg",
        "Announce Date": "announce_date",
        "Acquirer Name": "acquirer_name",
        "Target Name": "target_name",
    })
    deals["deal_id"] = range(1, len(deals) + 1)
    deals["acq_ticker_yf"] = deals["acq_ticker_bbg"].apply(bbg_to_yf)
    deals["benchmark"] = BENCHMARK_LABEL

    # Save deals master
    deals.to_csv(OUTPUT_DEALS, index=False)
    print(f"  -> Saved {OUTPUT_DEALS}  ({len(deals)} deals)")

    # ── 2. Determine global date range for S&P 500 download ─────────
    global_start = (deals["announce_date"].min() - timedelta(days=CALENDAR_PAD_BEFORE))
    global_end   = (deals["announce_date"].max() + timedelta(days=CALENDAR_PAD_AFTER))
    print(f"\nGlobal price window: {global_start.date()} -> {global_end.date()}")

    # ── 3. Download S&P 500 once for the entire range ───────────────
    print(f"\nDownloading benchmark ({BENCHMARK_YF}) ...")
    mkt_raw = yf.download(BENCHMARK_YF, start=global_start.strftime("%Y-%m-%d"),
                          end=global_end.strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
    if mkt_raw.empty:
        print("ERROR: Could not download S&P 500 data. Exiting.")
        sys.exit(1)

    # yfinance returns multi-level columns when single ticker; flatten
    if isinstance(mkt_raw.columns, pd.MultiIndex):
        mkt_raw.columns = mkt_raw.columns.get_level_values(0)
    mkt_prices = mkt_raw[["Close"]].rename(columns={"Close": "px_last"})
    mkt_prices.index = pd.to_datetime(mkt_prices.index).tz_localize(None)
    mkt_prices = mkt_prices.sort_index()
    print(f"  -> {len(mkt_prices)} trading days loaded for {BENCHMARK_YF}")

    # ── 4. Download acquirer prices in batches ──────────────────────
    unique_yf = deals["acq_ticker_yf"].unique().tolist()
    print(f"\nDownloading {len(unique_yf)} unique acquirer tickers in batches of {YF_BATCH_SIZE} ...")

    acq_price_store = {}   # ticker -> DataFrame with 'px_last' column
    failed_tickers = []

    for batch_start in range(0, len(unique_yf), YF_BATCH_SIZE):
        batch = unique_yf[batch_start : batch_start + YF_BATCH_SIZE]
        batch_num = batch_start // YF_BATCH_SIZE + 1
        total_batches = (len(unique_yf) + YF_BATCH_SIZE - 1) // YF_BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}  ({len(batch)} tickers) ...", end="", flush=True)

        try:
            data = yf.download(batch, start=global_start.strftime("%Y-%m-%d"),
                               end=global_end.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True,
                               threads=True)
        except Exception as e:
            print(f"  BATCH ERROR: {e}")
            failed_tickers.extend(batch)
            continue

        if data.empty:
            print(f"  empty result")
            failed_tickers.extend(batch)
            continue

        # Parse result – yf.download returns MultiIndex columns: (field, ticker)
        # for single ticker it may flatten
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker result
            for ticker in batch:
                try:
                    series = data["Close"][ticker].dropna()
                    if len(series) > 0:
                        tdf = pd.DataFrame({"px_last": series})
                        tdf.index = pd.to_datetime(tdf.index).tz_localize(None)
                        acq_price_store[ticker] = tdf.sort_index()
                    else:
                        failed_tickers.append(ticker)
                except (KeyError, TypeError):
                    failed_tickers.append(ticker)
        else:
            # Single-ticker result (batch size 1)
            ticker = batch[0]
            try:
                series = data["Close"].dropna()
                if len(series) > 0:
                    tdf = pd.DataFrame({"px_last": series})
                    tdf.index = pd.to_datetime(tdf.index).tz_localize(None)
                    acq_price_store[ticker] = tdf.sort_index()
                else:
                    failed_tickers.append(ticker)
            except (KeyError, TypeError):
                failed_tickers.append(ticker)

        got = sum(1 for t in batch if t in acq_price_store)
        print(f"  {got}/{len(batch)} ok")

        if batch_start + YF_BATCH_SIZE < len(unique_yf):
            time.sleep(YF_SLEEP)

    print(f"\n  Tickers loaded: {len(acq_price_store)}")
    if failed_tickers:
        print(f"  Tickers FAILED ({len(failed_tickers)}): {failed_tickers[:20]}{'...' if len(failed_tickers)>20 else ''}")

    # ── 5. Build long tidy time-series per deal ─────────────────────
    print(f"\nBuilding time-series for each deal ...")
    ts_rows = []
    skip_no_data = 0
    skip_no_day0 = 0
    skip_few_est = 0
    success = 0

    for _, deal in deals.iterrows():
        deal_id = deal["deal_id"]
        yf_ticker = deal["acq_ticker_yf"]
        ann_date = pd.Timestamp(deal["announce_date"])

        # Check acquirer data exists
        if yf_ticker not in acq_price_store:
            skip_no_data += 1
            continue

        acq_df = acq_price_store[yf_ticker].copy()
        mkt_df = mkt_prices.copy()

        # Trim to deal-specific window (generously)
        pull_start = ann_date - timedelta(days=CALENDAR_PAD_BEFORE)
        pull_end   = ann_date + timedelta(days=CALENDAR_PAD_AFTER)
        acq_df = acq_df.loc[(acq_df.index >= pull_start) & (acq_df.index <= pull_end)]
        mkt_df = mkt_df.loc[(mkt_df.index >= pull_start) & (mkt_df.index <= pull_end)]

        # Align on common trading days
        common_idx = acq_df.index.intersection(mkt_df.index)
        if len(common_idx) < MIN_EST_OBS + 11:
            skip_no_data += 1
            continue

        acq_df = acq_df.loc[common_idx]
        mkt_df = mkt_df.loc[common_idx]

        # Assign rel_day (trading-day based)
        acq_df = assign_rel_day(acq_df, ann_date)
        mkt_df = assign_rel_day(mkt_df, ann_date)

        if acq_df.empty or mkt_df.empty:
            skip_no_day0 += 1
            continue

        # Compute log returns
        acq_df["ret_1d"] = np.log(acq_df["px_last"] / acq_df["px_last"].shift(1))
        mkt_df["ret_1d"] = np.log(mkt_df["px_last"] / mkt_df["px_last"].shift(1))

        # Drop the first row (NaN return)
        acq_df = acq_df.dropna(subset=["ret_1d"])
        mkt_df = mkt_df.dropna(subset=["ret_1d"])

        # Re-synchronise rel_day after dropping first row
        # (rel_day should still be correct since it was assigned before the drop)

        # Flag windows
        acq_df["window_flag"] = acq_df["rel_day"].apply(flag_window)
        mkt_df["window_flag"] = mkt_df["rel_day"].apply(flag_window)

        # Quality check: enough estimation observations?
        n_est = (acq_df["window_flag"] == "EST").sum()
        n_evt = (acq_df["window_flag"] == "EVENT").sum()
        if n_est < MIN_EST_OBS:
            skip_few_est += 1
            continue

        # Keep only estimation + event window rows (drop GAP to save space)
        acq_out = acq_df[acq_df["window_flag"].isin(["EST", "EVENT"])].copy()
        mkt_out = mkt_df[mkt_df["window_flag"].isin(["EST", "EVENT"])].copy()

        # Tag with deal metadata
        for df_out, role, sec_label in [
            (acq_out, "ACQUIRER", deal["acq_ticker_bbg"]),
            (mkt_out, "BENCHMARK", BENCHMARK_LABEL),
        ]:
            df_out["deal_id"]       = deal_id
            df_out["deal_key"]      = deal["deal_key"]
            df_out["security_role"] = role
            df_out["security"]      = sec_label
            df_out["ann_date"]      = ann_date
            df_out.index.name       = "trading_date"
            ts_rows.append(df_out.reset_index())

        success += 1
        if success % 500 == 0:
            print(f"  {success} deals processed ...")

    print(f"\n  Successful deals: {success}")
    print(f"  Skipped (no price data):       {skip_no_data}")
    print(f"  Skipped (no Day 0 found):      {skip_no_day0}")
    print(f"  Skipped (< {MIN_EST_OBS} est obs):    {skip_few_est}")

    if not ts_rows:
        print("\nERROR: No deals produced time-series data. Exiting.")
        sys.exit(1)

    # ── 6. Concatenate and save ─────────────────────────────────────
    ts = pd.concat(ts_rows, ignore_index=True)

    # Reorder columns for clarity
    col_order = [
        "deal_id", "deal_key", "security_role", "security",
        "trading_date", "px_last", "ret_1d",
        "ann_date", "rel_day", "window_flag",
    ]
    ts = ts[col_order]
    ts.to_csv(OUTPUT_TS, index=False)

    print(f"\n  -> Saved {OUTPUT_TS}  ({len(ts)} rows)")

    # ── 7. Summary statistics ───────────────────────────────────────
    est_rows = ts[ts["window_flag"] == "EST"]
    evt_rows = ts[ts["window_flag"] == "EVENT"]
    acq_evt  = evt_rows[evt_rows["security_role"] == "ACQUIRER"]

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Deals in master:          {len(deals)}")
    print(f"  Deals with time-series:   {success}")
    print(f"  Total time-series rows:   {len(ts)}")
    print(f"    Estimation rows:        {len(est_rows)}")
    print(f"    Event rows:             {len(evt_rows)}")
    print(f"  Acquirer event rows/deal: {acq_evt.groupby('deal_id').size().mean():.1f} (mean)")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  1. {OUTPUT_DEALS}")
    print(f"  2. {OUTPUT_TS}")
    print(f"\nNext step: Run OLS market model on estimation window,")
    print(f"           then compute CAR in event window.")


if __name__ == "__main__":
    main()

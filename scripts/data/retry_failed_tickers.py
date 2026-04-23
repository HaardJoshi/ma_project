"""
retry_failed_tickers.py  —  Retry rate-limited ticker downloads
================================================================
Reads the existing timeseries_long.csv and deals_master.csv,
identifies deals that are missing, re-downloads those tickers
with longer delays to avoid rate-limiting, then appends to the
existing output files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import time
import sys

# ── CONFIG (must match pull_car_data.py) ────────────────────────────────────
DEAL_FILE       = "combined_financial_text.csv"
OUTPUT_DEALS    = "deals_master.csv"
OUTPUT_TS       = "timeseries_long.csv"
BENCHMARK_YF    = "^GSPC"
BENCHMARK_LABEL = "SPX Index"

EST_START  = -200
EST_END    =  -20
EVT_START  =   -5
EVT_END    =   +5

CALENDAR_PAD_BEFORE = 380
CALENDAR_PAD_AFTER  =  20
MIN_EST_OBS         = 120

# Smaller batches + longer delays for retry
YF_BATCH_SIZE       =  10
YF_SLEEP            =   5   # 5 seconds between batches
MAX_RETRIES         =   3
RETRY_BACKOFF       =  10   # extra seconds per retry


def bbg_to_yf(bbg_ticker):
    t = str(bbg_ticker).strip()
    for suffix in (" US Equity", " US", " Equity"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    t = t.replace("/", "-")
    return t


def assign_rel_day(price_df, announce_date):
    df = price_df.copy()
    df = df.sort_index()
    candidates = df.index[df.index >= announce_date]
    if len(candidates) == 0:
        return pd.DataFrame()
    day0 = candidates[0]
    day0_pos = df.index.get_loc(day0)
    df["rel_day"] = np.arange(len(df)) - day0_pos
    return df


def flag_window(rel_day):
    if EST_START <= rel_day <= EST_END:
        return "EST"
    elif EVT_START <= rel_day <= EVT_END:
        return "EVENT"
    else:
        return "GAP"


def main():
    print("=" * 70)
    print("  Retry Failed Ticker Downloads")
    print("=" * 70)

    # Load existing results
    deals = pd.read_csv(OUTPUT_DEALS)
    deals["announce_date"] = pd.to_datetime(deals["announce_date"])
    existing_ts = pd.read_csv(OUTPUT_TS)
    existing_deal_ids = set(existing_ts["deal_id"].unique())

    # Find missing deals
    missing_deals = deals[~deals["deal_id"].isin(existing_deal_ids)].copy()
    print(f"\nTotal deals: {len(deals)}")
    print(f"Already have data for: {len(existing_deal_ids)} deals")
    print(f"Missing deals: {len(missing_deals)}")

    # Get unique missing tickers (exclude ones that are clearly non-ticker IDs)
    missing_tickers = missing_deals["acq_ticker_yf"].unique().tolist()
    # Filter out tickers that look like CUSIP/Bloomberg IDs (contain mostly digits)
    valid_tickers = [t for t in missing_tickers
                     if isinstance(t, str)
                     and len(t) >= 1
                     and not any(c.isdigit() for c in t[:3])  # starts with letters
                     and t != "nan"
                     and " CN" not in t and " LN" not in t  # non-US exchanges
                     and len(t) <= 6]  # reasonable ticker length
    print(f"Missing tickers (all): {len(missing_tickers)}")
    print(f"Missing tickers (valid for retry): {len(valid_tickers)}")

    # Load S&P 500 data (already downloaded)
    global_start = (deals["announce_date"].min() - timedelta(days=CALENDAR_PAD_BEFORE))
    global_end   = (deals["announce_date"].max() + timedelta(days=CALENDAR_PAD_AFTER))

    print(f"\nDownloading benchmark ({BENCHMARK_YF}) ...")
    mkt_raw = yf.download(BENCHMARK_YF, start=global_start.strftime("%Y-%m-%d"),
                          end=global_end.strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
    if isinstance(mkt_raw.columns, pd.MultiIndex):
        mkt_raw.columns = mkt_raw.columns.get_level_values(0)
    mkt_prices = mkt_raw[["Close"]].rename(columns={"Close": "px_last"})
    mkt_prices.index = pd.to_datetime(mkt_prices.index).tz_localize(None)
    mkt_prices = mkt_prices.sort_index()
    print(f"  -> {len(mkt_prices)} trading days loaded")

    # Download missing tickers with retry logic
    print(f"\nDownloading {len(valid_tickers)} tickers in batches of {YF_BATCH_SIZE} ...")
    acq_price_store = {}
    still_failed = []

    for batch_start in range(0, len(valid_tickers), YF_BATCH_SIZE):
        batch = valid_tickers[batch_start : batch_start + YF_BATCH_SIZE]
        batch_num = batch_start // YF_BATCH_SIZE + 1
        total_batches = (len(valid_tickers) + YF_BATCH_SIZE - 1) // YF_BATCH_SIZE

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  Batch {batch_num}/{total_batches} (attempt {attempt}) ...", end="", flush=True)
            try:
                data = yf.download(batch, start=global_start.strftime("%Y-%m-%d"),
                                   end=global_end.strftime("%Y-%m-%d"),
                                   progress=False, auto_adjust=True,
                                   threads=False)  # single-threaded for reliability
            except Exception as e:
                print(f" ERROR: {e}")
                time.sleep(YF_SLEEP + RETRY_BACKOFF * attempt)
                continue

            if data.empty:
                print(f" empty")
                time.sleep(YF_SLEEP + RETRY_BACKOFF * attempt)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                batch_got = 0
                for ticker in batch:
                    try:
                        series = data["Close"][ticker].dropna()
                        if len(series) > 0:
                            tdf = pd.DataFrame({"px_last": series})
                            tdf.index = pd.to_datetime(tdf.index).tz_localize(None)
                            acq_price_store[ticker] = tdf.sort_index()
                            batch_got += 1
                    except (KeyError, TypeError):
                        pass
                print(f" {batch_got}/{len(batch)} ok")
            else:
                ticker = batch[0]
                try:
                    series = data["Close"].dropna()
                    if len(series) > 0:
                        tdf = pd.DataFrame({"px_last": series})
                        tdf.index = pd.to_datetime(tdf.index).tz_localize(None)
                        acq_price_store[ticker] = tdf.sort_index()
                        print(f" 1/1 ok")
                    else:
                        print(f" empty series")
                except (KeyError, TypeError):
                    print(f" parse error")
            break  # success, move to next batch

        time.sleep(YF_SLEEP)

    print(f"\n  New tickers loaded: {len(acq_price_store)}")

    # Build time-series for missing deals
    print(f"\nBuilding time-series for missing deals ...")
    ts_rows = []
    success = 0
    skip_no_data = 0
    skip_few_est = 0

    for _, deal in missing_deals.iterrows():
        deal_id = deal["deal_id"]
        yf_ticker = deal["acq_ticker_yf"]
        ann_date = pd.Timestamp(deal["announce_date"])

        if yf_ticker not in acq_price_store:
            skip_no_data += 1
            continue

        acq_df = acq_price_store[yf_ticker].copy()
        mkt_df = mkt_prices.copy()

        pull_start = ann_date - timedelta(days=CALENDAR_PAD_BEFORE)
        pull_end   = ann_date + timedelta(days=CALENDAR_PAD_AFTER)
        acq_df = acq_df.loc[(acq_df.index >= pull_start) & (acq_df.index <= pull_end)]
        mkt_df = mkt_df.loc[(mkt_df.index >= pull_start) & (mkt_df.index <= pull_end)]

        common_idx = acq_df.index.intersection(mkt_df.index)
        if len(common_idx) < MIN_EST_OBS + 11:
            skip_no_data += 1
            continue

        acq_df = acq_df.loc[common_idx]
        mkt_df = mkt_df.loc[common_idx]

        acq_df = assign_rel_day(acq_df, ann_date)
        mkt_df = assign_rel_day(mkt_df, ann_date)

        if acq_df.empty or mkt_df.empty:
            skip_no_data += 1
            continue

        acq_df["ret_1d"] = np.log(acq_df["px_last"] / acq_df["px_last"].shift(1))
        mkt_df["ret_1d"] = np.log(mkt_df["px_last"] / mkt_df["px_last"].shift(1))
        acq_df = acq_df.dropna(subset=["ret_1d"])
        mkt_df = mkt_df.dropna(subset=["ret_1d"])

        acq_df["window_flag"] = acq_df["rel_day"].apply(flag_window)
        mkt_df["window_flag"] = mkt_df["rel_day"].apply(flag_window)

        n_est = (acq_df["window_flag"] == "EST").sum()
        if n_est < MIN_EST_OBS:
            skip_few_est += 1
            continue

        acq_out = acq_df[acq_df["window_flag"].isin(["EST", "EVENT"])].copy()
        mkt_out = mkt_df[mkt_df["window_flag"].isin(["EST", "EVENT"])].copy()

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

    print(f"\n  New successful deals: {success}")
    print(f"  Skipped (no data): {skip_no_data}")
    print(f"  Skipped (< {MIN_EST_OBS} est obs): {skip_few_est}")

    if ts_rows:
        new_ts = pd.concat(ts_rows, ignore_index=True)
        col_order = [
            "deal_id", "deal_key", "security_role", "security",
            "trading_date", "px_last", "ret_1d",
            "ann_date", "rel_day", "window_flag",
        ]
        new_ts = new_ts[col_order]

        # Append to existing file
        combined_ts = pd.concat([existing_ts, new_ts], ignore_index=True)
        combined_ts.to_csv(OUTPUT_TS, index=False)
        print(f"\n  -> Appended to {OUTPUT_TS}")
        print(f"  Total rows now: {len(combined_ts)}")
        total_deals = combined_ts["deal_id"].nunique()
        print(f"  Total unique deals now: {total_deals}")
    else:
        print("\nNo new data to append.")


if __name__ == "__main__":
    main()

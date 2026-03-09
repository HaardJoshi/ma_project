import pandas as pd
ts = pd.read_csv('timeseries_long.csv')
bbg_rows = ts.iloc[826032:]
acq = bbg_rows[bbg_rows['security_role'] == 'ACQUIRER']
counts = acq.groupby('deal_id')['security'].nunique()
bad = counts[counts > 1]
print(f"Bad deals: {len(bad)}")
print("\nTop 5 bad deals:")
if len(bad) > 0:
    for did in bad.index.tolist()[:5]:
        bdf = acq[acq['deal_id'] == did]
        secs = bdf['security'].unique()
        print(f"Deal {did}: {secs}")

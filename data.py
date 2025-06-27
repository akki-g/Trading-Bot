#!/usr/bin/env python3
import pandas as pd

# 1) Load the master CSV
df = pd.read_csv("fx_data_5y_daily_1y_1min.csv", parse_dates=["datetime"])

# 2) Group by the 'symbol' column and write each group to its own CSV
for symbol, group in df.groupby("symbol"):
    out_name = f"fx_data_{symbol}.csv"
    group.to_csv(out_name, index=False)
    print(f"Wrote {len(group)} rows to {out_name}")
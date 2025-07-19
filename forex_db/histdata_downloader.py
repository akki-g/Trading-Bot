#!/usr/bin/env python3
import os
import zipfile
import datetime
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
YEAR     = "2025"
SOURCE   = "histdata"
INTERVAL = "1m"

# ─── LOAD ENV & CONNECT ───────────────────────────────────────────────────────────
load_dotenv()  # requires python-dotenv
conn = psycopg2.connect(
    host     = os.getenv("DB_HOST"),
    port     = os.getenv("DB_PORT"),
    dbname   = os.getenv("DB_NAME"),
    user     = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
)
conn.autocommit = False

# ─── INTROSPECT TABLE COLUMNS ────────────────────────────────────────────────────
with conn.cursor() as cur:
    cur.execute("""
        SELECT column_name
          FROM information_schema.columns
         WHERE table_name = 'forex_data'
      ORDER BY ordinal_position
    """)
    table_cols = [row[0] for row in cur.fetchall()]

print("Will insert into columns:", table_cols)

# ─── COLLECT ALL RECORDS ─────────────────────────────────────────────────────────
records = []
now = datetime.datetime.now(datetime.timezone.utc)

for fname in os.listdir("."):
    if fname.startswith("DAT_ASCII_") and f"_M1_{YEAR}" in fname and fname.endswith(".zip"):
        pair = fname.split("_")[2]  # e.g. 'EURUSD'
        print("Parsing", fname)
        with zipfile.ZipFile(fname, "r") as z:
            for member in z.namelist():
                if member.lower().endswith(".csv"):
                    df = pd.read_csv(
                        z.open(member),
                        sep=";",
                        header=None,
                        names=["datetime_str","open","high","low","close","tick_volume"]
                    )
                    df["timestamp"] = pd.to_datetime(
                        df["datetime_str"], format="%Y%m%d %H%M%S", utc=True
                    )
                    # build a dict of all possible fields
                    df["pair"]       = pair
                    df["volume"]     = None
                    df["spread"]     = None
                    df["source"]     = SOURCE
                    df["created_at"] = now
                    df["interval"]   = INTERVAL
                    # drop the helper column
                    df.drop(columns=["datetime_str"], inplace=True)

                    # for each row, emit a tuple matching table_cols
                    for _, row in df.iterrows():
                        rec = {col: row[col] if col in row.index else None
                               for col in table_cols}
                        # ensure non-nullable keys exist
                        # e.g., timestamp, pair, open, high, low, close, source, created_at, interval
                        records.append(tuple(rec[col] for col in table_cols))

print(f"Prepared {len(records)} rows for INSERT.")

# ─── BULK INSERT ─────────────────────────────────────────────────────────────────
cols_sql = ",".join(table_cols)
insert_sql = f"""
    INSERT INTO forex_data ({cols_sql})
    VALUES %s
    ON CONFLICT DO NOTHING
"""

with conn.cursor() as cur:
    execute_values(cur, insert_sql, records, page_size=1000)

conn.commit()
conn.close()
print("✅ Done inserting!")

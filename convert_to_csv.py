"""Convert SWaT Excel files to CSV once, so later runs load in seconds instead of minutes.

Usage:
  python convert_to_csv.py data/SWaT_Dataset_Attack_v0.xlsx data/attack.csv
  python convert_to_csv.py data/SWaT_Dataset_Normal_v0.xlsx data/normal.csv --drop-first 1800

--drop-first 1800 removes the first 30 minutes (tank drainage) from Normal v0,
which is how iTrust derived Normal v1.
"""
import argparse

from data import read_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop-first", type=int, default=0, help="rows to drop from the start")
    a = ap.parse_args()

    print(f"Reading {a.src} ...")
    df = read_table(a.src)
    if a.drop_first:
        df = df.iloc[a.drop_first:]
    df.to_csv(a.dst, index=False)
    print(f"Wrote {a.dst}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()

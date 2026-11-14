"""Check the SWaT files against the numbers reported in the base paper.

Usage:
  python check_data.py                       # checks data/attack.csv and data/normal.csv
  python check_data.py --files data/attack.csv
"""
import argparse
from pathlib import Path

from data import attack_segments, load_swat

EXPECTED_ATTACK = {"rows": 449_919, "attack_rows": 54_621, "columns": 51}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=["data/attack.csv", "data/normal.csv"])
    a = ap.parse_args()

    for f in a.files:
        if not Path(f).exists():
            print(f"{f}: not found, skipped")
            continue
        slow = " (Excel: this can take a few minutes)" if f.lower().endswith((".xlsx", ".xls")) else ""
        print(f"\nReading {f}{slow}...")
        X, y, missing, t = load_swat(f, return_time=True)
        print(f"  rows: {len(y):,} | attack rows: {int(y.sum()):,} | columns: {X.shape[1]} "
              f"| attack segments: {int(attack_segments(y).max())} | missing values filled: {missing}")
        if t is not None:
            print(f"  first timestamp: {t.iloc[0]} | last: {t.iloc[-1]}")
        if "attack" in Path(f).name.lower():
            found = {"rows": len(y), "attack_rows": int(y.sum()), "columns": X.shape[1]}
            for k, v in EXPECTED_ATTACK.items():
                status = "OK" if found[k] == v else f"DIFFERENT (paper: {v:,})"
                print(f"  {k}: {found[k]:,} -> {status}")


if __name__ == "__main__":
    main()

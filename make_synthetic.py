"""Generate a small synthetic SWaT-like CSV for SMOKE TESTS ONLY.

It mimics the real file's layout (51 tag names, ' Timestamp', 'Normal/Attack' with the
'A ttack' typo) so the whole pipeline can be checked before the real data arrives.
Results on this file mean nothing scientifically.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import ACTUATOR_RE

TAGS = ("FIT101 LIT101 MV101 P101 P102 AIT201 AIT202 AIT203 FIT201 MV201 P201 P202 P203 P204 "
        "P205 P206 DPIT301 FIT301 LIT301 MV301 MV302 MV303 MV304 P301 P302 AIT401 AIT402 FIT401 "
        "LIT401 P401 P402 P403 P404 UV401 AIT501 AIT502 AIT503 AIT504 FIT501 FIT502 FIT503 "
        "FIT504 P501 P502 PIT501 PIT502 PIT503 FIT601 P601 P602 P603").split()


def _h(tag):
    return sum(ord(ch) for ch in tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--segments", type=int, default=30)
    ap.add_argument("--out", default="data/synthetic_swat.csv")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.rows

    data = {}
    for tag in TAGS:
        if tag.startswith("MV"):
            data[tag] = np.where(rng.random(n) < 0.002, 0, 1 + (np.arange(n) // 3000 + _h(tag) % 2) % 2)
        elif ACTUATOR_RE.match(tag):
            data[tag] = 1 + (np.arange(n) // (2000 + _h(tag) * 37 % 3000) + _h(tag) % 2) % 2
        else:
            e = rng.normal(0, 1, n)
            x = np.empty(n); x[0] = 0
            for t in range(1, n):
                x[t] = 0.98 * x[t - 1] + e[t]
            data[tag] = 100 + 5 * np.sin(np.arange(n) / 1500) + x
    df = pd.DataFrame(data)

    y = np.zeros(n, dtype=int)
    starts = np.sort(rng.choice(np.arange(1000, n - 3000), a.segments, replace=False))
    cont = [t for t in TAGS if not ACTUATOR_RE.match(t)]
    for s in starts:
        length = int(rng.integers(150, 1500))
        tags = rng.choice(cont, size=int(rng.integers(1, 4)), replace=False)
        for tag in tags:
            ramp = np.linspace(0, rng.choice([-1, 1]) * rng.uniform(3, 12), length)
            df.loc[s:s + length - 1, tag] += ramp
        y[s:s + length] = 1

    labels = np.where(y == 1, "Attack", "Normal").astype(object)
    typo = np.flatnonzero(y == 1)[:: 50]
    labels[typo] = "A ttack"
    df.insert(0, " Timestamp", pd.date_range("2015-12-28 10:00:00", periods=n, freq="s"))
    df["Normal/Attack"] = labels
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {n} rows, {y.sum()} attack rows in {a.segments} segments")


if __name__ == "__main__":
    main()

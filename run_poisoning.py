"""Poisoning study: how corrupted analyst labels affect the closed-loop IDS, and which
lightweight defences recover performance.

Corruption scenarios (applied only to the labels the analyst returns):
  random        - each label flipped with probability r (honest mistakes)
  missed_attack - each true attack label turned into "normal" with probability r
  targeted      - labels of ONE attack segment turned into "normal" with probability r,
                  repeated for every eligible segment present in the test fold.
                  Rate 0 gives the clean reference for the same segment.

Defences: none, frozen (no updates at all: the lower reference), filter, clip, rollback,
with optional parameters and '+' combinations, e.g. filter@0.99, clip@0.05, rollback@0.1,
filter@0.99+clip@0.05.

Example (pilot subset, used to choose defence parameters):
  python run_poisoning.py --attack-csv data/attack.csv --block-selection random \
      --block-seed 99 --reps 1 --features raw --rates 0.1 0.3 \
      --defences none frozen filter@0.95 filter@0.99 clip@0.02 clip@0.05 \
                 rollback@0.05 rollback@0.1 --out results_pilot
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import SubsetConfig, build_subset, iter_folds
from evaluation import compare_to_reference, summary_table
from features import build_features, pump_workload
from online import Corruption, Defence, LoopConfig, run_closed_loop


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--attack-csv", required=True)
    p.add_argument("--features", nargs="+", choices=["raw", "improved"], default=["raw", "improved"])
    p.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3],
                   help="rates for the random and missed_attack modes")
    p.add_argument("--modes", nargs="+", default=["random", "missed_attack", "targeted"])
    p.add_argument("--targeted-rates", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--target-min-rows", type=int, default=300,
                   help="a segment needs at least this many rows in the subset to be a target")
    p.add_argument("--target-max-frac", type=float, default=0.3,
                   help="skip segments holding more than this share of the subset's attack rows")
    p.add_argument("--defences", nargs="+", default=["none", "frozen", "filter", "clip", "rollback"])
    p.add_argument("--block-selection", choices=["spread", "random", "all"], default="spread")
    p.add_argument("--block-seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--eta0", type=float, default=2.0, help="chosen on the pilot subset")
    p.add_argument("--update-epochs", type=int, default=5, help="chosen on the pilot subset")
    p.add_argument("--query", choices=["random", "uncertainty", "deviation"], default="random")
    p.add_argument("--persistence", default=None, help="k,n alarm rule, e.g. 3,5")
    p.add_argument("--workload", choices=["none", "pumps"], default="none")
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    persistence = tuple(int(v) for v in a.persistence.split(",")) if a.persistence else None

    ds, info = build_subset(a.attack_csv, SubsetConfig(selection=a.block_selection,
                                                       seed=a.block_seed))
    print("Data:", info)
    W = pump_workload(ds.X) if a.workload == "pumps" else None

    # eligible target segments: large enough to measure, small enough not to dominate
    seg_sizes = pd.Series(ds.attack_id[ds.y == 1]).value_counts()
    total = int(ds.y.sum())
    eligible = [int(s) for s, c in seg_sizes.items()
                if c >= a.target_min_rows and c <= a.target_max_frac * total]
    excluded = {int(s): int(c) for s, c in seg_sizes.items() if int(s) not in eligible}
    print(f"Target segments, eligible {{id: rows}}: { {s: int(seg_sizes[s]) for s in eligible} }")
    print(f"Excluded (too small or dominant): {excluded}")

    scenarios = [("none", 0.0)]
    for mode in a.modes:
        if mode == "targeted":
            if eligible:
                scenarios += [("targeted", r) for r in a.targeted_rates]
            else:
                print("No eligible target segments; targeted mode skipped")
        else:
            scenarios += [(mode, r) for r in a.rates]

    base = LoopConfig(eta0=a.eta0, update_epochs=a.update_epochs, query=a.query)
    runs = []   # (label, config, defence)
    for spec in a.defences:
        if spec == "frozen":
            runs.append(("frozen", dataclasses.replace(base, incremental=False), Defence()))
        else:
            d = Defence.parse(spec)
            runs.append((d.label, base, d))

    feats = {fs: build_features(ds.X, ds.block, fs)[0] for fs in a.features}
    print(f"{len(feats)} feature sets x {len(scenarios)} scenarios x {len(runs)} defences "
          f"x {5 * a.reps} folds (targeted: one run per eligible segment in each test fold)")

    rows, t_start = [], time.time()
    for rep, fold, tr, te in iter_folds(ds.y, ds.block, reps=a.reps):
        seed = 1000 * rep + fold
        meta = {"rep": rep, "fold": fold, "n_train": len(tr), "n_test": len(te)}
        targets_here = [t for t in eligible if np.any(ds.attack_id[te] == t)]
        kw = dict(seed=seed, attack_id_te=ds.attack_id[te], block_te=ds.block[te],
                  workload_te=None if W is None else W[te], persistence=persistence)
        for fs, F in feats.items():
            for mode, rate in scenarios:
                corrs = ([(t, Corruption("targeted", rate, (t,))) for t in targets_here]
                         if mode == "targeted" else [(None, Corruption(mode, rate))])
                for target, corr in corrs:
                    for label, cfg, dfn in runs:
                        res = run_closed_loop(F[tr], ds.y[tr], F[te], ds.y[te], cfg,
                                              corruption=corr, defence=dfn, **kw)
                        rows.append({"features": fs, "corruption": mode, "rate": rate,
                                     "defence": label, "target_id": target, **meta, **res})
        print(f"rep {rep} fold {fold} done ({time.time() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "poisoning_folds.csv", index=False)

    metrics = ("F1", "PR_AUC", "FPR", "recall", "n_labels_used")
    table = summary_table(df, ["features", "corruption", "rate", "defence"], metrics)
    table.to_csv(out / "poisoning_summary.csv", index=False)
    print(table.to_string(index=False))

    tgt = df[df.corruption == "targeted"]
    if not tgt.empty:
        t_table = summary_table(tgt, ["features", "rate", "defence"],
                                ("target_recall", "nontarget_recall", "F1"))
        t_table.to_csv(out / "poisoning_targeted_recall.csv", index=False)
        print("\nTargeted poisoning, per target segment (rate 0 = clean reference for the "
              "same segment):")
        print(t_table.to_string(index=False))

    if df["rep"].nunique() * df["fold"].nunique() >= 6:
        tests = []
        for (fs, mode, rate), grp in df.groupby(["features", "corruption", "rate"]):
            t = compare_to_reference(grp, "defence", "none", "F1")
            t.insert(0, "scenario", f"{fs}/{mode}/{rate}")
            tests.append(t)
        tests = pd.concat(tests)
        tests.to_csv(out / "poisoning_defence_tests.csv", index=False)
        print("\nDefence vs no defence (F1). mean_diff = none - defence, so negative means the "
              "defence helps. Holm-adjusted within each scenario:")
        print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Poisoning study: how corrupted analyst labels affect the closed-loop IDS, and which
lightweight defences recover performance.

Corruption scenarios (applied only to the labels the analyst returns):
  random        - each label flipped with probability r (honest mistakes)
  missed_attack - each true attack label turned into "normal" with probability r
  targeted      - every label of the chosen attack segments turned into "normal"
Defences: none, filter, clip, rollback, and '+' combinations (e.g. filter+clip).

Example:
  python run_poisoning.py --attack-csv data/SWaT_Dataset_Attack_v0.csv \
      --features raw improved --rates 0.05 0.1 0.2 0.3 \
      --defences none filter clip rollback filter+clip
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    p.add_argument("--modes", nargs="+", default=["random", "missed_attack", "targeted"])
    p.add_argument("--targeted-k", type=int, default=3,
                   help="number of longest attack segments in the subset used as targets")
    p.add_argument("--defences", nargs="+", default=["none", "filter", "clip", "rollback"])
    p.add_argument("--block-selection", choices=["spread", "random", "all"], default="spread")
    p.add_argument("--block-seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--eta0", type=float, default=0.01)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--query", choices=["random", "uncertainty", "deviation"], default="random")
    p.add_argument("--persistence", default=None, help="k,n alarm rule, e.g. 3,5")
    p.add_argument("--workload", choices=["none", "pumps"], default="none",
                   help="W_t source for the threshold rule (paper does not specify one for SWaT)")
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
    seg_sizes = pd.Series(ds.attack_id[ds.y == 1]).value_counts()
    targets = tuple(int(t) for t in seg_sizes.index[: a.targeted_k])
    print(f"Targeted attack segments (longest {a.targeted_k} in subset): {targets} "
          f"with {seg_sizes.iloc[: a.targeted_k].tolist()} rows")

    scenarios = [("none", 0.0)]
    for mode in a.modes:
        if mode == "targeted":
            scenarios.append(("targeted", 1.0))
        else:
            scenarios += [(mode, r) for r in a.rates]
    defences = [Defence.parse(s) for s in a.defences]
    cfg = LoopConfig(eta0=a.eta0, update_epochs=a.update_epochs, query=a.query)
    feats = {fs: build_features(ds.X, ds.block, fs)[0] for fs in a.features}
    print(f"{len(feats)} feature sets x {len(scenarios)} scenarios x {len(defences)} defences "
          f"x {5 * a.reps} folds")

    rows, t_start = [], time.time()
    for rep, fold, tr, te in iter_folds(ds.y, ds.block, reps=a.reps):
        seed = 1000 * rep + fold
        meta = {"rep": rep, "fold": fold, "n_train": len(tr), "n_test": len(te)}
        for fs, F in feats.items():
            for mode, rate in scenarios:
                corr = Corruption(mode=mode, rate=rate,
                                  target_ids=targets if mode in ("targeted", "none") else ())
                for dfn in defences:
                    res = run_closed_loop(F[tr], ds.y[tr], F[te], ds.y[te], cfg, seed=seed,
                                          attack_id_te=ds.attack_id[te], block_te=ds.block[te],
                                          workload_te=None if W is None else W[te],
                                          corruption=corr, defence=dfn, persistence=persistence)
                    rows.append({"features": fs, "corruption": mode, "rate": rate,
                                 "defence": dfn.label, **meta, **res})
        print(f"rep {rep} fold {fold} done ({time.time() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / "poisoning_folds.csv", index=False)

    metrics = ("F1", "PR_AUC", "FPR", "recall")
    table = summary_table(df, ["features", "corruption", "rate", "defence"], metrics)
    table.to_csv(out / "poisoning_summary.csv", index=False)
    print(table.to_string(index=False))

    tgt = df[df.corruption.isin(["none", "targeted"])]
    if "target_recall" in tgt:
        t_table = summary_table(tgt, ["features", "corruption", "defence"],
                                ("target_recall", "nontarget_recall"))
        t_table.to_csv(out / "poisoning_targeted_recall.csv", index=False)
        print("\nTargeted attack recall (clean vs poisoned):")
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

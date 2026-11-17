"""Pilot tuning of the online learner (eta0, update_epochs), which the base paper does not report.

Run it on a PILOT block subset that is not used for the reported results, pick one setting,
then fix it for every reported experiment. Example:

  python tune_online.py --attack-csv data/attack.csv --block-selection random --block-seed 99

For each setting it prints the full system and the same system without updates, so you can
see how much the analyst labels actually change the model.
"""
import argparse
import dataclasses
import itertools

import pandas as pd

from data import SubsetConfig, build_subset, iter_folds
from features import build_features
from online import LoopConfig, run_closed_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-csv", required=True)
    ap.add_argument("--block-selection", choices=["spread", "random"], default="random")
    ap.add_argument("--block-seed", type=int, default=99)
    ap.add_argument("--features", choices=["raw", "improved"], default="raw")
    ap.add_argument("--eta0", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.5, 1.0])
    ap.add_argument("--epochs", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()

    ds, info = build_subset(a.attack_csv, SubsetConfig(selection=a.block_selection,
                                                       seed=a.block_seed))
    print("Pilot subset:", info)
    F, _ = build_features(ds.X, ds.block, a.features)
    folds = list(iter_folds(ds.y, ds.block, reps=a.reps))

    rows = []
    for eta0, epochs in itertools.product(a.eta0, a.epochs):
        cfg = LoopConfig(eta0=eta0, update_epochs=epochs)
        no_upd = dataclasses.replace(cfg, incremental=False)
        for rep, fold, tr, te in folds:
            seed = 1000 * rep + fold
            kw = dict(seed=seed, attack_id_te=ds.attack_id[te], block_te=ds.block[te])
            full = run_closed_loop(F[tr], ds.y[tr], F[te], ds.y[te], cfg, **kw)
            static = run_closed_loop(F[tr], ds.y[tr], F[te], ds.y[te], no_upd, **kw)
            rows.append({"eta0": eta0, "epochs": epochs, "F1_full": full["F1"],
                         "F1_no_updates": static["F1"], "FPR_full": full["FPR"],
                         "PR_AUC_full": full["PR_AUC"],
                         "rel_update": full["mean_rel_update"]})
        r = pd.DataFrame(rows).query("eta0 == @eta0 and epochs == @epochs").mean()
        print(f"eta0={eta0:<5} epochs={epochs}: F1 full {r.F1_full:.4f} | no updates "
              f"{r.F1_no_updates:.4f} | gain {r.F1_full - r.F1_no_updates:+.4f} | "
              f"FPR {r.FPR_full:.4f} | mean relative weight change per update {r.rel_update:.4f}")

    import os
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tune_online.csv", index=False)


if __name__ == "__main__":
    main()

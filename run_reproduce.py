"""Reproduce the confirmatory SWaT experiments of Al Ghamdi et al. (Sci. Rep. 2026).

  * Experiment 9  - parallel full-minus-one ablation of the closed-loop system
  * Experiment 10 - batch machine-learning baselines on the same 25 folds

Example:
  python run_reproduce.py --attack-csv data/SWaT_Dataset_Attack_v0.csv
  python run_reproduce.py --attack-csv data/SWaT_Dataset_Attack_v0.csv --features improved \
      --persistence 3,5 --calibrate
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import MODEL_NAMES, run_batch
from data import SubsetConfig, build_subset, iter_folds
from evaluation import compare_to_reference, summary_table
from features import build_features, pump_workload
from online import LoopConfig, run_closed_loop


def ablation_configs(base: LoopConfig) -> dict[str, LoopConfig]:
    r = dataclasses.replace
    return {
        "Full system": base,
        "Full minus context thresholding": r(base, context_threshold=False),
        "Full minus behavioural profiling": r(base, profiling=False),
        "Full minus error-rate feedback": r(base, error_feedback=False),
        "Full minus incremental SGD": r(base, incremental=False),
        "Fixed-threshold LR": r(base, context_threshold=False, profiling=False,
                                error_feedback=False, incremental=False),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--attack-csv", required=True)
    p.add_argument("--features", choices=["raw", "improved"], default="raw")
    p.add_argument("--block-selection", choices=["spread", "random", "all"], default="spread")
    p.add_argument("--block-seed", type=int, default=0)
    p.add_argument("--n-attack-blocks", type=int, default=20)
    p.add_argument("--n-normal-blocks", type=int, default=20)
    p.add_argument("--reps", type=int, default=5, help="repetitions of 5-fold CV (max 5)")
    p.add_argument("--eta0", type=float, default=0.01)
    p.add_argument("--update-epochs", type=int, default=1)
    p.add_argument("--query", choices=["random", "uncertainty", "deviation"], default="random")
    p.add_argument("--persistence", default=None, help="k,n alarm rule, e.g. 3,5")
    p.add_argument("--workload", choices=["none", "pumps"], default="none",
                   help="W_t source for the threshold rule (paper does not specify one for SWaT)")
    p.add_argument("--calibrate", action="store_true", help="tune baseline thresholds on validation")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    persistence = tuple(int(v) for v in a.persistence.split(",")) if a.persistence else None
    tag = f"{a.features}" + (f"_persist{a.persistence.replace(',', 'of')}" if persistence else "")

    ds, info = build_subset(a.attack_csv, SubsetConfig(
        n_attack_blocks=a.n_attack_blocks, n_normal_blocks=a.n_normal_blocks,
        selection=a.block_selection, seed=a.block_seed))
    print("Data:", info)
    W = pump_workload(ds.X) if a.workload == "pumps" else None
    print("Base paper reports: 449,919 file rows, 54,621 attack rows, 51 columns, "
          "subset 39,919 rows / 12,952 attack rows")
    F, names = build_features(ds.X, ds.block, a.features)
    print(f"Feature set '{a.features}': {F.shape[1]} features")

    configs = ablation_configs(LoopConfig(eta0=a.eta0, update_epochs=a.update_epochs, query=a.query))
    rows, t_start = [], time.time()
    for rep, fold, tr, te in iter_folds(ds.y, ds.block, reps=a.reps):
        seed = 1000 * rep + fold
        meta = {"rep": rep, "fold": fold, "n_train": len(tr), "n_test": len(te)}
        for name, cfg in configs.items():
            res = run_closed_loop(F[tr], ds.y[tr], F[te], ds.y[te], cfg, seed=seed,
                                  attack_id_te=ds.attack_id[te], block_te=ds.block[te],
                                  workload_te=None if W is None else W[te],
                                  persistence=persistence)
            rows.append({"experiment": "ablation", "config": name, **meta, **res})
        if not a.skip_baselines:
            for name in MODEL_NAMES:
                res = run_batch(name, F[tr], ds.y[tr], F[te], ds.y[te], seed=seed,
                                groups_tr=ds.block[tr], block_te=ds.block[te],
                                calibrate=a.calibrate, persistence=persistence)
                rows.append({"experiment": "baseline", "config": name, **meta, **res})
        print(f"rep {rep} fold {fold} done ({time.time() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"reproduce_{tag}_folds.csv", index=False)

    for exp, ref in (("ablation", "Full system"), ("baseline", "LogisticRegression")):
        sub = df[df.experiment == exp]
        if sub.empty:
            continue
        print(f"\n=== {exp}: mean ± sd [95% CI] over {sub.groupby(['rep', 'fold']).ngroups} folds ===")
        table = summary_table(sub, "config")
        print(table.to_string(index=False))
        table.to_csv(out / f"reproduce_{tag}_{exp}_summary.csv", index=False)
        if sub["rep"].nunique() * sub["fold"].nunique() >= 6:
            tests = pd.concat([compare_to_reference(sub, "config", ref, m)
                               for m in ("F1", "FPR", "PR_AUC")])
            print(tests.round(4).to_string(index=False))
            tests.to_csv(out / f"reproduce_{tag}_{exp}_tests.csv", index=False)


if __name__ == "__main__":
    main()

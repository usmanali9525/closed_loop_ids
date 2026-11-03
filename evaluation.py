"""Metrics, alarm persistence, summary tables and paired statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    pr_auc = average_precision_score(y, score) if 0 < y.sum() < len(y) else np.nan
    return {"F1": f1, "precision": precision, "recall": recall, "FPR": fpr,
            "PR_AUC": pr_auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def apply_persistence(pred: np.ndarray, block: np.ndarray, k: int, n: int) -> np.ndarray:
    """Raise an alarm only if at least k of the last n predictions (same block) are positive."""
    out = np.zeros_like(pred)
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        trailing = np.convolve(pred[idx].astype(int), np.ones(n, dtype=int))[: len(idx)]
        out[idx] = (trailing >= k).astype(pred.dtype)
    return out


def ci95(values) -> tuple[float, float, float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        m = float(v.mean()) if len(v) else np.nan
        return m, np.nan, np.nan, np.nan
    m, sd = v.mean(), v.std(ddof=1)
    half = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return m, sd, m - half, m + half


def summary_table(df: pd.DataFrame, by, metrics=("F1", "PR_AUC", "FPR")) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, grp in df.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(grp)
        for m in metrics:
            mean, sd, lo, hi = ci95(grp[m])
            row[m] = f"{mean:.4f} ± {sd:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def nadeau_bengio_p(diff: np.ndarray, n_train: float, n_test: float, k: int, r: int) -> float:
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(diff, float)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1.0 / (k * r) + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k * r - 1))


def compare_to_reference(df: pd.DataFrame, label_col: str, reference: str, metric: str,
                         k: int = 5) -> pd.DataFrame:
    """Paired comparison of every configuration against `reference` on matched folds.

    Reports Wilcoxon signed-rank (as in the base paper) and the Nadeau-Bengio corrected
    t-test, each Holm-adjusted across the comparisons in this table.
    """
    wide = df.pivot_table(index=["rep", "fold"], columns=label_col, values=metric)
    r = df["rep"].nunique()
    n_train, n_test = df["n_train"].mean(), df["n_test"].mean()
    rows = []
    for other in [c for c in wide.columns if c != reference]:
        pair = wide[[reference, other]].dropna()
        d = (pair[reference] - pair[other]).to_numpy()
        p_w = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue)
        rows.append({"comparison": f"{reference} vs {other}", "metric": metric,
                     "mean_diff": d.mean(), "n_pairs": len(d), "p_wilcoxon": p_w,
                     "p_nadeau_bengio": nadeau_bengio_p(d, n_train, n_test, k, r)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
        out["p_nb_holm"] = holm(out["p_nadeau_bengio"])
    return out

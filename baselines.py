"""Batch baselines (Experiment 10 of the base paper) on the same folds and preprocessing."""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import apply_persistence, binary_metrics

MODEL_NAMES = ("LogisticRegression", "DecisionTree", "RandomForest", "LinearSVM",
               "ExtraTrees", "MLP")


def make_model(name: str, seed: int):
    if name == "LogisticRegression":
        return SGDClassifier(loss="log_loss", penalty="l1", alpha=1e-4, max_iter=1000,
                             tol=1e-3, average=True, random_state=seed)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "LinearSVM":
        return LinearSVC(max_iter=5000, random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True,
                             random_state=seed)
    raise ValueError(f"Unknown model '{name}'")


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], 0.5
    return model.decision_function(X), 0.0


def _best_f1_threshold(y, s, default):
    if not 0 < y.sum() < len(y):
        return default
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    pred = s[None, :] >= cands[:, None]
    tp = (pred & (y == 1)).sum(1); fp = (pred & (y == 0)).sum(1); fn = ((~pred) & (y == 1)).sum(1)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    return float(cands[np.argmax(f1)])


def run_batch(name, Xtr, ytr, Xte, yte, *, seed, groups_tr=None, block_te=None,
              calibrate=False, persistence=None) -> dict:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float64)
    Xte_s = scaler.transform(Xte).astype(np.float64)
    model = make_model(name, seed)

    threshold = None
    if calibrate:   # choose the F1-optimal threshold on a block-grouped split of the training fold
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(Xtr_s, ytr, groups_tr))
        m = clone(model).fit(Xtr_s[itr], ytr[itr])
        s_val, default = _scores(m, Xtr_s[iva])
        threshold = _best_f1_threshold(ytr[iva], s_val, default)

    t0 = time.perf_counter()
    model.fit(Xtr_s, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    s, default = _scores(model, Xte_s)
    infer_ms = 1000 * (time.perf_counter() - t0) / len(yte)
    pred = (s >= (default if threshold is None else threshold)).astype(np.int8)
    if persistence:
        pred = apply_persistence(pred, block_te, *persistence)

    res = binary_metrics(yte, pred, s)
    res.update({"threshold": default if threshold is None else threshold,
                "fit_seconds": fit_s, "inference_ms_per_sample": infer_ms})
    return res

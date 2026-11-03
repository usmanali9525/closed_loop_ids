"""Feature sets.

raw      : the 51 numeric sensor/actuator values, as in the base paper.
improved : continuous sensors + one-hot actuator states + causal rolling
           statistics computed *within each temporal block* (past rows only),
           so no information crosses block (and therefore fold) boundaries.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ACTUATOR_RE = re.compile(r"^(MV|P|UV)\d{3}$")   # MV101, P101, UV401 ... (not PIT/FIT)
ACTUATOR_STATES = (0, 1, 2)                     # SWaT: MV 0/1/2, pumps and UV 1/2


def split_columns(columns):
    act = [c for c in columns if ACTUATOR_RE.match(c)]
    cont = [c for c in columns if c not in set(act)]
    return cont, act


def build_features(X: pd.DataFrame, block: np.ndarray, feature_set: str = "raw",
                   windows=(10, 30, 120)):
    """Return (feature matrix float32, feature names)."""
    if feature_set == "raw":
        return X.to_numpy(np.float32), list(X.columns)
    if feature_set != "improved":
        raise ValueError(f"Unknown feature set '{feature_set}'")

    X = X.reset_index(drop=True)
    cont, act = split_columns(list(X.columns))
    C = X[cont]
    g = C.groupby(block)
    parts = [C]

    if act:
        A = X[act].round().astype(int)
        onehot = {f"{c}=={v}": (A[c] == v).astype(np.float32)
                  for c in act for v in ACTUATOR_STATES}
        parts.append(pd.DataFrame(onehot))

    parts.append((C - g.shift(1)).fillna(0.0).add_suffix("_d1"))
    for w in windows:
        mean = g.rolling(w, min_periods=1).mean().droplevel(0).sort_index()
        std = g.rolling(w, min_periods=2).std().droplevel(0).sort_index().fillna(0.0)
        diff = (C - g.shift(w)).fillna(0.0)
        parts += [mean.add_suffix(f"_mean{w}"), std.add_suffix(f"_std{w}"),
                  diff.add_suffix(f"_d{w}")]

    F = pd.concat(parts, axis=1)
    return F.to_numpy(np.float32), list(F.columns)


PUMP_RE = re.compile(r"^P\d{3}$")


def pump_workload(X: pd.DataFrame) -> np.ndarray:
    """Optional workload proxy W_t in [0, 1]: share of pumps running (SWaT state 2 = on).

    The base paper does not say how W_t was measured on SWaT; this is an assumption.
    """
    pumps = [c for c in X.columns if PUMP_RE.match(c)]
    if not pumps:
        return np.zeros(len(X))
    return (X[pumps].round() == 2).mean(axis=1).to_numpy(float)

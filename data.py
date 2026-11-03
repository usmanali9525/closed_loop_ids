"""Data loading and leakage-controlled temporal blocking for SWaT-style CSV files.

Follows the confirmatory protocol of Al Ghamdi et al. (Sci. Rep. 2026):
the attack file is cut into non-overlapping blocks of consecutive rows,
20 attack-containing and 20 normal-only blocks are selected, and folds are
built with StratifiedGroupKFold so that every block stays inside one fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

LABEL_CANDIDATES = ("Normal/Attack", "label", "attack")
TIME_CANDIDATES = ("Timestamp", "time")
FOLD_SEEDS = (42, 43, 44, 45, 46)  # seeds used in the base paper


@dataclass
class Dataset:
    X: pd.DataFrame        # numeric sensor/actuator columns, time order
    y: np.ndarray          # 1 = attack, 0 = normal
    block: np.ndarray      # temporal block id
    attack_id: np.ndarray  # 0 = normal, k = k-th contiguous attack segment in the file
    row: np.ndarray        # row index in the original file


@dataclass
class SubsetConfig:
    block_size: int = 1000
    n_attack_blocks: int = 20
    n_normal_blocks: int = 20
    selection: str = "spread"  # spread | random | all
    seed: int = 0


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def _find_col(columns, candidates):
    lookup = {_norm(c): c for c in columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


def load_swat(path: str):
    """Load a SWaT CSV. Handles stray spaces in headers and the 'A ttack' label typo."""
    df = pd.read_csv(path, low_memory=False)
    if _find_col(df.columns, LABEL_CANDIDATES) is None:
        df = pd.read_csv(path, header=1, low_memory=False)  # some copies have a title row
    df.columns = [str(c).strip() for c in df.columns]

    label_col = _find_col(df.columns, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"No label column found. First columns: {list(df.columns)[:8]}")
    time_col = _find_col(df.columns, TIME_CANDIDATES)

    labels = df[label_col].astype(str).str.replace(" ", "", regex=False).str.lower()
    unexpected = set(labels.unique()) - {"normal", "attack"}
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")
    y = (labels == "attack").to_numpy(dtype=np.int8)

    drop = [label_col] + ([time_col] if time_col else [])
    X = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        X = X.ffill().bfill()
    return X.astype(np.float32), y, n_missing


def attack_segments(y: np.ndarray) -> np.ndarray:
    """Give each contiguous run of attack rows its own id (1, 2, ...); normal rows get 0."""
    starts = np.diff(np.concatenate([[0], y.astype(int)])) == 1
    return np.where(y == 1, np.cumsum(starts), 0).astype(np.int32)


def select_blocks(y: np.ndarray, block: np.ndarray, cfg: SubsetConfig) -> np.ndarray:
    has_attack = pd.Series(y).groupby(block).max()
    att = has_attack.index[has_attack == 1].to_numpy()
    nor = has_attack.index[has_attack == 0].to_numpy()
    if cfg.selection == "all":
        return np.sort(np.concatenate([att, nor]))
    if cfg.n_attack_blocks > len(att) or cfg.n_normal_blocks > len(nor):
        raise ValueError(f"Asked for {cfg.n_attack_blocks}/{cfg.n_normal_blocks} attack/normal "
                         f"blocks but file has {len(att)}/{len(nor)}")
    if cfg.selection == "spread":   # deterministic, evenly spaced through time
        pick = lambda c, k: c[np.round(np.linspace(0, len(c) - 1, k)).astype(int)]
    elif cfg.selection == "random":
        rng = np.random.default_rng(cfg.seed)
        pick = lambda c, k: rng.choice(c, k, replace=False)
    else:
        raise ValueError(f"Unknown selection '{cfg.selection}'")
    return np.sort(np.concatenate([pick(att, cfg.n_attack_blocks), pick(nor, cfg.n_normal_blocks)]))


def build_subset(path: str, cfg: SubsetConfig = SubsetConfig()):
    X, y, n_missing = load_swat(path)
    block = np.arange(len(y)) // cfg.block_size
    aid = attack_segments(y)
    chosen = select_blocks(y, block, cfg)
    mask = np.isin(block, chosen)
    ds = Dataset(X=X.loc[mask].reset_index(drop=True), y=y[mask], block=block[mask],
                 attack_id=aid[mask], row=np.flatnonzero(mask))
    info = {
        "file_rows": len(y), "file_attack_rows": int(y.sum()), "n_columns": X.shape[1],
        "missing_values_filled": n_missing, "attack_segments_in_file": int(aid.max()),
        "subset_blocks": len(chosen), "subset_rows": int(mask.sum()),
        "subset_attack_rows": int(ds.y.sum()),
    }
    return ds, info


def iter_folds(y: np.ndarray, groups: np.ndarray, reps: int = 5, n_splits: int = 5):
    """Repeated StratifiedGroupKFold; yields (rep, fold, train_idx, test_idx) in time order."""
    for rep, seed in enumerate(FOLD_SEEDS[:reps]):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            yield rep, fold, np.sort(tr), np.sort(te)

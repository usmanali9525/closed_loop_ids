"""Closed-loop adaptive IDS (re-implementation of Al Ghamdi et al., Sci. Rep. 2026),
with hooks for corrupted analyst labels and lightweight defences.

Per test fold the stream is processed in time order, in `n_updates` equal chunks:
  1. score the chunk with the current model;
  2. for each event: update the behavioural profile, compute the adaptive threshold
     tau_t = clip(tau0 - a*L_t + b*W_t + g*FP_t - d*FN_t, 0.05, 0.95), decide;
  3. the analyst labels `label_budget / n_updates` events from the chunk
     (labels may be corrupted); those events are excluded from evaluation;
  4. the model is updated with one SGD pass on those labels (optionally defended).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from evaluation import apply_persistence, binary_metrics


@dataclass(frozen=True)
class LoopConfig:
    # components (switched off one at a time in the ablation)
    context_threshold: bool = True
    profiling: bool = True
    error_feedback: bool = True
    incremental: bool = True
    # threshold rule, coefficients as reported in the base paper
    tau0: float = 0.50
    alpha: float = 0.30
    beta: float = 0.20
    gamma: float = 0.10
    delta: float = 0.20
    fb_window: int = 200
    external_threat: float = 0.0   # SWaT has no external threat feed
    workload: float = 0.0          # SWaT has no workload signal
    # learning
    warmup_size: int = 256
    label_budget: int = 256
    n_updates: int = 16
    sgd_alpha: float = 1e-4
    eta0: float = 0.01             # not reported in the paper; fix it on a pilot subset
    update_epochs: int = 1
    query: str = "random"          # random | uncertainty | deviation
    # behavioural profiler
    profile_min_obs: int = 10
    profile_z_clip: float = 5.0
    profile_gate_pct: float = 80.0
    profile_gate_refresh: int = 50


@dataclass(frozen=True)
class Corruption:
    mode: str = "none"             # none | random | missed_attack | targeted
    rate: float = 0.0
    target_ids: tuple = ()         # attack-segment ids for the targeted mode


@dataclass(frozen=True)
class Defence:
    names: tuple = ()              # any of: filter, clip, rollback
    filter_margin: float = 0.9     # drop a label if the model gives the opposite class >= this
    clip_rel: float = 0.1          # max weight change per update, relative to the weight norm
    rollback_tol: float = 0.02     # max allowed drop in warm-up-set F1 per update
    spec: str = "none"

    PARAM = {"filter": "filter_margin", "clip": "clip_rel", "rollback": "rollback_tol"}

    @classmethod
    def parse(cls, spec: str) -> "Defence":
        """Parse e.g. 'none', 'filter', 'clip@0.05', 'filter@0.99+rollback@0.1'."""
        if spec in ("", "none"):
            return cls()
        names, params = [], {}
        for token in spec.split("+"):
            name, _, value = token.partition("@")
            if name not in cls.PARAM:
                raise ValueError(f"Unknown defence '{name}' in '{spec}'")
            names.append(name)
            if value:
                params[cls.PARAM[name]] = float(value)
        return cls(names=tuple(names), spec=spec, **params)

    @property
    def label(self) -> str:
        return self.spec


class Profiler:
    """Label-free behavioural baseline: running mean/variance (Welford) per feature."""

    def __init__(self, d: int, cfg: LoopConfig):
        self.cfg = cfg
        self.n = 0
        self.mean = np.zeros(d)
        self.m2 = np.zeros(d)
        self.history = deque(maxlen=1000)
        self.gate = np.inf
        self.steps = 0

    def step(self, x: np.ndarray) -> float:
        if self.n >= self.cfg.profile_min_obs:
            std = np.sqrt(self.m2 / (self.n - 1)) + 1e-6
            z = np.minimum(np.abs(x - self.mean) / std, self.cfg.profile_z_clip)
            dev = float(np.tanh(z.mean()))
        else:
            dev = 0.0
        # baseline refresh gated by the deviation score itself, never by labels
        if self.n < self.cfg.profile_min_obs or dev <= self.gate:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)
        self.history.append(dev)
        self.steps += 1
        if self.steps % self.cfg.profile_gate_refresh == 0:
            self.gate = float(np.percentile(self.history, self.cfg.profile_gate_pct))
        return dev


class ErrorWindow:
    """Rolling false-positive and false-negative rates over the last `size` events."""

    def __init__(self, size: int):
        self.size, self.buf = size, deque()
        self.tp = self.fp = self.fn = self.tn = 0

    def _count(self, y: int, h: int, s: int):
        if h and y: self.tp += s
        elif h: self.fp += s
        elif y: self.fn += s
        else: self.tn += s

    def push(self, y: int, h: int):
        self._count(y, h, 1)
        self.buf.append((y, h))
        if len(self.buf) > self.size:
            self._count(*self.buf.popleft(), -1)

    def rates(self):
        fpr = self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0
        fnr = self.fn / (self.fn + self.tp) if self.fn + self.tp else 0.0
        return fpr, fnr


def sample_warmup(y: np.ndarray, size: int, rng) -> np.ndarray:
    size = min(size, len(y))
    for _ in range(100):
        idx = rng.choice(len(y), size, replace=False)
        if 0 < y[idx].sum() < size:
            return np.sort(idx)
    raise ValueError("Could not draw a warm-up batch containing both classes")


def select_queries(chunk, p, tau, dev, strategy: str, k: int, rng) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if strategy == "random":
        return np.sort(rng.choice(chunk, k, replace=False))
    if strategy == "uncertainty":
        order = np.argsort(np.abs(p - tau))
    elif strategy == "deviation":
        order = np.argsort(-dev)
    else:
        raise ValueError(f"Unknown query strategy '{strategy}'")
    return np.sort(chunk[order[:k]])


def corrupt_labels(y: np.ndarray, aid: np.ndarray, c: Corruption, rng) -> np.ndarray:
    """Simulate analyst errors or poisoning on the labels the analyst returns."""
    y = y.copy()
    if c.mode == "none" or len(y) == 0:
        return y
    u = rng.random(len(y))
    if c.mode == "random":
        flip = u < c.rate
        y[flip] = 1 - y[flip]
    elif c.mode == "missed_attack":
        y[(y == 1) & (u < c.rate)] = 0
    elif c.mode == "targeted":
        y[(y == 1) & np.isin(aid, c.target_ids) & (u < c.rate)] = 0
    else:
        raise ValueError(f"Unknown corruption mode '{c.mode}'")
    return y


def _get_w(m):
    return np.concatenate([m.coef_.ravel(), m.intercept_.ravel()])


def _set_w(m, w):
    m.coef_ = w[:-1].reshape(1, -1).copy()
    m.intercept_ = w[-1:].copy()


def _anchor_f1(m, X, y, tau: float) -> float:
    return f1_score(y, (m.predict_proba(X)[:, 1] >= tau).astype(int), zero_division=0)


def run_closed_loop(Xtr, ytr, Xte, yte, cfg: LoopConfig, *, seed: int,
                    attack_id_te=None, block_te=None, workload_te=None,
                    corruption: Corruption = Corruption(),
                    defence: Defence = Defence(), persistence=None) -> dict:
    rng_warm = np.random.default_rng([seed, 0])   # same warm-up for every configuration
    rng_query = np.random.default_rng([seed, 1])  # same random queries for every configuration
    rng_corrupt = np.random.default_rng([seed, 2])
    if attack_id_te is None:
        attack_id_te = np.zeros(len(yte), dtype=int)

    scaler = StandardScaler().fit(Xtr)            # fitted on the training fold only
    Xtr_s = scaler.transform(Xtr).astype(np.float64)
    Xte_s = scaler.transform(Xte).astype(np.float64)

    # warm-up batch (trusted; also used as the rollback anchor)
    w_idx = sample_warmup(ytr, cfg.warmup_size, rng_warm)
    Xw, yw = Xtr_s[w_idx], ytr[w_idx]
    batch = SGDClassifier(loss="log_loss", penalty="l1", alpha=cfg.sgd_alpha, max_iter=1000,
                          tol=1e-3, learning_rate="optimal", average=True, random_state=seed)
    batch.fit(Xw, yw)
    model = SGDClassifier(loss="log_loss", penalty="l1", alpha=cfg.sgd_alpha,
                          learning_rate="constant", eta0=cfg.eta0, random_state=seed)
    model.partial_fit(Xw, yw, classes=np.array([0, 1]))
    _set_w(model, _get_w(batch))                  # online phase starts from the batch model

    n, d = Xte_s.shape
    prob, tau_all, dev_all = np.empty(n), np.empty(n), np.zeros(n)
    pred = np.zeros(n, dtype=np.int8)
    queried = np.zeros(n, dtype=bool)
    profiler = Profiler(d, cfg) if cfg.profiling else None
    window = ErrorWindow(cfg.fb_window)
    per_update = cfg.label_budget // cfg.n_updates
    log = {"n_corrupted": 0, "n_filtered": 0, "n_clipped": 0, "n_rollbacks": 0,
           "n_labels_used": 0}
    rel_updates = []   # diagnostic: relative weight change per update

    for chunk in np.array_split(np.arange(n), cfg.n_updates):
        if len(chunk) == 0:
            continue
        prob[chunk] = model.predict_proba(Xte_s[chunk])[:, 1]

        for i in chunk:
            dev = profiler.step(Xte_s[i]) if profiler else 0.0
            tau = cfg.tau0
            if cfg.context_threshold:
                threat = 0.5 * dev + 0.5 * cfg.external_threat if profiler else cfg.external_threat
                load = workload_te[i] if workload_te is not None else cfg.workload
                tau += -cfg.alpha * threat + cfg.beta * load
            if cfg.error_feedback:
                fpr_w, fnr_w = window.rates()
                tau += cfg.gamma * fpr_w - cfg.delta * fnr_w
            tau = min(max(tau, 0.05), 0.95)
            yhat = int(prob[i] >= tau)
            dev_all[i], tau_all[i], pred[i] = dev, tau, yhat
            window.push(int(yte[i]), yhat)  # paper assumes every alert outcome is confirmed

        # analyst labels: queried in every configuration so evaluation sets stay matched
        k = min(per_update, len(chunk))
        q = select_queries(chunk, prob[chunk], tau_all[chunk], dev_all[chunk], cfg.query, k,
                           rng_query)
        queried[q] = True
        if not cfg.incremental or len(q) == 0:
            continue

        y_given = corrupt_labels(yte[q], attack_id_te[q], corruption, rng_corrupt)
        log["n_corrupted"] += int((y_given != yte[q]).sum())
        keep = np.ones(len(q), dtype=bool)
        if "filter" in defence.names:   # skip labels the current model strongly disputes
            pq = prob[q]
            keep = ~(((y_given == 0) & (pq >= defence.filter_margin)) |
                     ((y_given == 1) & (pq <= 1 - defence.filter_margin)))
            log["n_filtered"] += int((~keep).sum())
        if not keep.any():
            continue

        w_old = _get_w(model)
        f1_before = _anchor_f1(model, Xw, yw, cfg.tau0) if "rollback" in defence.names else None
        for _ in range(cfg.update_epochs):
            model.partial_fit(Xte_s[q][keep], y_given[keep])
        if "clip" in defence.names:     # bound how far one update can move the weights
            step = _get_w(model) - w_old
            limit = defence.clip_rel * max(np.linalg.norm(w_old), 1e-3)
            norm = np.linalg.norm(step)
            if norm > limit:
                _set_w(model, w_old + step * (limit / norm))
                log["n_clipped"] += 1
        if "rollback" in defence.names:  # undo updates that hurt the trusted warm-up set
            if _anchor_f1(model, Xw, yw, cfg.tau0) < f1_before - defence.rollback_tol:
                _set_w(model, w_old)
                log["n_rollbacks"] += 1
                continue
        log["n_labels_used"] += int(keep.sum())
        rel_updates.append(np.linalg.norm(_get_w(model) - w_old) / max(np.linalg.norm(w_old), 1e-12))

    final_pred = apply_persistence(pred, block_te, *persistence) if persistence else pred
    ev = ~queried
    res = binary_metrics(yte[ev], final_pred[ev], prob[ev])
    res.update(log)
    res["n_eval"] = int(ev.sum())
    res["mean_tau"] = float(tau_all.mean())
    res["mean_rel_update"] = float(np.mean(rel_updates)) if rel_updates else 0.0
    if corruption.target_ids:
        is_target = np.isin(attack_id_te, corruption.target_ids)
        t_mask = ev & (yte == 1) & is_target
        o_mask = ev & (yte == 1) & ~is_target
        res["target_recall"] = float(final_pred[t_mask].mean()) if t_mask.any() else np.nan
        res["nontarget_recall"] = float(final_pred[o_mask].mean()) if o_mask.any() else np.nan
    return res
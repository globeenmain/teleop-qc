"""Is a judge's confidence worth anything? Discrimination and calibration of a probabilistic judge against human
labels, recalibration, and the thresholds that decide which episodes skip human review.

Everywhere, p is the judge's probability that an episode succeeded and y the human label (1 = success).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _arrays(p, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("p and y must be 1-d arrays of the same length")
    if ((p < 0) | (p > 1)).any() or not np.isin(y, (0, 1)).all():
        raise ValueError("p must lie in [0, 1] and y in {0, 1}")
    return p, y


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    confidence: float  # mean predicted probability in the bin
    frequency: float  # observed share of successes in the bin


def reliability_bins(p, y, n_bins: int = 10, strategy: str = "uniform") -> list[Bin]:
    """Bins for a reliability diagram; "uniform" splits [0, 1] evenly, "quantile" gives bins of equal count."""
    p, y = _arrays(p, y)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for i in range(len(edges) - 1):
        mask = index == i
        if mask.any():
            bins.append(Bin(float(edges[i]), float(edges[i + 1]), int(mask.sum()), float(p[mask].mean()),
                            float(y[mask].mean())))
    return bins


def ece(p, y, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: the count-weighted gap between confidence and observed frequency."""
    p, _ = _arrays(p, y)
    return float(sum(b.count * abs(b.frequency - b.confidence) for b in reliability_bins(p, y, n_bins, strategy))
                 / len(p))


def brier(p, y) -> float:
    p, y = _arrays(p, y)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    p, y = _arrays(p, y)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(p, y) -> float:
    """Probability that a random success gets a higher score than a random failure (ties count half)."""
    p, y = _arrays(p, y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cohen_kappa(a, b) -> float:
    """Agreement between two labelings beyond what chance would give."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    observed = float(np.mean(a == b))
    expected = float(sum(np.mean(a == k) * np.mean(b == k) for k in labels))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * z))  # no overflow for large |z|


def fit_platt(p, y, iters: int = 100) -> Callable[[np.ndarray], np.ndarray]:
    """Platt scaling on the logit of p, with Platt's smoothed targets so that separable data stays finite."""
    p, y = _arrays(p, y)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    x = _logit(p)

    def loss(a: float, b: float) -> float:
        z = a * x + b
        return float(np.sum(np.logaddexp(0, z) - t * z))  # cross-entropy, computed stably

    a, b = 1.0, 0.0
    for _ in range(iters):  # damped Newton steps on the cross-entropy
        q = _sigmoid(a * x + b)
        g = np.array([np.sum((q - t) * x), np.sum(q - t)])
        w = q * (1 - q)
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        scale, current = 1.0, loss(a, b)
        while scale > 1e-8 and loss(a - scale * step[0], b - scale * step[1]) > current:
            scale /= 2
        a, b = a - scale * step[0], b - scale * step[1]
        if np.abs(scale * step).max() < 1e-10:
            break
    return lambda new: _sigmoid(a * _logit(np.asarray(new, dtype=np.float64)) + b)


def fit_isotonic(p, y) -> Callable[[np.ndarray], np.ndarray]:
    """Isotonic regression by pool-adjacent-violators; predictions interpolate linearly between block ends."""
    p, y = _arrays(p, y)
    xs, inverse = np.unique(p, return_inverse=True)
    weights = np.bincount(inverse).astype(np.float64)
    values = np.bincount(inverse, weights=y) / weights
    blocks: list[list[float]] = []  # [value, weight, first x index, last x index]
    for i, (v, w) in enumerate(zip(values, weights, strict=True)):
        blocks.append([v, w, i, i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, _, last = blocks.pop()
            v1, w1, first, _ = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, first, last])
    knots_x, knots_y = [], []
    for v, _, first, last in blocks:
        knots_x += [xs[first], xs[last]]
        knots_y += [v, v]
    knots_x, knots_y = np.asarray(knots_x), np.asarray(knots_y)
    return lambda new: np.interp(np.asarray(new, dtype=np.float64), knots_x, knots_y)


def cross_fit(p, y, method: str = "platt", k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold recalibrated probabilities: each episode is recalibrated by a map fitted on the other folds."""
    p, y = _arrays(p, y)
    fit = {"platt": fit_platt, "isotonic": fit_isotonic}[method]
    folds = np.random.default_rng(seed).permutation(len(p)) % k
    out = np.empty_like(p)
    for f in range(k):
        test = folds == f
        out[test] = fit(p[~test], y[~test])(p[test])
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class Threshold:
    value: float  # accept the episodes with p >= value (or reject those with p <= value)
    precision: float  # share of correct decisions among the episodes decided automatically
    coverage: float  # share of all episodes decided automatically
    count: int


def selective_threshold(p, y, target_precision: float = 0.95, accept: bool = True) -> Threshold | None:
    """Most permissive threshold whose automatic decisions reach target_precision.

    accept=True: episodes with p >= t are accepted as successes without review.
    accept=False: episodes with p <= t are rejected as failures without review.
    None when no threshold reaches the target.
    """
    p, y = _arrays(p, y)
    scores, wanted = (p, y == 1) if accept else (-p, y == 0)
    best = None
    for t in np.unique(scores):
        decided = scores >= t
        precision = float(wanted[decided].mean())
        if precision >= target_precision and (best is None or decided.sum() > best.count):
            value = float(t if accept else -t)
            best = Threshold(value, precision, float(decided.mean()), int(decided.sum()))
    return best


def bootstrap_ci(metric: Callable, p, y, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of metric(p, y), resampling episodes."""
    p, y = _arrays(p, y)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(p), len(p))
        value = metric(p[i], y[i])
        if np.isfinite(value):
            values.append(value)
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def summary(p, y) -> dict[str, float]:
    p, y = _arrays(p, y)
    verdict = (p >= 0.5).astype(int)
    return {
        "n": float(len(p)),
        "success_rate": float(y.mean()),
        "accuracy": float(np.mean(verdict == y)),
        "kappa": cohen_kappa(verdict, y),
        "auroc": auroc(p, y),
        "brier": brier(p, y),
        "log_loss": log_loss(p, y),
        "ece": ece(p, y),
        "ece_quantile": ece(p, y, strategy="quantile"),
    }

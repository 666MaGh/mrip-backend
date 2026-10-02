"""Calibration of classifier probabilities: scores, reliability, temperature, isotonic, abstain.

Pure functions on numpy arrays. ``probs`` is (n, k) with rows summing to 1,
``labels`` is (n,) of true class indices in [0, k), ``confidences`` is the (n,)
top-label probability and ``correct`` the (n,) "top label was right" flag.
Rules: no hidden state, no randomness; invalid input raises ValueError; zero
probabilities stay zero under temperature scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

_ROW_SUM_TOL = 1e-6
_MIN_FIT_N = 10


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    temperature: float
    n: int
    nll_before: float  # mean NLL at T = 1
    nll_after: float  # mean NLL at the fitted T


@dataclass(frozen=True, slots=True)
class AbstainFit:
    threshold: float
    precision: float  # accuracy of the accepted set {confidence >= threshold}
    coverage: float  # accepted / n
    n_accepted: int


@dataclass(frozen=True, slots=True)
class IsotonicMap:
    knots_x: tuple[float, ...]
    knots_y: tuple[float, ...]

    def apply(self, confidence: float | np.ndarray) -> float | np.ndarray:
        """Linear interpolation between knots; end values are held outside the range."""
        xs = np.asarray(self.knots_x, dtype=float)
        ys = np.asarray(self.knots_y, dtype=float)
        out = np.interp(np.asarray(confidence, dtype=float), xs, ys)
        if np.ndim(out) == 0:
            return float(out)
        return out


def _check_probs_labels(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 1:
        raise ValueError("probs must be a non-empty (n, k) array")
    if y.ndim != 1 or y.shape[0] != p.shape[0]:
        raise ValueError("labels must be a (n,) array matching probs")
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("labels must be integer class indices")
    y = y.astype(int)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("probs must be finite and non-negative")
    if np.any(np.abs(p.sum(axis=1) - 1.0) > _ROW_SUM_TOL):
        raise ValueError("rows of probs must sum to 1")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("labels out of range [0, k)")
    return p, y


def _check_conf_correct(
    confidences: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct)
    if c.ndim != 1 or c.shape[0] == 0 or ok.shape != c.shape:
        raise ValueError("confidences and correct must be non-empty (n,) arrays of equal length")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("confidences must lie in [0, 1]")
    if not np.all((ok == 0) | (ok == 1)):
        raise ValueError("correct must be boolean or 0/1")
    return c, ok.astype(float)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier: mean over samples of sum_k (p_k - 1[k == label])^2."""
    p, y = _check_probs_labels(probs, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log of the probability assigned to the true class (clipped at eps)."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    p, y = _check_probs_labels(probs, labels)
    true_p = np.clip(p[np.arange(p.shape[0]), y], eps, 1.0)
    return float(-np.mean(np.log(true_p)))


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Equal-width bins on [0, 1]; the last bin includes 1.0; empty bins are omitted."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    c, ok = _check_conf_correct(confidences, correct)
    idx = np.minimum(np.floor(c * n_bins).astype(int), n_bins - 1)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                lower=b / n_bins,
                upper=(b + 1) / n_bins,
                n=count,
                mean_confidence=float(c[mask].mean()),
                accuracy=float(ok[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = sum over bins of (n_bin / n) * |accuracy - mean_confidence|."""
    c = np.asarray(confidences, dtype=float)
    n = c.shape[0] if c.ndim == 1 else 0
    bins = reliability_bins(confidences, correct, n_bins)
    return float(sum(b.n / n * abs(b.accuracy - b.mean_confidence) for b in bins))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling p_i^(1/T), renormalised per row, computed in log space.

    T > 0 required. T = 1 returns the input values; zero probabilities stay zero.
    """
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2:
        raise ValueError("probs must be a (n, k) array")
    if temperature == 1.0:
        return p.copy()
    with np.errstate(divide="ignore"):
        scaled = np.log(p) / temperature  # zeros become -inf
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)  # exp(-inf) == 0
    return weights / weights.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> TemperatureFit:
    """Fit T minimising the mean NLL of the temperature-scaled distribution.

    The search runs over log T (symmetric around T = 1) inside ``bounds``.
    """
    lo, hi = bounds
    if not (0.0 < lo < hi) or not np.isfinite(hi):
        raise ValueError("bounds must satisfy 0 < lower < upper")
    p, y = _check_probs_labels(probs, labels)
    if p.shape[0] < _MIN_FIT_N:
        raise ValueError(f"need at least {_MIN_FIT_N} samples")

    def nll(log_t: float) -> float:
        return log_loss(apply_temperature(p, float(np.exp(log_t))), y)

    res = optimize.minimize_scalar(
        nll, bounds=(float(np.log(lo)), float(np.log(hi))), method="bounded",
        options={"xatol": 1e-9},
    )
    t = float(np.exp(res.x))
    return TemperatureFit(
        temperature=t, n=int(p.shape[0]), nll_before=log_loss(p, y), nll_after=nll(res.x)
    )


def fit_isotonic(confidences: np.ndarray, correct: np.ndarray) -> IsotonicMap:
    """Pool-adjacent-violators fit of correct (0/1) on confidence, non-decreasing.

    Tied confidences are merged into one point (mean outcome, weight = count).
    Each pooled block contributes knots at its first and last confidence.
    """
    c, ok = _check_conf_correct(confidences, correct)
    if c.shape[0] < _MIN_FIT_N:
        raise ValueError(f"need at least {_MIN_FIT_N} samples")
    xs, inverse = np.unique(c, return_inverse=True)
    weights = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=ok)

    # Each block: [value, weight, first_x_index, last_x_index]
    blocks: list[list[float]] = []
    for i in range(xs.shape[0]):
        blocks.append([sums[i] / weights[i], weights[i], i, i])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, _, last = blocks.pop()
            v1, w1, first, _ = blocks.pop()
            w = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / w, w, first, last])

    kx: list[float] = []
    ky: list[float] = []
    for value, _, first, last in blocks:
        kx.append(float(xs[int(first)]))
        ky.append(float(value))
        if last != first:
            kx.append(float(xs[int(last)]))
            ky.append(float(value))
    return IsotonicMap(knots_x=tuple(kx), knots_y=tuple(ky))


def fit_abstain_threshold(
    confidences: np.ndarray,
    correct: np.ndarray,
    target_precision: float,
    min_coverage: float = 0.0,
) -> AbstainFit | None:
    """Lowest observed confidence t whose accepted set {confidence >= t} qualifies.

    Qualifies means precision >= target_precision and coverage >= min_coverage.
    Returns None if no candidate qualifies. Because a lower threshold always has
    higher coverage, ``min_coverage`` can only turn a result into None.
    """
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must lie in [0, 1]")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1]")
    c, ok = _check_conf_correct(confidences, correct)
    n = c.shape[0]
    order = np.argsort(c, kind="stable")
    c_sorted, ok_sorted = c[order], ok[order]
    # hits_from[i] = number of correct among sorted samples i..n-1
    hits_from = np.cumsum(ok_sorted[::-1])[::-1]
    thresholds, first_idx = np.unique(c_sorted, return_index=True)
    tol = 1e-12
    for t, i in zip(thresholds, first_idx):
        accepted = n - int(i)
        precision = float(hits_from[i]) / accepted
        coverage = accepted / n
        if precision >= target_precision - tol and coverage >= min_coverage - tol:
            return AbstainFit(
                threshold=float(t), precision=precision, coverage=coverage, n_accepted=accepted
            )
    return None

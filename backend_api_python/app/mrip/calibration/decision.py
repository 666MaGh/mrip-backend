"""Segmented calibration of semantic decisions (LAYA) from observed labels (work 010).

Temperature scaling is label-agnostic, so a fitted temperature applies to any
decision's probability vector. Each segment (decision family, relationship type,
VIX regime, ...) also gets an ABSTAIN threshold on the CALIBRATED confidence that
meets a target precision on the calibration data. The UI shows the calibrated
confidence (e.g. raw 0.87 -> calibrated 0.69), never the raw one.

Samples must carry TRUE labels from observed outcomes or independent structured
sources; a model's own output is never a label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping, Sequence

import numpy as np

from app.mrip.calibration import classification as cls
from app.mrip.calibration.segments import SegmentEntry, fit_segments, resolve_segment
from app.mrip.calibration.store import CalibrationError, CalibrationKind, CalibrationStore, StoredEntry
from app.mrip.decision.types import AbstainReason, Decision

POLICY_VERSION = "decision-calibration-v0"


@dataclass(frozen=True, slots=True)
class DecisionSample:
    probabilities: Mapping[str, float]  # provider probability per label
    true_label: str  # observed truth
    attrs: Mapping[str, str]  # segment attributes (decision_family, relationship_type, vix_regime, ...)


@dataclass(frozen=True, slots=True)
class DecisionCalibrationParams:
    temperature: float
    abstain_threshold: float | None  # on calibrated confidence; None = no threshold reaches the target precision
    n: int


@dataclass(frozen=True, slots=True)
class FittedDecisionCalibration:
    entries: Mapping[str, SegmentEntry[DecisionCalibrationParams]]
    dims: tuple[str, ...]
    min_samples: int
    target_precision: float
    fit_as_of: date
    n_total: int
    version: str = "unsaved"
    metrics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CalibratedDecision:
    decision: Decision
    answer: str | None  # None when abstained (raw or after calibration)
    abstained: bool
    calibrated_confidence: float | None
    calibrated_probabilities: Mapping[str, float]
    raw_confidence: float | None
    calibrated: bool
    segment_key: str | None
    calibration_version: str


def _matrix(samples: Sequence[DecisionSample]) -> tuple[np.ndarray, np.ndarray]:
    labels = sorted({l for s in samples for l in s.probabilities} | {s.true_label for s in samples})
    index = {l: i for i, l in enumerate(labels)}
    probs = np.zeros((len(samples), len(labels)))
    for r, s in enumerate(samples):
        for label, p in s.probabilities.items():
            probs[r, index[label]] = p
        probs[r] /= probs[r].sum()
    return probs, np.array([index[s.true_label] for s in samples])


def _fit_params(samples: list[DecisionSample], target_precision: float) -> DecisionCalibrationParams:
    probs, y = _matrix(samples)
    fit = cls.fit_temperature(probs, y)
    scaled = cls.apply_temperature(probs, fit.temperature)
    confidence, correct = scaled.max(axis=1), scaled.argmax(axis=1) == y
    abstain = cls.fit_abstain_threshold(confidence, correct, target_precision)
    return DecisionCalibrationParams(fit.temperature, abstain.threshold if abstain else None, len(samples))


def fit_decision_calibration(
    samples: Sequence[DecisionSample],
    *,
    dims: Sequence[str],
    min_samples: int = 50,
    target_precision: float = 0.9,
    fit_as_of: date,
) -> FittedDecisionCalibration:
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be within [0, 1]")
    if min_samples < 10:
        raise ValueError("min_samples must be >= 10 for a temperature fit")
    entries = fit_segments(
        [(s.attrs, s) for s in samples], dims, min_samples, lambda ss: _fit_params(ss, target_precision)
    )
    if not entries:
        raise CalibrationError(f"no segment has {min_samples} labelled decisions ({len(samples)} samples)")
    return FittedDecisionCalibration(entries, tuple(dims), min_samples, target_precision, fit_as_of, len(samples))


def apply_decision_calibration(
    fitted: FittedDecisionCalibration | None, decision: Decision, attrs: Mapping[str, str]
) -> CalibratedDecision:
    entry = resolve_segment(fitted.entries, fitted.dims, attrs) if fitted is not None else None
    if entry is None or not decision.probabilities:
        return CalibratedDecision(
            decision, decision.answer, decision.abstained, decision.raw_confidence,
            dict(decision.probabilities), decision.raw_confidence, False, None, "uncalibrated",
        )
    labels = list(decision.probabilities)
    raw = np.array([[decision.probabilities[l] for l in labels]], dtype=float)
    raw /= raw.sum()
    scaled = cls.apply_temperature(raw, entry.params.temperature)[0]
    calibrated = {l: float(p) for l, p in zip(labels, scaled)}
    top = labels[int(np.argmax(scaled))]
    confidence = float(scaled.max())
    below = entry.params.abstain_threshold is not None and confidence < entry.params.abstain_threshold
    abstained = decision.abstained or below
    return CalibratedDecision(
        decision, None if abstained else top, abstained, confidence, calibrated, decision.raw_confidence,
        True, entry.key, fitted.version,
    )


def save_decision_calibration(
    store: CalibrationStore, fitted: FittedDecisionCalibration, metrics: Mapping[str, object] | None = None
) -> FittedDecisionCalibration:
    stored = store.save(
        CalibrationKind.DECISION_TEMPERATURE,
        fit_as_of=fitted.fit_as_of, dims=fitted.dims, min_samples=fitted.min_samples, n_total=fitted.n_total,
        policy_version=POLICY_VERSION,
        entries=[
            StoredEntry(e.key, e.level, e.n, {
                "temperature": e.params.temperature, "abstain_threshold": e.params.abstain_threshold, "n": e.params.n,
            })
            for e in fitted.entries.values()
        ],
        metrics={"target_precision": fitted.target_precision, **dict(metrics or {})},
    )
    return FittedDecisionCalibration(
        fitted.entries, fitted.dims, fitted.min_samples, fitted.target_precision, fitted.fit_as_of, fitted.n_total,
        stored.version, dict(stored.metrics),
    )


def load_decision_calibration(store: CalibrationStore) -> FittedDecisionCalibration | None:
    stored = store.latest(CalibrationKind.DECISION_TEMPERATURE)
    if stored is None:
        return None
    entries = {
        e.segment_key: SegmentEntry(
            e.segment_key, e.level, e.n,
            DecisionCalibrationParams(float(e.params["temperature"]), e.params["abstain_threshold"], int(e.params["n"])),
        )
        for e in stored.entries
    }
    return FittedDecisionCalibration(
        entries, stored.dims, stored.min_samples, float(stored.metrics.get("target_precision", 0.9)),
        stored.fit_as_of, stored.n_total, stored.version, dict(stored.metrics),
    )


@dataclass(frozen=True, slots=True)
class DecisionCalibrationEvaluation:
    n_train: int
    n_test: int
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    log_loss_before: float
    log_loss_after: float


def evaluate_decision_calibration(
    samples: Sequence[DecisionSample],
    *,
    dims: Sequence[str],
    min_samples: int = 50,
    train_fraction: float = 0.6,
    fit_as_of: date,
) -> DecisionCalibrationEvaluation:
    """Fit on the first part (in the given order, assumed chronological), score the rest before vs after."""
    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be within [0.2, 0.9]")
    split = int(len(samples) * train_fraction)
    fitted = fit_decision_calibration(samples[:split], dims=dims, min_samples=min_samples, fit_as_of=fit_as_of)
    test = [s for s in samples[split:] if resolve_segment(fitted.entries, fitted.dims, s.attrs) is not None]
    if not test:
        raise CalibrationError("no test sample falls into a calibrated segment")
    labels = sorted({l for s in test for l in s.probabilities} | {s.true_label for s in test})
    index = {l: i for i, l in enumerate(labels)}
    raw = np.zeros((len(test), len(labels)))
    scaled = np.zeros_like(raw)
    for r, s in enumerate(test):
        for l, p in s.probabilities.items():
            raw[r, index[l]] = p
        raw[r] /= raw[r].sum()
        t = resolve_segment(fitted.entries, fitted.dims, s.attrs).params.temperature
        scaled[r] = cls.apply_temperature(raw[r : r + 1], t)[0]
    y = np.array([index[s.true_label] for s in test])
    ece = lambda p: cls.expected_calibration_error(p.max(axis=1), p.argmax(axis=1) == y)  # noqa: E731
    return DecisionCalibrationEvaluation(
        n_train=split, n_test=len(test), ece_before=ece(raw), ece_after=ece(scaled),
        brier_before=cls.brier_score(raw, y), brier_after=cls.brier_score(scaled, y),
        log_loss_before=cls.log_loss(raw, y), log_loss_after=cls.log_loss(scaled, y),
    )

"""Segmented calibration of forecast quantiles from resolved outcomes (work 010).

Fit only on outcomes that were already KNOWN at ``as_of`` (window_end <= as_of);
apply to new forecasts by the most specific fitted segment (horizon, model, ...),
backing off to broader ones; with no usable segment the forecast passes through
flagged ``calibrated=False``. The UI shows the calibrated quantiles; raw values
stay available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

from app.mrip.calibration.quantiles import CalibrationInput, QuantileShift, apply_quantile_shift, fit_quantile_shift
from app.mrip.calibration.segments import SegmentEntry, fit_segments, resolve_segment
from app.mrip.calibration.store import CalibrationError, CalibrationKind, CalibrationStore, StoredEntry
from app.mrip.forecast.evaluation import pinball_loss
from app.mrip.forecast.types import QUANTILES, SCENARIO_QUANTILES, ForecastResult, Scenarios
from app.mrip.outcomes.predictions import horizon_kind_for
from app.mrip.outcomes.types import Outcome, OutcomeStatus, Prediction, PredictionType

POLICY_VERSION = "forecast-calibration-v0"
DEFAULT_DIMS = ("horizon_kind", "model_version")  # most general first
_ET = ZoneInfo("America/New_York")

AttrsFn = Callable[[Prediction], Mapping[str, str]]


def prediction_attrs(p: Prediction) -> dict[str, str]:
    attrs = {"horizon_kind": p.horizon_kind.value, "model_version": p.model_version, "subject": p.subject}
    regime = (p.payload.get("market_regime") or {}).get("regime")
    if regime:
        attrs["vix_regime"] = str(regime)  # available as a segment dimension when logged with the prediction
    return attrs


def result_attrs(result: ForecastResult, vix_regime: str | None = None) -> dict[str, str]:
    attrs = {
        "horizon_kind": horizon_kind_for(result.horizon_days).value,
        "model_version": result.provider_version,
        "subject": result.symbol,
    }
    if vix_regime:
        attrs["vix_regime"] = vix_regime
    return attrs


@dataclass(frozen=True, slots=True)
class FittedForecastCalibration:
    entries: Mapping[str, SegmentEntry[QuantileShift]]
    dims: tuple[str, ...]
    min_samples: int
    fit_as_of: date
    n_total: int
    version: str = "unsaved"
    metrics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CalibratedForecast:
    raw: ForecastResult
    return_quantiles: Mapping[float, float]
    calibrated: bool
    segment_key: str | None
    n_calibration: int
    calibration_version: str

    def price_quantiles(self) -> dict[float, float]:
        return {q: self.raw.last_price * (1.0 + r) for q, r in self.return_quantiles.items()}

    @property
    def scenarios(self) -> Scenarios:
        q = self.return_quantiles
        return Scenarios(bear=q[SCENARIO_QUANTILES["bear"]], base=q[SCENARIO_QUANTILES["base"]], bull=q[SCENARIO_QUANTILES["bull"]])


def _usable(pairs: Sequence[tuple[Prediction, Outcome]], as_of: date | None) -> list[tuple[Prediction, Outcome]]:
    rows = []
    for p, o in pairs:
        if p.prediction_type is not PredictionType.FORECAST:
            raise CalibrationError(f"prediction {p.id} is not a forecast")
        if o.status is not OutcomeStatus.RESOLVED or o.actual_return is None or o.window_end is None:
            continue
        if as_of is not None and o.window_end > as_of:
            continue  # not yet known at as_of: using it would leak the future
        rows.append((p, o))
    return rows


def fit_forecast_calibration(
    pairs: Sequence[tuple[Prediction, Outcome]],
    *,
    dims: Sequence[str] = DEFAULT_DIMS,
    min_samples: int = 30,
    as_of: date | None = None,
    attrs_fn: AttrsFn = prediction_attrs,
) -> FittedForecastCalibration:
    rows = _usable(pairs, as_of)
    items = [
        (attrs_fn(p), CalibrationInput({float(q): float(v) for q, v in p.payload["quantiles"].items()}, o.actual_return))
        for p, o in rows
    ]
    entries = fit_segments(items, dims, min_samples, lambda rs: fit_quantile_shift(rs, min_samples))
    if not entries:
        raise CalibrationError(f"no segment has {min_samples} resolved outcomes ({len(rows)} usable rows)")
    return FittedForecastCalibration(
        entries=entries, dims=tuple(dims), min_samples=min_samples,
        fit_as_of=as_of or max(o.window_end for _, o in rows), n_total=len(rows),
    )


def apply_forecast_calibration(
    fitted: FittedForecastCalibration | None, result: ForecastResult, *, attrs: Mapping[str, str] | None = None
) -> CalibratedForecast:
    entry = None
    if fitted is not None:
        entry = resolve_segment(fitted.entries, fitted.dims, attrs if attrs is not None else result_attrs(result))
    if entry is None:
        return CalibratedForecast(result, dict(result.return_quantiles), False, None, 0, "uncalibrated")
    return CalibratedForecast(
        result, apply_quantile_shift(result.return_quantiles, entry.params), True, entry.key, entry.n,
        fitted.version,
    )


def save_forecast_calibration(
    store: CalibrationStore, fitted: FittedForecastCalibration, metrics: Mapping[str, object] | None = None
) -> FittedForecastCalibration:
    stored = store.save(
        CalibrationKind.FORECAST_QUANTILE_SHIFT,
        fit_as_of=fitted.fit_as_of, dims=fitted.dims, min_samples=fitted.min_samples, n_total=fitted.n_total,
        policy_version=POLICY_VERSION,
        entries=[
            StoredEntry(e.key, e.level, e.n, {"n": e.params.n, "shifts": {str(q): v for q, v in e.params.shifts.items()}})
            for e in fitted.entries.values()
        ],
        metrics=metrics,
    )
    return FittedForecastCalibration(
        fitted.entries, fitted.dims, fitted.min_samples, fitted.fit_as_of, fitted.n_total, stored.version,
        dict(metrics or {}),
    )


def load_forecast_calibration(store: CalibrationStore) -> FittedForecastCalibration | None:
    stored = store.latest(CalibrationKind.FORECAST_QUANTILE_SHIFT)
    if stored is None:
        return None
    entries = {
        e.segment_key: SegmentEntry(
            e.segment_key, e.level, e.n,
            QuantileShift({float(q): float(v) for q, v in e.params["shifts"].items()}, int(e.params["n"])),
        )
        for e in stored.entries
    }
    return FittedForecastCalibration(
        entries, stored.dims, stored.min_samples, stored.fit_as_of, stored.n_total, stored.version, dict(stored.metrics)
    )


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    n_train: int
    n_test: int
    n_calibrated_test: int
    pinball_before: float
    pinball_after: float
    coverage_p10_p90_before: float
    coverage_p10_p90_after: float
    coverage_p25_p75_before: float
    coverage_p25_p75_after: float
    hit_before: Mapping[float, float]  # nominal level -> observed frequency, raw
    hit_after: Mapping[float, float]

    def improves(self, min_relative_gain: float = 0.02, coverage_slack: float = 0.02) -> bool:
        """Promotion gate (initial, uncalibrated): out-of-sample pinball must fall by at least
        ``min_relative_gain`` and the 80 % interval coverage must not drift further from 0.80
        by more than ``coverage_slack``. A calibration that fails this is not promoted."""
        gain = self.pinball_after <= self.pinball_before * (1.0 - min_relative_gain)
        coverage = abs(self.coverage_p10_p90_after - 0.8) <= abs(self.coverage_p10_p90_before - 0.8) + coverage_slack
        return gain and coverage


def evaluate_forecast_calibration(
    pairs: Sequence[tuple[Prediction, Outcome]],
    *,
    dims: Sequence[str] = DEFAULT_DIMS,
    min_samples: int = 30,
    train_fraction: float = 0.6,
    attrs_fn: AttrsFn = prediction_attrs,
) -> CalibrationEvaluation:
    """Out-of-sample check: fit on the earlier part, score the later part before vs after calibration.

    Training uses only outcomes already known when the first test forecast was made
    (purged), so no test-period information leaks into the fit.
    """
    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be within [0.2, 0.9]")
    ordered = sorted(_usable(pairs, None), key=lambda po: (po[0].made_at, po[0].id))
    split = int(len(ordered) * train_fraction)
    test = ordered[split:]
    if split < min_samples or not test:
        raise CalibrationError("not enough resolved outcomes for a train/test evaluation")
    test_start = test[0][0].made_at.astimezone(_ET).date()
    fitted = fit_forecast_calibration(
        ordered[:split], dims=dims, min_samples=min_samples, as_of=test_start, attrs_fn=attrs_fn
    )
    before_loss, after_loss = [], []
    c80 = [[], []]
    c50 = [[], []]
    hits = [{q: [] for q in QUANTILES}, {q: [] for q in QUANTILES}]
    for p, o in test:
        entry = resolve_segment(fitted.entries, fitted.dims, attrs_fn(p))
        if entry is None:
            continue
        raw = {float(q): float(v) for q, v in p.payload["quantiles"].items()}
        cal = apply_quantile_shift(raw, entry.params)
        for i, qs in enumerate((raw, cal)):
            (before_loss, after_loss)[i].append(pinball_loss(o.actual_return, qs))
            c80[i].append(qs[0.1] <= o.actual_return <= qs[0.9])
            c50[i].append(qs[0.25] <= o.actual_return <= qs[0.75])
            for q in QUANTILES:
                hits[i][q].append(o.actual_return <= qs[q])
    n_cal = len(before_loss)
    if n_cal == 0:
        raise CalibrationError("no test row fell into a calibrated segment")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return CalibrationEvaluation(
        n_train=split, n_test=len(test), n_calibrated_test=n_cal,
        pinball_before=mean(before_loss), pinball_after=mean(after_loss),
        coverage_p10_p90_before=mean(c80[0]), coverage_p10_p90_after=mean(c80[1]),
        coverage_p25_p75_before=mean(c50[0]), coverage_p25_p75_after=mean(c50[1]),
        hit_before={q: mean(v) for q, v in hits[0].items()}, hit_after={q: mean(v) for q, v in hits[1].items()},
    )

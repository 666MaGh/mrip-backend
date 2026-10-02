"""Quantile recalibration of forecasts by per-level additive shifts (conformal style).

For each nominal level tau, the shift is the tau-quantile of the realised residuals
``actual - predicted_quantile_tau`` on calibration data. By construction
``P(actual <= predicted_tau + shift_tau)`` is about tau on that data. Shifts are in
return space, so segments (horizon, model, regime) must be fitted separately.
The recalibrated quantiles are re-sorted so they never cross.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from app.mrip.forecast.types import QUANTILES


@dataclass(frozen=True, slots=True)
class QuantileShift:
    shifts: Mapping[float, float]  # level -> additive return shift
    n: int


@dataclass(frozen=True, slots=True)
class CalibrationInput:
    quantiles: Mapping[float, float]  # predicted return quantiles
    actual: float  # observed return


def fit_quantile_shift(rows: Sequence[CalibrationInput], min_samples: int = 30) -> QuantileShift:
    if len(rows) < min_samples:
        raise ValueError(f"need at least {min_samples} calibration rows, got {len(rows)}")
    shifts = {}
    for tau in QUANTILES:
        residuals = np.array([r.actual - r.quantiles[tau] for r in rows], dtype=float)
        shifts[tau] = float(np.quantile(residuals, tau))
    return QuantileShift(shifts=shifts, n=len(rows))


def apply_quantile_shift(quantiles: Mapping[float, float], shift: QuantileShift) -> dict[float, float]:
    adjusted = [quantiles[tau] + shift.shifts[tau] for tau in QUANTILES]
    return dict(zip(QUANTILES, (float(v) for v in np.sort(adjusted))))

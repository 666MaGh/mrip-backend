"""VIX-based volatility regimes and regime-conditional statistics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd


class Regime(str, Enum):
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    STRESS = "STRESS"
    CRISIS = "CRISIS"


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """VIX cut-offs between regimes.

    Initial defaults, to be calibrated later by the learning policy.
    """

    low_normal: float = 15.0
    normal_elevated: float = 20.0
    elevated_stress: float = 30.0
    stress_crisis: float = 40.0
    version: str = "v0-uncalibrated"

    def __post_init__(self) -> None:
        cuts = (self.low_normal, self.normal_elevated, self.elevated_stress, self.stress_crisis)
        if not all(a < b for a, b in zip(cuts, cuts[1:])):
            raise ValueError("regime thresholds must be strictly increasing")


def _classify(value: float, t: RegimeThresholds) -> Regime | float:
    if np.isnan(value):
        return np.nan
    if value < t.low_normal:
        return Regime.LOW_VOL
    if value < t.normal_elevated:
        return Regime.NORMAL
    if value < t.elevated_stress:
        return Regime.ELEVATED
    if value < t.stress_crisis:
        return Regime.STRESS
    return Regime.CRISIS


def classify_vix(vix: pd.Series, thresholds: RegimeThresholds = RegimeThresholds()) -> pd.Series:
    """Map VIX levels to regimes (object dtype); NaN input stays NaN."""
    labels = [_classify(float(v), thresholds) for v in vix.to_numpy(dtype=float)]
    return pd.Series(labels, index=vix.index, dtype=object)


def segment(data: pd.DataFrame, regimes: pd.Series) -> dict[Regime, pd.DataFrame]:
    """Split `data` by regime on the common index; rows without a regime are dropped."""
    common = data.index.intersection(regimes.index)
    labels = regimes.loc[common].dropna()
    rows = data.loc[labels.index]
    out: dict[Regime, pd.DataFrame] = {}
    for regime in Regime:
        part = rows.loc[(labels == regime).to_numpy()]
        if len(part) > 0:
            out[regime] = part
    return out


def by_regime(
    data: pd.DataFrame,
    regimes: pd.Series,
    stat: Callable[[pd.DataFrame], float],
    min_obs: int = 30,
) -> dict[Regime, float | None]:
    """Apply `stat` per regime; None where a regime has fewer than `min_obs` rows."""
    parts = segment(data, regimes)
    result: dict[Regime, float | None] = {}
    for regime in Regime:
        part = parts.get(regime)
        result[regime] = stat(part) if part is not None and len(part) >= min_obs else None
    return result

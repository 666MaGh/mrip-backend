"""Gamma regime and amplification assessment (MODELED / ESTIMATED, versioned, uncalibrated).

Both classifiers are deterministic rule sets. Their thresholds are initial
defaults that the learning policy may calibrate later; they are never tuned by
hand to produce more alarming output. LAYA never computes any of this: it may
later judge whether a result materially changes a research thesis.

The system does not predict a "gamma squeeze". It reports an amplification
setup with contributing and contradicting factors, and says UNCERTAIN when the
data cannot support an assessment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.mrip.options.gex import GexProfile


class GammaRegime(str, Enum):
    POSITIVE_GAMMA = "POSITIVE_GAMMA"
    NEUTRAL_GAMMA = "NEUTRAL_GAMMA"
    NEGATIVE_GAMMA = "NEGATIVE_GAMMA"
    EXTREME_NEGATIVE_GAMMA = "EXTREME_NEGATIVE_GAMMA"
    UNKNOWN = "UNKNOWN"  # no usable gamma


class AmplificationLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNCERTAIN = "UNCERTAIN"


class Direction(str, Enum):
    STABILIZING = "STABILIZING"
    AMPLIFYING = "AMPLIFYING"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class RegimeMethod:
    """Classify by the GEX tilt (net / gross), which is comparable across underlyings."""

    version: str = "gamma-regime-v0-uncalibrated"
    positive_tilt: float = 0.2
    negative_tilt: float = -0.2
    extreme_negative_tilt: float = -0.6

    def __post_init__(self) -> None:
        if not self.extreme_negative_tilt < self.negative_tilt < self.positive_tilt:
            raise ValueError("thresholds must satisfy extreme_negative < negative < positive")


def classify_gamma_regime(tilt: float | None, method: RegimeMethod = RegimeMethod()) -> GammaRegime:
    if tilt is None:
        return GammaRegime.UNKNOWN
    if tilt <= method.extreme_negative_tilt:
        return GammaRegime.EXTREME_NEGATIVE_GAMMA
    if tilt <= method.negative_tilt:
        return GammaRegime.NEGATIVE_GAMMA
    if tilt >= method.positive_tilt:
        return GammaRegime.POSITIVE_GAMMA
    return GammaRegime.NEUTRAL_GAMMA


@dataclass(frozen=True, slots=True)
class AmplificationMethod:
    version: str = "amplification-v0-uncalibrated"
    min_contracts_used: int = 50
    min_usable_fraction: float = 0.5  # usable contracts / contracts with open interest
    flip_near_pct: float = 0.02
    zero_dte_share_high: float = 0.30
    short_dated_share_high: float = 0.60
    call_wall_near_pct: float = 0.02
    momentum_high: float = 0.05  # |return| over the caller's lookback
    realized_vol_high: float = 0.40  # annualised


@dataclass(frozen=True, slots=True)
class ExternalContext:
    """Optional non-options inputs (computed elsewhere, deterministically)."""

    momentum: float | None = None  # trailing return of the underlying (+ up, - down)
    realized_vol: float | None = None  # annualised


@dataclass(frozen=True, slots=True)
class AmplificationAssessment:
    level: AmplificationLevel
    direction: Direction
    regime: GammaRegime
    contributing: tuple[str, ...] = field(default_factory=tuple)
    contradicting: tuple[str, ...] = field(default_factory=tuple)
    uncertain_because: tuple[str, ...] = field(default_factory=tuple)
    method_version: str = ""


def assess_amplification(
    gex: GexProfile,
    regime: GammaRegime,
    context: ExternalContext = ExternalContext(),
    method: AmplificationMethod = AmplificationMethod(),
) -> AmplificationAssessment:
    """Rule-based. Positive modeled gamma -> LOW/STABILIZING. Negative gamma starts at 1 point (2 if extreme)
    and each risk factor adds 1: 1 -> LOW, 2 -> MODERATE, 3 -> HIGH, 4+ -> EXTREME."""
    problems: list[str] = []
    eligible = gex.contracts_total - gex.excluded.get("no_open_interest", 0) - gex.excluded.get("expired", 0)
    if regime is GammaRegime.UNKNOWN:
        problems.append("no usable gamma")
    if gex.contracts_used < method.min_contracts_used:
        problems.append(f"only {gex.contracts_used} usable contracts (< {method.min_contracts_used})")
    if eligible > 0 and gex.contracts_used / eligible < method.min_usable_fraction:
        problems.append(f"only {gex.contracts_used}/{eligible} contracts with open interest are usable")
    if problems:
        return AmplificationAssessment(
            AmplificationLevel.UNCERTAIN, Direction.UNCERTAIN, regime, uncertain_because=tuple(problems),
            method_version=method.version,
        )

    contributing: list[str] = []
    contradicting: list[str] = []
    if regime is GammaRegime.POSITIVE_GAMMA:
        return AmplificationAssessment(
            AmplificationLevel.LOW, Direction.STABILIZING, regime,
            contributing=("positive modeled net gamma",), method_version=method.version,
        )
    if regime is GammaRegime.NEUTRAL_GAMMA:
        return AmplificationAssessment(
            AmplificationLevel.UNCERTAIN, Direction.UNCERTAIN, regime,
            uncertain_because=("modeled net gamma is close to neutral",), method_version=method.version,
        )

    points = 2 if regime is GammaRegime.EXTREME_NEGATIVE_GAMMA else 1
    contributing.append(f"{regime.value.lower().replace('_', ' ')} (modeled)")
    flip = gex.flip
    if flip.status == "found" and flip.distance_pct is not None and abs(flip.distance_pct) <= method.flip_near_pct:
        points += 1
        contributing.append(f"spot within {method.flip_near_pct:.0%} of the modeled gamma flip")
    if gex.zero_dte_share is not None and gex.zero_dte_share >= method.zero_dte_share_high:
        points += 1
        contributing.append(f"0DTE holds {gex.zero_dte_share:.0%} of gross gamma")
    elif gex.short_dated_share is not None and gex.short_dated_share >= method.short_dated_share_high:
        points += 1
        contributing.append(f"short-dated options hold {gex.short_dated_share:.0%} of gross gamma")
    wall = gex.walls.call_wall_distance_pct
    if wall is not None and wall <= method.call_wall_near_pct:
        points += 1
        contributing.append("spot near the estimated call wall")
    if context.momentum is not None:
        if abs(context.momentum) >= method.momentum_high:
            points += 1
            contributing.append(f"strong underlying momentum ({context.momentum:+.1%})")
        else:
            contradicting.append("weak underlying momentum")
    if context.realized_vol is not None:
        if context.realized_vol >= method.realized_vol_high:
            points += 1
            contributing.append(f"high realized volatility ({context.realized_vol:.0%})")
        else:
            contradicting.append("moderate realized volatility")
    if flip.status == "found" and flip.distance_pct is not None and abs(flip.distance_pct) > 0.10:
        contradicting.append("modeled gamma flip is far from spot")

    level = (
        AmplificationLevel.EXTREME if points >= 4
        else AmplificationLevel.HIGH if points == 3
        else AmplificationLevel.MODERATE if points == 2
        else AmplificationLevel.LOW
    )
    return AmplificationAssessment(
        level, Direction.AMPLIFYING, regime, tuple(contributing), tuple(contradicting), (), method.version
    )

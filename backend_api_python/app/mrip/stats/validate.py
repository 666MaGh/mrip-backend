"""Decide whether a hypothesized relationship has statistical support (work 006).

The rule set is explicit and versioned (``ValidationPolicy.version``); the numbers
below are initial, UNCALIBRATED defaults that the learning policy may later tune.

Version ``validation-v1-uncalibrated`` (work 022, ADR pending) adds two controls on
top of v0 ("v0-uncalibrated", kept for history and for tests of the v0 rules):

  * sector control: the partial correlation also controls for the equal-weighted mean
    return of the OTHER stored-price members of the pair's sector (both pair members
    excluded). Without >= 5 peers the control is "unavailable" and the market-only
    partial is used.
  * same-sector null gate: the signed partial correlation in the expected direction
    must exceed the 95th percentile of the null distribution (see
    ``app.mrip.stats.null_distribution``) for the same controls. The null is the
    same-sector null, or the random-pair null when no sector control applies.

Decision logic, in order:
  1. Too little data                                  -> INCONCLUSIVE (neutral)
  2. Lead/lag with a corrected significant correlation:
     - sign opposite to ``expected_sign``             -> REJECTED   (contradict)
  3. Support requires ALL of: significant after Bonferroni over lags; expected
     sign (if given) and lead/lag direction (for LEADS/LAGS) confirmed; partial
     correlation still significant after controlling for the market (and sector);
     v1 only: partial correlation above the null p95 in the expected direction;
     and walk-forward sign consistency >= policy threshold over >= min_folds folds
                                                      -> VALIDATED  (support)
  4. Anything else                                    -> INCONCLUSIVE (neutral)

Absence of significance is never a rejection: a hypothesis stays a hypothesis.
Only a significant effect with the wrong sign rejects it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app.mrip.evidence.types import Stance
from app.mrip.relationships.types import RelationType
from app.mrip.stats.correlation import CorrResult, lead_lag
from app.mrip.stats.null_distribution import NullDistribution, partial_at_lag
from app.mrip.stats.transforms import align
from app.mrip.stats.walkforward import lead_lag_walk_forward, sign_consistency


class Verdict(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    version: str = "validation-v1-uncalibrated"
    min_observations: int = 300
    alpha: float = 0.05
    max_lag: int = 5
    train_size: int = 150
    test_size: int = 50
    min_folds: int = 3
    min_oos_consistency: float = 0.6
    sector_control: bool = True
    null_gate: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or self.max_lag < 0 or self.min_folds < 1:
            raise ValueError("invalid validation policy")
        if self.min_observations < self.train_size + self.test_size:
            raise ValueError("min_observations must cover at least one train+test fold")


# The pre-v1 rules: market-only partial correlation, no null gate. Kept so the v0 history
# and its rules can still be exercised; the production default is ``ValidationPolicy()``.
POLICY_V0 = ValidationPolicy(version="v0-uncalibrated", sector_control=False, null_gate=False)


@dataclass(frozen=True, slots=True)
class SectorControl:
    """Sector peer average return aligned with the pair (pair members already excluded)."""

    series: pd.Series
    sector: str
    peers: int


@dataclass(frozen=True, slots=True)
class ValidationResult:
    verdict: Verdict
    stance: Stance
    reasons: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    policy_version: str = ""


def _result(verdict: Verdict, reasons: list[str], metrics: dict[str, Any], policy: ValidationPolicy) -> ValidationResult:
    stance = {
        Verdict.VALIDATED: Stance.SUPPORT,
        Verdict.REJECTED: Stance.CONTRADICT,
        Verdict.INCONCLUSIVE: Stance.NEUTRAL,
    }[verdict]
    return ValidationResult(verdict, stance, tuple(reasons), metrics, policy.version)


def validate_relationship(
    x: pd.Series,
    y: pd.Series,
    *,
    market: pd.Series | None = None,
    sector: SectorControl | None = None,
    sector_status: str = "unavailable",
    null: NullDistribution | None = None,
    expected_sign: int | None = None,
    relation_type: RelationType | None = None,
    policy: ValidationPolicy = ValidationPolicy(),
) -> ValidationResult:
    """``x`` and ``y`` are return series; the hypothesis is "x relates to y" (x may lead y).

    ``sector`` is applied only together with ``market`` and only when ``policy.sector_control``
    is set; otherwise ``sector_status`` ("unavailable" or "not_applicable") is reported.
    ``null`` is required by the v1 gate (``policy.null_gate``); its absence is reported as
    INCONCLUSIVE rather than silently skipping the gate.
    """
    if expected_sign not in (None, -1, 1):
        raise ValueError("expected_sign must be -1, 1 or None")
    sector_used = policy.sector_control and sector is not None and market is not None
    series = [x.rename("x"), y.rename("y")]
    if market is not None:
        series.append(market.rename("market"))
    sector_series = sector.series if (sector_used and sector is not None) else None
    if sector_series is not None:
        series.append(sector_series.rename("sector"))
    aligned = align(*series)
    xs, ys = aligned[0], aligned[1]
    n = len(xs)
    if not policy.sector_control:
        sector_control_status = "disabled"
    elif sector_used:
        sector_control_status = "used"
    else:
        sector_control_status = sector_status
    metrics: dict[str, Any] = {
        "n_obs": n,
        "start": str(xs.index[0].date()) if n else None,
        "end": str(xs.index[-1].date()) if n else None,
        "expected_sign": expected_sign,
        "relation_type": relation_type.value if relation_type else None,
        "max_lag": policy.max_lag,
        "alpha": policy.alpha,
        "sector_control": sector_control_status,
        "sector": sector.sector if sector_series is not None and sector is not None else None,
        "peers_used": sector.peers if sector_series is not None and sector is not None else 0,
    }
    if n < policy.min_observations:
        return _result(Verdict.INCONCLUSIVE, [f"insufficient observations ({n} < {policy.min_observations})"], metrics, policy)

    ll = lead_lag(xs, ys, policy.max_lag)
    sign = int(np.sign(ll.best.r))
    metrics.update(
        best_lag=ll.best_lag, direction=ll.direction, r=round(ll.best.r, 6), p_corrected=ll.corrected_p_value,
        lags_tested=ll.lags_tested,
    )
    significant = ll.corrected_p_value < policy.alpha
    if not significant:
        return _result(Verdict.INCONCLUSIVE, ["no significant correlation after correcting for lags searched"], metrics, policy)

    if expected_sign is not None and sign != expected_sign:
        return _result(
            Verdict.REJECTED,
            [f"significant correlation has sign {sign:+d}, expected {expected_sign:+d}"],
            metrics,
            policy,
        )

    reasons: list[str] = []
    if relation_type is RelationType.LEADS and ll.best_lag <= 0:
        reasons.append("LEADS not confirmed: best lag does not have x leading y")
    if relation_type is RelationType.LAGS and ll.best_lag >= 0:
        reasons.append("LAGS not confirmed: best lag does not have y leading x")

    partial: CorrResult | None = None
    control_names = "market and sector" if sector_series is not None else "market"
    if market is not None:
        # Controls at both paired dates (t and t-|lag|); the same statistic the same-sector null uses.
        m = aligned[2]
        s = aligned[3] if sector_series is not None else None
        partial = partial_at_lag(xs, ys, m, s, ll.best_lag)
        controls_used = ["market_t"] + (["market_lagged"] if ll.best_lag != 0 else [])
        if s is not None:
            controls_used += ["sector_t"] + (["sector_lagged"] if ll.best_lag != 0 else [])
        metrics.update(partial_r=round(partial.r, 6), partial_p=partial.p_value, controls=controls_used)
        if not partial.p_value < policy.alpha or int(np.sign(partial.r)) != sign:
            reasons.append(f"relationship does not survive controlling for the {control_names}")
    else:
        metrics["controls"] = []

    if policy.null_gate:
        if partial is None or null is None or not np.isfinite(partial.r):
            reasons.append("same-sector null unavailable: no partial correlation to compare")
        else:
            direction = expected_sign if expected_sign is not None else sign
            measure = direction * partial.r
            # Null p95 of the signed partial in the expected direction; for -1 the p95 of -partial is -p05.
            threshold = null.p95 if direction > 0 else -null.p05
            metrics.update(
                null_label=null.label, null_n=null.pairs_used, null_p95=round(threshold, 6),
                effect_measure=round(measure, 6),
            )
            if not measure > threshold:
                reasons.append(
                    f"does not exceed {null.label} (partial_r {measure:.3f} < p95 {threshold:.3f}, n={null.pairs_used})"
                )

    folds = lead_lag_walk_forward(
        xs, ys, policy.max_lag, train_size=policy.train_size, test_size=policy.test_size
    )
    consistency = sign_consistency(folds)
    metrics.update(folds=len(folds), oos_consistency=None if consistency is None else round(consistency, 4))
    if len(folds) < policy.min_folds or consistency is None:
        reasons.append(f"too few walk-forward folds ({len(folds)} < {policy.min_folds})")
    elif consistency < policy.min_oos_consistency:
        reasons.append(f"out-of-sample sign consistency {consistency:.2f} < {policy.min_oos_consistency:.2f}")

    if reasons:
        return _result(Verdict.INCONCLUSIVE, reasons, metrics, policy)
    return _result(
        Verdict.VALIDATED,
        ["significant after lag correction, survives market control, stable out-of-sample"],
        metrics,
        policy,
    )

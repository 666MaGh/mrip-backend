"""Decide whether a hypothesized relationship has statistical support (work 006).

The rule set is explicit and versioned (``ValidationPolicy.version``); the numbers
below are initial, UNCALIBRATED defaults that the learning policy may later tune.

Decision logic, in order:
  1. Too little data                                  -> INCONCLUSIVE (neutral)
  2. Lead/lag with a corrected significant correlation:
     - sign opposite to ``expected_sign``             -> REJECTED   (contradict)
  3. Support requires ALL of: significant after Bonferroni over lags; expected
     sign (if given) and lead/lag direction (for LEADS/LAGS) confirmed; partial
     correlation still significant after controlling for the market; and
     walk-forward sign consistency >= policy threshold over >= min_folds folds
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
from app.mrip.stats.correlation import lead_lag, partial_correlation, lagged_pair
from app.mrip.stats.transforms import align
from app.mrip.stats.walkforward import lead_lag_walk_forward, sign_consistency


class Verdict(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    version: str = "v0-uncalibrated"
    min_observations: int = 300
    alpha: float = 0.05
    max_lag: int = 5
    train_size: int = 150
    test_size: int = 50
    min_folds: int = 3
    min_oos_consistency: float = 0.6

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1 or self.max_lag < 0 or self.min_folds < 1:
            raise ValueError("invalid validation policy")
        if self.min_observations < self.train_size + self.test_size:
            raise ValueError("min_observations must cover at least one train+test fold")


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
    expected_sign: int | None = None,
    relation_type: RelationType | None = None,
    policy: ValidationPolicy = ValidationPolicy(),
) -> ValidationResult:
    """``x`` and ``y`` are return series; the hypothesis is "x relates to y" (x may lead y)."""
    if expected_sign not in (None, -1, 1):
        raise ValueError("expected_sign must be -1, 1 or None")
    series = [x.rename("x"), y.rename("y")] + ([market.rename("market")] if market is not None else [])
    aligned = align(*series)
    xs, ys = aligned[0], aligned[1]
    n = len(xs)
    metrics: dict[str, Any] = {
        "n_obs": n,
        "start": str(xs.index[0].date()) if n else None,
        "end": str(xs.index[-1].date()) if n else None,
        "expected_sign": expected_sign,
        "relation_type": relation_type.value if relation_type else None,
        "max_lag": policy.max_lag,
        "alpha": policy.alpha,
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

    if market is not None:
        x_at_lag, y_now = lagged_pair(xs, ys, ll.best_lag)
        m = aligned[2]
        # Control for the market at both paired dates (t and t-|lag|): it can drive either leg.
        controls = pd.DataFrame({"market_t": m})
        if ll.best_lag != 0:
            controls["market_lagged"] = m.shift(abs(ll.best_lag))
        pairs = pd.concat([x_at_lag.rename("x"), y_now.rename("y")], axis=1).dropna()
        partial = partial_correlation(pairs["x"], pairs["y"], controls.loc[pairs.index].dropna())
        metrics.update(partial_r=round(partial.r, 6), partial_p=partial.p_value)
        if not partial.p_value < policy.alpha or int(np.sign(partial.r)) != sign:
            reasons.append("relationship does not survive controlling for the market")

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

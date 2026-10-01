"""Options structure analysis: observed data and modeled estimates, kept apart.

``OptionsAnalysis.observed`` holds what the chain actually states (volume, open
interest, IV and quantities derived directly from them). ``OptionsAnalysis.modeled``
holds everything that rests on assumptions (dealer GEX, gamma flip, walls, gamma
regime, amplification) and carries the label ``MODELED / ESTIMATED``. The UI must
never present the modeled part as observed fact. ``data_quality`` always accompanies
both, including what is NOT known (e.g. the open-interest as-of date).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Sequence

import numpy as np

from app.mrip.data.models import Latency, OptionsChainSnapshot
from app.mrip.options.gex import GexAssumptions, GexProfile, compute_gex
from app.mrip.options.regime import (
    AmplificationAssessment,
    AmplificationMethod,
    ExternalContext,
    GammaRegime,
    RegimeMethod,
    assess_amplification,
    classify_gamma_regime,
)
from app.mrip.options.vol_surface import (
    Anomaly,
    PutCallRatios,
    Skew,
    TermPoint,
    atm_iv_term_structure,
    iv_skew,
    put_call_ratios,
    snapshot_date,
    term_structure_slope,
    volume_oi_anomalies,
    volume_zscore,
)

MODELED_LABEL = "MODELED / ESTIMATED"


class LookAheadError(Exception):
    """A snapshot taken after the requested as-of time was supplied."""


@dataclass(frozen=True, slots=True)
class DataQuality:
    source: str
    gateway: str
    endpoint: str
    latency: Latency
    snapshot_timestamp: datetime
    underlying_timestamp: datetime | None
    oi_effective_date: date | None
    oi_status: str  # "unknown" | "current" | "stale"
    contracts: int
    coverage: dict[str, float]  # fractions of contracts with OI / IV / greeks
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ObservedOptions:
    underlying_price: float | None
    total_volume: float
    total_open_interest: float
    put_call: PutCallRatios
    term_structure: tuple[TermPoint, ...]
    term_slope: float | None
    skew: Skew | None
    anomalies: tuple[Anomaly, ...]
    volume_zscore: float | None


@dataclass(frozen=True, slots=True)
class ModeledOptions:
    label: str
    assumptions: dict[str, object]
    gex: GexProfile
    regime: GammaRegime
    regime_method_version: str
    amplification: AmplificationAssessment


@dataclass(frozen=True, slots=True)
class OptionsAnalysis:
    underlying: str
    data_quality: DataQuality
    observed: ObservedOptions
    modeled: ModeledOptions

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for lineage storage (enums, dates and dataclasses flattened)."""
        return _plain(self)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(_plain(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _oi_status(snapshot: OptionsChainSnapshot, warnings: list[str]) -> str:
    if snapshot.oi_effective_date is None:
        warnings.append("open-interest as-of date is not stated by the provider; treat OI as possibly stale")
        return "unknown"
    gap = int(np.busday_count(snapshot.oi_effective_date, snapshot_date(snapshot)))
    if gap > 1:
        warnings.append(f"open interest is {gap} business days old")
        return "stale"
    return "current"


def assert_not_after(snapshot: OptionsChainSnapshot, as_of: datetime) -> None:
    """Look-ahead guard: a snapshot taken after ``as_of`` must never inform an as-of analysis."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if snapshot.snapshot_timestamp > as_of:
        raise LookAheadError(
            f"snapshot taken {snapshot.snapshot_timestamp.isoformat()} is after as_of {as_of.isoformat()}"
        )


def analyze_options(
    snapshot: OptionsChainSnapshot,
    *,
    assumptions: GexAssumptions = GexAssumptions(),
    regime_method: RegimeMethod = RegimeMethod(),
    amplification_method: AmplificationMethod = AmplificationMethod(),
    context: ExternalContext = ExternalContext(),
    volume_history: Sequence[float] | None = None,
    as_of: datetime | None = None,
) -> OptionsAnalysis:
    if as_of is not None:
        assert_not_after(snapshot, as_of)
    contracts = snapshot.contracts
    n = len(contracts)
    cov = snapshot.coverage
    warnings: list[str] = []
    oi_status = _oi_status(snapshot, warnings)
    if snapshot.provenance.latency is Latency.DELAYED:
        warnings.append("quotes are delayed")
    if snapshot.underlying_timestamp is None:
        warnings.append("underlying timestamp missing")
    coverage = {
        "open_interest": cov.with_open_interest / n if n else 0.0,
        "implied_volatility": cov.with_implied_volatility / n if n else 0.0,
        "greeks": cov.with_greeks / n if n else 0.0,
    }
    if coverage["greeks"] < 0.5:
        warnings.append("less than half of the contracts have greeks")

    total_volume = float(sum(c.volume or 0 for c in contracts))
    gex = compute_gex(snapshot, assumptions)
    regime = classify_gamma_regime(gex.tilt, regime_method)
    term = tuple(atm_iv_term_structure(snapshot)) if snapshot.underlying_price else ()
    prov = snapshot.provenance
    return OptionsAnalysis(
        underlying=snapshot.underlying,
        data_quality=DataQuality(
            source=prov.provider, gateway=prov.gateway, endpoint=prov.endpoint, latency=prov.latency,
            snapshot_timestamp=snapshot.snapshot_timestamp, underlying_timestamp=snapshot.underlying_timestamp,
            oi_effective_date=snapshot.oi_effective_date, oi_status=oi_status, contracts=n,
            coverage=coverage, warnings=tuple(warnings),
        ),
        observed=ObservedOptions(
            underlying_price=snapshot.underlying_price,
            total_volume=total_volume,
            total_open_interest=float(sum(c.open_interest or 0 for c in contracts)),
            put_call=put_call_ratios(snapshot),
            term_structure=term,
            term_slope=term_structure_slope(term),
            skew=iv_skew(snapshot),
            anomalies=tuple(volume_oi_anomalies(snapshot)),
            volume_zscore=volume_zscore(total_volume, volume_history) if volume_history is not None else None,
        ),
        modeled=ModeledOptions(
            label=MODELED_LABEL,
            assumptions=assumptions.describe(),
            gex=gex,
            regime=regime,
            regime_method_version=regime_method.version,
            amplification=assess_amplification(gex, regime, context, amplification_method),
        ),
    )

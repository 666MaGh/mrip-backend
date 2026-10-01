"""Normalized market-data models for MRIP.

Everything downstream of the data gateway consumes these types, never raw
provider payloads. Observed facts and modeled estimates are kept apart: this
module only holds observed data plus its provenance and quality metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Literal


class Latency(str, Enum):
    REALTIME = "realtime"
    DELAYED = "delayed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a dataset came from. Always attached, never optional."""

    provider: str
    gateway: str
    endpoint: str
    fetched_at: datetime
    latency: Latency = Latency.UNKNOWN
    gateway_version: str | None = None


@dataclass(frozen=True, slots=True)
class PriceBar:
    ts: datetime | date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


@dataclass(frozen=True, slots=True)
class PriceSeries:
    symbol: str
    interval: str
    bars: tuple[PriceBar, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class OptionContract:
    contract_symbol: str | None
    expiry: date
    strike: float
    option_type: Literal["call", "put"]
    open_interest: float | None = None
    volume: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    contract_multiplier: int = 100


@dataclass(frozen=True, slots=True)
class OptionsChainSnapshot:
    """One point-in-time options chain (observed data only).

    ``oi_effective_date`` is None when the provider does not state it; callers
    must treat that as unknown and never assume it. Old chains must never be
    reconstructed from current open interest.
    """

    underlying: str
    underlying_price: float | None
    underlying_timestamp: datetime | None
    snapshot_timestamp: datetime
    oi_effective_date: date | None
    contracts: tuple[OptionContract, ...]
    provenance: Provenance

    @property
    def coverage(self) -> "OptionsCoverage":
        total = len(self.contracts)
        return OptionsCoverage(
            contracts=total,
            with_open_interest=sum(c.open_interest is not None for c in self.contracts),
            with_greeks=sum(c.gamma is not None for c in self.contracts),
            with_implied_volatility=sum(c.implied_volatility is not None for c in self.contracts),
        )


@dataclass(frozen=True, slots=True)
class OptionsCoverage:
    contracts: int
    with_open_interest: int
    with_greeks: int
    with_implied_volatility: int


@dataclass(frozen=True, slots=True)
class CotRecord:
    report_date: date
    market: str
    open_interest: float | None
    long_positions: dict[str, float | None] = field(default_factory=dict)
    short_positions: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CotSeries:
    market: str
    records: tuple[CotRecord, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class MacroPoint:
    date: date
    value: float | None


@dataclass(frozen=True, slots=True)
class MacroSeries:
    series_id: str
    points: tuple[MacroPoint, ...]
    provenance: Provenance

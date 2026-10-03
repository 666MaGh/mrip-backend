"""Universe member models and exceptions.

A UniverseMember represents a security that belongs to a trading universe
(e.g. S&P 500 constituents). The member carries both display and data-series
metadata: exchange and currency for the security itself, and provider/symbol
for the price history data source.
"""
from __future__ import annotations

from dataclasses import dataclass


class UniverseError(Exception):
    """Invalid universe operation (parse error, network failure, bad data)."""


@dataclass(frozen=True, slots=True)
class UniverseMember:
    """A security in a trading universe.

    Attributes:
        universe: Universe name (e.g. "SP500").
        symbol: Display symbol as used in the universe (e.g. "BRK.B").
        name: Human-readable name (e.g. "Berkshire Hathaway Inc. Class B").
        exchange: Exchange code (e.g. "US").
        currency: Currency (e.g. "USD").
        sector: GICS sector, or None if unknown.
        sub_industry: GICS sub-industry, or None if unknown.
        data_symbol: Symbol used for price history queries (may differ from symbol).
        data_provider: Price data provider (e.g. "cboe").
        isin: International Securities Identification Number, or None.
    """

    universe: str
    symbol: str
    name: str
    exchange: str
    currency: str
    sector: str | None
    sub_industry: str | None
    data_symbol: str
    data_provider: str
    isin: str | None = None

    @property
    def node_key(self) -> str:
        """The relationship graph node key: '{exchange}:{symbol}'."""
        return f"{self.exchange}:{self.symbol}"

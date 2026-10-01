"""Interface the rest of MRIP uses to obtain financial data."""
from __future__ import annotations

from datetime import date
from typing import Protocol

from app.mrip.data.models import CotSeries, MacroSeries, OptionsChainSnapshot, PriceSeries


class DataUnavailable(Exception):
    """A provider returned no usable data for the request."""


class FinancialDataGateway(Protocol):
    """Normalized, provenance-carrying access to market, options, COT and macro data."""

    def price_history(
        self, symbol: str, start: date | None = None, end: date | None = None, interval: str = "1d"
    ) -> PriceSeries: ...

    def vix_history(self, start: date | None = None, end: date | None = None) -> PriceSeries: ...

    def options_chain(self, symbol: str) -> OptionsChainSnapshot: ...

    def cot(self, market: str, start: date | None = None, end: date | None = None) -> CotSeries: ...

    def macro_series(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> MacroSeries: ...

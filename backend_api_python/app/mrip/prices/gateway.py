"""Gateway that reads prices from local storage, optionally backed by live data for other types."""
from __future__ import annotations

from datetime import date
from typing import Any, Callable

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import CotSeries, MacroSeries, OptionsChainSnapshot, PriceSeries
from app.mrip.prices.store import PriceStore


class StoredDataGateway:
    """Implements FinancialDataGateway: price_history from local store, others delegated to live."""

    def __init__(
        self,
        store: PriceStore,
        *,
        provider: str = "cboe",
        live: FinancialDataGateway | None = None,
        provider_for: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the gateway.

        Args:
            store: PriceStore instance for reading local prices.
            provider: Default provider name for stored prices.
            live: Optional live FinancialDataGateway to delegate non-price requests to.
            provider_for: Optional callable to determine provider by symbol.
        """
        self._store = store
        self._provider = provider
        self._live = live
        self._provider_for = provider_for

    def price_history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: str = "1d",
    ) -> PriceSeries:
        """Retrieve price history from local storage.

        Args:
            symbol: Stock symbol.
            start: Optional start date (inclusive).
            end: Optional end date (inclusive).
            interval: Must be "1d"; other intervals raise ValueError.

        Returns:
            PriceSeries from local storage.

        Raises:
            ValueError: If interval is not "1d".
            DataUnavailable: If no bars are found for the symbol.
        """
        if interval != "1d":
            raise ValueError(f"only 1d bars are supported, got {interval!r}")

        # Determine which provider to use for this symbol.
        provider = self._provider_for(symbol) if self._provider_for else self._provider

        # Retrieve from store.
        series = self._store.series(provider, symbol, start=start, end=end)
        if series is None:
            raise DataUnavailable(f"no price data found for {symbol}")

        return series

    def index_history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> PriceSeries:
        """Delegate to live gateway if available."""
        if self._live is None:
            raise DataUnavailable("not stored locally and no live gateway configured")
        return self._live.index_history(symbol, start=start, end=end)

    def vix_history(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> PriceSeries:
        """Delegate to live gateway if available."""
        if self._live is None:
            raise DataUnavailable("not stored locally and no live gateway configured")
        return self._live.vix_history(start=start, end=end)

    def options_chain(self, symbol: str) -> OptionsChainSnapshot:
        """Delegate to live gateway if available."""
        if self._live is None:
            raise DataUnavailable("not stored locally and no live gateway configured")
        return self._live.options_chain(symbol)

    def cot(
        self,
        market: str,
        start: date | None = None,
        end: date | None = None,
    ) -> CotSeries:
        """Delegate to live gateway if available."""
        if self._live is None:
            raise DataUnavailable("not stored locally and no live gateway configured")
        return self._live.cot(market, start=start, end=end)

    def macro_series(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> MacroSeries:
        """Delegate to live gateway if available."""
        if self._live is None:
            raise DataUnavailable("not stored locally and no live gateway configured")
        return self._live.macro_series(series_id, start=start, end=end)

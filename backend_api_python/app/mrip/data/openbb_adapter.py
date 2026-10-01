"""OpenBBAdapter: primary financial data gateway for MRIP (ADR-0003).

Maps OpenBB (v5, provider-first command tree) onto the normalized models in
``app.mrip.data.models``. Nothing outside this module sees OpenBB types.

Provider choices and their verified behaviour (2026-10-01, OpenBB 5.0.0):
- ``cboe.options.chains``       options chain with OI/IV/greeks, no key, ~15 min delayed.
- ``cboe.index.historical``     VIX history, no key.
- ``cboe.equity.historical``    daily history for CBOE-listed US securities, no key.
- ``cftc.cot``                  CFTC Commitments of Traders, no key, ``code=`` CFTC market code.
- ``fred.economy.fred_series``  macro series, needs a FRED API key (not verified live).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import (
    CotRecord,
    CotSeries,
    Latency,
    MacroPoint,
    MacroSeries,
    OptionContract,
    OptionsChainSnapshot,
    PriceBar,
    PriceSeries,
    Provenance,
)

_GATEWAY = "openbb"
_DEFAULT_MULTIPLIER = 100
_GREEK_FIELDS = ("delta", "gamma", "theta", "vega", "rho")

# COT trader groups: group -> (long field candidates, short field candidates).
_COT_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "non_commercial": (("non_commercial_positions_long_all",), ("non_commercial_positions_short_all",)),
    "commercial": (("commercial_positions_long_all",), ("commercial_positions_short_all",)),
    "non_reportable": (("non_reportable_positions_long_all",), ("non_reportable_positions_short_all",)),
    "managed_money": (("managed_money_positions_long_all",), ("managed_money_positions_short_all",)),
    "dealer": (("dealer_positions_long_all",), ("dealer_positions_short_all",)),
    "asset_manager": (("asset_manager_positions_long",), ("asset_manager_positions_short",)),
    "swap": (("swap_positions_long_all",), ("swap_positions_short_all",)),
}


def _load_obb() -> Any:
    try:
        from openbb import obb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DataUnavailable(
            "OpenBB is not installed; install backend_api_python/requirements-mrip.txt"
        ) from exc
    return obb


def _rows(results: Any) -> list[dict[str, Any]]:
    """Normalize OpenBB results to row dicts.

    Handles a list of models, a model whose ``model_dump()`` yields row dicts
    (CBOE options chains, verified live) and, defensively, a dict of columns.
    """
    if results is None:
        return []
    if isinstance(results, list):
        return [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in results]
    dumped = results.model_dump() if hasattr(results, "model_dump") else results
    if isinstance(dumped, list):
        return [dict(r) for r in dumped]
    sized = [v for v in dumped.values() if isinstance(v, list) and v]
    if not sized:
        return []
    n = max(len(v) for v in sized)
    return [
        {k: (v[i] if isinstance(v, list) and i < len(v) else None) for k, v in dumped.items()}
        for i in range(n)
    ]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN -> None


def _to_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _to_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class OpenBBAdapter:
    """Implements ``FinancialDataGateway`` on top of OpenBB."""

    def __init__(
        self,
        obb: Any | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._obb = obb
        self._clock = clock

    @property
    def obb(self) -> Any:
        if self._obb is None:
            self._obb = _load_obb()
        return self._obb

    # -- public interface -------------------------------------------------

    def price_history(
        self, symbol: str, start: date | None = None, end: date | None = None, interval: str = "1d"
    ) -> PriceSeries:
        if interval != "1d":
            raise ValueError(f"only 1d bars are supported, got {interval!r}")
        response = self._call(
            "cboe.equity.historical",
            lambda: self.obb.cboe.equity.historical(symbol, start_date=start, end_date=end),
        )
        return PriceSeries(
            symbol=symbol,
            interval=interval,
            bars=self._bars(response, symbol),
            provenance=self._provenance(response, "cboe", "cboe.equity.historical"),
        )

    def vix_history(self, start: date | None = None, end: date | None = None) -> PriceSeries:
        response = self._call(
            "cboe.index.historical",
            lambda: self.obb.cboe.index.historical("VIX", start_date=start, end_date=end),
        )
        return PriceSeries(
            symbol="VIX",
            interval="1d",
            bars=self._bars(response, "VIX"),
            provenance=self._provenance(response, "cboe", "cboe.index.historical"),
        )

    def options_chain(self, symbol: str) -> OptionsChainSnapshot:
        response = self._call("cboe.options.chains", lambda: self.obb.cboe.options.chains(symbol))
        rows = _rows(getattr(response, "results", None))
        contracts = tuple(c for c in (self._contract(r) for r in rows) if c is not None)
        if not contracts:
            raise DataUnavailable(f"no option contracts returned for {symbol}")
        meta = (getattr(response, "extra", None) or {}).get("results_metadata") or {}
        underlying_price = _num(rows[0].get("underlying_price")) if rows else None
        if underlying_price is None:
            underlying_price = _num(meta.get("current_price"))
        return OptionsChainSnapshot(
            underlying=symbol,
            underlying_price=underlying_price,
            # Provider-local (exchange) time, tz not stated by the provider: kept naive.
            underlying_timestamp=_to_datetime(meta.get("last_trade_timestamp")),
            snapshot_timestamp=self._clock(),
            # The provider does not state the OI as-of date; never assume it.
            oi_effective_date=None,
            contracts=contracts,
            provenance=self._provenance(response, "cboe", "cboe.options.chains"),
        )

    def cot(self, market: str, start: date | None = None, end: date | None = None) -> CotSeries:
        """``market`` is a CFTC contract market code, e.g. ``"085692"`` (copper)."""
        response = self._call(
            "cftc.cot", lambda: self.obb.cftc.cot(code=market, start_date=start, end_date=end)
        )
        records = tuple(
            r for r in (self._cot_record(row, market) for row in _rows(getattr(response, "results", None)))
            if r is not None
        )
        if not records:
            raise DataUnavailable(f"no COT data returned for {market}")
        return CotSeries(
            market=market,
            records=records,
            provenance=self._provenance(response, "cftc", "cftc.cot"),
        )

    def macro_series(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> MacroSeries:
        response = self._call(
            "fred.economy.fred_series",
            lambda: self.obb.fred.economy.fred_series(series_id, start_date=start, end_date=end),
        )
        points = []
        for row in _rows(getattr(response, "results", None)):
            point_date = _to_date(row.get("date"))
            if point_date is None:
                continue
            raw = row[series_id] if series_id in row else row.get("value")
            points.append(MacroPoint(date=point_date, value=_num(raw)))
        if not points:
            raise DataUnavailable(f"no macro data returned for {series_id}")
        return MacroSeries(
            series_id=series_id,
            points=tuple(points),
            provenance=self._provenance(response, "fred", "fred.economy.fred_series"),
        )

    # -- helpers ----------------------------------------------------------

    def _call(self, endpoint: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except DataUnavailable:
            raise
        except Exception as exc:  # provider boundary: any OpenBB/provider failure
            raise DataUnavailable(f"{endpoint} failed: {exc}") from exc

    def _provenance(self, response: Any, provider: str, endpoint: str) -> Provenance:
        used = getattr(response, "provider", None) or provider
        return Provenance(
            provider=str(used),
            gateway=_GATEWAY,
            endpoint=endpoint,
            fetched_at=self._clock(),
            # CBOE data via OpenBB is ~15 min delayed (checked 2026-10-01: last trade
            # 12:35 vs request 12:50 exchange time). Other providers: unknown/not applicable.
            latency=Latency.DELAYED if used == "cboe" else Latency.UNKNOWN,
            gateway_version=self._gateway_version(),
        )

    def _gateway_version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("openbb")
        except Exception:  # package metadata missing (e.g. injected fake in tests)
            return None

    def _bars(self, response: Any, symbol: str) -> tuple[PriceBar, ...]:
        bars = []
        for row in _rows(getattr(response, "results", None)):
            raw = row.get("date")
            ts = raw if isinstance(raw, date) else _to_datetime(raw)
            if ts is None:
                continue
            bars.append(
                PriceBar(
                    ts=ts,
                    open=_num(row.get("open")),
                    high=_num(row.get("high")),
                    low=_num(row.get("low")),
                    close=_num(row.get("close")),
                    volume=_num(row.get("volume")),
                )
            )
        if not bars:
            raise DataUnavailable(f"no price data returned for {symbol}")
        return tuple(bars)

    def _contract(self, row: dict[str, Any]) -> OptionContract | None:
        expiry = _to_date(row.get("expiration"))
        strike = _num(row.get("strike"))
        option_type = str(row.get("option_type") or "").lower()
        if expiry is None or strike is None or option_type not in ("call", "put"):
            return None
        iv = _num(row.get("implied_volatility"))
        greeks: dict[str, float | None] = {name: _num(row.get(name)) for name in _GREEK_FIELDS}
        # CBOE reports IV 0.0 when no IV can be computed; its greeks are then a
        # degenerate model output. Treat all of them as missing, never as zero.
        if iv is None or iv <= 0:
            iv = None
            greeks = {name: None for name in _GREEK_FIELDS}
        multiplier = _num(row.get("contract_size"))
        return OptionContract(
            contract_symbol=row.get("contract_symbol"),
            expiry=expiry,
            strike=strike,
            option_type=option_type,  # type: ignore[arg-type]
            open_interest=_num(row.get("open_interest")),
            volume=_num(row.get("volume")),
            implied_volatility=iv,
            contract_multiplier=int(multiplier) if multiplier else _DEFAULT_MULTIPLIER,
            **greeks,
        )

    def _cot_record(self, row: dict[str, Any], market: str) -> CotRecord | None:
        report_date = _to_date(row.get("date"))
        if report_date is None:
            return None
        longs: dict[str, float | None] = {}
        shorts: dict[str, float | None] = {}
        for group, (long_fields, short_fields) in _COT_GROUPS.items():
            long_value = self._first(row, long_fields)
            short_value = self._first(row, short_fields)
            if long_value is not None or short_value is not None:
                longs[group] = long_value
                shorts[group] = short_value
        return CotRecord(
            report_date=report_date,
            market=str(row.get("market_and_exchange_names") or market),
            open_interest=_num(row.get("open_interest_all")),
            long_positions=longs,
            short_positions=shorts,
        )

    @staticmethod
    def _first(row: dict[str, Any], fields: Iterable[str]) -> float | None:
        for name in fields:
            value = _num(row.get(name))
            if value is not None:
                return value
        return None

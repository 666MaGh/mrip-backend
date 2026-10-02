"""OpenBBAdapter normalization tests against a fake OpenBB client (no network)."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency
from app.mrip.data.openbb_adapter import OpenBBAdapter

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class ColumnarModel:
    """Mimics CBOE's columnar options-chain result object."""

    def __init__(self, **columns):
        self._columns = columns

    def model_dump(self):
        return dict(self._columns)


class RecordsModel:
    """Mimics the real CBOE chain object: model_dump() returns a list of row dicts."""

    def __init__(self, rows):
        self._rows = rows

    def model_dump(self):
        return list(self._rows)


class RowModel:
    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):
        return dict(self._fields)


def response(results, provider="cboe", extra=None):
    return SimpleNamespace(results=results, provider=provider, extra=extra or {})


def make_adapter(**namespaces):
    return OpenBBAdapter(obb=SimpleNamespace(**namespaces), clock=lambda: NOW)


def chain_response(records=False):
    columns = dict(
            underlying_symbol=["SPY"] * 5,
            underlying_price=[100.0] * 5,
            contract_symbol=["C100", "P100", "C0DTE", "BAD", "C_NOIV"],
            expiration=[date(2026, 10, 16), date(2026, 10, 16), date(2026, 10, 1), None, date(2026, 11, 20)],
            strike=[100.0, 100.0, 99.0, 98.0, 120.0],
            option_type=["call", "put", "call", "call", "CALL"],
            open_interest=[500, 400, 0, 1, 10],
            volume=[10, 20, 0, 0, 1],
            implied_volatility=[0.25, 0.27, 0.0, 0.3, 0.0],
            delta=[0.5, -0.5, 1.0, 0.4, 1.0],
            gamma=[0.04, 0.04, 0.0, 0.03, 0.0],
            theta=[-0.1, -0.1, 0.0, -0.1, 0.0],
            vega=[0.2, 0.2, 0.0, 0.2, 0.0],
            rho=[0.05, -0.05, 0.0, 0.05, 0.0],
            contract_size=[],  # empty column: provider gave no multiplier
    )
    if records:
        n = 5
        data = RecordsModel(
            [{k: (v[i] if i < len(v) else None) for k, v in columns.items()} for i in range(n)]
        )
    else:
        data = ColumnarModel(**columns)
    return response(
        data,
        extra={"results_metadata": {"current_price": 100.5, "last_trade_timestamp": "2026-10-01 12:35:05"}},
    )


@pytest.mark.parametrize("records", [True, False], ids=["row-dicts (real shape)", "columns"])
def test_options_chain_normalizes_and_keeps_observed_data_honest(records):
    chains = lambda symbol: chain_response(records)
    adapter = make_adapter(cboe=SimpleNamespace(options=SimpleNamespace(chains=chains)))

    snap = adapter.options_chain("SPY")

    assert snap.underlying == "SPY"
    assert snap.underlying_price == 100.0
    assert snap.underlying_timestamp == datetime(2026, 10, 1, 12, 35, 5)
    assert snap.snapshot_timestamp == NOW
    assert snap.oi_effective_date is None  # provider does not state it; never assumed
    assert [c.contract_symbol for c in snap.contracts] == ["C100", "P100", "C0DTE", "C_NOIV"]  # BAD dropped
    call = snap.contracts[0]
    assert (call.option_type, call.strike, call.gamma, call.contract_multiplier) == ("call", 100.0, 0.04, 100)
    assert snap.contracts[3].option_type == "call"  # case-normalized


def test_options_zero_iv_means_missing_not_zero_greeks():
    adapter = make_adapter(
        cboe=SimpleNamespace(options=SimpleNamespace(chains=lambda s: chain_response(True)))
    )
    snap = adapter.options_chain("SPY")
    zero_iv = snap.contracts[2]
    assert zero_iv.implied_volatility is None
    assert (zero_iv.delta, zero_iv.gamma, zero_iv.theta, zero_iv.vega, zero_iv.rho) == (None,) * 5
    cov = snap.coverage
    assert (cov.contracts, cov.with_open_interest, cov.with_greeks, cov.with_implied_volatility) == (4, 4, 2, 2)


def test_options_provenance_marks_cboe_as_delayed():
    adapter = make_adapter(
        cboe=SimpleNamespace(options=SimpleNamespace(chains=lambda s: chain_response()))
    )
    p = adapter.options_chain("SPY").provenance
    assert (p.provider, p.gateway, p.endpoint, p.latency) == ("cboe", "openbb", "cboe.options.chains", Latency.DELAYED)
    assert p.fetched_at == NOW


def test_options_empty_chain_raises_data_unavailable():
    empty = response(ColumnarModel(contract_symbol=[], strike=[]))
    adapter = make_adapter(cboe=SimpleNamespace(options=SimpleNamespace(chains=lambda s: empty)))
    with pytest.raises(DataUnavailable):
        adapter.options_chain("ZZZZ")


def test_provider_failure_becomes_data_unavailable_with_cause():
    def boom(symbol):
        raise RuntimeError("upstream 503")

    adapter = make_adapter(cboe=SimpleNamespace(options=SimpleNamespace(chains=boom)))
    with pytest.raises(DataUnavailable, match="cboe.options.chains failed: upstream 503") as err:
        adapter.options_chain("SPY")
    assert isinstance(err.value.__cause__, RuntimeError)


def test_price_history_and_vix():
    rows = [
        RowModel(date=date(2026, 9, 30), open=1.0, high=2.0, low=0.5, close=1.5, volume=1000),
        RowModel(date=date(2026, 10, 1), open=1.5, high=2.5, low=1.0, close=float("nan"), volume=None),
    ]
    adapter = make_adapter(
        cboe=SimpleNamespace(
            equity=SimpleNamespace(historical=lambda sym, start_date=None, end_date=None: response(rows)),
            index=SimpleNamespace(historical=lambda sym, start_date=None, end_date=None: response(rows)),
        )
    )
    series = adapter.price_history("SPY", date(2026, 9, 1))
    assert series.symbol == "SPY" and series.interval == "1d"
    assert series.bars[0].close == 1.5
    assert series.bars[1].close is None and series.bars[1].volume is None  # NaN/None -> None
    assert adapter.vix_history().symbol == "VIX"


def test_index_history_passes_the_symbol_and_vix_is_an_alias():
    seen = []

    def historical(sym, start_date=None, end_date=None):
        seen.append(sym)
        return response([RowModel(date=date(2026, 9, 30), open=1.0, high=2.0, low=0.5, close=1.5, volume=None)])

    adapter = make_adapter(cboe=SimpleNamespace(index=SimpleNamespace(historical=historical)))
    assert adapter.index_history("TNX").symbol == "TNX"
    assert adapter.vix_history().symbol == "VIX" and seen == ["TNX", "VIX"]


def test_price_history_rejects_unsupported_interval():
    with pytest.raises(ValueError):
        make_adapter().price_history("SPY", interval="1m")


def test_cot_maps_trader_groups():
    rows = [
        RowModel(
            date=date(2026, 6, 2),
            market_and_exchange_names="COPPER- #1 - COMMODITY EXCHANGE INC.",
            open_interest_all=313585,
            non_commercial_positions_long_all=111505,
            non_commercial_positions_short_all=31906,
            commercial_positions_long_all=92456,
            commercial_positions_short_all=182262,
            managed_money_positions_long_all=None,
            managed_money_positions_short_all=None,
        )
    ]
    seen = {}

    def cot(code, start_date=None, end_date=None):
        seen["code"] = code
        return response(rows, provider="cftc")

    adapter = make_adapter(cftc=SimpleNamespace(cot=cot))
    series = adapter.cot("085692")

    assert seen["code"] == "085692"
    rec = series.records[0]
    assert rec.open_interest == 313585
    assert rec.long_positions == {"non_commercial": 111505, "commercial": 92456}
    assert rec.short_positions == {"non_commercial": 31906, "commercial": 182262}  # managed_money absent
    assert series.provenance.latency is Latency.UNKNOWN


@pytest.mark.parametrize("value_key", ["DGS10", "value"])
def test_macro_series_accepts_series_id_or_value_column(value_key):
    rows = [RowModel(date=date(2026, 9, 30), **{value_key: 4.1}), RowModel(date="2026-10-01", **{value_key: None})]
    fred = SimpleNamespace(
        economy=SimpleNamespace(fred_series=lambda sid, start_date=None, end_date=None: response(rows, provider="fred"))
    )
    series = make_adapter(fred=fred).macro_series("DGS10")
    assert [(p.date, p.value) for p in series.points] == [(date(2026, 9, 30), 4.1), (date(2026, 10, 1), None)]


def test_missing_openbb_install_is_reported_as_data_unavailable(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openbb":
            raise ImportError("no openbb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DataUnavailable, match="not installed"):
        OpenBBAdapter(clock=lambda: NOW).vix_history()

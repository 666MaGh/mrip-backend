"""Vol-surface and activity features on small hand-built chains."""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from app.mrip.data.models import OptionContract, OptionsChainSnapshot, Provenance
from app.mrip.options.vol_surface import (
    TermPoint,
    atm_iv_term_structure,
    iv_skew,
    put_call_ratios,
    snapshot_date,
    term_structure_slope,
    volume_oi_anomalies,
    volume_zscore,
)

TS = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)  # 11:00 New York -> 2026-10-01
PROV = Provenance(provider="t", gateway="t", endpoint="t", fetched_at=TS)

E7 = date(2026, 10, 8)  # dte 7
E30 = date(2026, 10, 31)  # dte 30
E90 = date(2026, 12, 30)  # dte 90


def c(expiry: date, strike: float, typ: str, **kw: object) -> OptionContract:
    return OptionContract(contract_symbol=None, expiry=expiry, strike=strike, option_type=typ, **kw)  # type: ignore[arg-type]


def snap(contracts: list[OptionContract], price: float | None = 100.0, ts: datetime = TS) -> OptionsChainSnapshot:
    return OptionsChainSnapshot(
        underlying="X",
        underlying_price=price,
        underlying_timestamp=None,
        snapshot_timestamp=ts,
        oi_effective_date=None,
        contracts=tuple(contracts),
        provenance=PROV,
    )


def test_snapshot_date_new_york_boundary() -> None:
    early = datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)  # 22:00 EDT on 09-30
    assert snapshot_date(snap([], ts=early)) == date(2026, 9, 30)
    assert snapshot_date(snap([], ts=TS)) == date(2026, 10, 1)
    # Winter time (EST, UTC-5): 04:59 UTC is still the previous day, 05:00 is not.
    assert snapshot_date(snap([], ts=datetime(2026, 12, 2, 4, 59, tzinfo=timezone.utc))) == date(2026, 12, 1)
    assert snapshot_date(snap([], ts=datetime(2026, 12, 2, 5, 0, tzinfo=timezone.utc))) == date(2026, 12, 2)


def test_dte_uses_new_york_date() -> None:
    early = datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc)
    pts = atm_iv_term_structure(snap([c(E7, 100, "call", implied_volatility=0.2)], ts=early))
    assert pts[0].dte == 8  # 2026-09-30 -> 2026-10-08


def test_atm_selection_mean_of_call_and_put() -> None:
    s = snap([
        c(E30, 95, "call", implied_volatility=0.30), c(E30, 95, "put", implied_volatility=0.32),
        c(E30, 101, "call", implied_volatility=0.20), c(E30, 101, "put", implied_volatility=0.22),
        c(E30, 110, "call", implied_volatility=0.18),
    ], price=100.4)
    (p,) = atm_iv_term_structure(s)
    assert p == TermPoint(expiry=E30, dte=30, atm_iv=pytest.approx(0.21), n_contracts=5)  # type: ignore[arg-type]


def test_atm_tie_breaks_to_lower_strike() -> None:
    s = snap([
        c(E30, 95, "call", implied_volatility=0.30),
        c(E30, 105, "call", implied_volatility=0.10),
    ], price=100.0)
    assert atm_iv_term_structure(s)[0].atm_iv == pytest.approx(0.30)


def test_atm_single_side_and_missing_iv_skipped() -> None:
    s = snap([
        # Nearest strike has no IV at all, so it is ignored.
        c(E30, 100, "call"), c(E30, 100, "put"),
        c(E30, 98, "put", implied_volatility=0.25),
        # Expiry with no IV anywhere is skipped.
        c(E7, 100, "call"), c(E7, 100, "put", implied_volatility=None),
        # Put-only IV at nearest strike while call has none.
        c(E90, 100, "call"), c(E90, 100, "put", implied_volatility=0.4),
    ])
    pts = atm_iv_term_structure(s)
    assert [p.expiry for p in pts] == [E30, E90]
    assert pts[0].atm_iv == pytest.approx(0.25) and pts[0].n_contracts == 1
    assert pts[1].atm_iv == pytest.approx(0.4)


def test_term_structure_sorted_and_min_dte() -> None:
    s = snap([
        c(E90, 100, "call", implied_volatility=0.3),
        c(E7, 100, "call", implied_volatility=0.2),
        c(E30, 100, "call", implied_volatility=0.25),
    ])
    assert [p.dte for p in atm_iv_term_structure(s)] == [7, 30, 90]
    assert [p.dte for p in atm_iv_term_structure(s, min_dte=8)] == [30, 90]


def test_atm_requires_underlying_price() -> None:
    with pytest.raises(ValueError):
        atm_iv_term_structure(snap([c(E30, 100, "call", implied_volatility=0.2)], price=None))


def _tp(dte: int, iv: float) -> TermPoint:
    return TermPoint(expiry=date(2026, 10, 1).fromordinal(date(2026, 10, 1).toordinal() + dte), dte=dte, atm_iv=iv, n_contracts=2)


def test_term_slope_selection() -> None:
    pts = [_tp(3, 0.50), _tp(10, 0.30), _tp(30, 0.28), _tp(55, 0.26), _tp(90, 0.20)]
    # short = smallest dte >= 7 -> 10 (0.30); long = largest dte <= 60 -> 55 (0.26)
    assert term_structure_slope(pts) == pytest.approx(-0.04)
    assert term_structure_slope(pts, short_dte=0, long_dte=100) == pytest.approx(0.20 - 0.50)


def test_term_slope_none_cases() -> None:
    assert term_structure_slope([]) is None
    assert term_structure_slope([_tp(30, 0.2)]) is None  # same point on both ends
    assert term_structure_slope([_tp(3, 0.5), _tp(5, 0.4)]) is None  # nothing >= 7
    assert term_structure_slope([_tp(70, 0.5), _tp(90, 0.4)]) is None  # nothing <= 60


def _skew_chain() -> list[OptionContract]:
    return [
        c(E30, 90, "put", implied_volatility=0.32, delta=-0.20),
        c(E30, 92, "put", implied_volatility=0.30, delta=-0.27),
        c(E30, 95, "put", implied_volatility=0.28, delta=-0.40),
        c(E30, 105, "call", implied_volatility=0.21, delta=0.30),
        c(E30, 108, "call", implied_volatility=0.19, delta=0.24),
        c(E30, 112, "call", implied_volatility=0.18, delta=0.15),
        # Other expiry must be ignored (30 is closest to target 30).
        c(E90, 92, "put", implied_volatility=0.9, delta=-0.25),
        c(E90, 108, "call", implied_volatility=0.1, delta=0.25),
    ]


def test_iv_skew_picks_nearest_deltas() -> None:
    sk = iv_skew(snap(_skew_chain()))
    assert sk is not None
    assert (sk.expiry, sk.dte) == (E30, 30)
    assert (sk.put_strike, sk.call_strike) == (92, 108)
    assert (sk.put_delta, sk.call_delta) == (-0.27, 0.24)
    assert (sk.put_iv, sk.call_iv) == (0.30, 0.19)
    assert sk.skew == pytest.approx(0.11)


def test_iv_skew_expiry_selection_and_tie_to_shorter() -> None:
    chain = [
        c(E7, 90, "put", implied_volatility=0.5, delta=-0.25), c(E7, 110, "call", implied_volatility=0.4, delta=0.25),
        c(E90, 90, "put", implied_volatility=0.3, delta=-0.25), c(E90, 110, "call", implied_volatility=0.2, delta=0.25),
    ]
    # dte 7 vs 90: target 48.5 is not reachable; use target 48 -> |7-48|=41 < |90-48|=42 -> E7.
    assert iv_skew(snap(chain), target_dte=48).expiry == E7  # type: ignore[union-attr]
    assert iv_skew(snap(chain), target_dte=49).expiry == E90  # type: ignore[union-attr]
    # Exact tie (dte 7 and 90 are 41.5 from 48.5): choose integer-tie setup instead.
    tie = [
        c(date(2026, 10, 11), 90, "put", implied_volatility=0.5, delta=-0.25),  # dte 10
        c(date(2026, 10, 11), 110, "call", implied_volatility=0.4, delta=0.25),
        c(date(2026, 10, 31), 90, "put", implied_volatility=0.3, delta=-0.25),  # dte 30
        c(date(2026, 10, 31), 110, "call", implied_volatility=0.2, delta=0.25),
    ]
    assert iv_skew(snap(tie), target_dte=20).expiry == date(2026, 10, 11)  # type: ignore[union-attr]


def test_iv_skew_ignores_expired_and_same_day() -> None:
    chain = [
        c(date(2026, 10, 1), 90, "put", implied_volatility=0.5, delta=-0.25),
        c(date(2026, 10, 1), 110, "call", implied_volatility=0.4, delta=0.25),
    ]
    assert iv_skew(snap(chain)) is None  # dte 0 is excluded


def test_iv_skew_none_when_side_missing_or_deltas_absent() -> None:
    puts_only = [x for x in _skew_chain() if x.option_type == "put"]
    assert iv_skew(snap(puts_only)) is None
    no_delta = [
        c(E30, 90, "put", implied_volatility=0.3), c(E30, 110, "call", implied_volatility=0.2),
    ]
    assert iv_skew(snap(no_delta)) is None
    no_iv_calls = [
        c(E30, 90, "put", implied_volatility=0.3, delta=-0.25), c(E30, 110, "call", delta=0.25),
    ]
    assert iv_skew(snap(no_iv_calls)) is None
    assert iv_skew(snap([])) is None


def test_put_call_ratios() -> None:
    s = snap([
        c(E7, 100, "call", volume=100, open_interest=1000),
        c(E7, 100, "put", volume=150, open_interest=500),
        c(E90, 100, "call", volume=None, open_interest=3000),
        c(E90, 100, "put", volume=50, open_interest=None),
    ])
    r = put_call_ratios(s)
    assert (r.call_volume, r.put_volume, r.call_open_interest, r.put_open_interest) == (100, 200, 4000, 500)
    assert r.volume == pytest.approx(2.0)
    assert r.open_interest == pytest.approx(0.125)
    near = put_call_ratios(s, max_dte=30)
    assert (near.call_volume, near.put_volume) == (100, 150)
    assert near.volume == pytest.approx(1.5) and near.open_interest == pytest.approx(0.5)


def test_put_call_ratios_none_without_calls() -> None:
    s = snap([c(E7, 100, "put", volume=10, open_interest=10), c(E7, 100, "call", volume=0, open_interest=None)])
    r = put_call_ratios(s)
    assert r.volume is None and r.open_interest is None
    assert r.put_volume == 10 and r.call_volume == 0
    empty = put_call_ratios(s, max_dte=-1)
    assert empty.volume is None and empty.put_volume == 0


def _anoms() -> list[OptionContract]:
    return [
        c(E30, 100, "call", volume=500, open_interest=100),  # ratio 5
        c(E30, 100, "put", volume=300, open_interest=100),  # ratio 3
        c(E7, 105, "call", volume=900, open_interest=None),  # None, vol 900
        c(E7, 95, "put", volume=2000, open_interest=0),  # None, vol 2000
        c(E7, 110, "call", volume=50, open_interest=None),  # below min_volume
        c(E7, 90, "put", volume=1000, open_interest=1000),  # ratio 1.0 -> not > 1.0
        c(E7, 120, "call", volume=None, open_interest=5),  # no volume
        c(E90, 100, "call", volume=200, open_interest=100),  # ratio 2
        c(E90, 100, "put", volume=200, open_interest=100),  # ratio 2 (tie with the call)
        c(E30, 99, "put", volume=100, open_interest=100),  # ratio 1.0 -> excluded; volume == min_volume ok
        c(E30, 98, "put", volume=101, open_interest=100),  # ratio 1.01
    ]


def test_anomaly_ordering_threshold_and_ties() -> None:
    out = volume_oi_anomalies(snap(_anoms()), top_n=100)
    got = [(a.expiry, a.strike, a.option_type, a.volume_oi_ratio) for a in out]
    assert got == [
        (E7, 95, "put", None),  # None ratio first, biggest volume first
        (E7, 105, "call", None),
        (E30, 100, "call", 5.0),
        (E30, 100, "put", 3.0),
        (E90, 100, "call", 2.0),  # tie: (expiry, strike, option_type) -> call before put
        (E90, 100, "put", 2.0),
        (E30, 98, "put", pytest.approx(1.01)),
    ]
    assert out[0].open_interest == 0 and out[1].open_interest is None


def test_anomaly_top_n_min_volume_and_threshold_params() -> None:
    s = snap(_anoms())
    assert len(volume_oi_anomalies(s, top_n=3)) == 3
    assert volume_oi_anomalies(s, top_n=0) == []
    hi = volume_oi_anomalies(s, min_volume=500, ratio_threshold=4.0, top_n=10)
    assert [(a.strike, a.volume) for a in hi] == [(95, 2000), (105, 900), (100, 500)]
    assert [a.volume_oi_ratio for a in hi] == [None, None, 5.0]


def test_anomaly_deterministic_regardless_of_input_order() -> None:
    a = volume_oi_anomalies(snap(_anoms()), top_n=100)
    b = volume_oi_anomalies(snap(list(reversed(_anoms()))), top_n=100)
    assert a == b


def test_volume_zscore_matches_numpy() -> None:
    hist = [float(x) for x in np.random.default_rng(0).normal(1000, 100, 30)]
    expected = (1300.0 - np.mean(hist)) / np.std(hist, ddof=1)
    assert volume_zscore(1300.0, hist) == pytest.approx(float(expected), abs=1e-12)


def test_volume_zscore_none_cases() -> None:
    assert volume_zscore(5.0, [1.0] * 19) is None  # too short
    assert volume_zscore(5.0, [3.0] * 25) is None  # zero std
    assert volume_zscore(5.0, [1.0, 2.0, 3.0], min_history=3) == pytest.approx(3.0)
    assert volume_zscore(5.0, [1.0, 2.0, float("nan")], min_history=3) is None
    assert volume_zscore(5.0, [1.0], min_history=1) is None  # std undefined

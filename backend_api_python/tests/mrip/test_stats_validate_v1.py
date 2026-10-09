"""validation-v1-uncalibrated: sector control and same-sector null on synthetic factor models (no DB)."""
import json

import numpy as np
import pandas as pd

from app.mrip.evidence.types import Stance
from app.mrip.relationships.types import RelationType
from app.mrip.stats.null_distribution import (
    GLOBAL_SECTOR,
    MIN_SECTOR_PEERS,
    compute_null,
    sector_peer_average,
)
from app.mrip.stats.validate import POLICY_V0, SectorControl, ValidationPolicy, Verdict, validate_relationship

N = 700
DATES = pd.bdate_range("2020-01-01", periods=N)


def series(seed: int, scale: float = 1.0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0, scale, N), index=DATES)


class Factors:
    """One market factor, one sector factor, idiosyncratic noise per name (seeded)."""

    def __init__(self, sector_members: int = 12):
        self.market = series(1)
        self.sector = series(2)
        self.idio = {i: series(100 + i, 0.7) for i in range(sector_members + 4)}

    def stock(self, i: int) -> pd.Series:
        return self.market + self.sector + self.idio[i]

    def peers(self, names: list[int]) -> pd.DataFrame:
        return pd.DataFrame({f"S{i}": self.stock(i) for i in names})


def sector_control(factors: Factors, pair: tuple[int, int], members: list[int]) -> SectorControl:
    frame = factors.peers(members)
    peer = sector_peer_average(frame, [f"S{pair[0]}", f"S{pair[1]}"])
    assert peer is not None
    return SectorControl(series=peer, sector="Tech", peers=len(members) - 2)


def same_sector_null(factors: Factors, members: list[int], seed: int = 7):
    frame_symbols = [f"S{i}" for i in members]
    returns = {s: factors.stock(int(s[1:])) for s in frame_symbols}
    pool = [(a, b) for i, a in enumerate(frame_symbols) for b in frame_symbols[i + 1:]]
    null = compute_null(
        sector="Tech", as_of=pd.Timestamp("2026-10-01").date(), policy_version=ValidationPolicy().version,
        returns=returns, market=factors.market, pool=pool, max_lag=ValidationPolicy().max_lag,
        sector_members=frame_symbols, n=200, seed=seed,
    )
    assert null is not None
    return null


def test_common_sector_and_market_factor_is_validated_under_v0_but_not_v1():
    factors = Factors(12)
    members = list(range(12))
    x, y = factors.stock(0), factors.stock(1)  # only shared factors, no pair-specific link
    v0 = validate_relationship(x, y, market=factors.market, expected_sign=1, policy=POLICY_V0)
    assert v0.verdict is Verdict.VALIDATED  # the old rules call this a relationship

    control = sector_control(factors, (0, 1), members)
    null = same_sector_null(factors, members)
    v1 = validate_relationship(
        x, y, market=factors.market, sector=control, null=null, expected_sign=1,
        policy=ValidationPolicy(),
    )
    assert v1.verdict is Verdict.INCONCLUSIVE and v1.stance is Stance.NEUTRAL
    assert v1.policy_version == "validation-v1-uncalibrated"
    assert v1.metrics["sector_control"] == "used"
    assert abs(v1.metrics["partial_r"]) < null.p95  # sector peer average absorbs the shared factor
    assert v1.metrics["controls"] == ["market_t", "sector_t"]


def test_genuine_idiosyncratic_lagged_link_is_validated_under_v1():
    factors = Factors(12)
    members = list(range(12))
    x = factors.stock(0)
    # Pair-specific dependency at lag 2, strong enough to dominate the shared-factor correlation at lag 0.
    y = (factors.stock(1) + 2.0 * x.shift(2)).dropna()
    control = sector_control(factors, (0, 1), members)
    null = same_sector_null(factors, members)
    res = validate_relationship(
        x, y, market=factors.market, sector=control, null=null, expected_sign=1,
        relation_type=RelationType.LEADS, policy=ValidationPolicy(),
    )
    assert res.verdict is Verdict.VALIDATED, res.reasons
    assert res.metrics["best_lag"] == 2 and res.metrics["direction"] == "x_leads"
    assert res.metrics["effect_measure"] > res.metrics["null_p95"]
    assert res.metrics["null_p95"] == round(null.p95, 6)
    assert res.metrics["null_label"] == "same-sector null" and res.metrics["null_n"] == null.pairs_used


def test_inconclusive_reason_names_the_null_and_its_numbers():
    factors = Factors(12)
    members = list(range(12))
    x, y = factors.stock(0), factors.stock(1)  # shared factors only: the sector control removes the link
    control = sector_control(factors, (0, 1), members)
    null = same_sector_null(factors, members)
    res = validate_relationship(x, y, market=factors.market, sector=control, null=null, expected_sign=1, policy=ValidationPolicy())
    assert res.verdict is Verdict.INCONCLUSIVE
    gate = [r for r in res.reasons if r.startswith("does not exceed")]
    assert len(gate) == 1, res.reasons
    assert gate[0].startswith("does not exceed same-sector null (partial_r ")
    assert f"p95 {null.p95:.3f}" in gate[0] and f"n={null.pairs_used})" in gate[0]


def test_missing_sector_control_is_labelled_and_market_only():
    factors = Factors(3)
    res = validate_relationship(
        factors.stock(0), factors.stock(1), market=factors.market, sector=None, sector_status="unavailable",
        null=None, expected_sign=1, policy=ValidationPolicy(),
    )
    assert res.metrics["sector_control"] == "unavailable"
    assert res.metrics["controls"] == ["market_t"]
    assert res.metrics["peers_used"] == 0 and res.metrics["sector"] is None


def test_v1_without_null_is_inconclusive_never_silently_validated():
    factors = Factors(12)
    res = validate_relationship(factors.stock(0), factors.stock(1) + 0.9 * factors.stock(0).shift(1).fillna(0),
                                market=factors.market, expected_sign=1, policy=ValidationPolicy())
    assert res.verdict is Verdict.INCONCLUSIVE
    assert any("null unavailable" in r for r in res.reasons)


def test_sector_control_is_skipped_below_the_peer_minimum():
    factors = Factors(6)
    frame = factors.peers([0, 1, 2, 3, 4])  # pair (0,1) leaves 3 peers
    assert sector_peer_average(frame, ["S0", "S1"]) is None
    assert MIN_SECTOR_PEERS == 5


def test_sector_peer_average_excludes_the_pair_and_needs_enough_peers():
    frame = pd.DataFrame({"A": series(1), "B": series(2), "P1": series(3), "P2": series(4), "P3": series(5),
                          "P4": series(6), "P5": series(7)})
    peer = sector_peer_average(frame, ["A", "B"])
    assert peer is not None
    expected = frame[["P1", "P2", "P3", "P4", "P5"]].mean(axis=1)
    pd.testing.assert_series_equal(peer, expected, check_names=False)


def test_null_is_seeded_and_deterministic():
    factors = Factors(25)  # 300 pairs > NULL_SAMPLE_SIZE, so the seed decides which pairs are sampled
    members = list(range(25))
    a = same_sector_null(factors, members, seed=7)
    b = same_sector_null(factors, members, seed=7)
    c = same_sector_null(factors, members, seed=8)
    assert a == b
    assert (a.p05, a.p95) != (c.p05, c.p95)
    assert a.p05 < 0 < a.p95 and a.label == "same-sector null"


def test_global_null_is_labelled_random_pair_and_small_pools_are_unavailable():
    factors = Factors(4)
    symbols = [f"S{i}" for i in range(4)]
    returns = {s: factors.stock(int(s[1:])) for s in symbols}
    pool = [(a, b) for i, a in enumerate(symbols) for b in symbols[i + 1:]]  # 6 pairs < MIN_NULL_PAIRS
    assert compute_null(sector=GLOBAL_SECTOR, as_of=DATES[-1].date(), policy_version="x", returns=returns,
                        market=factors.market, pool=pool, max_lag=5) is None


def test_metrics_are_json_friendly_and_name_the_evidence_fields():
    factors = Factors(12)
    members = list(range(12))
    res = validate_relationship(factors.stock(0), factors.stock(1), market=factors.market,
                                sector=sector_control(factors, (0, 1), members),
                                null=same_sector_null(factors, members), expected_sign=1, policy=ValidationPolicy())
    payload = json.loads(json.dumps(res.metrics))  # raises if a numpy type leaks through
    for key in ("partial_r", "null_p95", "null_n", "null_label", "sector", "peers_used", "controls", "sector_control"):
        assert key in payload
    assert payload["sector_control"] == "used" and payload["peers_used"] == 10

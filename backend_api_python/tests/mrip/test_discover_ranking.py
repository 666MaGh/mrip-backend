from datetime import date

import pytest

from app.mrip.discover.ranking import RankingPolicy, rank
from app.mrip.discover.types import Kind, Observation


def test_ranking_score_unknown_evidence_and_conflict():
    obs = Observation(Kind.COT_CROWDING, "Gold", date(2026, 1, 1), "crowded", 1.0, evidence={"support": 1, "contradict": 1}, reliability=1, confidence=1, liquidity=1)
    item = rank([obs])[0]
    assert item.components["evidence"] == .25
    assert item.score == 88.75
    unknown = rank([Observation(Kind.REGIME_SHIFT, "MARKET", date.today(), "shift", .5)])[0]
    assert "evidence" in unknown.unknown and unknown.components["evidence"] == .5


def test_ranking_weight_validation_and_tie_break():
    with pytest.raises(ValueError):
        RankingPolicy(magnitude=.9)
    items = [Observation(Kind.REGIME_SHIFT, "Z", date.today(), "x", .5), Observation(Kind.COT_CROWDING, "A", date.today(), "x", .5)]
    result = rank(items)
    assert [(x.observation.kind.value, x.observation.subject) for x in result] == [("COT_CROWDING", "A"), ("REGIME_SHIFT", "Z")]


def test_all_weights_sum_to_one_and_bad_total_is_rejected():
    policy = RankingPolicy()
    assert sum((policy.magnitude, policy.evidence, policy.novelty, policy.reliability, policy.confidence, policy.liquidity, policy.data_quality)) == pytest.approx(1.)
    with pytest.raises(ValueError, match="sum to 1"):
        RankingPolicy(magnitude=.31)


def test_unknown_components_are_neutral_and_listed():
    observation = Observation(Kind.REGIME_SHIFT, "MARKET", date(2026, 1, 1), "shift", .8, data_quality={})
    item = rank([observation])[0]
    assert item.components["evidence"] == item.components["reliability"] == item.components["confidence"] == item.components["liquidity"] == .5
    assert item.unknown == ("evidence", "reliability", "confidence", "liquidity")


def test_conflicting_evidence_penalty_and_novelty_decay():
    date_seen = date(2026, 1, 1)
    clean = Observation(Kind.COT_CROWDING, "Gold", date_seen, "x", .5, evidence={"support": 1, "contradict": 0})
    mixed = Observation(Kind.COT_CROWDING, "Gold", date_seen, "x", .5, evidence={"support": 1, "contradict": 1})
    clean_item, mixed_item = rank([clean, mixed])
    assert clean_item.components["evidence"] == 1.
    assert mixed_item.components["evidence"] == .25
    fresh = rank([clean])[0]
    seen = rank([clean], recent={(Kind.COT_CROWDING.value, "Gold"): 3})[0]
    old = rank([clean], recent={(Kind.COT_CROWDING.value, "Gold"): 20})[0]
    assert fresh.components["novelty"] == 1
    assert seen.components["novelty"] == pytest.approx(4 / 7)
    assert old.components["novelty"] == .2


def test_hand_computed_score_and_deterministic_tie_breaks():
    item = Observation(Kind.COT_CROWDING, "Gold", date(2026, 1, 1), "x", .8, data_quality={"completeness": .6}, evidence={"support": 3, "contradict": 1}, reliability=.7, confidence=.9, liquidity=.2)
    ranked = rank([item])[0]
    expected = round(100 * (.30 * .8 + .15 * .5 + .15 * 1 + .10 * .7 + .10 * .9 + .10 * .2 + .10 * .6), 4)
    assert ranked.score == expected == 70.5
    same_kind_z = Observation(Kind.COT_CROWDING, "Z", date(2026, 1, 1), "z", .5)
    same_kind_a = Observation(Kind.COT_CROWDING, "A", date(2026, 1, 1), "a", .5)
    tied = rank([same_kind_z, same_kind_a])
    assert [item.observation.subject for item in tied] == ["A", "Z"]
    assert tied == rank([same_kind_a, same_kind_z])

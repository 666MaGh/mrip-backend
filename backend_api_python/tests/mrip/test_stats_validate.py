"""Decision rules of validate_relationship on synthetic relationships with known truth."""
import numpy as np
import pandas as pd
import pytest

from app.mrip.evidence.types import Stance
from app.mrip.relationships.types import RelationType
from app.mrip.stats.validate import POLICY_V0, ValidationPolicy, Verdict, validate_relationship

# These tests pin the v0 rules (market-only control, no null gate); v1 is covered in test_stats_validate_v1.py.

N = 700


def noise(seed, n=N, scale=1.0):
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=pd.bdate_range("2020-01-01", periods=n))


def leading_pair(sign=1.0, lag=2, seed=0):
    x = noise(seed + 1)
    y = sign * 0.5 * x.shift(lag) + noise(seed + 2, scale=0.8)
    return x, y


def test_stable_leading_relationship_with_expected_sign_is_validated():
    x, y = leading_pair()
    res = validate_relationship(x, y, expected_sign=1, relation_type=RelationType.LEADS, market=noise(9), policy=POLICY_V0)
    assert res.verdict is Verdict.VALIDATED and res.stance is Stance.SUPPORT
    assert res.metrics["best_lag"] == 2 and res.metrics["direction"] == "x_leads"
    assert res.metrics["oos_consistency"] >= 0.6 and res.policy_version == "v0-uncalibrated"  # v0 rules
    assert res.metrics["n_obs"] == N - 2 and res.metrics["partial_p"] < 0.05  # two rows lost to the lag


def test_significant_effect_with_the_wrong_sign_is_rejected():
    x, y = leading_pair(sign=-1.0)
    res = validate_relationship(x, y, expected_sign=1)
    assert res.verdict is Verdict.REJECTED and res.stance is Stance.CONTRADICT
    assert "expected +1" in res.reasons[0]


def test_matching_negative_sign_validates():
    x, y = leading_pair(sign=-1.0)
    assert validate_relationship(x, y, expected_sign=-1, policy=POLICY_V0).verdict is Verdict.VALIDATED


def test_independent_series_stay_inconclusive_never_rejected():
    res = validate_relationship(noise(100), noise(101), expected_sign=1)
    assert res.verdict is Verdict.INCONCLUSIVE and res.stance is Stance.NEUTRAL
    assert "no significant correlation" in res.reasons[0]


def test_too_little_data_is_inconclusive():
    res = validate_relationship(noise(1, 200), noise(2, 200))
    assert res.verdict is Verdict.INCONCLUSIVE and "insufficient observations" in res.reasons[0]


def test_leads_requires_x_to_lead_and_lags_requires_y_to_lead():
    x, y = leading_pair()  # x leads y
    assert validate_relationship(x, y, relation_type=RelationType.LEADS, policy=POLICY_V0).verdict is Verdict.VALIDATED
    wrong = validate_relationship(x, y, relation_type=RelationType.LAGS, policy=POLICY_V0)
    assert wrong.verdict is Verdict.INCONCLUSIVE and any("LAGS not confirmed" in r for r in wrong.reasons)
    assert validate_relationship(y, x, relation_type=RelationType.LAGS, policy=POLICY_V0).verdict is Verdict.VALIDATED
    flipped = validate_relationship(y, x, relation_type=RelationType.LEADS, policy=POLICY_V0)
    assert flipped.verdict is Verdict.INCONCLUSIVE and any("LEADS not confirmed" in r for r in flipped.reasons)


def test_relationship_explained_by_the_market_does_not_survive_control():
    m = noise(200)
    x, y = m + noise(201, scale=0.7), m + noise(202, scale=0.7)
    assert validate_relationship(x, y, policy=POLICY_V0).verdict is Verdict.VALIDATED  # looks real without a control
    controlled = validate_relationship(x, y, market=m, policy=POLICY_V0)
    assert controlled.verdict is Verdict.INCONCLUSIVE
    assert any("controlling for the market" in r for r in controlled.reasons)


def test_relationship_that_flips_over_time_fails_out_of_sample_consistency():
    x = noise(300)
    y = 0.5 * x.shift(1) + noise(301, scale=0.8)
    y.iloc[N // 2 :] = (-0.5 * x.shift(1).iloc[N // 2 :]).to_numpy() + noise(302, N - N // 2, 0.8).to_numpy()
    res = validate_relationship(x, y, policy=POLICY_V0)
    assert res.verdict is not Verdict.VALIDATED


def test_input_validation_and_policy_validation():
    with pytest.raises(ValueError):
        validate_relationship(noise(1), noise(2), expected_sign=2)
    with pytest.raises(ValueError):
        ValidationPolicy(alpha=1.5)
    with pytest.raises(ValueError):
        ValidationPolicy(min_observations=100, train_size=150, test_size=50)


def test_result_is_deterministic_and_metrics_are_json_friendly():
    import json

    x, y = leading_pair()
    a = validate_relationship(x, y, expected_sign=1, market=noise(9), policy=POLICY_V0)
    b = validate_relationship(x, y, expected_sign=1, market=noise(9), policy=POLICY_V0)
    assert a == b
    json.dumps(a.metrics)  # raises if a numpy type leaks through

"""Known-answer tests for classifier calibration (scores, bins, temperature, isotonic, abstain)."""
from __future__ import annotations

import numpy as np
import pytest

from app.mrip.calibration.classification import (
    AbstainFit,
    IsotonicMap,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_abstain_threshold,
    fit_isotonic,
    fit_temperature,
    log_loss,
    reliability_bins,
)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sample_labels(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    cum = probs.cumsum(axis=1)
    u = rng.random((probs.shape[0], 1))
    return np.minimum((u > cum).sum(axis=1), probs.shape[1] - 1)


def _conf_correct(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return probs.max(axis=1), probs.argmax(axis=1) == labels


# ---------------------------------------------------------------- scores


def test_brier_hand_computed() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 2])
    # row 1: 0.09 + 0.04 + 0.01 = 0.14 ; row 2: 0.01 + 0.01 + 0.04 = 0.06
    assert brier_score(probs, labels) == pytest.approx(0.10)


def test_brier_extremes() -> None:
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier_score(perfect, np.array([0, 1])) == pytest.approx(0.0)
    assert brier_score(perfect, np.array([1, 0])) == pytest.approx(2.0)


def test_log_loss_hand_computed() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 2])
    expected = -(np.log(0.7) + np.log(0.8)) / 2.0
    assert log_loss(probs, labels) == pytest.approx(expected)


def test_log_loss_clips_zero_probability() -> None:
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    labels = np.array([1, 0])
    expected = -(np.log(1e-12) + np.log(0.5)) / 2.0
    assert log_loss(probs, labels) == pytest.approx(expected)
    assert np.isfinite(log_loss(probs, labels))


def test_scores_reject_bad_input() -> None:
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        brier_score(probs, np.array([0, 2]))
    with pytest.raises(ValueError):
        log_loss(probs, np.array([0]))
    with pytest.raises(ValueError):
        brier_score(np.array([[0.5, 0.6]]), np.array([0]))


# ------------------------------------------------------- reliability / ECE


def test_reliability_bins_edges_and_empty_omitted() -> None:
    conf = np.array([0.05, 0.15, 0.15, 0.95, 1.0, 1.0])
    correct = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
    bins = reliability_bins(conf, correct, n_bins=10)
    assert [b.n for b in bins] == [1, 2, 3]
    assert [(b.lower, b.upper) for b in bins] == [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.9, 1.0),
    ]
    assert bins[0].mean_confidence == pytest.approx(0.05)
    assert bins[0].accuracy == pytest.approx(1.0)
    assert bins[1].mean_confidence == pytest.approx(0.15)
    assert bins[1].accuracy == pytest.approx(0.5)
    # 1.0 lands in the last bin, not in a bin of its own
    assert bins[2].mean_confidence == pytest.approx((0.95 + 1.0 + 1.0) / 3)
    assert bins[2].accuracy == pytest.approx(2 / 3)


def test_reliability_bins_counts_sum_to_n() -> None:
    rng = np.random.default_rng(1)
    conf = rng.random(500)
    correct = rng.random(500) < conf
    assert sum(b.n for b in reliability_bins(conf, correct, 7)) == 500


def test_ece_hand_computed() -> None:
    conf = np.array([0.05, 0.15, 0.15, 0.95, 1.0, 1.0])
    correct = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
    expected = (
        1 / 6 * abs(1.0 - 0.05)
        + 2 / 6 * abs(0.5 - 0.15)
        + 3 / 6 * abs(2 / 3 - (0.95 + 1.0 + 1.0) / 3)
    )
    assert expected_calibration_error(conf, correct) == pytest.approx(expected)


def test_ece_near_zero_when_calibrated() -> None:
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = rng.random(20000) < conf
    assert expected_calibration_error(conf, correct) < 0.02


def test_ece_large_when_overconfident() -> None:
    conf = np.full(100, 0.9)
    correct = np.zeros(100, dtype=bool)
    correct[:50] = True
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4)


def test_reliability_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        reliability_bins(np.array([0.5, 1.2]), np.array([True, False]))
    with pytest.raises(ValueError):
        reliability_bins(np.array([0.5]), np.array([True, False]))
    with pytest.raises(ValueError):
        reliability_bins(np.array([0.5]), np.array([True]), n_bins=0)


# ------------------------------------------------------------ temperature


def test_apply_temperature_identity_at_one() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.0, 0.4, 0.6]])
    np.testing.assert_allclose(apply_temperature(probs, 1.0), probs, atol=1e-12)


def test_apply_temperature_flattens_and_sharpens() -> None:
    probs = np.array([[0.7, 0.2, 0.1]])
    hot = apply_temperature(probs, 2.0)
    cold = apply_temperature(probs, 0.5)
    assert hot.max() < probs.max() < cold.max()
    # hand check at T = 2: sqrt weights normalised
    w = np.sqrt(probs[0])
    np.testing.assert_allclose(hot[0], w / w.sum(), atol=1e-12)
    # ordering of classes is preserved
    assert np.argmax(hot) == np.argmax(cold) == 0


def test_apply_temperature_rows_sum_to_one_and_zeros_stay_zero() -> None:
    probs = np.array([[0.0, 0.3, 0.7], [1.0, 0.0, 0.0], [0.2, 0.2, 0.6]])
    for t in (0.1, 0.5, 2.0, 10.0):
        out = apply_temperature(probs, t)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
        assert out[0, 0] == 0.0
        assert out[1, 1] == 0.0 and out[1, 2] == 0.0
        assert np.all(np.isfinite(out))


def test_apply_temperature_stable_for_extreme_temperatures() -> None:
    probs = np.array([[0.9, 0.1, 1e-300]])
    out = apply_temperature(probs, 1e-3)
    assert np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("t", [0.0, -1.0, float("nan")])
def test_apply_temperature_rejects_non_positive(t: float) -> None:
    with pytest.raises(ValueError):
        apply_temperature(np.array([[0.5, 0.5]]), t)


def test_fit_temperature_recovers_known_temperature() -> None:
    rng = np.random.default_rng(42)
    n, k, true_t = 6000, 4, 2.0
    z = rng.normal(0.0, 2.0, size=(n, k))
    labels = _sample_labels(rng, _softmax(z / true_t))
    probs = _softmax(z)  # model is overconfident by a factor of 2

    half = n // 2
    fit = fit_temperature(probs[:half], labels[:half])
    assert fit.n == half
    assert fit.temperature == pytest.approx(true_t, rel=0.15)
    assert fit.nll_after < fit.nll_before
    assert fit.nll_before == pytest.approx(log_loss(probs[:half], labels[:half]))

    # held-out half: calibration improves
    held_probs, held_labels = probs[half:], labels[half:]
    before = expected_calibration_error(*_conf_correct(held_probs, held_labels))
    scaled = apply_temperature(held_probs, fit.temperature)
    after = expected_calibration_error(*_conf_correct(scaled, held_labels))
    assert after < before
    assert log_loss(scaled, held_labels) < log_loss(held_probs, held_labels)


def test_fit_temperature_near_one_when_calibrated() -> None:
    rng = np.random.default_rng(3)
    n, k = 6000, 4
    z = rng.normal(0.0, 2.0, size=(n, k))
    probs = _softmax(z)
    labels = _sample_labels(rng, probs)
    fit = fit_temperature(probs, labels)
    assert fit.temperature == pytest.approx(1.0, abs=0.1)
    assert fit.nll_after <= fit.nll_before + 1e-9


def test_fit_temperature_respects_bounds() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 2.0, size=(2000, 3))
    labels = _sample_labels(rng, _softmax(z / 4.0))
    fit = fit_temperature(_softmax(z), labels, bounds=(0.5, 1.5))
    assert 0.5 <= fit.temperature <= 1.5
    assert fit.temperature == pytest.approx(1.5, abs=1e-3)


def test_fit_temperature_value_errors() -> None:
    rng = np.random.default_rng(0)
    probs = _softmax(rng.normal(size=(20, 3)))
    labels = rng.integers(0, 3, size=20)
    with pytest.raises(ValueError):  # n < 10
        fit_temperature(probs[:9], labels[:9])
    with pytest.raises(ValueError):  # shape mismatch
        fit_temperature(probs, labels[:19])
    with pytest.raises(ValueError):  # rows do not sum to 1
        fit_temperature(probs * 0.9, labels)
    with pytest.raises(ValueError):  # labels out of range
        fit_temperature(probs, labels + 3)
    with pytest.raises(ValueError):  # negative label
        fit_temperature(probs, labels - 3)
    with pytest.raises(ValueError):  # bad bounds
        fit_temperature(probs, labels, bounds=(2.0, 1.0))


# --------------------------------------------------------------- isotonic


def test_isotonic_hand_worked_pav_example() -> None:
    x = np.arange(1, 11) / 10.0
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
    iso = fit_isotonic(x, y)
    # PAV: (1,0) at x=.3,.4 pool to 0.5; (1,1,0) at x=.5,.6,.7 pool to 2/3
    fitted = iso.apply(x)
    np.testing.assert_allclose(fitted, [0, 0, 0.5, 0.5, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1], atol=1e-12)
    # linear interpolation between blocks: between x=.4 (0.5) and x=.5 (2/3)
    assert iso.apply(0.45) == pytest.approx((0.5 + 2 / 3) / 2)
    # constant inside a pooled block
    assert iso.apply(0.35) == pytest.approx(0.5)
    assert iso.knots_x == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0))
    assert iso.knots_y == pytest.approx((0, 0, 0.5, 0.5, 2 / 3, 2 / 3, 1, 1, 1))


def test_isotonic_output_is_monotone() -> None:
    rng = np.random.default_rng(11)
    conf = rng.uniform(0.0, 1.0, 400)
    correct = rng.random(400) < 0.5
    iso = fit_isotonic(conf, correct)
    assert all(a < b for a, b in zip(iso.knots_x, iso.knots_x[1:]))
    assert all(a <= b for a, b in zip(iso.knots_y, iso.knots_y[1:]))
    grid = np.linspace(-0.2, 1.2, 501)
    assert np.all(np.diff(iso.apply(grid)) >= -1e-15)
    assert all(0.0 <= v <= 1.0 for v in iso.knots_y)


def test_isotonic_already_monotone_data_is_unchanged() -> None:
    x = np.arange(1, 11) / 10.0
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    iso = fit_isotonic(x, y)
    np.testing.assert_allclose(iso.apply(x), y, atol=1e-12)


def test_isotonic_clips_outside_knot_range() -> None:
    x = np.linspace(0.4, 0.9, 12)
    y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1])
    iso = fit_isotonic(x, y)
    assert iso.apply(0.0) == pytest.approx(iso.knots_y[0])
    assert iso.apply(1.0) == pytest.approx(iso.knots_y[-1])
    out = iso.apply(np.array([-1.0, 2.0]))
    assert out[0] == pytest.approx(iso.knots_y[0])
    assert out[1] == pytest.approx(iso.knots_y[-1])


def test_isotonic_ties_are_merged() -> None:
    conf = np.array([0.2] * 5 + [0.8] * 5)
    correct = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    iso = fit_isotonic(conf, correct)
    assert iso.knots_x == (0.2, 0.8)
    assert iso.knots_y == pytest.approx((0.2, 0.8))


def test_isotonic_ties_that_violate_order_are_pooled() -> None:
    conf = np.array([0.3] * 5 + [0.7] * 5)
    correct = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])  # 0.8 then 0.2
    iso = fit_isotonic(conf, correct)
    assert iso.knots_x == (0.3, 0.7)
    assert iso.knots_y == pytest.approx((0.5, 0.5))


def test_isotonic_apply_returns_scalar_for_scalar() -> None:
    iso = IsotonicMap(knots_x=(0.0, 1.0), knots_y=(0.0, 1.0))
    assert isinstance(iso.apply(0.25), float)
    assert iso.apply(0.25) == pytest.approx(0.25)
    assert isinstance(iso.apply(np.array([0.25])), np.ndarray)


def test_isotonic_reduces_ece_on_miscalibrated_confidences() -> None:
    rng = np.random.default_rng(21)
    n = 8000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf**2  # true accuracy is below stated confidence
    half = n // 2
    iso = fit_isotonic(conf[:half], correct[:half])
    held_conf, held_correct = conf[half:], correct[half:]
    before = expected_calibration_error(held_conf, held_correct)
    after = expected_calibration_error(np.asarray(iso.apply(held_conf)), held_correct)
    assert before > 0.1
    assert after < before / 3
    assert after < 0.03


def test_isotonic_value_errors() -> None:
    with pytest.raises(ValueError):
        fit_isotonic(np.linspace(0.1, 0.9, 9), np.array([0, 1] * 4 + [1]))
    with pytest.raises(ValueError):
        fit_isotonic(np.linspace(0.1, 0.9, 10), np.array([0, 1] * 4 + [1]))
    with pytest.raises(ValueError):
        fit_isotonic(np.linspace(0.1, 1.9, 10), np.array([0, 1] * 5))
    with pytest.raises(ValueError):
        fit_isotonic(np.linspace(0.1, 0.9, 10), np.array([0, 2] * 5))


# ---------------------------------------------------------------- abstain

_ABS_CONF = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5])
_ABS_OK = np.array([1, 1, 1, 0, 1, 0, 1, 0], dtype=bool)
# threshold: accepted / correct / precision
#   0.50: 8 / 5 / .625    0.55: 7 / 5 / .714    0.60: 6 / 4 / .667    0.70: 5 / 4 / .80
#   0.80: 4 / 3 / .75     0.85: 3 / 3 / 1.0     0.90: 2 / 2 / 1.0     0.95: 1 / 1 / 1.0


def test_abstain_picks_lowest_qualifying_threshold() -> None:
    fit = fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=0.75)
    assert fit == AbstainFit(threshold=0.7, precision=0.8, coverage=5 / 8, n_accepted=5)
    # 0.55 qualifies for 0.7 even though 0.6 (in between) does not
    low = fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=0.7)
    assert low is not None and low.threshold == 0.55
    assert low.precision == pytest.approx(5 / 7)
    assert low.n_accepted == 7


def test_abstain_target_exactly_met_counts() -> None:
    fit = fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=0.8)
    assert fit is not None and fit.threshold == 0.7


def test_abstain_min_coverage_changes_the_answer() -> None:
    free = fit_abstain_threshold(_ABS_CONF, _ABS_OK, 0.9)
    assert free is not None and free.threshold == 0.85 and free.coverage == pytest.approx(3 / 8)
    # coverage exactly at the limit still qualifies
    edge = fit_abstain_threshold(_ABS_CONF, _ABS_OK, 0.9, min_coverage=3 / 8)
    assert edge == free
    # a higher coverage floor makes the same target unattainable
    assert fit_abstain_threshold(_ABS_CONF, _ABS_OK, 0.9, min_coverage=0.5) is None


def test_abstain_none_when_unattainable() -> None:
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([0, 0, 0, 0], dtype=bool)
    assert fit_abstain_threshold(conf, correct, target_precision=0.5) is None


def test_abstain_zero_target_accepts_everything() -> None:
    fit = fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=0.0)
    assert fit is not None and fit.threshold == 0.5 and fit.coverage == 1.0


def test_abstain_handles_tied_confidences() -> None:
    conf = np.array([0.9, 0.9, 0.9, 0.5, 0.5])
    correct = np.array([1, 1, 0, 1, 0], dtype=bool)
    fit = fit_abstain_threshold(conf, correct, target_precision=0.6)
    # t=0.5 accepts all 5 with precision 3/5 = 0.6
    assert fit is not None and fit.threshold == 0.5 and fit.n_accepted == 5
    strict = fit_abstain_threshold(conf, correct, target_precision=0.65)
    assert strict is not None and strict.threshold == 0.9 and strict.n_accepted == 3
    assert strict.precision == pytest.approx(2 / 3)


def test_abstain_validation_errors() -> None:
    with pytest.raises(ValueError):
        fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=-0.1)
    with pytest.raises(ValueError):
        fit_abstain_threshold(_ABS_CONF, _ABS_OK, target_precision=1.1)
    with pytest.raises(ValueError):
        fit_abstain_threshold(_ABS_CONF, _ABS_OK, 0.5, min_coverage=-0.1)
    with pytest.raises(ValueError):
        fit_abstain_threshold(_ABS_CONF, _ABS_OK, 0.5, min_coverage=1.5)
    with pytest.raises(ValueError):
        fit_abstain_threshold(_ABS_CONF, _ABS_OK[:-1], 0.5)
    with pytest.raises(ValueError):
        fit_abstain_threshold(np.array([]), np.array([], dtype=bool), 0.5)

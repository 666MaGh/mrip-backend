"""Segmented calibration of semantic decisions on synthetic overconfident classifiers."""
import os
from datetime import date
from pathlib import Path as FsPath

import numpy as np
import pytest

from app.mrip.calibration import decision as dc
from app.mrip.calibration.store import CalibrationError, CalibrationStore
from app.mrip.decision.types import AbstainReason, Decision

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_calibration.sql"
LABELS = ["supplies", "competes", "unrelated"]


def synth(n, true_temperature, family, seed):
    """The classifier reports softmax(z) but the truth follows softmax(z / T): overconfident when T > 1."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2.0, (n, 3))
    reported = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    true_p = np.exp(z / true_temperature) / np.exp(z / true_temperature).sum(axis=1, keepdims=True)
    truth = [rng.choice(3, p=row) for row in true_p]
    return [dc.DecisionSample(dict(zip(LABELS, row)), LABELS[t], {"decision_family": family}) for row, t in zip(reported, truth)]


DIMS = ("decision_family",)


def mixed(seed=0):
    a, b = synth(600, 2.5, "relationship_type", seed), synth(600, 1.0, "materiality", seed + 1)
    return [s for pair in zip(a, b) for s in pair]  # interleaved, as if chronological


def test_each_segment_gets_its_own_temperature():
    fitted = dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=100, fit_as_of=date(2026, 10, 1))
    t_rel = fitted.entries["decision_family=relationship_type"].params.temperature
    t_mat = fitted.entries["decision_family=materiality"].params.temperature
    assert 2.0 < t_rel < 3.0 and 0.85 < t_mat < 1.2
    assert "" in fitted.entries and fitted.entries[""].n == 1200 and fitted.version == "unsaved"


def test_calibrated_confidence_is_lower_than_raw_for_an_overconfident_family():
    fitted = dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=100, fit_as_of=date(2026, 10, 1))
    decision = Decision("relationship_type", "supplies", False, 0.87, "laya:x", {"supplies": 0.87, "competes": 0.08, "unrelated": 0.05})
    out = dc.apply_decision_calibration(fitted, decision, {"decision_family": "relationship_type"})
    assert out.calibrated and out.segment_key == "decision_family=relationship_type" and out.raw_confidence == 0.87
    assert out.calibrated_confidence < 0.87 and sum(out.calibrated_probabilities.values()) == pytest.approx(1.0)
    assert out.answer in LABELS or out.answer is None
    same = dc.apply_decision_calibration(fitted, decision, {"decision_family": "materiality"})
    assert abs(same.calibrated_confidence - 0.87) < 0.06  # a calibrated family is left almost alone


def test_calibrated_abstain_threshold_enforces_target_precision_on_calibration_data():
    samples = mixed(3)
    fitted = dc.fit_decision_calibration(samples, dims=DIMS, min_samples=100, target_precision=0.7, fit_as_of=date(2026, 10, 1))
    params = fitted.entries["decision_family=relationship_type"].params
    assert params.abstain_threshold is not None
    kept = correct = 0
    for s in samples:
        if s.attrs["decision_family"] != "relationship_type":
            continue
        top = max(s.probabilities, key=s.probabilities.get)
        d = Decision("f", top, False, s.probabilities[top], "v", dict(s.probabilities))
        out = dc.apply_decision_calibration(fitted, d, s.attrs)
        if not out.abstained:
            kept += 1
            correct += out.answer == s.true_label
    assert kept > 20 and correct / kept >= 0.68  # precision of what is answered, near the 0.7 target
    # A target the (noisy) data cannot reach yields no threshold rather than a fake one.
    strict = dc.fit_decision_calibration(samples, dims=DIMS, min_samples=100, target_precision=0.999, fit_as_of=date(2026, 10, 1))
    assert strict.entries["decision_family=relationship_type"].params.abstain_threshold is None


def test_low_calibrated_confidence_becomes_abstain_and_raw_abstain_is_kept():
    fitted = dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=100, target_precision=0.7, fit_as_of=date(2026, 10, 1))
    flat = Decision("f", "supplies", False, 0.4, "v", {"supplies": 0.4, "competes": 0.35, "unrelated": 0.25})
    out = dc.apply_decision_calibration(fitted, flat, {"decision_family": "relationship_type"})
    assert out.abstained and out.answer is None and out.calibrated
    raw_abstain = Decision("f", None, True, 0.2, "v", {"supplies": 0.4, "competes": 0.35, "unrelated": 0.25}, AbstainReason.LOW_CONFIDENCE)
    assert dc.apply_decision_calibration(fitted, raw_abstain, {"decision_family": "materiality"}).abstained


def test_uncalibrated_passthrough_without_a_segment_or_probabilities():
    d = Decision("f", "supplies", False, 0.87, "v", {"supplies": 0.87, "competes": 0.13})
    out = dc.apply_decision_calibration(None, d, {"decision_family": "x"})
    assert not out.calibrated and out.calibrated_confidence == 0.87 and out.answer == "supplies" and out.calibration_version == "uncalibrated"
    fitted = dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=100, fit_as_of=date(2026, 10, 1))
    no_probs = Decision("f", "supplies", False, 0.9, "v")
    assert not dc.apply_decision_calibration(fitted, no_probs, {"decision_family": "relationship_type"}).calibrated


def test_out_of_sample_evaluation_improves_calibration_metrics():
    ev = dc.evaluate_decision_calibration(mixed(5), dims=DIMS, min_samples=100, fit_as_of=date(2026, 10, 1))
    assert ev.n_train == 720 and ev.n_test == 480
    assert ev.ece_after < ev.ece_before and ev.log_loss_after < ev.log_loss_before and ev.brier_after <= ev.brier_before


def test_validation_errors():
    with pytest.raises(CalibrationError, match="no segment"):
        dc.fit_decision_calibration(synth(30, 2.0, "f", 1), dims=DIMS, min_samples=50, fit_as_of=date(2026, 1, 1))
    with pytest.raises(ValueError):
        dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=5, fit_as_of=date(2026, 1, 1))
    with pytest.raises(ValueError):
        dc.fit_decision_calibration(mixed(), dims=DIMS, target_precision=1.5, fit_as_of=date(2026, 1, 1))
    with pytest.raises(ValueError):
        dc.evaluate_decision_calibration(mixed(), dims=DIMS, train_fraction=0.05, fit_as_of=date(2026, 1, 1))


@DB
def test_decision_calibration_round_trips_through_the_database():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_calibration_entries, mrip_calibration_sets RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    store = CalibrationStore(db.get_db_connection)
    assert dc.load_decision_calibration(store) is None
    fitted = dc.save_decision_calibration(
        store, dc.fit_decision_calibration(mixed(), dims=DIMS, min_samples=100, fit_as_of=date(2026, 10, 1))
    )
    loaded = dc.load_decision_calibration(store)
    assert loaded.version == fitted.version == "decision_temperature#1" and loaded.target_precision == 0.9
    key = "decision_family=relationship_type"
    assert loaded.entries[key].params == fitted.entries[key].params
    d = Decision("f", "supplies", False, 0.87, "v", {"supplies": 0.87, "competes": 0.08, "unrelated": 0.05})
    a = dc.apply_decision_calibration(fitted, d, {"decision_family": "relationship_type"})
    b = dc.apply_decision_calibration(loaded, d, {"decision_family": "relationship_type"})
    assert a.calibrated_confidence == pytest.approx(b.calibrated_confidence) and b.calibration_version == "decision_temperature#1"

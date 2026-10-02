"""Segmented quantile recalibration of forecasts: math, leakage control, persistence."""
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path as FsPath

import numpy as np
import pytest
from scipy import stats

from app.mrip.calibration import forecast as cal
from app.mrip.calibration.quantiles import CalibrationInput, QuantileShift, apply_quantile_shift, fit_quantile_shift
from app.mrip.calibration.segments import SegmentEntry, fit_segments, resolve_segment, segment_key
from app.mrip.calibration.store import CalibrationError, CalibrationKind, CalibrationStore, StoredEntry
from app.mrip.forecast.types import QUANTILES, ForecastResult
from app.mrip.outcomes.types import HorizonKind, Outcome, OutcomeStatus, Prediction, PredictionType

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_calibration.sql"
SIGMA = 0.04
T0 = datetime(2020, 1, 31, 23, tzinfo=timezone.utc)


def normal_quantiles(mu=0.0, sigma=SIGMA):
    return {q: float(mu + stats.norm.ppf(q) * sigma) for q in QUANTILES}


def make_pairs(n, bias=0.02, seed=0, kind=HorizonKind.M1, model="naive-v1", start=0, spacing=30, horizon_days=30):
    """Forecasts centred at 0 while outcomes are centred at ``bias`` (a systematically wrong forecaster)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        made = T0 + timedelta(days=spacing * (start + i))
        pred = Prediction(
            start + i + 1, PredictionType.FORECAST, "SPY", made, kind, model,
            {"quantiles": {str(q): v for q, v in normal_quantiles().items()}},
        )
        out = Outcome(
            start + i + 1, pred.id, OutcomeStatus.RESOLVED, made, "outcome-v0",
            window_start=made.date(), window_end=(made + timedelta(days=horizon_days)).date(),
            actual_return=float(rng.normal(bias, SIGMA)),
        )
        pairs.append((pred, out))
    return pairs


def rows(pairs):
    return [CalibrationInput({float(q): v for q, v in p.payload["quantiles"].items()}, o.actual_return) for p, o in pairs]


def test_shift_recovers_a_constant_bias_at_every_level_and_keeps_order():
    fit = fit_quantile_shift(rows(make_pairs(3000, bias=0.02)))
    assert fit.n == 3000
    assert all(abs(fit.shifts[q] - 0.02) < 0.004 for q in QUANTILES)
    out = apply_quantile_shift(normal_quantiles(), fit)
    vals = [out[q] for q in QUANTILES]
    assert vals == sorted(vals) and out[0.5] == pytest.approx(0.02, abs=0.004)


def test_shift_fixes_hit_rates_out_of_sample_and_widens_when_too_narrow():
    train, test = make_pairs(2000, bias=0.0, seed=1), make_pairs(2000, bias=0.0, seed=2)
    narrow = {str(q): v * 0.5 for q, v in normal_quantiles().items()}  # forecaster too confident
    narrow_pairs = lambda ps: [(Prediction(p.id, p.prediction_type, p.subject, p.made_at, p.horizon_kind, p.model_version, {"quantiles": narrow}), o) for p, o in ps]  # noqa: E731
    fit = fit_quantile_shift(rows(narrow_pairs(train)))
    for p, o in narrow_pairs(test)[:1]:
        pass
    hits = {q: np.mean([o.actual_return <= apply_quantile_shift({float(k): v for k, v in p.payload["quantiles"].items()}, fit)[q]
                        for p, o in narrow_pairs(test)]) for q in QUANTILES}
    raw = {q: np.mean([o.actual_return <= p.payload["quantiles"][str(q)] for p, o in narrow_pairs(test)]) for q in QUANTILES}
    assert abs(raw[0.1] - 0.1) > 0.04  # the raw forecast is miscalibrated...
    assert all(abs(hits[q] - q) < 0.035 for q in QUANTILES)  # ...and the recalibrated one is not


def test_fit_needs_enough_samples():
    with pytest.raises(ValueError, match="at least 30"):
        fit_quantile_shift(rows(make_pairs(10)))


def test_segment_keys_fit_levels_and_backoff():
    attrs = {"horizon_kind": "M1", "model_version": "a", "vix": "LOW"}
    assert segment_key(attrs, ()) == "" and segment_key(attrs, ("horizon_kind", "model_version")) == "horizon_kind=M1|model_version=a"
    with pytest.raises(KeyError):
        segment_key(attrs, ("nope",))
    items = [({"horizon_kind": "M1", "model_version": "a"}, 1.0)] * 40 + [({"horizon_kind": "M1", "model_version": "b"}, 3.0)] * 5
    dims = ("horizon_kind", "model_version")
    entries = fit_segments(items, dims, 30, lambda xs: sum(xs) / len(xs))
    assert set(entries) == {"", "horizon_kind=M1", "horizon_kind=M1|model_version=a"} and entries[""].n == 45
    assert resolve_segment(entries, dims, {"horizon_kind": "M1", "model_version": "a"}).level == 2
    assert resolve_segment(entries, dims, {"horizon_kind": "M1", "model_version": "b"}).key == "horizon_kind=M1"  # too few samples -> back off
    assert resolve_segment(entries, dims, {"horizon_kind": "M3", "model_version": "a"}).key == ""  # unseen horizon -> global
    assert resolve_segment(entries, dims, {"horizon_kind": "M1"}).key == "horizon_kind=M1"  # missing attr -> broader level
    assert resolve_segment({}, dims, {"horizon_kind": "M1", "model_version": "a"}) is None
    with pytest.raises(ValueError):
        fit_segments(items, dims, 0, len)


def test_fit_uses_only_outcomes_known_at_as_of_and_ignores_unresolved():
    pairs = make_pairs(80, bias=0.02)
    unresolved = Outcome(999, 1, OutcomeStatus.UNRESOLVABLE, T0, "v", note="x")
    cutoff = pairs[49][1].window_end  # known outcomes: the first 50
    fitted = cal.fit_forecast_calibration(pairs + [(pairs[0][0], unresolved)], as_of=cutoff)
    assert fitted.n_total == 50 and fitted.fit_as_of == cutoff and fitted.version == "unsaved"
    assert fitted.entries["horizon_kind=M1|model_version=naive-v1"].n == 50
    with pytest.raises(CalibrationError, match="no segment"):
        cal.fit_forecast_calibration(pairs[:10])
    with pytest.raises(CalibrationError, match="not a forecast"):
        wrong = Prediction(1, PredictionType.OPTIONS_EVENT, "SPY", T0, HorizonKind.W1, "m", {})
        cal.fit_forecast_calibration([(wrong, pairs[0][1])])


def result(horizon=21, model="naive-v1"):
    return ForecastResult("naive", model, "SPY", date(2026, 9, 30), horizon, 700.0, normal_quantiles(), 60)


def test_apply_uses_the_most_specific_segment_and_passes_through_otherwise():
    fitted = cal.fit_forecast_calibration(make_pairs(200, bias=0.02))
    out = cal.apply_forecast_calibration(fitted, result())
    assert out.calibrated and out.segment_key == "horizon_kind=M1|model_version=naive-v1" and out.n_calibration == 200
    assert out.return_quantiles[0.5] == pytest.approx(0.02, abs=0.01) and out.raw.return_quantiles[0.5] == 0.0
    s = out.scenarios
    assert s.bear < s.base < s.bull and out.price_quantiles()[0.5] == pytest.approx(700 * (1 + out.return_quantiles[0.5]))
    other_model = cal.apply_forecast_calibration(fitted, result(model="other-v1"))
    assert other_model.segment_key == "horizon_kind=M1"  # backs off to the horizon level
    other_horizon = cal.apply_forecast_calibration(fitted, result(horizon=63))
    assert other_horizon.segment_key == "" and other_horizon.calibrated  # global level
    passthrough = cal.apply_forecast_calibration(None, result())
    assert not passthrough.calibrated and passthrough.calibration_version == "uncalibrated"
    assert passthrough.return_quantiles == dict(result().return_quantiles)


def test_out_of_sample_evaluation_improves_a_biased_forecaster_and_purges_overlap(monkeypatch):
    pairs = make_pairs(300, bias=0.025, seed=5)
    ev = cal.evaluate_forecast_calibration(pairs, train_fraction=0.6)
    assert ev.n_train == 180 and ev.n_test == 120 and ev.n_calibrated_test == 120
    assert ev.pinball_after < ev.pinball_before and ev.improves()
    assert abs(ev.hit_after[0.5] - 0.5) < abs(ev.hit_before[0.5] - 0.5)
    assert abs(ev.coverage_p10_p90_after - 0.8) < abs(ev.coverage_p10_p90_before - 0.8)
    seen = {}
    real = cal.fit_forecast_calibration
    monkeypatch.setattr(cal, "fit_forecast_calibration", lambda *a, **k: seen.update(as_of=k["as_of"]) or real(*a, **k))
    cal.evaluate_forecast_calibration(pairs, train_fraction=0.6)
    assert seen["as_of"] == pairs[180][0].made_at.date()  # the fit only sees outcomes known before the first test forecast
    with pytest.raises(CalibrationError, match="not enough"):
        cal.evaluate_forecast_calibration(make_pairs(40))
    with pytest.raises(ValueError):
        cal.evaluate_forecast_calibration(pairs, train_fraction=0.99)


# -- persistence ---------------------------------------------------------------

def test_kind_list_in_sql_matches_python():
    sql = MIGRATION.read_text(encoding="utf-8")
    m = re.search(r"kind VARCHAR\(\d+\) NOT NULL CHECK \(kind IN \((.*?)\)\)", sql, re.S)
    assert set(re.findall(r"'([^']+)'", m.group(1))) == {k.value for k in CalibrationKind}


@pytest.fixture()
def store():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_calibration_entries, mrip_calibration_sets RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    return CalibrationStore(db.get_db_connection)


@DB
def test_calibration_sets_round_trip_and_newer_sets_do_not_overwrite_older(store):
    assert cal.load_forecast_calibration(store) is None
    first = cal.save_forecast_calibration(store, cal.fit_forecast_calibration(make_pairs(120, bias=0.02)), {"note": "v1"})
    assert first.version == "forecast_quantile_shift#1"
    loaded = cal.load_forecast_calibration(store)
    assert loaded.version == first.version and loaded.dims == first.dims and loaded.metrics == {"note": "v1"}
    key = "horizon_kind=M1|model_version=naive-v1"
    assert loaded.entries[key].params.shifts == pytest.approx(first.entries[key].params.shifts)
    a, b = cal.apply_forecast_calibration(first, result()), cal.apply_forecast_calibration(loaded, result())
    assert a.return_quantiles == pytest.approx(b.return_quantiles) and b.calibration_version == "forecast_quantile_shift#1"

    second = cal.save_forecast_calibration(store, cal.fit_forecast_calibration(make_pairs(120, bias=-0.03, seed=9)))
    assert second.version == "forecast_quantile_shift#2" and cal.load_forecast_calibration(store).version == second.version
    with store._connect() as conn:  # the first set is still there, untouched
        cur = conn.cursor()
        cur.execute("SELECT count(*) AS n FROM mrip_calibration_sets WHERE kind = 'forecast_quantile_shift'")
        assert cur.fetchone()["n"] == 2
        cur.close()
    with pytest.raises(CalibrationError, match="empty"):
        store.save(CalibrationKind.FORECAST_QUANTILE_SHIFT, fit_as_of=date(2026, 1, 1), dims=(), min_samples=1,
                   n_total=0, policy_version="v", entries=[])


def test_promotion_gate_rejects_a_calibration_that_does_not_help():
    already_calibrated = make_pairs(300, bias=0.0, seed=7)
    ev = cal.evaluate_forecast_calibration(already_calibrated, train_fraction=0.6)
    assert not ev.improves(min_relative_gain=0.05)  # nothing to fix: no meaningful gain, so it must not be promoted
    base = cal.CalibrationEvaluation(100, 60, 60, 0.0100, 0.0095, 0.80, 0.80, 0.5, 0.5, {}, {})
    assert base.improves(min_relative_gain=0.04) and not base.improves(min_relative_gain=0.06)
    worse_cov = cal.CalibrationEvaluation(100, 60, 60, 0.0100, 0.0090, 0.80, 0.70, 0.5, 0.5, {}, {})
    assert not worse_cov.improves()  # better loss but the 80 % interval drifted from 0.80 by more than the slack

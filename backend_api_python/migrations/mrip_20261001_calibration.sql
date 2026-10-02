-- MRIP calibration sets (work 010). Idempotent; applied as a bootstrap component by
-- app/utils/db.py. Kind list must match app/mrip/calibration/store.py (enforced by tests).
--
-- A calibration set is one immutable fit: parameters per segment, the data window it was
-- fitted on (fit_as_of: only outcomes known by then) and before/after metrics. Newer fits
-- never overwrite older ones, so any analysis can name the exact calibration_version it used.

CREATE TABLE IF NOT EXISTS mrip_calibration_sets (
    id BIGSERIAL PRIMARY KEY,
    kind VARCHAR(32) NOT NULL CHECK (kind IN ('forecast_quantile_shift', 'decision_temperature')),
    fit_as_of DATE NOT NULL,
    dims JSONB NOT NULL,
    min_samples INTEGER NOT NULL CHECK (min_samples >= 1),
    n_total INTEGER NOT NULL CHECK (n_total >= 0),
    policy_version VARCHAR(80) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mrip_calibration_entries (
    id BIGSERIAL PRIMARY KEY,
    set_id BIGINT NOT NULL REFERENCES mrip_calibration_sets(id),
    segment_key TEXT NOT NULL,
    level INTEGER NOT NULL CHECK (level >= 0),
    n INTEGER NOT NULL CHECK (n >= 1),
    params JSONB NOT NULL,
    UNIQUE (set_id, segment_key)
);

CREATE INDEX IF NOT EXISTS idx_mrip_calibration_sets_kind ON mrip_calibration_sets (kind, id DESC);

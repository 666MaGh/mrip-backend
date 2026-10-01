-- MRIP Outcome Resolver (work 009). Idempotent; applied as a bootstrap component by
-- app/utils/db.py. Value lists must match app/mrip/outcomes/types.py (enforced by tests).
--
-- A prediction is logged once, immutably; its outcome is resolved later from OBSERVED
-- market data and stored separately (one outcome per prediction). Labels for calibration
-- and learning are built only from resolved outcomes, never from the prediction itself.

CREATE TABLE IF NOT EXISTS mrip_predictions (
    id BIGSERIAL PRIMARY KEY,
    prediction_type VARCHAR(24) NOT NULL CHECK (prediction_type IN ('forecast', 'options_event')),
    subject VARCHAR(40) NOT NULL,
    benchmark VARCHAR(40),
    made_at TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION CHECK (entry_price IS NULL OR entry_price > 0),
    horizon_kind VARCHAR(16) NOT NULL CHECK (horizon_kind IN (
        'W1', 'M1', 'M3', 'M6', 'M12', 'EOD', 'NEXT_SESSION', 'EXPIRY'
    )),
    expiry_date DATE,
    model_version VARCHAR(300) NOT NULL,
    lineage JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (horizon_kind <> 'EXPIRY' OR expiry_date IS NOT NULL),
    CHECK (horizon_kind NOT IN ('EOD', 'NEXT_SESSION') OR entry_price IS NOT NULL),
    UNIQUE (prediction_type, subject, made_at, horizon_kind, model_version)
);

CREATE INDEX IF NOT EXISTS idx_mrip_predictions_subject ON mrip_predictions (subject, made_at);

CREATE TABLE IF NOT EXISTS mrip_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL UNIQUE REFERENCES mrip_predictions(id),
    status VARCHAR(16) NOT NULL CHECK (status IN ('resolved', 'unresolvable')),
    resolved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_start DATE,
    window_end DATE,
    actual_return DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    realized_volatility DOUBLE PRECISION,
    max_adverse_excursion DOUBLE PRECISION,
    max_favorable_excursion DOUBLE PRECISION,
    measures JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolver_version VARCHAR(80) NOT NULL,
    note TEXT,
    CHECK (status <> 'resolved' OR (actual_return IS NOT NULL AND window_start IS NOT NULL AND window_end IS NOT NULL))
);

-- MRIP point-in-time options snapshots (work 007). Idempotent; applied as a bootstrap
-- component by app/utils/db.py.
--
-- Snapshots are immutable records of a chain exactly as observed at snapshot_timestamp.
-- A historical chain must NEVER be reconstructed from current open interest; backtests
-- of GEX, gamma flip and wall effects read these rows (latest at or before a given time).
-- The contract list is stored as gzip-compressed JSON with its SHA-256 for integrity.

CREATE TABLE IF NOT EXISTS mrip_options_snapshots (
    id BIGSERIAL PRIMARY KEY,
    underlying VARCHAR(40) NOT NULL,
    snapshot_timestamp TIMESTAMPTZ NOT NULL,
    underlying_price DOUBLE PRECISION,
    underlying_timestamp TIMESTAMP,
    oi_effective_date DATE,
    n_contracts INTEGER NOT NULL CHECK (n_contracts >= 0),
    provider VARCHAR(80) NOT NULL,
    gateway VARCHAR(80) NOT NULL,
    endpoint VARCHAR(160) NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    latency VARCHAR(16) NOT NULL,
    gateway_version VARCHAR(80),
    payload BYTEA NOT NULL,
    payload_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (underlying, snapshot_timestamp, provider)
);

CREATE INDEX IF NOT EXISTS idx_mrip_options_snapshots_lookup
    ON mrip_options_snapshots (underlying, snapshot_timestamp DESC);

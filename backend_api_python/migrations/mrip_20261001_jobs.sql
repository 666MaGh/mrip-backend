-- MRIP scheduled job run tracking (work 014).
-- Idempotent; applied as a bootstrap component by app/utils/db.py.
--
-- mrip_job_runs tracks the execution history of scheduled background jobs, including
-- their status, duration, report data, and error messages for debugging and monitoring.

CREATE TABLE IF NOT EXISTS mrip_job_runs (
    id BIGSERIAL PRIMARY KEY,
    job VARCHAR(80) NOT NULL,
    universe VARCHAR(40) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status VARCHAR(16) NOT NULL CHECK (status IN ('running','completed','halted','skipped','failed')),
    report JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_mrip_job_runs_job_universe_id
    ON mrip_job_runs (job, universe, id DESC);

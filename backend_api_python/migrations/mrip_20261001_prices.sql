-- MRIP price bar storage and incremental sync tracking (work 013).
-- Idempotent; applied as a bootstrap component by app/utils/db.py.
--
-- mrip_price_bars is the authoritative store of daily OHLCV data for analysis.
-- mrip_price_sync tracks the last successful fetch and consecutive failures per (provider, symbol).
-- Providers may revise their history; ingestion uses ON CONFLICT ... DO UPDATE to overwrite.

CREATE TABLE IF NOT EXISTS mrip_price_bars (
    provider VARCHAR(40) NOT NULL,
    symbol VARCHAR(60) NOT NULL,
    bar_date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION NOT NULL CHECK (close > 0),
    volume DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, symbol, bar_date)
);

CREATE INDEX IF NOT EXISTS idx_mrip_price_bars_symbol_date
    ON mrip_price_bars (symbol, bar_date DESC);

CREATE INDEX IF NOT EXISTS idx_mrip_price_bars_ingested
    ON mrip_price_bars (provider, symbol, ingested_at DESC);


CREATE TABLE IF NOT EXISTS mrip_price_sync (
    provider VARCHAR(40) NOT NULL,
    symbol VARCHAR(60) NOT NULL,
    last_bar_date DATE,
    last_success_at TIMESTAMPTZ,
    last_attempt_at TIMESTAMPTZ NOT NULL,
    last_error TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, symbol)
);

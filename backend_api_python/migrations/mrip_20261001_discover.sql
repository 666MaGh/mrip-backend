CREATE TABLE IF NOT EXISTS mrip_discover_items (
    id BIGSERIAL PRIMARY KEY,
    as_of DATE NOT NULL,
    kind VARCHAR(40) NOT NULL CHECK (kind IN ('RELATIONSHIP_DIVERGENCE','DELAYED_REACTION','COT_CROWDING','COT_DIVERGENCE','REGIME_SHIFT','TERM_BACKWARDATION','PEER_DIVERGENCE','GAMMA_REGIME_CHANGE','NEAR_GAMMA_FLIP','WALL_PROXIMITY','UNUSUAL_OPTIONS_ACTIVITY','HIGH_ZERO_DTE')),
    subject VARCHAR(160) NOT NULL,
    headline TEXT NOT NULL,
    magnitude DOUBLE PRECISION NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    components JSONB NOT NULL DEFAULT '{}'::jsonb,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_quality JSONB NOT NULL DEFAULT '{}'::jsonb,
    modeled BOOLEAN NOT NULL DEFAULT FALSE,
    policy_version VARCHAR(60) NOT NULL,
    status VARCHAR(12) NOT NULL DEFAULT 'open' CHECK (status IN ('open','dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (as_of, kind, subject)
);
CREATE INDEX IF NOT EXISTS mrip_discover_items_as_of_score_idx ON mrip_discover_items (as_of, score DESC);

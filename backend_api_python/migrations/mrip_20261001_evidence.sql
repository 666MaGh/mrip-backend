-- MRIP Evidence Engine (work 005). Idempotent; applied as a bootstrap component
-- by app/utils/db.py. Value lists must match app/mrip/evidence/types.py and the
-- relation list in mrip_20261001_relationships.sql (enforced by tests).
--
-- Evidence attaches to the logical relationship (src, dst, relation_type), not to
-- an edge row: edge rows are replaced when a status changes. Evidence is
-- append-only; a wrong item is retracted, never deleted or edited. Point-in-time
-- reads use available_at (when the information became public), ingested_at
-- (when we learned it) and retracted_at.

CREATE TABLE IF NOT EXISTS mrip_rel_evidence (
    id BIGSERIAL PRIMARY KEY,
    src_id BIGINT NOT NULL REFERENCES mrip_rel_nodes(id),
    dst_id BIGINT NOT NULL REFERENCES mrip_rel_nodes(id),
    relation_type VARCHAR(32) NOT NULL CHECK (relation_type IN (
        'SUPPLIES', 'CONSUMES', 'DEPENDS_ON', 'CUSTOMER_OF', 'SUPPLIER_OF', 'COMPETES_WITH',
        'BENEFITS_FROM', 'HURT_BY', 'EXPOSED_TO', 'LEADS', 'LAGS', 'CORRELATED_WITH',
        'SUBSTITUTE_FOR', 'FINANCED_BY'
    )),
    stance VARCHAR(16) NOT NULL CHECK (stance IN ('support', 'contradict', 'neutral')),
    source_type VARCHAR(32) NOT NULL CHECK (source_type IN (
        'ANNUAL_REPORT', 'QUARTERLY_REPORT', 'EARNINGS_TRANSCRIPT', 'INVESTOR_PRESENTATION',
        'REGULATORY_FILING', 'COMPANY_STATEMENT', 'MACRO_DATA', 'COMMODITY_DATA', 'MARKET_DATA',
        'COT', 'OPTIONS_DATA', 'REGULATORY_DATA', 'NEWS', 'STATISTICAL_TEST'
    )),
    source_uri TEXT NOT NULL CHECK (length(btrim(source_uri)) > 0),
    source_title TEXT,
    publisher VARCHAR(200),
    excerpt TEXT,
    excerpt_hash CHAR(64) NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assessed_by VARCHAR(120) NOT NULL CHECK (length(btrim(assessed_by)) > 0),
    model_version VARCHAR(160),
    assessor_confidence DOUBLE PRECISION CHECK (assessor_confidence IS NULL OR assessor_confidence BETWEEN 0 AND 1),
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    retracted_at TIMESTAMPTZ,
    retract_reason TEXT,
    CHECK (src_id <> dst_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_mrip_rel_evidence_active
    ON mrip_rel_evidence (src_id, dst_id, relation_type, source_type, source_uri, excerpt_hash)
    WHERE retracted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_mrip_rel_evidence_rel
    ON mrip_rel_evidence (src_id, dst_id, relation_type, available_at);

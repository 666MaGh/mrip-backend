-- MRIP Relationship Engine (work 004). Idempotent; applied as a bootstrap
-- component by app/utils/db.py. Node/relation/status lists must match
-- app/mrip/relationships/types.py (enforced by tests).
--
-- Graph versioning: every mutation bumps mrip_rel_graph_state.version (a single
-- row, so concurrent writers serialize). Edges are never updated or deleted:
-- they carry created_version / retired_version, so the graph as of any version
-- can be reproduced (data lineage: relationship_graph_version).

CREATE TABLE IF NOT EXISTS mrip_rel_graph_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO mrip_rel_graph_state (id, version) VALUES (1, 0) ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS mrip_rel_nodes (
    id BIGSERIAL PRIMARY KEY,
    node_type VARCHAR(32) NOT NULL CHECK (node_type IN (
        'THEME', 'SECTOR', 'INDUSTRY', 'COMPANY', 'SECURITY', 'COMMODITY', 'CURRENCY',
        'RATE', 'INDEX', 'MACRO_SERIES', 'ECONOMIC_DRIVER', 'TECHNOLOGY', 'PRODUCT', 'GEOGRAPHY'
    )),
    node_key VARCHAR(200) NOT NULL,
    name VARCHAR(300) NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (node_type, node_key)
);

CREATE TABLE IF NOT EXISTS mrip_rel_edges (
    id BIGSERIAL PRIMARY KEY,
    src_id BIGINT NOT NULL REFERENCES mrip_rel_nodes(id),
    dst_id BIGINT NOT NULL REFERENCES mrip_rel_nodes(id),
    relation_type VARCHAR(32) NOT NULL CHECK (relation_type IN (
        'SUPPLIES', 'CONSUMES', 'DEPENDS_ON', 'CUSTOMER_OF', 'SUPPLIER_OF', 'COMPETES_WITH',
        'BENEFITS_FROM', 'HURT_BY', 'EXPOSED_TO', 'LEADS', 'LAGS', 'CORRELATED_WITH',
        'SUBSTITUTE_FOR', 'FINANCED_BY'
    )),
    -- A hypothesis is not a strong relationship until statistics validate it.
    status VARCHAR(16) NOT NULL DEFAULT 'hypothesis' CHECK (status IN ('hypothesis', 'validated', 'rejected')),
    source VARCHAR(80) NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_version BIGINT NOT NULL,
    retired_version BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (src_id <> dst_id),
    CHECK (retired_version IS NULL OR retired_version > created_version)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_mrip_rel_edges_active
    ON mrip_rel_edges (src_id, dst_id, relation_type) WHERE retired_version IS NULL;
CREATE INDEX IF NOT EXISTS idx_mrip_rel_edges_src ON mrip_rel_edges (src_id, created_version);
CREATE INDEX IF NOT EXISTS idx_mrip_rel_edges_dst ON mrip_rel_edges (dst_id, created_version);

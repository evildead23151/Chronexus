-- ============================================================
-- Sentinel: PostgreSQL + PostGIS + pgvector Schema
-- Refined: RawEvent → DerivedEvent → Inference separation
-- Immutable raw layer, versioned graph, embedding ANN
-- ============================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for embedding ANN

-- ═══════════════════════════════════════════════════════════
-- ENUM TYPES
-- ═══════════════════════════════════════════════════════════

CREATE TYPE raw_event_type AS ENUM (
    'frame', 'lpr_capture', 'phone_ping', 'call_log',
    'transaction_log', 'sensor_reading', 'audio_capture',
    'manual_entry'
);

CREATE TYPE derived_event_type AS ENUM (
    'detection', 'track_update', 'face_embedding',
    'vehicle_embedding', 'plate_read', 'audio_transcript',
    'pose_estimate'
);

CREATE TYPE entity_type AS ENUM (
    'person', 'vehicle', 'camera', 'phone',
    'place', 'transaction'
);

CREATE TYPE resolution_method AS ENUM (
    'exact_match', 'near_match', 'embedding_similarity',
    'temporal_consistency', 'graph_structural', 'manual'
);

CREATE TYPE audit_action AS ENUM (
    'create', 'read', 'update', 'delete',
    'inference', 'export', 'login', 'query',
    'entity_merge', 'entity_split', 'model_load'
);

-- ═══════════════════════════════════════════════════════════
-- LAYER 0: INFRASTRUCTURE (cameras, sources)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    source_url TEXT,
    location GEOMETRY(Point, 4326),
    camera_type VARCHAR(50) DEFAULT 'fixed',
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cameras_location ON cameras USING GIST (location);

-- Model registry — tracks every model version used
CREATE TABLE model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'detector', 'tracker', 'face_embed', 'lpr', 'vehicle_embed'
    model_hash VARCHAR(128),          -- SHA256 of model weights
    config JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    license VARCHAR(50),               -- 'AGPL-3.0', 'MIT', 'Apache-2.0'
    is_active BOOLEAN DEFAULT true,
    loaded_at TIMESTAMPTZ DEFAULT NOW(),
    retired_at TIMESTAMPTZ,
    UNIQUE(model_name, model_version)
);

-- ═══════════════════════════════════════════════════════════
-- LAYER 1: RAW EVENTS (immutable, exactly what sensor produced)
-- Never overwritten. Chain-of-custody starts here.
-- ═══════════════════════════════════════════════════════════

CREATE TABLE raw_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type raw_event_type NOT NULL,
    source_id VARCHAR(255) NOT NULL,          -- camera id, sensor id, etc.
    timestamp TIMESTAMPTZ NOT NULL,
    location GEOMETRY(Point, 4326),
    location_accuracy_m REAL,
    raw_payload JSONB NOT NULL,               -- exact sensor output, untouched
    file_path TEXT,                            -- path to raw frame/audio file
    file_hash VARCHAR(128),                   -- SHA256 of raw file
    provenance_hash VARCHAR(128) NOT NULL,    -- HMAC signature for integrity
    provenance_signer VARCHAR(100),
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    -- IMMUTABLE: no updated_at column. Raw events are append-only.
    CONSTRAINT raw_events_no_update CHECK (true)  -- enforced at app level
);

CREATE INDEX idx_raw_events_type ON raw_events (type);
CREATE INDEX idx_raw_events_source ON raw_events (source_id);
CREATE INDEX idx_raw_events_timestamp ON raw_events (timestamp);
CREATE INDEX idx_raw_events_location ON raw_events USING GIST (location);
CREATE INDEX idx_raw_events_hash ON raw_events (file_hash);

-- Prevent UPDATE/DELETE on raw_events (immutability enforcement)
CREATE OR REPLACE FUNCTION prevent_raw_event_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'raw_events table is immutable. Updates and deletes are prohibited.';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_raw_events_no_update
    BEFORE UPDATE OR DELETE ON raw_events
    FOR EACH ROW
    EXECUTE FUNCTION prevent_raw_event_mutation();

-- ═══════════════════════════════════════════════════════════
-- LAYER 2: DERIVED EVENTS (model outputs, linked to raw source)
-- Each derived event traces back to raw event + model version
-- ═══════════════════════════════════════════════════════════

CREATE TABLE derived_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_event_id UUID NOT NULL REFERENCES raw_events(id),
    type derived_event_type NOT NULL,
    model_id UUID NOT NULL REFERENCES model_registry(id),
    model_version VARCHAR(50) NOT NULL,

    -- Detection fields
    class_name VARCHAR(50),
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    track_id INTEGER,

    -- Embedding (pgvector for ANN search)
    embedding vector(512),  -- face/vehicle embedding vector

    -- OCR / LPR
    ocr_text TEXT,
    ocr_confidence REAL,

    -- Full output
    output_payload JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_derived_raw ON derived_events (raw_event_id);
CREATE INDEX idx_derived_type ON derived_events (type);
CREATE INDEX idx_derived_model ON derived_events (model_id);
CREATE INDEX idx_derived_class ON derived_events (class_name);
CREATE INDEX idx_derived_confidence ON derived_events (confidence);
CREATE INDEX idx_derived_track ON derived_events (track_id);

-- pgvector ANN index for embedding similarity search
CREATE INDEX idx_derived_embedding ON derived_events
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ═══════════════════════════════════════════════════════════
-- LAYER 3: ENTITIES (resolved identities)
-- Only resolved entities get promoted to Neo4j graph
-- ═══════════════════════════════════════════════════════════

CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type entity_type NOT NULL,
    label VARCHAR(255),
    canonical_embedding vector(512),    -- cluster centroid embedding
    attributes JSONB DEFAULT '{}',
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    sighting_count INTEGER DEFAULT 0,
    is_resolved BOOLEAN DEFAULT false,  -- only resolved entities go to Neo4j
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_entities_type ON entities (type);
CREATE INDEX idx_entities_resolved ON entities (is_resolved);
CREATE INDEX idx_entities_label ON entities USING gin (label gin_trgm_ops);
CREATE INDEX idx_entities_embedding ON entities
    USING ivfflat (canonical_embedding vector_cosine_ops) WITH (lists = 50);

-- Entity resolution log — tracks how entities were merged/linked
CREATE TABLE entity_resolutions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES entities(id),
    method resolution_method NOT NULL,
    derived_event_ids UUID[] NOT NULL,     -- source evidence
    confidence REAL NOT NULL,
    details JSONB DEFAULT '{}',            -- edit distance, cosine sim, etc.
    resolved_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_resolutions_entity ON entity_resolutions (entity_id);
CREATE INDEX idx_resolutions_method ON entity_resolutions (method);

-- ═══════════════════════════════════════════════════════════
-- LAYER 4: LICENSE PLATES (with entity resolution)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE license_plates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_text VARCHAR(20) NOT NULL,
    normalized_text VARCHAR(20) NOT NULL,
    vehicle_entity_id UUID REFERENCES entities(id),
    derived_event_id UUID REFERENCES derived_events(id),
    confidence REAL NOT NULL,
    edit_distance_from_canonical INTEGER DEFAULT 0,  -- for near-match resolution
    region VARCHAR(10),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_plates_text ON license_plates (normalized_text);
CREATE INDEX idx_plates_vehicle ON license_plates (vehicle_entity_id);
CREATE INDEX idx_plates_confidence ON license_plates (confidence DESC);

-- ═══════════════════════════════════════════════════════════
-- LAYER 5: HYPOTHESES & EVIDENCE
-- ═══════════════════════════════════════════════════════════

CREATE TABLE hypotheses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    analyst_id VARCHAR(100),
    confidence_score REAL DEFAULT 0.0,
    status VARCHAR(20) DEFAULT 'draft',
    raw_event_ids UUID[] DEFAULT '{}',
    derived_event_ids UUID[] DEFAULT '{}',
    entity_ids UUID[] DEFAULT '{}',
    scoring_breakdown JSONB DEFAULT '{}',
    constraints JSONB DEFAULT '{}',
    reasoning_stage VARCHAR(20) DEFAULT 'deterministic',  -- deterministic → probabilistic → graph
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hypotheses_status ON hypotheses (status);
CREATE INDEX idx_hypotheses_confidence ON hypotheses (confidence_score DESC);

CREATE TABLE evidence_packages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hypothesis_id UUID REFERENCES hypotheses(id),
    file_path TEXT NOT NULL,
    file_hash VARCHAR(128) NOT NULL,
    signature TEXT,                         -- HMAC signature of package
    signer VARCHAR(255),
    raw_event_ids UUID[] DEFAULT '{}',
    derived_event_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════
-- AUDIT LOG (append-only, immutable)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    action audit_action NOT NULL,
    actor VARCHAR(100) NOT NULL DEFAULT 'system',
    resource_type VARCHAR(50),
    resource_id UUID,
    model_id UUID REFERENCES model_registry(id),   -- which model produced this
    model_version VARCHAR(50),
    details JSONB DEFAULT '{}',
    ip_address INET,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Immutability trigger
CREATE OR REPLACE FUNCTION prevent_audit_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_log table is immutable.';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_audit_no_mutation
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_mutation();

CREATE INDEX idx_audit_timestamp ON audit_log (timestamp);
CREATE INDEX idx_audit_actor ON audit_log (actor);
CREATE INDEX idx_audit_resource ON audit_log (resource_type, resource_id);
CREATE INDEX idx_audit_model ON audit_log (model_id);

-- ═══════════════════════════════════════════════════════════
-- FUNCTIONS
-- ═══════════════════════════════════════════════════════════

-- Find raw events within radius and time window
CREATE OR REPLACE FUNCTION find_raw_events_near(
    p_lat DOUBLE PRECISION,
    p_lon DOUBLE PRECISION,
    p_radius_m DOUBLE PRECISION,
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ
)
RETURNS SETOF raw_events AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM raw_events
    WHERE ST_DWithin(
        location::geography,
        ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography,
        p_radius_m
    )
    AND timestamp BETWEEN p_start AND p_end
    ORDER BY timestamp;
END;
$$ LANGUAGE plpgsql;

-- Find similar embeddings (ANN search via pgvector)
CREATE OR REPLACE FUNCTION find_similar_embeddings(
    query_embedding vector(512),
    similarity_threshold REAL DEFAULT 0.7,
    max_results INTEGER DEFAULT 20
)
RETURNS TABLE(
    derived_event_id UUID,
    raw_event_id UUID,
    similarity REAL,
    class_name VARCHAR,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        de.id,
        de.raw_event_id,
        (1 - (de.embedding <=> query_embedding))::REAL AS similarity,
        de.class_name,
        de.created_at
    FROM derived_events de
    WHERE de.embedding IS NOT NULL
    AND (1 - (de.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY de.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Speed violation check (deterministic constraint)
CREATE OR REPLACE FUNCTION check_speed_violation(
    p_entity_id UUID,
    p_max_speed_mps REAL DEFAULT 35.0  -- ~80 mph
)
RETURNS TABLE(
    event_a_id UUID,
    event_b_id UUID,
    time_delta_s REAL,
    distance_m REAL,
    speed_mps REAL,
    is_violation BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH entity_sightings AS (
        SELECT
            re.id AS event_id,
            re.timestamp,
            re.location
        FROM raw_events re
        JOIN derived_events de ON de.raw_event_id = re.id
        JOIN entity_resolutions er ON de.id = ANY(er.derived_event_ids)
        WHERE er.entity_id = p_entity_id
        AND re.location IS NOT NULL
        ORDER BY re.timestamp
    ),
    pairs AS (
        SELECT
            a.event_id AS a_id,
            b.event_id AS b_id,
            EXTRACT(EPOCH FROM (b.timestamp - a.timestamp))::REAL AS dt,
            ST_Distance(a.location::geography, b.location::geography)::REAL AS dist
        FROM entity_sightings a
        JOIN entity_sightings b ON b.timestamp > a.timestamp
        AND b.timestamp < a.timestamp + INTERVAL '2 hours'
    )
    SELECT
        p.a_id,
        p.b_id,
        p.dt,
        p.dist,
        CASE WHEN p.dt > 0 THEN p.dist / p.dt ELSE 0 END AS spd,
        CASE WHEN p.dt > 0 AND (p.dist / p.dt) > p_max_speed_mps THEN true ELSE false END
    FROM pairs p
    WHERE p.dt > 0;
END;
$$ LANGUAGE plpgsql;

-- Auto-audit derived event creation
CREATE OR REPLACE FUNCTION log_derived_event()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (action, actor, resource_type, resource_id, model_id, model_version, details)
    VALUES (
        'inference',
        'system',
        'derived_event',
        NEW.id,
        NEW.model_id,
        NEW.model_version,
        jsonb_build_object(
            'type', NEW.type,
            'raw_event_id', NEW.raw_event_id,
            'confidence', NEW.confidence
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_derived_event_audit
    AFTER INSERT ON derived_events
    FOR EACH ROW
    EXECUTE FUNCTION log_derived_event();

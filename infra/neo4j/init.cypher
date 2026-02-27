// ============================================================
// Sentinel: Neo4j Graph Schema — Resolved Entities Only
// Versioned nodes/edges with model provenance
// ============================================================

// ─── CONSTRAINTS ────────────────────────────────────────────
CREATE CONSTRAINT person_id IF NOT EXISTS
FOR (p:Person) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT vehicle_id IF NOT EXISTS
FOR (v:Vehicle) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT camera_id IF NOT EXISTS
FOR (c:Camera) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT phone_id IF NOT EXISTS
FOR (ph:Phone) REQUIRE ph.id IS UNIQUE;

CREATE CONSTRAINT place_id IF NOT EXISTS
FOR (pl:Place) REQUIRE pl.id IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

// ─── INDEXES ────────────────────────────────────────────────
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX vehicle_plate IF NOT EXISTS FOR (v:Vehicle) ON (v.plate);
CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp);
CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.type);
CREATE INDEX camera_name IF NOT EXISTS FOR (c:Camera) ON (c.name);
CREATE INDEX place_name IF NOT EXISTS FOR (pl:Place) ON (pl.name);

// ─── NODE VERSION INDEXES ───────────────────────────────────
// All nodes MUST have: model_version, confidence, created_at, source_event_ids
CREATE INDEX node_model_version IF NOT EXISTS FOR (n:Person) ON (n.model_version);
CREATE INDEX node_created IF NOT EXISTS FOR (n:Person) ON (n.created_at);

// ─── SEED DATA (synthetic, versioned) ───────────────────────

// Cameras (infrastructure — no model version needed)
CREATE (c1:Camera {
  id: 'cam-001',
  name: 'Main St & 5th Ave',
  lat: 40.7580, lon: -73.9855,
  type: 'fixed', status: 'active',
  created_at: '2025-01-01T00:00:00Z'
})
CREATE (c2:Camera {
  id: 'cam-002',
  name: 'Park Entrance North',
  lat: 40.7614, lon: -73.9776,
  type: 'fixed', status: 'active',
  created_at: '2025-01-01T00:00:00Z'
})
CREATE (c3:Camera {
  id: 'cam-003',
  name: 'Highway I-95 Mile 42',
  lat: 40.7282, lon: -73.7949,
  type: 'traffic', status: 'active',
  created_at: '2025-01-01T00:00:00Z'
});

// Places
CREATE (p1:Place {
  id: 'place-001', name: 'Central Business District',
  lat: 40.7580, lon: -73.9855, type: 'district',
  created_at: '2025-01-01T00:00:00Z'
})
CREATE (p2:Place {
  id: 'place-002', name: 'City Park',
  lat: 40.7614, lon: -73.9776, type: 'park',
  created_at: '2025-01-01T00:00:00Z'
})
CREATE (p3:Place {
  id: 'place-003', name: 'Interstate Highway',
  lat: 40.7282, lon: -73.7949, type: 'highway',
  created_at: '2025-01-01T00:00:00Z'
});

// Camera→Place edges (with full provenance)
MATCH (c:Camera {id: 'cam-001'}), (p:Place {id: 'place-001'})
CREATE (c)-[:LOCATED_AT {
  confidence: 1.0,
  reason: 'Manual placement record',
  created_by_model: 'manual',
  created_at: '2025-01-01T00:00:00Z'
}]->(p);

MATCH (c:Camera {id: 'cam-002'}), (p:Place {id: 'place-002'})
CREATE (c)-[:LOCATED_AT {
  confidence: 1.0,
  reason: 'Manual placement record',
  created_by_model: 'manual',
  created_at: '2025-01-01T00:00:00Z'
}]->(p);

MATCH (c:Camera {id: 'cam-003'}), (p:Place {id: 'place-003'})
CREATE (c)-[:LOCATED_AT {
  confidence: 1.0,
  reason: 'Manual placement record',
  created_by_model: 'manual',
  created_at: '2025-01-01T00:00:00Z'
}]->(p);

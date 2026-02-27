# Sentinel — Service Contracts

Every service in Sentinel must define its contract before implementation.
This prevents integration chaos at Sprint 3.

---

## Contract Template

```yaml
service: <name>
version: <semver>
input:
  schema: <Pydantic model or JSON schema>
  transport: <Redis Stream | REST | gRPC>
  stream_name: <if Redis>
output:
  schema: <Pydantic model or JSON schema>
  transport: <Redis Stream | REST>
  stream_name: <if Redis>
failure_modes:
  - <mode>: <behavior>
latency_budget: <max acceptable latency>
backpressure: <strategy>
```

---

## 1. Ingestion Service

```yaml
service: ingest
version: 1.0.0
input:
  schema: RTSP URL | MP4 file path | MJPEG stream
  transport: REST (POST /api/v1/ingest/source)
  config:
    fps: 0.5 (default) | 2.0 (incident mode)
    max_resolution: 1920x1080
output:
  schema: RawEvent (frame type)
  transport: Redis Stream
  stream_name: sentinel:raw_events
  storage: Frame file → disk (./data/frames/<source>/<timestamp>.jpg)
failure_modes:
  - source_unreachable: retry 3x with exponential backoff, then mark source as 'error'
  - storage_full: emit backpressure metric, drop frames, log provenance of dropped frames
  - corrupt_frame: skip frame, log warning with source_id and timestamp
latency_budget: < 200ms per frame (decode + hash + store + enqueue)
backpressure:
  strategy: drop-oldest
  queue_max_length: 10000
  metric: sentinel_queue_depth{stream="sentinel:raw_events"}
```

## 2. Detector Service (AGPL-isolated container)

```yaml
service: detector
version: 1.0.0
license: AGPL-3.0 (isolated in separate Docker container)
input:
  schema: RawEvent (frame type) with file_path
  transport: Redis Stream
  stream_name: sentinel:raw_events
  consumer_group: detector-workers
output:
  schema: DerivedEvent (detection type)
  transport: Redis Stream
  stream_name: sentinel:derived_events
  fields: [class_name, confidence, bbox, raw_event_id, model_id, model_version]
failure_modes:
  - model_load_fail: exit with code 1, log error, do not process
  - inference_error: skip frame, log error with raw_event_id, continue
  - gpu_oom: fallback to CPU with ONNX, log degradation
latency_budget: < 100ms/frame (GPU) | < 500ms/frame (CPU/ONNX)
backpressure:
  strategy: consumer-group lag monitoring
  alert_threshold: lag > 1000 messages
  metric: sentinel_inference_latency_seconds{model_name="yolov8n"}
```

## 3. Tracker Service

```yaml
service: tracker
version: 1.0.0
input:
  schema: DerivedEvent (detection type) — sequential per source
  transport: Redis Stream
  stream_name: sentinel:derived_events
  consumer_group: tracker-workers
output:
  schema: DerivedEvent (track_update type) with track_id
  transport: Redis Stream
  stream_name: sentinel:track_events
failure_modes:
  - id_switch: log and emit metric (sentinel_id_switches_total)
  - track_lost: mark track as 'lost' after N frames, emit event
latency_budget: < 50ms per frame (tracking is lightweight)
```

## 4. Embedding Service

```yaml
service: embeddings
version: 1.0.0
input:
  schema: DerivedEvent (detection type) with bbox crop
  transport: Redis Stream
  stream_name: sentinel:derived_events
  consumer_group: embedding-workers
output:
  schema: DerivedEvent (face_embedding | vehicle_embedding) with embedding vector
  transport: Redis Stream
  stream_name: sentinel:embedding_events
  storage: embeddings stored in PostgreSQL pgvector column
failure_modes:
  - no_face_detected: skip, log info
  - embedding_dim_mismatch: reject, log error
  - model_version_change: log model transition in audit_log
latency_budget: < 150ms per crop (GPU) | < 800ms (CPU)
```

## 5. LPR Service

```yaml
service: lpr
version: 1.0.0
input:
  schema: DerivedEvent (detection type) with class_name='vehicle' bbox crop
  transport: Redis Stream
  stream_name: sentinel:derived_events
  consumer_group: lpr-workers
output:
  schema: DerivedEvent (plate_read type) with ocr_text, ocr_confidence
  transport: Redis Stream
  stream_name: sentinel:lpr_events
failure_modes:
  - no_plate_region: skip, not an error
  - low_confidence_read: emit with confidence < threshold, flag for review
  - ambiguous_characters: emit with edit_distance alternatives
latency_budget: < 200ms per crop
```

## 6. Graph Ingestion Service

```yaml
service: graph-ingest
version: 1.0.0
input:
  schema: Resolved Entity (from entity resolution) + edges
  transport: Redis Stream
  stream_name: sentinel:resolved_entities
  consumer_group: graph-workers
output:
  target: Neo4j (Cypher upsert)
  node_properties: [id, label, type, model_version, confidence, source_event_ids, created_at]
  edge_properties: [confidence, reason, created_by_model, created_at, source_event_ids]
failure_modes:
  - neo4j_down: buffer in Redis, retry with backoff
  - constraint_violation: log error, skip duplicate
  - schema_mismatch: reject, log error
latency_budget: < 50ms per upsert
```

## 7. Reasoner Service

```yaml
service: reasoner
version: 1.0.0
reasoning_stages:
  stage_1_deterministic:
    - speed_violation_check: "Can entity move between A and B in time T?"
    - time_window_consistency: "Are all events within plausible time window?"
    - plate_exact_match: "Do plate strings match exactly?"
    - bbox_overlap: "Do detection bboxes overlap significantly?"
  stage_2_probabilistic:
    - bayesian_update: "Prior * likelihood (spatial/temporal proximity * sensor confidence)"
    - embedding_similarity: "Cosine similarity above threshold?"
  stage_3_graph:
    - path_search: "Multi-hop shortest path between suspects and events"
    - structural_consistency: "Is graph structure consistent with hypothesis?"
input:
  schema: Hypothesis with constraint set
  transport: REST (POST /api/v1/hypotheses/{id}/score)
output:
  schema: HypothesisResponse with scoring_breakdown
failure_modes:
  - insufficient_evidence: return score=0 with reason
  - contradiction_found: return negative indicators in breakdown
  - timeout: return partial score with warning
latency_budget: < 2s per hypothesis scoring
```

## 8. API Service

```yaml
service: api
version: 1.0.0
endpoints:
  - "GET  /                       → root info"
  - "GET  /health                 → service health"
  - "GET  /metrics                → Prometheus metrics"
  - "POST /api/v1/events          → create raw event"
  - "GET  /api/v1/events          → list/filter events"
  - "GET  /api/v1/entities        → list/filter entities"
  - "GET  /api/v1/timeline        → confidence-weighted timeline"
  - "GET  /api/v1/graph           → entity graph data"
  - "POST /api/v1/hypotheses      → create hypothesis"
  - "POST /api/v1/hypotheses/:id/score → score hypothesis"
  - "GET  /api/v1/audit           → audit log"
transport: REST (JSON)
latency_budget: < 200ms for reads, < 500ms for writes
```

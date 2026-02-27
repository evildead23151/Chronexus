"""
Sentinel — Pydantic Models (3-Layer Data Model)

Layer 1: RawEvent — immutable sensor output
Layer 2: DerivedEvent — model inference output (linked to raw + model version)
Layer 3: Inference — hypothesis-level objects

Never overwrite raw data. Chain-of-custody requires immutability.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════

class RawEventType(str, Enum):
    FRAME = "frame"
    LPR_CAPTURE = "lpr_capture"
    PHONE_PING = "phone_ping"
    CALL_LOG = "call_log"
    TRANSACTION_LOG = "transaction_log"
    SENSOR_READING = "sensor_reading"
    AUDIO_CAPTURE = "audio_capture"
    MANUAL_ENTRY = "manual_entry"


class DerivedEventType(str, Enum):
    DETECTION = "detection"
    TRACK_UPDATE = "track_update"
    FACE_EMBEDDING = "face_embedding"
    VEHICLE_EMBEDDING = "vehicle_embedding"
    PLATE_READ = "plate_read"
    AUDIO_TRANSCRIPT = "audio_transcript"
    POSE_ESTIMATE = "pose_estimate"


class EntityType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    CAMERA = "camera"
    PHONE = "phone"
    PLACE = "place"
    TRANSACTION = "transaction"


class ResolutionMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    NEAR_MATCH = "near_match"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    GRAPH_STRUCTURAL = "graph_structural"
    MANUAL = "manual"


class HypothesisStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    VERIFIED = "verified"
    DISMISSED = "dismissed"


class ReasoningStage(str, Enum):
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    GRAPH = "graph"


# ═══════════════════════════════════════════════════════════
# VALUE OBJECTS
# ═══════════════════════════════════════════════════════════

class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    alt: Optional[float] = None
    accuracy_m: Optional[float] = None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Provenance(BaseModel):
    source_hash: str             # SHA256 of raw file
    provenance_hash: str         # HMAC signature for integrity
    signer: Optional[str] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class ModelReference(BaseModel):
    """Every derived event must reference the model that produced it."""
    model_id: UUID
    model_name: str
    model_version: str
    model_hash: Optional[str] = None  # SHA256 of weights


# ═══════════════════════════════════════════════════════════
# LAYER 1: RAW EVENTS (immutable)
# ═══════════════════════════════════════════════════════════

class RawEventCreate(BaseModel):
    type: RawEventType
    source_id: str
    timestamp: datetime
    location: Optional[Location] = None
    location_accuracy_m: Optional[float] = None
    raw_payload: dict[str, Any]         # exact sensor output, untouched
    file_path: Optional[str] = None
    file_hash: Optional[str] = None


class RawEventResponse(RawEventCreate):
    id: UUID
    provenance_hash: str
    provenance_signer: Optional[str] = None
    ingested_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════
# LAYER 2: DERIVED EVENTS (model outputs)
# ═══════════════════════════════════════════════════════════

class DerivedEventCreate(BaseModel):
    raw_event_id: UUID
    type: DerivedEventType
    model: ModelReference

    # Detection fields
    class_name: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    track_id: Optional[int] = None

    # Embedding (as list of floats — stored as pgvector)
    embedding: Optional[list[float]] = None

    # OCR / LPR
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Full output payload
    output_payload: dict[str, Any] = Field(default_factory=dict)


class DerivedEventResponse(DerivedEventCreate):
    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════
# ENTITIES (resolved identities)
# ═══════════════════════════════════════════════════════════

class EntityCreate(BaseModel):
    type: EntityType
    label: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class EntityResponse(EntityCreate):
    id: UUID
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    sighting_count: int = 0
    is_resolved: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class EntityResolution(BaseModel):
    """Records how an entity was resolved/merged."""
    entity_id: UUID
    method: ResolutionMethod
    derived_event_ids: list[UUID]
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    resolved_by: str = "system"


# ═══════════════════════════════════════════════════════════
# GRAPH NODE/EDGE (versioned, traceable)
# ═══════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    model_version: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source_event_ids: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""                          # why this edge exists
    created_by_model: str = "unknown"         # which model/rule created it
    created_at: Optional[str] = None
    source_event_ids: list[str] = Field(default_factory=list)  # evidence trail
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# LAYER 3: HYPOTHESES (inference objects)
# ═══════════════════════════════════════════════════════════

class HypothesisCreate(BaseModel):
    title: str
    description: Optional[str] = None
    analyst_id: Optional[str] = None
    raw_event_ids: list[UUID] = Field(default_factory=list)
    derived_event_ids: list[UUID] = Field(default_factory=list)
    entity_ids: list[UUID] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class HypothesisResponse(HypothesisCreate):
    id: UUID
    confidence_score: float = 0.0
    status: HypothesisStatus = HypothesisStatus.DRAFT
    reasoning_stage: ReasoningStage = ReasoningStage.DETERMINISTIC
    scoring_breakdown: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ═══════════════════════════════════════════════════════════
# TIMELINE
# ═══════════════════════════════════════════════════════════

class TimelineEntry(BaseModel):
    id: UUID
    layer: str                      # 'raw', 'derived', 'inference'
    type: str                       # raw/derived event type
    timestamp: datetime
    source: str
    location: Optional[Location] = None
    confidence: float
    summary: str
    model_version: Optional[str] = None
    entity_ids: list[UUID] = Field(default_factory=list)
    raw_event_id: Optional[UUID] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineResponse(BaseModel):
    entries: list[TimelineEntry]
    total: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════════

class AuditEntry(BaseModel):
    id: int
    action: str
    actor: str
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None
    model_id: Optional[UUID] = None
    model_version: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


# ═══════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════

class ModelRegistryEntry(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    model_name: str
    model_version: str
    model_type: str             # 'detector', 'tracker', 'face_embed', 'lpr', 'vehicle_embed'
    model_hash: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)
    license: Optional[str] = None
    is_active: bool = True
    loaded_at: datetime = Field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════
# SERVICE HEALTH
# ═══════════════════════════════════════════════════════════

class ServiceHealth(BaseModel):
    service: str
    status: str                 # 'up', 'degraded', 'down'
    version: str
    uptime_seconds: float
    queue_depth: Optional[int] = None
    active_workers: Optional[int] = None
    models_loaded: list[str] = Field(default_factory=list)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

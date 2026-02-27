"""
Sentinel — Timeline Router (3-Layer Model)
Shows raw events, derived events, and inferences with model provenance
"""
from log import get_logger
from datetime import datetime
from typing import Optional
from uuid import uuid4
from fastapi import APIRouter, Query

from models import TimelineResponse, TimelineEntry, Location

logger = get_logger()
router = APIRouter()


# Synthetic demo data — 3-layer model with model versioning
DEMO_TIMELINE = [
    # Layer 1: Raw sensor events
    TimelineEntry(
        id=uuid4(),
        layer="raw",
        type="frame",
        timestamp=datetime(2026, 2, 26, 10, 15, 0),
        source="cam-001",
        location=Location(lat=40.7580, lon=-73.9855),
        confidence=1.0,  # raw events are always 1.0 confidence
        summary="Raw frame captured at Main St & 5th Ave",
        model_version=None,
        entity_ids=[],
        metadata={"resolution": "1920x1080", "file_hash": "sha256:a1b2c3..."},
    ),
    # Layer 2: Derived — detection from that frame
    TimelineEntry(
        id=uuid4(),
        layer="derived",
        type="detection",
        timestamp=datetime(2026, 2, 26, 10, 15, 0),
        source="cam-001",
        location=Location(lat=40.7580, lon=-73.9855),
        confidence=0.92,
        summary="Person detected (bbox 92% conf) + white sedan (bbox 88% conf)",
        model_version="yolov8n-8.3.0",
        entity_ids=[],
        raw_event_id=uuid4(),
        metadata={"detections": 3, "classes": ["person", "car", "car"]},
    ),
    # Layer 2: Derived — LPR from same frame
    TimelineEntry(
        id=uuid4(),
        layer="derived",
        type="plate_read",
        timestamp=datetime(2026, 2, 26, 10, 17, 30),
        source="cam-001",
        location=Location(lat=40.7580, lon=-73.9855),
        confidence=0.87,
        summary="License plate ABC-1234 recognized on white sedan",
        model_version="paddleocr-2.8.0",
        entity_ids=[],
        raw_event_id=uuid4(),
        metadata={"plate": "ABC-1234", "ocr_confidence": 0.87, "edit_distance_alternatives": ["ABC-I234"]},
    ),
    # Layer 2: Derived — face embedding match
    TimelineEntry(
        id=uuid4(),
        layer="derived",
        type="face_embedding",
        timestamp=datetime(2026, 2, 26, 10, 22, 0),
        source="cam-002",
        location=Location(lat=40.7614, lon=-73.9776),
        confidence=0.78,
        summary="Face match (cosine sim 0.78) with subject from cam-001",
        model_version="insightface-buffalo_l-0.7.3",
        entity_ids=[],
        raw_event_id=uuid4(),
        metadata={"cosine_similarity": 0.78, "embedding_dim": 512},
    ),
    # Layer 1: Raw sensor event — phone ping
    TimelineEntry(
        id=uuid4(),
        layer="raw",
        type="phone_ping",
        timestamp=datetime(2026, 2, 26, 10, 25, 0),
        source="cell-tower-42",
        location=Location(lat=40.7600, lon=-73.9800, accuracy_m=150),
        confidence=1.0,
        summary="Phone ping detected — 150m accuracy radius",
        model_version=None,
        entity_ids=[],
        metadata={"signal_strength": -72, "cell_tower": "tower-42"},
    ),
    # Layer 2: Derived — second LPR on highway
    TimelineEntry(
        id=uuid4(),
        layer="derived",
        type="plate_read",
        timestamp=datetime(2026, 2, 26, 10, 45, 0),
        source="cam-003",
        location=Location(lat=40.7282, lon=-73.7949),
        confidence=0.94,
        summary="License plate ABC-1234 on Highway I-95 Mile 42 — exact match",
        model_version="paddleocr-2.8.0",
        entity_ids=[],
        raw_event_id=uuid4(),
        metadata={"plate": "ABC-1234", "speed_estimate_mph": 65, "direction": "eastbound"},
    ),
    # Layer 3: Inference — deterministic constraint check
    TimelineEntry(
        id=uuid4(),
        layer="inference",
        type="constraint_check",
        timestamp=datetime(2026, 2, 26, 10, 50, 0),
        source="reasoner-v1",
        confidence=0.85,
        summary="Speed check PASSED: 28 min travel, ~18 miles = ~38 mph (within 80 mph limit)",
        model_version="reasoner-deterministic-1.0",
        entity_ids=[],
        metadata={
            "check_type": "speed_violation",
            "stage": "deterministic",
            "distance_miles": 18.2,
            "time_minutes": 28,
            "speed_mph": 39,
            "max_speed_mph": 80,
            "result": "feasible",
        },
    ),
    # Layer 2: Derived — low confidence detection
    TimelineEntry(
        id=uuid4(),
        layer="derived",
        type="face_embedding",
        timestamp=datetime(2026, 2, 26, 11, 5, 0),
        source="cam-002",
        location=Location(lat=40.7614, lon=-73.9776),
        confidence=0.45,
        summary="Possible face match (0.45) — below threshold, flagged for review",
        model_version="insightface-buffalo_l-0.7.3",
        entity_ids=[],
        raw_event_id=uuid4(),
        metadata={"cosine_similarity": 0.45, "occlusion": True, "below_threshold": True},
    ),
]


@router.get("/", response_model=TimelineResponse)
async def get_timeline(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    source: Optional[str] = None,
    layer: Optional[str] = None,  # 'raw', 'derived', 'inference'
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    event_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Get the confidence-weighted event timeline.
    Supports filtering by layer (raw/derived/inference), confidence, source, and time window.
    """
    entries = DEMO_TIMELINE.copy()

    if start_time:
        entries = [e for e in entries if e.timestamp >= start_time]
    if end_time:
        entries = [e for e in entries if e.timestamp <= end_time]
    if source:
        entries = [e for e in entries if e.source == source]
    if layer:
        entries = [e for e in entries if e.layer == layer]
    if min_confidence > 0:
        entries = [e for e in entries if e.confidence >= min_confidence]
    if event_type:
        entries = [e for e in entries if e.type == event_type]

    entries.sort(key=lambda e: e.timestamp)
    total = len(entries)
    entries = entries[offset:offset + limit]

    return TimelineResponse(
        entries=entries,
        total=total,
        start_time=entries[0].timestamp if entries else None,
        end_time=entries[-1].timestamp if entries else None,
    )


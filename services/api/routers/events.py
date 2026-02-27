"""
Sentinel — Events Router (3-Layer Model)
CRUD for raw_events and derived_events
"""
from log import get_logger
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from models import (
    RawEventCreate, RawEventResponse, RawEventType,
    DerivedEventCreate, DerivedEventResponse, DerivedEventType,
    Location, Provenance,
)

logger = get_logger()
router = APIRouter()

# In-memory stores (replaced by PostgreSQL when Docker is up)
_raw_events: dict[UUID, dict] = {}
_derived_events: dict[UUID, dict] = {}


# ─── RAW EVENTS ─────────────────────────────────────────────

@router.post("/raw", status_code=201)
async def create_raw_event(event: RawEventCreate):
    """Create a new raw event (immutable once created)."""
    event_id = uuid4()
    now = datetime.now(timezone.utc)

    record = {
        "id": event_id,
        "created_at": now,
        **event.model_dump(),
    }
    _raw_events[event_id] = record

    logger.info(f"raw_event.created id={event_id} type={event.type}")
    return {"id": str(event_id), "status": "created", "layer": "raw"}


@router.get("/raw")
async def list_raw_events(
    type: Optional[RawEventType] = None,
    source_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List raw events with optional filters."""
    results = list(_raw_events.values())

    if type:
        results = [e for e in results if e.get("type") == type]
    if source_id:
        results = [e for e in results if e.get("source_id") == source_id]

    results.sort(key=lambda e: e.get("timestamp", e.get("created_at")))
    return {"events": results[offset:offset + limit], "total": len(results)}


@router.get("/raw/{event_id}")
async def get_raw_event(event_id: UUID):
    """Get a single raw event by ID."""
    if event_id not in _raw_events:
        raise HTTPException(status_code=404, detail="Raw event not found")
    return _raw_events[event_id]


# ─── DERIVED EVENTS ─────────────────────────────────────────

@router.post("/derived", status_code=201)
async def create_derived_event(event: DerivedEventCreate):
    """Create a new derived event (linked to a raw event)."""
    event_id = uuid4()
    now = datetime.now(timezone.utc)

    record = {
        "id": event_id,
        "created_at": now,
        **event.model_dump(),
    }
    _derived_events[event_id] = record

    logger.info(f"derived_event.created id={event_id} type={event.type}")
    return {"id": str(event_id), "status": "created", "layer": "derived"}


@router.get("/derived")
async def list_derived_events(
    type: Optional[DerivedEventType] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List derived events with optional filters."""
    results = list(_derived_events.values())

    if type:
        results = [e for e in results if e.get("type") == type]

    return {"events": results[offset:offset + limit], "total": len(results)}


@router.get("/derived/{event_id}")
async def get_derived_event(event_id: UUID):
    """Get a single derived event by ID."""
    if event_id not in _derived_events:
        raise HTTPException(status_code=404, detail="Derived event not found")
    return _derived_events[event_id]


# ─── SPATIAL SEARCH (placeholder) ───────────────────────────

@router.get("/near/search")
async def search_events_near(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(500, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    """Search events near a geographic point. PostGIS query in full mode."""
    all_events = list(_raw_events.values()) + list(_derived_events.values())
    return {
        "query": {"lat": lat, "lon": lon, "radius_m": radius_m},
        "results": all_events[:limit],
        "total": len(all_events),
        "note": "PostGIS spatial query pending Docker setup",
    }

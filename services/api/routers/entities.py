"""
Sentinel — Entities Router
CRUD for persons, vehicles, cameras, phones, places
"""
from log import get_logger
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from models import EntityCreate, EntityResponse, EntityType

logger = get_logger()
router = APIRouter()

# In-memory store for Sprint 0
_entities_store: dict[UUID, dict] = {}


@router.post("/", response_model=EntityResponse, status_code=201)
async def create_entity(entity: EntityCreate):
    """Create a new entity (person, vehicle, camera, etc.)."""
    entity_id = uuid4()
    now = datetime.utcnow()

    record = {
        "id": entity_id,
        "created_at": now,
        "first_seen": None,
        "last_seen": None,
        "sighting_count": 0,
        **entity.model_dump(),
    }
    _entities_store[entity_id] = record

    logger.info("entity.created", entity_id=str(entity_id), type=entity.type)
    return EntityResponse(**record)


@router.get("/", response_model=list[EntityResponse])
async def list_entities(
    type: Optional[EntityType] = None,
    label: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List entities with optional type/label filter."""
    results = list(_entities_store.values())

    if type:
        results = [e for e in results if e["type"] == type]
    if label:
        results = [e for e in results if label.lower() in (e.get("label") or "").lower()]

    return [EntityResponse(**e) for e in results[offset:offset + limit]]


@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: UUID):
    """Get a single entity by ID."""
    if entity_id not in _entities_store:
        raise HTTPException(status_code=404, detail="Entity not found")
    return EntityResponse(**_entities_store[entity_id])


@router.put("/{entity_id}", response_model=EntityResponse)
async def update_entity(entity_id: UUID, entity: EntityCreate):
    """Update an entity."""
    if entity_id not in _entities_store:
        raise HTTPException(status_code=404, detail="Entity not found")

    _entities_store[entity_id].update(entity.model_dump())
    _entities_store[entity_id]["updated_at"] = datetime.utcnow()

    logger.info("entity.updated", entity_id=str(entity_id))
    return EntityResponse(**_entities_store[entity_id])


@router.delete("/{entity_id}", status_code=204)
async def delete_entity(entity_id: UUID):
    """Delete an entity."""
    if entity_id not in _entities_store:
        raise HTTPException(status_code=404, detail="Entity not found")
    del _entities_store[entity_id]
    logger.info("entity.deleted", entity_id=str(entity_id))


"""
Sentinel — Audit Router
Immutable audit log for all actions and inferences
"""
from log import get_logger
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query

from models import AuditEntry

logger = get_logger()
router = APIRouter()

# In-memory append-only log for Sprint 0
_audit_log: list[dict] = [
    {
        "id": 1,
        "action": "create",
        "actor": "system",
        "resource_type": "camera",
        "resource_id": None,
        "details": {"cameras_initialized": 3, "source": "seed_data"},
        "timestamp": datetime(2026, 2, 26, 9, 0, 0),
    },
    {
        "id": 2,
        "action": "inference",
        "actor": "detector-v1",
        "resource_type": "detection",
        "resource_id": None,
        "details": {"model": "yolov8n", "frames_processed": 45, "detections": 12},
        "timestamp": datetime(2026, 2, 26, 10, 15, 0),
    },
    {
        "id": 3,
        "action": "inference",
        "actor": "face-embedder",
        "resource_type": "face_match",
        "resource_id": None,
        "details": {"model": "insightface-buffalo_l", "matches_found": 2, "avg_confidence": 0.78},
        "timestamp": datetime(2026, 2, 26, 10, 20, 0),
    },
    {
        "id": 4,
        "action": "inference",
        "actor": "lpr-engine",
        "resource_type": "license_plate",
        "resource_id": None,
        "details": {"model": "paddleocr", "plates_read": 2, "plate": "ABC-1234", "confidence": 0.91},
        "timestamp": datetime(2026, 2, 26, 10, 17, 30),
    },
    {
        "id": 5,
        "action": "create",
        "actor": "analyst-demo",
        "resource_type": "hypothesis",
        "resource_id": None,
        "details": {"title": "Subject A traveled from CBD to Highway via Park"},
        "timestamp": datetime(2026, 2, 26, 11, 30, 0),
    },
]
_audit_counter = len(_audit_log)


@router.get("/", response_model=list[AuditEntry])
async def list_audit_entries(
    action: Optional[str] = None,
    actor: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List audit log entries (append-only, immutable)."""
    results = _audit_log.copy()

    if action:
        results = [a for a in results if a["action"] == action]
    if actor:
        results = [a for a in results if a["actor"] == actor]
    if resource_type:
        results = [a for a in results if a["resource_type"] == resource_type]
    if start_time:
        results = [a for a in results if a["timestamp"] >= start_time]
    if end_time:
        results = [a for a in results if a["timestamp"] <= end_time]

    results.sort(key=lambda a: a["timestamp"], reverse=True)
    return [AuditEntry(**a) for a in results[offset:offset + limit]]


@router.get("/stats")
async def audit_stats():
    """Get audit log statistics."""
    actions = {}
    actors = {}
    for entry in _audit_log:
        actions[entry["action"]] = actions.get(entry["action"], 0) + 1
        actors[entry["actor"]] = actors.get(entry["actor"], 0) + 1

    return {
        "total_entries": len(_audit_log),
        "actions_breakdown": actions,
        "actors_breakdown": actors,
        "earliest": _audit_log[0]["timestamp"].isoformat() if _audit_log else None,
        "latest": _audit_log[-1]["timestamp"].isoformat() if _audit_log else None,
    }


"""
Sentinel — Hypotheses Router
Hypothesis creation, scoring, and management
"""
from log import get_logger
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from models import HypothesisCreate, HypothesisResponse, HypothesisStatus

logger = get_logger()
router = APIRouter()

# In-memory store for Sprint 0
_hypotheses_store: dict[UUID, dict] = {}

# Pre-populate with demo hypothesis
_demo_id = uuid4()
_hypotheses_store[_demo_id] = {
    "id": _demo_id,
    "title": "Subject A traveled from CBD to Highway via Park",
    "description": (
        "Hypothesis: Unknown Subject A was observed at Main St & 5th Ave at 10:15, "
        "then at Park Entrance North at 10:22, before the associated vehicle (ABC-1234) "
        "was detected on Highway I-95 at 10:45. Travel time and distance are consistent "
        "with vehicle transit."
    ),
    "analyst_id": "analyst-demo",
    "evidence_ids": [],
    "constraints": {
        "max_speed_mph": 80,
        "min_travel_time_min": 15,
        "face_match_required": True,
    },
    "confidence_score": 0.73,
    "status": HypothesisStatus.ACTIVE,
    "scoring_breakdown": {
        "temporal_consistency": 0.85,
        "spatial_feasibility": 0.90,
        "face_match_avg": 0.72,
        "lpr_match": 0.91,
        "phone_corroboration": 0.55,
        "combined_bayesian": 0.73,
    },
    "created_at": datetime(2026, 2, 26, 11, 30, 0),
    "updated_at": datetime(2026, 2, 26, 11, 30, 0),
}


@router.post("/", response_model=HypothesisResponse, status_code=201)
async def create_hypothesis(hypothesis: HypothesisCreate):
    """Create a new hypothesis for investigation."""
    hyp_id = uuid4()
    now = datetime.utcnow()

    record = {
        "id": hyp_id,
        "confidence_score": 0.0,
        "status": HypothesisStatus.DRAFT,
        "scoring_breakdown": {},
        "created_at": now,
        "updated_at": now,
        **hypothesis.model_dump(),
    }
    _hypotheses_store[hyp_id] = record

    logger.info("hypothesis.created", hyp_id=str(hyp_id), title=hypothesis.title)
    return HypothesisResponse(**record)


@router.get("/", response_model=list[HypothesisResponse])
async def list_hypotheses(
    status: Optional[HypothesisStatus] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all hypotheses, optionally filtered."""
    results = list(_hypotheses_store.values())

    if status:
        results = [h for h in results if h["status"] == status]
    if min_confidence > 0:
        results = [h for h in results if h["confidence_score"] >= min_confidence]

    results.sort(key=lambda h: h["confidence_score"], reverse=True)
    return [HypothesisResponse(**h) for h in results[:limit]]


@router.get("/{hypothesis_id}", response_model=HypothesisResponse)
async def get_hypothesis(hypothesis_id: UUID):
    """Get a single hypothesis."""
    if hypothesis_id not in _hypotheses_store:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return HypothesisResponse(**_hypotheses_store[hypothesis_id])


@router.post("/{hypothesis_id}/score")
async def score_hypothesis(hypothesis_id: UUID):
    """Re-score a hypothesis based on current evidence (placeholder for Sprint 5)."""
    if hypothesis_id not in _hypotheses_store:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    return {
        "hypothesis_id": str(hypothesis_id),
        "status": "scoring_queued",
        "message": "Bayesian scoring will be implemented in Sprint 5",
    }


@router.put("/{hypothesis_id}/status")
async def update_hypothesis_status(hypothesis_id: UUID, status: HypothesisStatus):
    """Update hypothesis status (draft → active → verified/dismissed)."""
    if hypothesis_id not in _hypotheses_store:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    _hypotheses_store[hypothesis_id]["status"] = status
    _hypotheses_store[hypothesis_id]["updated_at"] = datetime.utcnow()

    logger.info("hypothesis.status_updated", hyp_id=str(hypothesis_id), status=status)
    return HypothesisResponse(**_hypotheses_store[hypothesis_id])


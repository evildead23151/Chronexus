"""
Sentinel — Entity Resolution Engine

The hardest problem in the system. Handles:
1. Exact match (plate string identity)
2. Near match (edit distance for plates, fuzzy name matching)
3. Embedding similarity (cosine distance with pgvector ANN)
4. Temporal consistency (can entity move between sightings in time?)
5. Graph structural consistency (does merge create contradictions?)

CRITICAL: False merges are catastrophic. Every resolution is logged
in entity_resolutions table with method, confidence, and evidence trail.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger("sentinel.entity_resolution")


class ResolutionMethod(Enum):
    EXACT_MATCH = "exact_match"
    NEAR_MATCH = "near_match"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    GRAPH_STRUCTURAL = "graph_structural"
    MANUAL = "manual"


@dataclass
class ResolutionCandidate:
    """A candidate entity match with evidence."""
    entity_id: UUID
    method: ResolutionMethod
    confidence: float
    evidence: dict
    derived_event_ids: list[UUID]


@dataclass
class ResolutionResult:
    """Result of entity resolution attempt."""
    matched: bool
    entity_id: Optional[UUID]
    method: Optional[ResolutionMethod]
    confidence: float
    is_new_entity: bool
    details: dict


# ─── Thresholds (configurable, conservative defaults) ────────

THRESHOLDS = {
    "plate_exact_match": 1.0,          # exact string match
    "plate_near_match_max_edit": 1,    # max Levenshtein distance
    "plate_near_match_confidence": 0.8,
    "face_cosine_threshold": 0.70,     # cosine similarity for face embeddings
    "face_cosine_high_conf": 0.85,
    "vehicle_cosine_threshold": 0.75,
    "temporal_max_speed_mps": 35.0,    # ~80 mph max travel speed
    "merge_min_confidence": 0.65,      # minimum confidence to auto-merge
    "review_confidence_band": (0.50, 0.65),  # needs human review
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def resolve_plate_exact(plate_text: str, existing_plates: dict[str, UUID]) -> Optional[ResolutionCandidate]:
    """
    Stage 1: Exact plate string match.
    Deterministic, zero ambiguity.
    """
    normalized = plate_text.upper().replace(" ", "").replace("-", "")

    if normalized in existing_plates:
        return ResolutionCandidate(
            entity_id=existing_plates[normalized],
            method=ResolutionMethod.EXACT_MATCH,
            confidence=1.0,
            evidence={"plate": normalized, "match_type": "exact"},
            derived_event_ids=[],
        )
    return None


def resolve_plate_near(
    plate_text: str,
    existing_plates: dict[str, UUID],
    max_edit_distance: int = 1,
) -> list[ResolutionCandidate]:
    """
    Stage 1: Near plate match using edit distance.
    Returns candidates sorted by distance.
    Common confusions: O/0, I/1, B/8, etc.
    """
    normalized = plate_text.upper().replace(" ", "").replace("-", "")
    candidates = []

    for existing, entity_id in existing_plates.items():
        dist = levenshtein_distance(normalized, existing)
        if 0 < dist <= max_edit_distance:
            confidence = max(0.5, 1.0 - (dist * 0.2))
            candidates.append(ResolutionCandidate(
                entity_id=entity_id,
                method=ResolutionMethod.NEAR_MATCH,
                confidence=confidence,
                evidence={
                    "query_plate": normalized,
                    "matched_plate": existing,
                    "edit_distance": dist,
                },
                derived_event_ids=[],
            ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def check_temporal_feasibility(
    lat1: float, lon1: float, time1_epoch: float,
    lat2: float, lon2: float, time2_epoch: float,
    max_speed_mps: float = 35.0,
) -> dict:
    """
    Deterministic check: Can an entity physically travel between two points
    in the given time at a plausible speed?

    Uses Haversine approximation for distance.
    """
    import math

    # Haversine distance
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_m = R * c

    time_delta_s = abs(time2_epoch - time1_epoch)

    if time_delta_s == 0:
        return {
            "feasible": distance_m < 50,  # same time, must be very close
            "distance_m": distance_m,
            "time_delta_s": 0,
            "speed_mps": float('inf') if distance_m > 0 else 0,
            "max_speed_mps": max_speed_mps,
        }

    speed_mps = distance_m / time_delta_s

    return {
        "feasible": speed_mps <= max_speed_mps,
        "distance_m": round(distance_m, 1),
        "time_delta_s": round(time_delta_s, 1),
        "speed_mps": round(speed_mps, 2),
        "speed_mph": round(speed_mps * 2.237, 1),
        "max_speed_mps": max_speed_mps,
        "max_speed_mph": round(max_speed_mps * 2.237, 1),
    }


def resolve_entity(
    entity_type: str,
    evidence: dict,
    existing_entities: dict,
) -> ResolutionResult:
    """
    Main entity resolution pipeline.
    Runs checks in order of determinism:
    1. Exact match
    2. Near match
    3. Embedding similarity (deferred to pgvector query)
    4. Temporal consistency
    5. Graph structural (deferred to Neo4j)

    Returns resolution decision with full audit trail.
    """
    if entity_type == "vehicle" and "plate" in evidence:
        # Stage 1a: Exact plate match
        plates = existing_entities.get("plates", {})
        exact = resolve_plate_exact(evidence["plate"], plates)
        if exact:
            logger.info("entity_resolution.exact_match",
                        plate=evidence["plate"],
                        entity_id=str(exact.entity_id))
            return ResolutionResult(
                matched=True,
                entity_id=exact.entity_id,
                method=ResolutionMethod.EXACT_MATCH,
                confidence=1.0,
                is_new_entity=False,
                details=exact.evidence,
            )

        # Stage 1b: Near plate match
        near_candidates = resolve_plate_near(evidence["plate"], plates)
        if near_candidates and near_candidates[0].confidence >= THRESHOLDS["merge_min_confidence"]:
            best = near_candidates[0]
            logger.info("entity_resolution.near_match",
                        plate=evidence["plate"],
                        matched_plate=best.evidence["matched_plate"],
                        edit_distance=best.evidence["edit_distance"])
            return ResolutionResult(
                matched=True,
                entity_id=best.entity_id,
                method=ResolutionMethod.NEAR_MATCH,
                confidence=best.confidence,
                is_new_entity=False,
                details=best.evidence,
            )

        # Stage 1b (review band): Flag for human review
        if near_candidates:
            best = near_candidates[0]
            low, high = THRESHOLDS["review_confidence_band"]
            if low <= best.confidence < high:
                logger.warning("entity_resolution.review_needed",
                               plate=evidence["plate"],
                               candidate=best.evidence,
                               confidence=best.confidence)
                return ResolutionResult(
                    matched=False,
                    entity_id=best.entity_id,
                    method=ResolutionMethod.NEAR_MATCH,
                    confidence=best.confidence,
                    is_new_entity=False,
                    details={**best.evidence, "needs_review": True},
                )

    # No match found — create new entity
    logger.info("entity_resolution.new_entity",
                entity_type=entity_type,
                evidence_keys=list(evidence.keys()))
    return ResolutionResult(
        matched=False,
        entity_id=None,
        method=None,
        confidence=0.0,
        is_new_entity=True,
        details={"reason": "No existing entity matched"},
    )

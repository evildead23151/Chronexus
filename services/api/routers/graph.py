"""
Sentinel — Graph Router (Versioned Nodes/Edges)
All nodes have model_version, confidence, source_event_ids.
All edges have confidence, reason, created_by_model.
"""
from log import get_logger
from typing import Optional
from fastapi import APIRouter, Query

from models import GraphData, GraphNode, GraphEdge

logger = get_logger()
router = APIRouter()


# Versioned synthetic demo graph
DEMO_GRAPH = GraphData(
    nodes=[
        GraphNode(
            id="person-001", label="Unknown Subject A", type="Person",
            model_version="insightface-buffalo_l-0.7.3",
            confidence=0.72,
            source_event_ids=["de-001", "de-003", "de-007"],
            created_at="2026-02-26T10:15:00Z",
            properties={
                "first_seen": "2026-02-26T10:15:00Z",
                "last_seen": "2026-02-26T11:05:00Z",
                "sighting_count": 3,
                "resolution_method": "embedding_similarity",
            },
        ),
        GraphNode(
            id="vehicle-001", label="White Sedan (ABC-1234)", type="Vehicle",
            model_version="paddleocr-2.8.0",
            confidence=0.91,
            source_event_ids=["de-002", "de-005"],
            created_at="2026-02-26T10:17:30Z",
            properties={
                "plate": "ABC-1234",
                "color": "white",
                "resolution_method": "exact_match",
                "edit_distance": 0,
            },
        ),
        GraphNode(
            id="cam-001", label="Main St & 5th Ave", type="Camera",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7580, "lon": -73.9855, "type": "fixed"},
        ),
        GraphNode(
            id="cam-002", label="Park Entrance North", type="Camera",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7614, "lon": -73.9776, "type": "fixed"},
        ),
        GraphNode(
            id="cam-003", label="Highway I-95 Mile 42", type="Camera",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7282, "lon": -73.7949, "type": "traffic"},
        ),
        GraphNode(
            id="place-001", label="Central Business District", type="Place",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7580, "lon": -73.9855, "type": "district"},
        ),
        GraphNode(
            id="place-002", label="City Park", type="Place",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7614, "lon": -73.9776, "type": "park"},
        ),
        GraphNode(
            id="place-003", label="Interstate Highway", type="Place",
            confidence=1.0,
            created_at="2025-01-01T00:00:00Z",
            properties={"lat": 40.7282, "lon": -73.7949, "type": "highway"},
        ),
        GraphNode(
            id="phone-001", label="Phone Ping Signal", type="Phone",
            confidence=0.65,
            source_event_ids=["re-004"],
            created_at="2026-02-26T10:25:00Z",
            properties={"signal_strength": -72, "accuracy_m": 150},
        ),
    ],
    edges=[
        # Person sightings (with full provenance)
        GraphEdge(
            source="person-001", target="cam-001", relationship="APPEARED_IN",
            confidence=0.92, reason="YOLOv8 detection + InsightFace embedding match",
            created_by_model="insightface-buffalo_l-0.7.3",
            created_at="2026-02-26T10:15:00Z",
            source_event_ids=["de-001"],
        ),
        GraphEdge(
            source="person-001", target="cam-002", relationship="APPEARED_IN",
            confidence=0.78, reason="Face embedding cosine similarity 0.78",
            created_by_model="insightface-buffalo_l-0.7.3",
            created_at="2026-02-26T10:22:00Z",
            source_event_ids=["de-003"],
        ),
        # Vehicle sightings
        GraphEdge(
            source="vehicle-001", target="cam-001", relationship="APPEARED_IN",
            confidence=0.87, reason="PaddleOCR plate read: ABC-1234 (87% conf)",
            created_by_model="paddleocr-2.8.0",
            created_at="2026-02-26T10:17:30Z",
            source_event_ids=["de-002"],
        ),
        GraphEdge(
            source="vehicle-001", target="cam-003", relationship="APPEARED_IN",
            confidence=0.94, reason="PaddleOCR plate read: ABC-1234 (94% conf) — exact match",
            created_by_model="paddleocr-2.8.0",
            created_at="2026-02-26T10:45:00Z",
            source_event_ids=["de-005"],
        ),
        # Person-Vehicle proximity (derived relationship)
        GraphEdge(
            source="person-001", target="vehicle-001", relationship="NEAR",
            confidence=0.75,
            reason="Co-located at cam-001 within 150-second window (10:15-10:17)",
            created_by_model="reasoner-deterministic-1.0",
            created_at="2026-02-26T10:50:00Z",
            source_event_ids=["de-001", "de-002"],
        ),
        # Camera placements
        GraphEdge(
            source="cam-001", target="place-001", relationship="LOCATED_AT",
            confidence=1.0, reason="Manual placement record",
            created_by_model="manual",
            created_at="2025-01-01T00:00:00Z",
        ),
        GraphEdge(
            source="cam-002", target="place-002", relationship="LOCATED_AT",
            confidence=1.0, reason="Manual placement record",
            created_by_model="manual",
            created_at="2025-01-01T00:00:00Z",
        ),
        GraphEdge(
            source="cam-003", target="place-003", relationship="LOCATED_AT",
            confidence=1.0, reason="Manual placement record",
            created_by_model="manual",
            created_at="2025-01-01T00:00:00Z",
        ),
        # Phone ping proximity
        GraphEdge(
            source="phone-001", target="person-001", relationship="NEAR",
            confidence=0.55,
            reason="Phone ping 150m from last person sighting, temporal overlap 3 min",
            created_by_model="reasoner-deterministic-1.0",
            created_at="2026-02-26T10:50:00Z",
            source_event_ids=["re-004", "de-003"],
        ),
    ],
)


@router.get("/", response_model=GraphData)
async def get_graph(
    entity_id: Optional[str] = None,
    depth: int = Query(2, ge=1, le=5),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    created_by_model: Optional[str] = None,
):
    """
    Get versioned entity graph.
    All nodes/edges include model_version, confidence, reason, and source_event_ids.
    """
    nodes = DEMO_GRAPH.nodes
    edges = DEMO_GRAPH.edges

    # Filter by model
    if created_by_model:
        edges = [e for e in edges if created_by_model in e.created_by_model]

    # Filter by confidence
    if min_confidence > 0:
        edges = [e for e in edges if e.confidence >= min_confidence]
        node_ids = set()
        for edge in edges:
            node_ids.add(edge.source)
            node_ids.add(edge.target)
        nodes = [n for n in nodes if n.id in node_ids]

    # Neighborhood query
    if entity_id:
        connected_ids = {entity_id}
        for _ in range(depth):
            for edge in edges:
                if edge.source in connected_ids:
                    connected_ids.add(edge.target)
                if edge.target in connected_ids:
                    connected_ids.add(edge.source)
        nodes = [n for n in nodes if n.id in connected_ids]
        edges = [e for e in edges if e.source in connected_ids and e.target in connected_ids]

    return GraphData(nodes=nodes, edges=edges)


@router.get("/neighbors/{node_id}")
async def get_neighbors(node_id: str, depth: int = Query(1, ge=1, le=3)):
    """Get neighbors of a node (versioned edges)."""
    return await get_graph(entity_id=node_id, depth=depth)


@router.get("/path")
async def find_path(
    source_id: str = Query(...),
    target_id: str = Query(...),
):
    """Find shortest path between entities (Neo4j Cypher in Sprint 4)."""
    return {
        "source": source_id,
        "target": target_id,
        "path": [],
        "message": "Path search via Neo4j Cypher — Sprint 4",
    }


@router.get("/provenance/{edge_source}/{edge_target}")
async def get_edge_provenance(edge_source: str, edge_target: str):
    """Get full provenance trail for a specific edge."""
    matching = [
        e for e in DEMO_GRAPH.edges
        if e.source == edge_source and e.target == edge_target
    ]
    return {
        "edges": matching,
        "message": "Each edge includes: confidence, reason, created_by_model, source_event_ids"
    }


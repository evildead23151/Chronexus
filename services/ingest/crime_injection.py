"""
Sentinel — Synthetic Crime Injection Layer

Injects synthetic crime-like events into the data pipeline for validation.
NO REAL CRIMES. All scenarios are fabricated for system testing.

Three injection categories:
  A. Overlay Injection — synthetic bounding boxes, plates, timestamps
  B. Behavioral Simulation — impossible travel, cross-camera movement
  C. Adversarial Simulation — occlusion, lighting, tracker drift

Safety:
  - All entity IDs are synthetic (SYNTH- prefix)
  - No real individual identification
  - Outputs are hypotheses only
  - Human-in-the-loop assumed
"""
import hashlib
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════

class InjectionType(Enum):
    OVERLAY = "overlay"
    BEHAVIORAL = "behavioral"
    ADVERSARIAL = "adversarial"


class EntityClass(Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    PHONE = "phone"


@dataclass
class SyntheticBBox:
    """Synthetic bounding box overlay."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    track_id: int
    entity_class: EntityClass


@dataclass
class SyntheticPlate:
    """Synthetic license plate detection."""
    text: str
    confidence: float
    camera_id: str
    timestamp: str
    bbox: Optional[SyntheticBBox] = None


@dataclass
class SyntheticPhonePing:
    """Synthetic phone ping / cell tower signal."""
    tower_id: str
    signal_dbm: int
    accuracy_m: float
    latitude: float
    longitude: float
    timestamp: str


@dataclass
class SyntheticDetection:
    """A fully synthetic detection event injected into the pipeline."""
    detection_id: str
    camera_id: str
    timestamp: str
    entity_class: EntityClass
    entity_id: str          # always SYNTH- prefix
    confidence: float
    bbox: SyntheticBBox
    plate: Optional[SyntheticPlate] = None
    phone_ping: Optional[SyntheticPhonePing] = None
    injection_type: InjectionType = InjectionType.OVERLAY
    metadata: dict = field(default_factory=dict)


@dataclass
class SyntheticScenario:
    """A complete synthetic crime scenario with expected outcomes."""
    scenario_id: str
    name: str
    description: str
    detections: list[SyntheticDetection]
    expected: dict        # expected validation outcomes
    safety_note: str = "All entities are synthetic. No real individuals."


# ═══════════════════════════════════════════════════════════
# HELPER: HAVERSINE DISTANCE
# ═══════════════════════════════════════════════════════════

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def travel_feasible(lat1, lon1, t1_epoch, lat2, lon2, t2_epoch,
                    max_speed_mps=35.0) -> dict:
    """Check if travel between two points is physically possible."""
    dist = haversine_m(lat1, lon1, lat2, lon2)
    dt = abs(t2_epoch - t1_epoch)
    if dt == 0:
        return {"feasible": dist < 50, "speed_mps": float('inf'),
                "distance_m": dist, "time_s": 0}
    speed = dist / dt
    return {
        "feasible": speed <= max_speed_mps,
        "speed_mps": round(speed, 2),
        "speed_mph": round(speed * 2.237, 1),
        "distance_m": round(dist, 1),
        "time_s": round(dt, 1),
    }


# ═══════════════════════════════════════════════════════════
# CAMERA COORDINATES (from feed_adapters registry)
# ═══════════════════════════════════════════════════════════

CAMERA_COORDS = {
    "DEL-CP-001": (28.6315, 77.2167),   # Connaught Place
    "DEL-IG-002": (28.6129, 77.2295),   # India Gate
    "DEL-CC-003": (28.6507, 77.2334),   # Chandni Chowk
    "LON-TF-001": (51.5080, -0.1281),   # Trafalgar Square
    "LON-WB-002": (51.5007, -0.1220),   # Westminster Bridge
    "NYC-TS-001": (40.7580, -73.9855),  # Times Square
    "TKY-SB-001": (35.6595, 139.7004),  # Shibuya
    "BER-BG-001": (52.5163, 13.3777),   # Brandenburg Gate
    "SGP-MB-001": (1.2839, 103.8607),   # Marina Bay
}


def _ts(base: datetime, offset_min: float) -> str:
    return (base + timedelta(minutes=offset_min)).isoformat()


def _bbox(x=200, y=300, w=40, h=80, label="person", conf=0.9,
          track_id=1, cls=EntityClass.PERSON) -> SyntheticBBox:
    return SyntheticBBox(x, y, x+w, y+h, label, conf, track_id, cls)


# ═══════════════════════════════════════════════════════════
# SCENARIO 1: SINGLE SUSPECT MOVEMENT
# ═══════════════════════════════════════════════════════════

def scenario_single_suspect(base_time: Optional[datetime] = None) -> SyntheticScenario:
    """
    One synthetic person moves across a single camera's FOV for 20 frames.
    Expected: 1 entity, no false merge, stable tracking ID.
    """
    base = base_time or datetime(2026, 2, 27, 22, 0, 0, tzinfo=timezone.utc)
    entity_id = f"SYNTH-P-{uuid.uuid4().hex[:8]}"
    track_id = 42

    detections = []
    for i in range(20):
        x = 100 + i * 50
        conf = 0.85 + 0.05 * math.sin(i * 0.3)
        detections.append(SyntheticDetection(
            detection_id=f"sd-s1-{i:03d}",
            camera_id="DEL-CP-001",
            timestamp=_ts(base, i * 0.5),
            entity_class=EntityClass.PERSON,
            entity_id=entity_id,
            confidence=round(conf, 3),
            bbox=_bbox(x, 300, 40, 80, "person", conf, track_id),
            injection_type=InjectionType.OVERLAY,
        ))

    return SyntheticScenario(
        scenario_id="S1",
        name="Single Suspect Movement",
        description="One synthetic person traverses cam DEL-CP-001 FOV over 10 minutes",
        detections=detections,
        expected={
            "entity_count": 1,
            "false_merges": 0,
            "tracking_stable": True,
            "unique_track_ids": 1,
            "min_confidence": 0.80,
        },
    )


# ═══════════════════════════════════════════════════════════
# SCENARIO 2: CROSS-CAMERA MOVEMENT (Delhi)
# ═══════════════════════════════════════════════════════════

def scenario_cross_camera(base_time: Optional[datetime] = None) -> SyntheticScenario:
    """
    Synthetic person appears at Connaught Place, then India Gate 15 min later.
    CP → IG: ~2.5 km, 15 min = ~2.8 m/s = feasible walking speed.
    Expected: temporal check passes, valid path in graph.
    """
    base = base_time or datetime(2026, 2, 27, 22, 0, 0, tzinfo=timezone.utc)
    entity_id = f"SYNTH-P-{uuid.uuid4().hex[:8]}"
    vehicle_id = f"SYNTH-V-{uuid.uuid4().hex[:8]}"

    detections = [
        # Person at Connaught Place
        SyntheticDetection(
            detection_id="sd-s2-001",
            camera_id="DEL-CP-001",
            timestamp=_ts(base, 0),
            entity_class=EntityClass.PERSON,
            entity_id=entity_id,
            confidence=0.91,
            bbox=_bbox(200, 300, 40, 80, "person", 0.91, 1),
            injection_type=InjectionType.BEHAVIORAL,
        ),
        # Vehicle near person at CP
        SyntheticDetection(
            detection_id="sd-s2-002",
            camera_id="DEL-CP-001",
            timestamp=_ts(base, 0.5),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_id,
            confidence=0.88,
            bbox=_bbox(300, 350, 80, 45, "car", 0.88, 2, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL1CAB1234", 0.87, "DEL-CP-001", _ts(base, 0.5)),
            injection_type=InjectionType.BEHAVIORAL,
        ),
        # Person at India Gate (15 min later)
        SyntheticDetection(
            detection_id="sd-s2-003",
            camera_id="DEL-IG-002",
            timestamp=_ts(base, 15),
            entity_class=EntityClass.PERSON,
            entity_id=entity_id,
            confidence=0.78,
            bbox=_bbox(400, 280, 38, 76, "person", 0.78, 1),
            injection_type=InjectionType.BEHAVIORAL,
            metadata={"face_similarity": 0.78},
        ),
        # Vehicle at India Gate
        SyntheticDetection(
            detection_id="sd-s2-004",
            camera_id="DEL-IG-002",
            timestamp=_ts(base, 15),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_id,
            confidence=0.94,
            bbox=_bbox(350, 380, 85, 48, "car", 0.94, 2, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL1CAB1234", 0.94, "DEL-IG-002", _ts(base, 15)),
            injection_type=InjectionType.BEHAVIORAL,
        ),
        # Phone ping corroboration
        SyntheticDetection(
            detection_id="sd-s2-005",
            camera_id="DEL-IG-002",
            timestamp=_ts(base, 14),
            entity_class=EntityClass.PHONE,
            entity_id=entity_id,
            confidence=0.55,
            bbox=_bbox(0, 0, 0, 0, "phone", 0.55, 99, EntityClass.PHONE),
            phone_ping=SyntheticPhonePing(
                "tower-42", -72, 150.0, 28.6140, 77.2280, _ts(base, 14)
            ),
            injection_type=InjectionType.BEHAVIORAL,
        ),
    ]

    # Verify feasibility
    cp, ig = CAMERA_COORDS["DEL-CP-001"], CAMERA_COORDS["DEL-IG-002"]
    check = travel_feasible(cp[0], cp[1], 0, ig[0], ig[1], 900)  # 15 min

    return SyntheticScenario(
        scenario_id="S2",
        name="Cross-Camera Movement (Delhi)",
        description="Person + vehicle move from Connaught Place to India Gate in 15 min",
        detections=detections,
        expected={
            "temporal_feasible": True,
            "travel_speed_mps": check["speed_mps"],
            "travel_distance_m": check["distance_m"],
            "entity_count": 2,  # 1 person + 1 vehicle
            "plate_exact_match": True,
            "graph_path_valid": True,
        },
    )


# ═══════════════════════════════════════════════════════════
# SCENARIO 3: IMPOSSIBLE TRAVEL
# ═══════════════════════════════════════════════════════════

def scenario_impossible_travel(base_time: Optional[datetime] = None) -> SyntheticScenario:
    """
    Same vehicle appears in Delhi and Tokyo within 5 minutes.
    Delhi → Tokyo: ~5,800 km. 5 min = ~19,300 m/s = IMPOSSIBLE.
    Expected: temporal violation flagged, confidence reduced.
    """
    base = base_time or datetime(2026, 2, 27, 22, 0, 0, tzinfo=timezone.utc)
    vehicle_id = f"SYNTH-V-{uuid.uuid4().hex[:8]}"

    detections = [
        SyntheticDetection(
            detection_id="sd-s3-001",
            camera_id="DEL-CP-001",
            timestamp=_ts(base, 0),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_id,
            confidence=0.92,
            bbox=_bbox(200, 350, 85, 48, "car", 0.92, 10, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL9XYZ7777", 0.92, "DEL-CP-001", _ts(base, 0)),
            injection_type=InjectionType.BEHAVIORAL,
        ),
        SyntheticDetection(
            detection_id="sd-s3-002",
            camera_id="TKY-SB-001",
            timestamp=_ts(base, 5),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_id,
            confidence=0.89,
            bbox=_bbox(300, 330, 90, 50, "car", 0.89, 10, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL9XYZ7777", 0.89, "TKY-SB-001", _ts(base, 5)),
            injection_type=InjectionType.BEHAVIORAL,
        ),
    ]

    del_c, tky_c = CAMERA_COORDS["DEL-CP-001"], CAMERA_COORDS["TKY-SB-001"]
    check = travel_feasible(del_c[0], del_c[1], 0, tky_c[0], tky_c[1], 300)

    return SyntheticScenario(
        scenario_id="S3",
        name="Impossible Travel (Delhi → Tokyo in 5 min)",
        description="Vehicle with identical plate seen in Delhi then Tokyo 5 min later",
        detections=detections,
        expected={
            "temporal_feasible": False,
            "travel_speed_mps": check["speed_mps"],
            "distance_m": check["distance_m"],
            "violation_reason": "temporal_inconsistency",
            "confidence_reduced": True,
            "should_flag": True,
        },
    )


# ═══════════════════════════════════════════════════════════
# SCENARIO 4: PLATE NEAR-MATCH
# ═══════════════════════════════════════════════════════════

def scenario_plate_near_match(base_time: Optional[datetime] = None) -> SyntheticScenario:
    """
    Two vehicles with similar plates: DL1CAB1234 vs DL1CAB12O4 (O vs 0).
    Expected: near-match band, review status, NO auto-merge.
    """
    base = base_time or datetime(2026, 2, 27, 22, 0, 0, tzinfo=timezone.utc)
    vehicle_a = f"SYNTH-V-{uuid.uuid4().hex[:8]}"
    vehicle_b = f"SYNTH-V-{uuid.uuid4().hex[:8]}"

    detections = [
        SyntheticDetection(
            detection_id="sd-s4-001",
            camera_id="DEL-CP-001",
            timestamp=_ts(base, 0),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_a,
            confidence=0.90,
            bbox=_bbox(200, 350, 85, 48, "car", 0.90, 20, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL1CAB1234", 0.90, "DEL-CP-001", _ts(base, 0)),
            injection_type=InjectionType.BEHAVIORAL,
        ),
        SyntheticDetection(
            detection_id="sd-s4-002",
            camera_id="DEL-IG-002",
            timestamp=_ts(base, 8),
            entity_class=EntityClass.VEHICLE,
            entity_id=vehicle_b,
            confidence=0.85,
            bbox=_bbox(300, 340, 88, 50, "car", 0.85, 21, EntityClass.VEHICLE),
            plate=SyntheticPlate("DL1CAB12O4", 0.85, "DEL-IG-002", _ts(base, 8)),
            injection_type=InjectionType.BEHAVIORAL,
            metadata={"ocr_confusion": "0_vs_O", "edit_distance": 1},
        ),
    ]

    return SyntheticScenario(
        scenario_id="S4",
        name="Plate Near-Match (OCR Confusion)",
        description="Two vehicles with plates differing by 1 character (0 vs O)",
        detections=detections,
        expected={
            "exact_match": False,
            "near_match": True,
            "edit_distance": 1,
            "auto_merge": False,
            "review_status": True,
            "entity_count": 2,  # must remain separate
        },
    )


# ═══════════════════════════════════════════════════════════
# SCENARIO 5: TRACKER DRIFT / ID SWITCH SPIKE
# ═══════════════════════════════════════════════════════════

def scenario_tracker_drift(base_time: Optional[datetime] = None) -> SyntheticScenario:
    """
    Single person with unstable tracker — ID switches 6 times in 20 frames.
    Expected: ID switch count flagged, drift alert raised.
    """
    base = base_time or datetime(2026, 2, 27, 22, 0, 0, tzinfo=timezone.utc)
    entity_id = f"SYNTH-P-{uuid.uuid4().hex[:8]}"

    track_ids = [42, 42, 42, 43, 43, 42, 44, 44, 42, 42,
                 45, 42, 42, 46, 46, 42, 47, 42, 42, 42]

    detections = []
    for i, tid in enumerate(track_ids):
        x = 100 + i * 40
        conf = 0.80 + 0.10 * random.Random(i).random()
        detections.append(SyntheticDetection(
            detection_id=f"sd-s5-{i:03d}",
            camera_id="LON-TF-001",
            timestamp=_ts(base, i * 0.5),
            entity_class=EntityClass.PERSON,
            entity_id=entity_id,
            confidence=round(conf, 3),
            bbox=_bbox(x, 300, 38, 76, "person", conf, tid),
            injection_type=InjectionType.ADVERSARIAL,
            metadata={"track_id": tid, "frame": i},
        ))

    unique_tids = len(set(track_ids))
    switches = sum(1 for i in range(1, len(track_ids)) if track_ids[i] != track_ids[i-1])

    return SyntheticScenario(
        scenario_id="S5",
        name="Tracker Drift / ID Switch Spike",
        description=f"Single person with {switches} ID switches across 20 frames",
        detections=detections,
        expected={
            "id_switch_count": switches,
            "unique_track_ids": unique_tids,
            "drift_alert": True,
            "drift_rate": round(switches / len(track_ids), 3),
            "actual_entities": 1,  # should resolve to 1 despite switches
        },
    )


# ═══════════════════════════════════════════════════════════
# SCENARIO REGISTRY
# ═══════════════════════════════════════════════════════════

ALL_SCENARIOS = [
    scenario_single_suspect,
    scenario_cross_camera,
    scenario_impossible_travel,
    scenario_plate_near_match,
    scenario_tracker_drift,
]


def generate_all_scenarios(base_time: Optional[datetime] = None) -> list[SyntheticScenario]:
    """Generate all 5 synthetic scenarios."""
    return [fn(base_time) for fn in ALL_SCENARIOS]

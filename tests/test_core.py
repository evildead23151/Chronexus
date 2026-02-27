"""
Sentinel — Unit Tests: Ingestion Service + Provenance

Tests:
1. Provenance HMAC signing and verification
2. Entity resolution (exact, near, temporal)
3. Frame processor (hashing, storage)
4. Backpressure logic
5. Session logger (replay)
"""
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import numpy as np

# Add service paths
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "ingest"))


# ═══════════════════════════════════════════════════════════
# 1. Provenance Tests
# ═══════════════════════════════════════════════════════════

class TestProvenance:
    """HMAC provenance signing and verification."""

    def test_sign_raw_event_deterministic(self):
        from provenance import sign_raw_event
        sig1 = sign_raw_event(
            event_type="frame",
            source_id="cam-001",
            timestamp="2026-02-26T10:15:00Z",
            raw_payload={"width": 1920, "height": 1080},
            file_hash="sha256:abc123",
        )
        sig2 = sign_raw_event(
            event_type="frame",
            source_id="cam-001",
            timestamp="2026-02-26T10:15:00Z",
            raw_payload={"width": 1920, "height": 1080},
            file_hash="sha256:abc123",
        )
        assert sig1 == sig2, "Same input must produce same signature"
        assert sig1.startswith("hmac-sha256:")

    def test_sign_different_inputs(self):
        from provenance import sign_raw_event
        sig1 = sign_raw_event("frame", "cam-001", "2026-02-26T10:15:00Z", {"a": 1})
        sig2 = sign_raw_event("frame", "cam-002", "2026-02-26T10:15:00Z", {"a": 1})
        assert sig1 != sig2, "Different source must produce different signature"

    def test_verify_valid_event(self):
        from provenance import sign_raw_event, verify_raw_event
        payload = {"width": 1920, "height": 1080}
        sig = sign_raw_event("frame", "cam-001", "2026-02-26T10:15:00Z", payload, "sha256:x")
        assert verify_raw_event("frame", "cam-001", "2026-02-26T10:15:00Z", payload, sig, "sha256:x")

    def test_verify_tampered_event(self):
        from provenance import sign_raw_event, verify_raw_event
        payload = {"width": 1920, "height": 1080}
        sig = sign_raw_event("frame", "cam-001", "2026-02-26T10:15:00Z", payload)
        # Tamper with payload
        tampered = {"width": 1920, "height": 720}
        assert not verify_raw_event("frame", "cam-001", "2026-02-26T10:15:00Z", tampered, sig)

    def test_compute_file_hash(self):
        from provenance import compute_file_hash
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"test data for hashing")
            f.flush()
            path = f.name
        try:
            h = compute_file_hash(path)
            assert h.startswith("sha256:")
            assert len(h) > 10
        finally:
            os.unlink(path)

    def test_compute_payload_hash_sorted(self):
        from provenance import compute_payload_hash
        h1 = compute_payload_hash({"b": 2, "a": 1})
        h2 = compute_payload_hash({"a": 1, "b": 2})
        assert h1 == h2, "Key order must not affect payload hash"


# ═══════════════════════════════════════════════════════════
# 2. Entity Resolution Tests
# ═══════════════════════════════════════════════════════════

class TestEntityResolution:
    """Entity resolution pipeline tests."""

    def test_plate_exact_match(self):
        from entity_resolution import resolve_plate_exact
        plates = {"ABC1234": uuid4(), "XYZ9999": uuid4()}
        result = resolve_plate_exact("ABC-1234", plates)
        assert result is not None
        assert result.confidence == 1.0
        assert result.method.value == "exact_match"

    def test_plate_exact_no_match(self):
        from entity_resolution import resolve_plate_exact
        plates = {"ABC1234": uuid4()}
        result = resolve_plate_exact("DEF-5678", plates)
        assert result is None

    def test_plate_near_match(self):
        from entity_resolution import resolve_plate_near
        target_id = uuid4()
        plates = {"ABC1234": target_id}
        candidates = resolve_plate_near("ABC1Z34", plates, max_edit_distance=1)
        assert len(candidates) == 1
        assert candidates[0].entity_id == target_id
        assert candidates[0].method.value == "near_match"

    def test_plate_near_no_match_too_far(self):
        from entity_resolution import resolve_plate_near
        plates = {"ABC1234": uuid4()}
        candidates = resolve_plate_near("XYZ9999", plates, max_edit_distance=1)
        assert len(candidates) == 0

    def test_levenshtein_distance(self):
        from entity_resolution import levenshtein_distance
        assert levenshtein_distance("ABC", "ABC") == 0
        assert levenshtein_distance("ABC", "ABD") == 1
        assert levenshtein_distance("ABC", "ABCD") == 1
        assert levenshtein_distance("ABC", "XYZ") == 3

    def test_temporal_feasibility_possible(self):
        from entity_resolution import check_temporal_feasibility
        # 10 km in 600 seconds = ~16.7 m/s (~37 mph) — feasible
        result = check_temporal_feasibility(
            lat1=40.7580, lon1=-73.9855, time1_epoch=0,
            lat2=40.8480, lon2=-73.9855, time2_epoch=600,
        )
        assert result["feasible"] is True

    def test_temporal_feasibility_impossible(self):
        from entity_resolution import check_temporal_feasibility
        # ~100 km in 60 seconds = ~1667 m/s — impossible
        result = check_temporal_feasibility(
            lat1=40.0, lon1=-74.0, time1_epoch=0,
            lat2=41.0, lon2=-74.0, time2_epoch=60,
        )
        assert result["feasible"] is False


# ═══════════════════════════════════════════════════════════
# 3. Ingestion Components Tests
# ═══════════════════════════════════════════════════════════

class TestIngestionComponents:
    """Test individual ingestion components."""

    def test_session_logger_write_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Inline import to avoid cv2 at module level
            sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "ingest"))

            from rtsp_service import SessionLogger
            session_id = f"test_{uuid4().hex[:8]}"
            sl = SessionLogger(session_id, tmpdir)

            events = [
                {"id": str(uuid4()), "type": "frame", "timestamp": "2026-02-26T10:15:00Z",
                 "source_id": "cam-001", "raw_payload": {"width": 1920}},
                {"id": str(uuid4()), "type": "frame", "timestamp": "2026-02-26T10:16:00Z",
                 "source_id": "cam-001", "raw_payload": {"width": 1920}},
            ]
            for e in events:
                sl.log_event(e)

            # Load back
            loaded = SessionLogger.load_session(session_id, tmpdir)
            assert len(loaded) == 2
            assert loaded[0]["source_id"] == "cam-001"

    def test_ingest_metrics_prometheus_format(self):
        from rtsp_service import IngestMetrics
        m = IngestMetrics(
            frames_read=100,
            frames_stored=95,
            frames_dropped_backpressure=3,
            frames_dropped_corrupt=2,
            current_fps=0.5,
            session_id="test-session",
            source="cam-001",
        )
        output = m.to_prometheus()
        assert "sentinel_ingest_frames_read" in output
        assert "sentinel_ingest_current_fps" in output
        assert "sentinel_ingest_backpressure" in output
        assert "100" in output  # frames_read value

    def test_ingest_metrics_latency_tracking(self):
        from rtsp_service import IngestMetrics
        m = IngestMetrics()
        m.record_latency(50.0)
        m.record_latency(100.0)
        m.record_latency(150.0)
        assert m.avg_latency_ms == 100.0

    def test_frame_processor_valid_frame(self):
        """Test processing a valid synthetic frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from rtsp_service import FrameProcessor, IngestConfig, IngestMetrics
            config = IngestConfig(
                source_id="test-cam",
                frame_store=tmpdir,
                session_id="test-session",
            )
            metrics = IngestMetrics()
            proc = FrameProcessor(config, metrics)

            # Create synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            event = proc.process_frame(
                frame=frame,
                frame_number=1,
                timestamp=datetime.now(timezone.utc),
            )

            assert event is not None
            assert event["type"] == "frame"
            assert event["file_hash"].startswith("sha256:")
            assert event["provenance_hash"].startswith("hmac-sha256:")
            assert event["source_id"] == "test-cam"
            assert metrics.frames_stored == 1
            assert metrics.avg_latency_ms > 0

            # Verify file exists
            assert os.path.exists(event["file_path"])

    def test_frame_processor_null_frame(self):
        """Test that null/corrupt frames are dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from rtsp_service import FrameProcessor, IngestConfig, IngestMetrics
            config = IngestConfig(source_id="test-cam", frame_store=tmpdir)
            metrics = IngestMetrics()
            proc = FrameProcessor(config, metrics)

            event = proc.process_frame(
                frame=np.array([]),
                frame_number=1,
                timestamp=datetime.now(timezone.utc),
            )

            assert event is None
            assert metrics.frames_dropped_corrupt == 1


# ═══════════════════════════════════════════════════════════
# 4. Models / Schema Tests
# ═══════════════════════════════════════════════════════════

class TestModels:
    """Test Pydantic model validation."""

    def test_raw_event_create(self):
        from models import RawEventCreate, RawEventType
        event = RawEventCreate(
            type=RawEventType.FRAME,
            source_id="cam-001",
            timestamp=datetime.now(timezone.utc),
            raw_payload={"width": 1920, "height": 1080},
        )
        assert event.type == RawEventType.FRAME
        assert event.raw_payload["width"] == 1920

    def test_derived_event_create(self):
        from models import DerivedEventCreate, DerivedEventType, ModelReference, BoundingBox
        event = DerivedEventCreate(
            raw_event_id=uuid4(),
            type=DerivedEventType.DETECTION,
            model=ModelReference(
                model_id=uuid4(),
                model_name="yolov8n",
                model_version="8.3.0",
            ),
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
        )
        assert event.confidence == 0.92
        assert event.model.model_name == "yolov8n"

    def test_hypothesis_create(self):
        from models import HypothesisCreate
        hyp = HypothesisCreate(
            title="Subject A traveled from CBD to Highway",
            description="Cross-camera face match + plate match",
            analyst_id="analyst-01",
        )
        assert len(hyp.title) > 0

    def test_graph_edge_versioned(self):
        from models import GraphEdge
        edge = GraphEdge(
            source="person-001",
            target="cam-001",
            relationship="APPEARED_IN",
            confidence=0.92,
            reason="YOLOv8 detection + InsightFace match",
            created_by_model="insightface-0.7.3",
            created_at="2026-02-26T10:15:00Z",
            source_event_ids=["de-001"],
        )
        assert edge.created_by_model == "insightface-0.7.3"
        assert len(edge.source_event_ids) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

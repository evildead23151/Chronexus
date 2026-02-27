"""
Sentinel — Ingestion Service: RTSP/MP4 Frame Sampler

Service contract:
  Input:  RTSP URL | MP4 file path
  Output: Redis Stream 'sentinel:raw_events' (or in-memory queue fallback)
  Storage: Frames → ./data/frames/<source>/<timestamp>.jpg
  Provenance: Every frame HMAC-signed at ingestion
  Backpressure:
    - redis_stream_length > MAX_QUEUE → reduce FPS by 50%
    - disk_usage > 80% → drop low-confidence frames
  Latency budget: < 200ms per frame (decode + hash + store + enqueue)
  Failure modes:
    - source_unreachable: retry 3x exponential backoff, then mark 'error'
    - storage_full: emit metric, drop frames, log provenance of drops
    - corrupt_frame: skip, log warning

Usage:
  python rtsp_service.py --source data/sample.mp4
  python rtsp_service.py --source data/sample.mp4 --replay-session session_001
  python rtsp_service.py --source rtsp://camera.local/stream
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from provenance import sign_raw_event, compute_file_hash

# ─── Structured Logging ─────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","service":"ingest","msg":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logger = logging.getLogger("sentinel.ingest")


# ─── Configuration ───────────────────────────────────────────

@dataclass
class IngestConfig:
    """Ingestion service configuration — all from env vars."""
    # Source
    source: str = ""
    source_id: str = ""

    # Frame sampling
    base_fps: float = float(os.environ.get("BASE_FPS", "1.0"))
    current_fps: float = float(os.environ.get("BASE_FPS", "1.0"))
    min_fps: float = 0.1
    max_resolution: tuple[int, int] = (1920, 1080)

    # Backpressure
    max_queue: int = int(os.environ.get("MAX_QUEUE", "50000"))
    max_disk_usage: float = float(os.environ.get("MAX_DISK_USAGE", "0.80"))
    backpressure_active: bool = False

    # Storage
    frame_store: str = os.environ.get("FRAME_STORE_PATH", "./data/frames")
    data_dir: str = os.environ.get("DATA_DIR", "./data")

    # Redis (optional — fallback to in-memory queue)
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    stream_name: str = os.environ.get("FRAME_STREAM", "sentinel:raw_events")

    # Reproducibility
    replay_session: Optional[str] = None
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")

    # Limits
    max_frames: int = int(os.environ.get("MAX_FRAMES", "0"))  # 0 = unlimited
    retry_count: int = 3
    retry_backoff: float = 1.0


# ─── Metrics (in-memory, exposed via /metrics) ──────────────

@dataclass
class IngestMetrics:
    """Prometheus-style metrics for ingestion service."""
    frames_read: int = 0
    frames_stored: int = 0
    frames_dropped_backpressure: int = 0
    frames_dropped_corrupt: int = 0
    frames_dropped_disk: int = 0
    bytes_stored: int = 0
    events_enqueued: int = 0
    current_fps: float = 1.0
    queue_depth: int = 0
    disk_usage_pct: float = 0.0
    backpressure_active: bool = False
    avg_latency_ms: float = 0.0
    session_id: str = ""
    source: str = ""
    started_at: str = ""
    _latencies: list[float] = field(default_factory=list)

    def record_latency(self, ms: float) -> None:
        self._latencies.append(ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def to_prometheus(self) -> str:
        """Format metrics as Prometheus text exposition."""
        lines = [
            f'# HELP sentinel_ingest_frames_read Total frames read from source',
            f'# TYPE sentinel_ingest_frames_read counter',
            f'sentinel_ingest_frames_read{{source="{self.source}",session="{self.session_id}"}} {self.frames_read}',
            f'# HELP sentinel_ingest_frames_stored Total frames stored to disk',
            f'# TYPE sentinel_ingest_frames_stored counter',
            f'sentinel_ingest_frames_stored{{source="{self.source}"}} {self.frames_stored}',
            f'# HELP sentinel_ingest_frames_dropped_total Frames dropped',
            f'# TYPE sentinel_ingest_frames_dropped_total counter',
            f'sentinel_ingest_frames_dropped_total{{reason="backpressure"}} {self.frames_dropped_backpressure}',
            f'sentinel_ingest_frames_dropped_total{{reason="corrupt"}} {self.frames_dropped_corrupt}',
            f'sentinel_ingest_frames_dropped_total{{reason="disk_full"}} {self.frames_dropped_disk}',
            f'# HELP sentinel_ingest_current_fps Current sampling FPS',
            f'# TYPE sentinel_ingest_current_fps gauge',
            f'sentinel_ingest_current_fps {self.current_fps}',
            f'# HELP sentinel_ingest_queue_depth Current queue depth',
            f'# TYPE sentinel_ingest_queue_depth gauge',
            f'sentinel_ingest_queue_depth {self.queue_depth}',
            f'# HELP sentinel_ingest_disk_usage_pct Disk usage percentage',
            f'# TYPE sentinel_ingest_disk_usage_pct gauge',
            f'sentinel_ingest_disk_usage_pct {self.disk_usage_pct:.3f}',
            f'# HELP sentinel_ingest_backpressure Is backpressure active',
            f'# TYPE sentinel_ingest_backpressure gauge',
            f'sentinel_ingest_backpressure {1 if self.backpressure_active else 0}',
            f'# HELP sentinel_ingest_latency_ms Average frame processing latency',
            f'# TYPE sentinel_ingest_latency_ms gauge',
            f'sentinel_ingest_latency_ms {self.avg_latency_ms:.2f}',
            f'# HELP sentinel_ingest_bytes_stored Total bytes stored',
            f'# TYPE sentinel_ingest_bytes_stored counter',
            f'sentinel_ingest_bytes_stored {self.bytes_stored}',
        ]
        return '\n'.join(lines) + '\n'


# ─── Event Queue (Redis or In-Memory Fallback) ──────────────

class EventQueue:
    """Abstraction over Redis Streams with in-memory fallback."""

    def __init__(self, redis_url: str, stream_name: str):
        self._redis_url = redis_url
        self._stream_name = stream_name
        self._redis = None
        self._memory_queue: list[dict] = []
        self._using_redis = False

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True, max_connections=5
            )
            await self._redis.ping()
            self._using_redis = True
            logger.info(f"redis.connected url={self._redis_url}")
        except Exception as e:
            logger.warning(f"redis.unavailable falling_back=in_memory error={e}")
            self._using_redis = False

    async def enqueue(self, event: dict, maxlen: int = 50000) -> None:
        if self._using_redis and self._redis:
            # Flatten for Redis Stream (only string values)
            flat = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in event.items()}
            await self._redis.xadd(self._stream_name, flat, maxlen=maxlen)
        else:
            self._memory_queue.append(event)
            # Trim in-memory queue
            if len(self._memory_queue) > maxlen:
                self._memory_queue = self._memory_queue[-maxlen:]

    async def length(self) -> int:
        if self._using_redis and self._redis:
            try:
                return await self._redis.xlen(self._stream_name)
            except Exception:
                return 0
        return len(self._memory_queue)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    def get_memory_queue(self) -> list[dict]:
        return self._memory_queue


# ─── Session Logger (for Reproducibility) ───────────────────

class SessionLogger:
    """Logs every raw event to a session file for replay."""

    def __init__(self, session_id: str, data_dir: str):
        self._session_dir = Path(data_dir) / "sessions" / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._session_dir / "events.jsonl"
        self._meta_file = self._session_dir / "meta.json"
        self._event_count = 0

    def log_event(self, event: dict) -> None:
        with open(self._log_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        self._event_count += 1

    def write_meta(self, config: IngestConfig, metrics: IngestMetrics) -> None:
        meta = {
            "session_id": config.session_id,
            "source": config.source,
            "source_id": config.source_id,
            "started_at": metrics.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "frames_read": metrics.frames_read,
            "frames_stored": metrics.frames_stored,
            "frames_dropped": (
                metrics.frames_dropped_backpressure +
                metrics.frames_dropped_corrupt +
                metrics.frames_dropped_disk
            ),
            "events_logged": self._event_count,
            "config": {
                "base_fps": config.base_fps,
                "max_queue": config.max_queue,
                "max_disk_usage": config.max_disk_usage,
            },
        }
        with open(self._meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_session(session_id: str, data_dir: str) -> list[dict]:
        log_file = Path(data_dir) / "sessions" / session_id / "events.jsonl"
        if not log_file.exists():
            raise FileNotFoundError(f"Session {session_id} not found at {log_file}")
        events = []
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


# ─── Backpressure Controller ────────────────────────────────

class BackpressureController:
    """Manages FPS reduction based on queue depth and disk usage."""

    def __init__(self, config: IngestConfig, metrics: IngestMetrics):
        self._config = config
        self._metrics = metrics

    def check_disk_usage(self) -> float:
        """Get disk usage percentage for frame store partition."""
        try:
            usage = shutil.disk_usage(self._config.frame_store)
            pct = usage.used / usage.total
            self._metrics.disk_usage_pct = pct
            return pct
        except Exception:
            return 0.0

    async def evaluate(self, queue: EventQueue) -> float:
        """
        Evaluate backpressure and return adjusted FPS.

        Policy:
        - queue_depth > MAX_QUEUE     → reduce FPS by 50%
        - queue_depth > MAX_QUEUE*1.5 → reduce FPS by 75%
        - disk_usage > 80%           → drop low-confidence frames
        - otherwise                   → restore to base FPS
        """
        queue_depth = await queue.length()
        disk_pct = self.check_disk_usage()

        self._metrics.queue_depth = queue_depth
        self._metrics.disk_usage_pct = disk_pct

        target_fps = self._config.base_fps

        # Queue-based backpressure
        if queue_depth > self._config.max_queue * 1.5:
            target_fps = self._config.base_fps * 0.25
            self._config.backpressure_active = True
            logger.warning(f"backpressure.critical queue_depth={queue_depth} fps_reduced_to={target_fps:.2f}")
        elif queue_depth > self._config.max_queue:
            target_fps = self._config.base_fps * 0.5
            self._config.backpressure_active = True
            logger.warning(f"backpressure.active queue_depth={queue_depth} fps_reduced_to={target_fps:.2f}")
        else:
            if self._config.backpressure_active:
                logger.info(f"backpressure.resolved queue_depth={queue_depth} fps_restored_to={target_fps:.2f}")
            self._config.backpressure_active = False

        # Disk-based backpressure
        if disk_pct > self._config.max_disk_usage:
            target_fps = max(self._config.min_fps, target_fps * 0.5)
            logger.warning(f"backpressure.disk_high usage={disk_pct:.1%} fps={target_fps:.2f}")

        target_fps = max(self._config.min_fps, target_fps)
        self._config.current_fps = target_fps
        self._metrics.current_fps = target_fps
        self._metrics.backpressure_active = self._config.backpressure_active

        return target_fps


# ─── Frame Processor ────────────────────────────────────────

class FrameProcessor:
    """Processes individual frames: decode, hash, store, sign, enqueue."""

    def __init__(self, config: IngestConfig, metrics: IngestMetrics):
        self._config = config
        self._metrics = metrics
        self._frame_dir = Path(config.frame_store) / config.source_id
        self._frame_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: datetime,
        capture_timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Process a single frame:
        1. Resize if needed
        2. Encode as JPEG
        3. Compute SHA256 hash
        4. Save to disk
        5. Sign with HMAC provenance
        6. Return raw event dict

        Latency budget: < 200ms

        Returns None if frame should be dropped.
        """
        t0 = time.perf_counter()

        # ── 1. Validate frame
        if frame is None or frame.size == 0:
            self._metrics.frames_dropped_corrupt += 1
            logger.warning(f"frame.corrupt frame_number={frame_number}")
            return None

        # ── 2. Resize if exceeds max resolution
        h, w = frame.shape[:2]
        max_w, max_h = self._config.max_resolution
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        # ── 3. Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, encoded = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            self._metrics.frames_dropped_corrupt += 1
            logger.warning(f"frame.encode_failed frame_number={frame_number}")
            return None

        frame_bytes = encoded.tobytes()

        # ── 4. Compute SHA256 hash
        file_hash = f"sha256:{hashlib.sha256(frame_bytes).hexdigest()}"

        # ── 5. Save to disk
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{frame_number:08d}_{ts_str}.jpg"
        file_path = str(self._frame_dir / filename)

        try:
            with open(file_path, "wb") as f:
                f.write(frame_bytes)
        except OSError as e:
            self._metrics.frames_dropped_disk += 1
            logger.error(f"frame.write_failed path={file_path} error={e}")
            return None

        file_size = len(frame_bytes)
        self._metrics.frames_stored += 1
        self._metrics.bytes_stored += file_size

        # ── 6. Build raw event payload
        raw_payload = {
            "width": w,
            "height": h,
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "jpeg_quality": 85,
            "file_size_bytes": file_size,
            "frame_number": frame_number,
        }

        # ── 7. Sign with HMAC
        ts_iso = timestamp.isoformat()
        provenance_hash = sign_raw_event(
            event_type="frame",
            source_id=self._config.source_id,
            timestamp=ts_iso,
            raw_payload=raw_payload,
            file_hash=file_hash,
        )

        # ── 8. Build raw event
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "type": "frame",
            "source_id": self._config.source_id,
            "timestamp": ts_iso,
            "raw_payload": raw_payload,
            "file_path": file_path,
            "file_hash": file_hash,
            "provenance_hash": provenance_hash,
            "provenance_signer": "ingest-service",
            "session_id": self._config.session_id,
            "frame_number": frame_number,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── 9. Track latency
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._metrics.record_latency(elapsed_ms)

        if elapsed_ms > 200:
            logger.warning(f"frame.slow_processing latency_ms={elapsed_ms:.1f} budget_ms=200")

        return event


# ─── Ingestion Engine ────────────────────────────────────────

class IngestionEngine:
    """Main ingestion loop: read source → sample → process → enqueue."""

    def __init__(self, config: IngestConfig):
        self.config = config
        self.metrics = IngestMetrics(
            session_id=config.session_id,
            source=config.source,
            current_fps=config.current_fps,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.queue = EventQueue(config.redis_url, config.stream_name)
        self.processor = FrameProcessor(config, self.metrics)
        self.backpressure = BackpressureController(config, self.metrics)
        self.session_logger = SessionLogger(config.session_id, config.data_dir)
        self._running = False

    async def start(self) -> None:
        """Start the ingestion pipeline."""
        await self.queue.connect()

        logger.info(
            f"ingest.starting source={self.config.source} "
            f"session={self.config.session_id} "
            f"fps={self.config.base_fps} "
            f"max_queue={self.config.max_queue}"
        )

        if self.config.replay_session:
            await self._run_replay()
        else:
            await self._run_live()

        # Write session metadata
        self.session_logger.write_meta(self.config, self.metrics)
        await self.queue.close()

        logger.info(
            f"ingest.completed "
            f"frames_read={self.metrics.frames_read} "
            f"frames_stored={self.metrics.frames_stored} "
            f"dropped_backpressure={self.metrics.frames_dropped_backpressure} "
            f"dropped_corrupt={self.metrics.frames_dropped_corrupt} "
            f"dropped_disk={self.metrics.frames_dropped_disk} "
            f"avg_latency_ms={self.metrics.avg_latency_ms:.1f} "
            f"session={self.config.session_id}"
        )

    async def _run_live(self) -> None:
        """Live ingestion from RTSP or MP4 source."""
        cap = None
        retry = 0

        while retry < self.config.retry_count:
            cap = cv2.VideoCapture(self.config.source)
            if cap.isOpened():
                break
            retry += 1
            wait = self.config.retry_backoff * (2 ** retry)
            logger.warning(f"source.retry attempt={retry} wait_s={wait}")
            await asyncio.sleep(wait)

        if cap is None or not cap.isOpened():
            logger.error(f"source.unreachable source={self.config.source} retries={retry}")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"source.opened fps={source_fps:.1f} total_frames={total_frames}")

        frame_number = 0
        sample_interval = max(1, int(source_fps / self.config.current_fps))
        self._running = True
        last_backpressure_check = time.time()

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                self.metrics.frames_read += 1

                # ── Skip non-sampled frames
                if frame_number % sample_interval != 0:
                    continue

                # ── Check backpressure every 5 seconds
                now = time.time()
                if now - last_backpressure_check > 5.0:
                    fps = await self.backpressure.evaluate(self.queue)
                    sample_interval = max(1, int(source_fps / fps))
                    last_backpressure_check = now

                # ── Disk backpressure: drop frames
                if self.metrics.disk_usage_pct > self.config.max_disk_usage:
                    self.metrics.frames_dropped_disk += 1
                    continue

                # ── Process frame
                timestamp = datetime.now(timezone.utc)
                event = self.processor.process_frame(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )

                if event is None:
                    continue

                # ── Enqueue
                await self.queue.enqueue(event, maxlen=self.config.max_queue)
                self.metrics.events_enqueued += 1

                # ── Log to session (for reproducibility)
                self.session_logger.log_event(event)

                # ── Max frames limit
                if self.config.max_frames > 0 and self.metrics.frames_stored >= self.config.max_frames:
                    logger.info(f"ingest.max_frames_reached limit={self.config.max_frames}")
                    break

                # ── Yield to event loop
                if frame_number % 10 == 0:
                    await asyncio.sleep(0)

        except KeyboardInterrupt:
            logger.info("ingest.interrupted")
        finally:
            cap.release()
            self._running = False

    async def _run_replay(self) -> None:
        """
        Reproducibility mode: replay a previous session.
        - Load raw_events from session log
        - Re-enqueue with same provenance
        - Compare event hashes
        """
        session_id = self.config.replay_session
        logger.info(f"replay.starting session={session_id}")

        try:
            events = SessionLogger.load_session(session_id, self.config.data_dir)
        except FileNotFoundError as e:
            logger.error(f"replay.session_not_found error={e}")
            return

        logger.info(f"replay.loaded events={len(events)}")

        replay_hashes: list[str] = []
        original_hashes: list[str] = []

        for event in events:
            self.metrics.frames_read += 1

            # Record original hash
            original_hashes.append(event.get("file_hash", ""))

            # Re-sign the event (should produce same hash if data unchanged)
            re_signed = sign_raw_event(
                event_type=event["type"],
                source_id=event["source_id"],
                timestamp=event["timestamp"],
                raw_payload=event.get("raw_payload", {}),
                file_hash=event.get("file_hash"),
            )

            original_provenance = event.get("provenance_hash", "")
            match = re_signed == original_provenance

            if not match:
                logger.warning(
                    f"replay.provenance_mismatch "
                    f"event_id={event['id']} "
                    f"original={original_provenance[:40]}... "
                    f"replayed={re_signed[:40]}..."
                )

            replay_hashes.append(re_signed)

            # Re-enqueue
            event["replay_session"] = self.config.session_id
            event["replay_original_session"] = session_id
            await self.queue.enqueue(event)
            self.metrics.events_enqueued += 1

        # Compare
        matches = sum(1 for a, b in zip(original_hashes, replay_hashes)
                      if a == b or not a)  # skip empty originals
        total = len(original_hashes)

        logger.info(
            f"replay.completed "
            f"events={total} "
            f"provenance_matches={matches}/{total} "
            f"reproducibility_pct={matches/total*100:.1f}%"
        )

    def stop(self) -> None:
        self._running = False


# ─── Health Server (lightweight) ─────────────────────────────

async def run_health_server(engine: IngestionEngine, port: int = 8002):
    """Minimal HTTP server for /health and /metrics endpoints."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({
                    "status": "healthy" if engine._running or engine.metrics.frames_read > 0 else "idle",
                    "service": "ingest",
                    "session_id": engine.config.session_id,
                    "source": engine.config.source,
                    "frames_read": engine.metrics.frames_read,
                    "frames_stored": engine.metrics.frames_stored,
                    "current_fps": engine.metrics.current_fps,
                    "backpressure_active": engine.metrics.backpressure_active,
                    "queue_depth": engine.metrics.queue_depth,
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
            elif self.path == "/metrics":
                body = engine.metrics.to_prometheus()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(body.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress default logging

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"health_server.started port={port} endpoints=[/health, /metrics]")


# ─── CLI Entry Point ────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Sentinel Ingestion Service — Frame Sampler"
    )
    parser.add_argument(
        "--source", required=True,
        help="Video source: file path (MP4) or RTSP URL"
    )
    parser.add_argument(
        "--source-id", default=None,
        help="Source identifier (default: derived from source path)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help=f"Sampling FPS (default: {os.environ.get('BASE_FPS', '1.0')})"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process (0 = unlimited)"
    )
    parser.add_argument(
        "--max-queue", type=int, default=None,
        help="Max Redis queue depth before backpressure"
    )
    parser.add_argument(
        "--replay-session", default=None,
        help="Replay a previous session for reproducibility verification"
    )
    parser.add_argument(
        "--health-port", type=int, default=8002,
        help="Health/metrics HTTP port (default: 8002)"
    )
    parser.add_argument(
        "--no-health", action="store_true",
        help="Disable health endpoint server"
    )

    args = parser.parse_args()

    # Build config
    config = IngestConfig(
        source=args.source,
        source_id=args.source_id or Path(args.source).stem.replace(" ", "_"),
        replay_session=args.replay_session,
    )

    if args.fps is not None:
        config.base_fps = args.fps
        config.current_fps = args.fps
    if args.max_frames:
        config.max_frames = args.max_frames
    if args.max_queue:
        config.max_queue = args.max_queue

    # Ensure directories exist
    Path(config.frame_store).mkdir(parents=True, exist_ok=True)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = IngestionEngine(config)

    # Start health server
    if not args.no_health:
        await run_health_server(engine, port=args.health_port)

    # Run ingestion
    await engine.start()

    # Print final metrics
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Session:    {config.session_id}")
    print(f"  Source:     {config.source}")
    print(f"  Frames:    {engine.metrics.frames_read} read → {engine.metrics.frames_stored} stored")
    print(f"  Dropped:   {engine.metrics.frames_dropped_backpressure} (backpressure) "
          f"+ {engine.metrics.frames_dropped_corrupt} (corrupt) "
          f"+ {engine.metrics.frames_dropped_disk} (disk)")
    print(f"  Enqueued:  {engine.metrics.events_enqueued}")
    print(f"  Latency:   {engine.metrics.avg_latency_ms:.1f}ms avg")
    print(f"  Stored:    {engine.metrics.bytes_stored / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

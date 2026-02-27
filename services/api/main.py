"""
Sentinel — FastAPI Backend (Sprint 1 — boots with available packages only)

Graceful degradation:
- structlog → stdlib logging
- redis → in-memory queue
- neo4j → demo data
- postgres → demo data
"""
import time
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# ─── Logging (structlog fallback) ────────────────────────────
try:
    import structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )
    logger = logging.getLogger("sentinel")

# Service status tracker
_status = {"postgres": "down", "neo4j": "down", "redis": "down"}
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup — connect what's available, skip what's not."""
    logger.info("sentinel.starting — Sprint 1 validation mode")

    for svc, init_fn in [
        ("postgres", "_try_postgres"),
        ("neo4j", "_try_neo4j"),
        ("redis", "_try_redis"),
    ]:
        try:
            if svc == "postgres":
                from db.postgres import init_postgres
                await init_postgres()
            elif svc == "neo4j":
                from db.neo4j import init_neo4j
                await init_neo4j()
            elif svc == "redis":
                from db.redis import init_redis
                await init_redis()
            _status[svc] = "up"
        except Exception as e:
            _status[svc] = "down"
            logger.warning(f"{svc}.unavailable — using demo fallback — {e}")

    up = [k for k, v in _status.items() if v == "up"]
    logger.info(f"sentinel.ready — services_up={up} mode={'full' if len(up)==3 else 'demo'}")
    yield

    for svc in ["postgres", "neo4j", "redis"]:
        if _status[svc] == "up":
            try:
                if svc == "postgres":
                    from db.postgres import close_postgres
                    await close_postgres()
                elif svc == "neo4j":
                    from db.neo4j import close_neo4j
                    await close_neo4j()
                elif svc == "redis":
                    from db.redis import close_redis
                    await close_redis()
            except Exception:
                pass
    logger.info("sentinel.shutdown")


app = FastAPI(
    title="Sentinel — Spatiotemporal Entity Reasoning Engine",
    description="3-layer data model: RawEvent → DerivedEvent → Inference",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Metrics (prometheus_client available) ───────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

    req_counter = Counter("sentinel_api_requests_total", "Total API requests", ["method", "path", "status"])
    req_latency = Histogram("sentinel_api_latency_seconds", "API latency", ["method", "path"],
                            buckets=[.01, .05, .1, .25, .5, 1, 2.5])
    frames_total = Counter("sentinel_frames_processed_total", "Frames processed")
    detections_total = Counter("sentinel_detections_total", "Detections produced")
    queue_depth = Gauge("sentinel_queue_depth", "Queue depth")
    _prom = True
except ImportError:
    _prom = False


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur = time.perf_counter() - start
    if _prom and not request.url.path.startswith(("/docs", "/openapi", "/redoc")):
        req_counter.labels(method=request.method, path=request.url.path, status=str(response.status_code)).inc()
        req_latency.labels(method=request.method, path=request.url.path).observe(dur)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "sentinel-api",
        "version": "1.0.0",
        "uptime_s": round(time.time() - _start_time, 1),
        "services": _status,
        "mode": "full" if all(v == "up" for v in _status.values()) else "demo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status")
async def status():
    return {
        "service": "sentinel-api",
        "version": "1.0.0",
        "data_model": "3-layer (raw → derived → inference)",
        "services": _status,
        "sprint": "1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if _prom:
    from fastapi.responses import Response

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ═══════════════════════════════════════════════════════════
# ROUTERS
# ═══════════════════════════════════════════════════════════

from routers import events, entities, timeline, graph, hypotheses, audit, health as health_router

# UI (embedded HTML)
try:
    from routers.ui import router as ui_router
    app.include_router(ui_router, tags=["UI"])
except Exception:
    pass

app.include_router(health_router.router, tags=["Health"])
app.include_router(events.router, prefix="/api/v1/events", tags=["Events"])
app.include_router(entities.router, prefix="/api/v1/entities", tags=["Entities"])
app.include_router(timeline.router, prefix="/api/v1/timeline", tags=["Timeline"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])
app.include_router(hypotheses.router, prefix="/api/v1/hypotheses", tags=["Hypotheses"])
app.include_router(audit.router, prefix="/api/v1/audit", tags=["Audit"])

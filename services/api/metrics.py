"""
Sentinel — Prometheus Metrics

Exposes /metrics endpoint for observability.
Tracks queue depth, inference latency, model versions, backpressure.
"""
import time
from functools import wraps
from typing import Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import APIRouter, Response


router = APIRouter()

# ═══════════════════════════════════════════════════════════
# COUNTERS
# ═══════════════════════════════════════════════════════════

raw_events_ingested = Counter(
    "sentinel_raw_events_ingested_total",
    "Total raw events ingested",
    ["type", "source"]
)

derived_events_created = Counter(
    "sentinel_derived_events_total",
    "Total derived events created by models",
    ["type", "model_name", "model_version"]
)

entity_resolutions = Counter(
    "sentinel_entity_resolutions_total",
    "Total entity resolutions performed",
    ["method"]
)

hypothesis_operations = Counter(
    "sentinel_hypothesis_operations_total",
    "Hypothesis create/score/verify operations",
    ["operation", "stage"]
)

audit_log_entries = Counter(
    "sentinel_audit_entries_total",
    "Total audit log entries",
    ["action", "actor"]
)

api_requests = Counter(
    "sentinel_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

# ═══════════════════════════════════════════════════════════
# HISTOGRAMS
# ═══════════════════════════════════════════════════════════

inference_latency = Histogram(
    "sentinel_inference_latency_seconds",
    "Model inference latency",
    ["model_name", "model_version"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

embedding_search_latency = Histogram(
    "sentinel_embedding_search_latency_seconds",
    "pgvector ANN search latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

api_request_latency = Histogram(
    "sentinel_api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# ═══════════════════════════════════════════════════════════
# GAUGES
# ═══════════════════════════════════════════════════════════

queue_depth = Gauge(
    "sentinel_queue_depth",
    "Current Redis stream queue depth",
    ["stream"]
)

active_workers = Gauge(
    "sentinel_active_workers",
    "Number of active processing workers",
    ["service"]
)

models_loaded = Gauge(
    "sentinel_models_loaded",
    "Number of models currently loaded",
    ["service"]
)

backpressure_drops = Counter(
    "sentinel_backpressure_drops_total",
    "Frames/events dropped due to backpressure",
    ["stream"]
)

entity_count = Gauge(
    "sentinel_entity_count",
    "Total entities by type and resolution status",
    ["type", "is_resolved"]
)

# ═══════════════════════════════════════════════════════════
# INFO
# ═══════════════════════════════════════════════════════════

build_info = Info(
    "sentinel_build",
    "Build information"
)
build_info.info({
    "version": "1.0.0",
    "service": "api",
    "data_model_version": "3-layer",
})


# ═══════════════════════════════════════════════════════════
# METRICS ENDPOINT
# ═══════════════════════════════════════════════════════════

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ═══════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════

def track_inference(model_name: str, model_version: str):
    """Decorator to track inference latency and count."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                derived_events_created.labels(
                    type="inference",
                    model_name=model_name,
                    model_version=model_version,
                ).inc()
                return result
            finally:
                duration = time.perf_counter() - start
                inference_latency.labels(
                    model_name=model_name,
                    model_version=model_version,
                ).observe(duration)
        return wrapper
    return decorator

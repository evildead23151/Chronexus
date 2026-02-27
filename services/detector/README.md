# Sentinel Detector Service

**⚠ AGPL-3.0 LICENSE ISOLATION**

This service runs YOLOv8 (AGPL-3.0) inside its own Docker container.
It communicates with the main system **only** via Redis Streams.
This architectural boundary prevents AGPL license infection of the rest of the codebase.

## Alternative (MIT-licensed)
If commercializing, replace with:
- RT-DETR (Apache-2.0)
- DAMO-YOLO (Apache-2.0)

## Interface
- **Input**: Redis Stream `sentinel:raw_events` (frame type with file_path)
- **Output**: Redis Stream `sentinel:derived_events` (detection type)
- **Metrics**: `GET /metrics` (Prometheus)
- **Health**: `GET /health`

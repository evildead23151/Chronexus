# Sentinel Ingestion Service

Connectors for RTSP streams, MP4 files, and metadata feeds.
Frame sampling with backpressure monitoring.

## Interface
- **Input**: REST `POST /api/v1/ingest/source` or RTSP URL
- **Output**: Redis Stream `sentinel:raw_events`
- **Storage**: Frames → `./data/frames/<source>/<timestamp>.jpg`
- **Provenance**: Every frame gets HMAC-signed at ingestion
- **Backpressure**: Drop-oldest strategy, queue max 10,000, metric exposed

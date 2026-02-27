# 🛡️ Sentinel — Incident Reconstruction Engine v1

> **Confidence-weighted timeline & entity graph for incident analysis.**

Sentinel ingests CCTV frames, public traffic camera streams, synthetic data, and structured metadata (phone pings, license-plate reads, timestamps) to produce a confidence-weighted timeline and an entity graph linking people, vehicles, places, and events. An analyst UI enables timeline + graph exploration and exports signed evidence packages.

---

## ⚡ Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for UI development)
- Python 3.11+ (for service development)
- GPU recommended (but CPU fallback supported via ONNX)

### 1. Start Infrastructure
```bash
cd infra
docker-compose up -d
```
This starts PostgreSQL+PostGIS, Neo4j, and Redis.

### 2. Start Backend API
```bash
cd services/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start Frontend
```bash
cd ui
npm install
npm run dev
```

### 4. Open Analyst UI
Navigate to `http://localhost:5173`

---

## 🏗️ Architecture

```
Data Sources → Ingestion → Processing Pipeline → Storage → Reasoning → Analyst UI
     ↓              ↓              ↓                 ↓          ↓           ↓
  CCTV/RTSP     Frame       YOLOv8/ByteTrack    Neo4j      Bayesian    Timeline
  Metadata      Sampler     InsightFace          PostGIS    Scoring     Graph
  Synthetic     Redis Q     PaddleOCR            Redis      Rules       Evidence
```

## 📁 Project Structure

```
sentinel/
├── infra/                   # Docker, DB init scripts
├── services/
│   ├── api/                 # FastAPI backend
│   ├── ingest/              # Connectors & frame sampler
│   ├── detector/            # YOLOv8 wrappers
│   ├── tracker/             # ByteTrack
│   ├── embeddings/          # Face & vehicle embeddings
│   ├── lpr/                 # License plate recognition
│   ├── graph_ingest/        # Neo4j ingestion
│   ├── reasoner/            # Scoring engine
│   └── normalizer/          # Event normalization
├── ui/                      # React + TypeScript frontend
├── experiments/             # Notebooks, eval scripts
├── data/                    # Sample datasets
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── tests/                   # Integration tests
```

## 🔒 Ethics & Privacy

- **Human-in-the-loop**: No automated accusations — only ranked hypotheses with provenance
- **Audit logs**: Immutable logs for every inference and data action
- **Data minimization**: Retain only what's needed; enable deletion workflows
- **Bias testing**: Evaluate models across demographics; flag uncertain results

## 📜 License

Private — All rights reserved.

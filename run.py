"""
Sentinel — Standalone Server

Boots the entire stack from a single Python command:
  python run.py

Starts:
  - FastAPI API on port 8000 (with embedded UI)
  - No Docker, Node, or external services needed
  - In-memory data stores for Sprint 0/1 demo
  - Serves the Analyst UI at /

Usage:
  cd P11
  pip install fastapi uvicorn pydantic pydantic-settings structlog prometheus-client opencv-python-headless numpy
  python run.py
"""
import os
import sys

# Set working directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Add service paths
sys.path.insert(0, os.path.join(project_root, "services", "api"))

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  🛡️  SENTINEL — Spatiotemporal Entity Reasoning Engine")
    print("=" * 60)
    print(f"  API:      http://localhost:8000/docs")
    print(f"  UI:       http://localhost:8000/")
    print(f"  Health:   http://localhost:8000/health")
    print(f"  Metrics:  http://localhost:8000/metrics")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.join(project_root, "services", "api")],
    )

"""
Sentinel — Health Check Router
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def root():
    return {
        "service": "sentinel",
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "services": {
            "api": "up",
            "postgres": "pending",
            "neo4j": "pending",
            "redis": "pending",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

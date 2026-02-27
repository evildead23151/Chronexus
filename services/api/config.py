"""
Sentinel — Application Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "sentinel"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_PORT: int = 8000
    APP_HOST: str = "0.0.0.0"

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sentinel"
    POSTGRES_USER: str = "sentinel"
    POSTGRES_PASSWORD: str = "sentinel_dev_2026"
    DATABASE_URL: str = "postgresql+asyncpg://sentinel:sentinel_dev_2026@localhost:5432/sentinel"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "sentinel_dev_2026"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"

    # Streams
    FRAME_STREAM: str = "sentinel:frames"
    EVENT_STREAM: str = "sentinel:events"
    DETECTION_STREAM: str = "sentinel:detections"
    GRAPH_STREAM: str = "sentinel:graph"

    # Storage
    FRAME_STORE_PATH: str = "./data/frames"
    EVIDENCE_STORE_PATH: str = "./data/evidence"

    # Models
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.4
    FACE_MODEL: str = "buffalo_l"
    LPR_MODEL: str = "en_PP-OCRv4"

    # Frame Sampling
    DEFAULT_FPS: float = 0.5
    INCIDENT_FPS: float = 2.0
    MAX_QUEUE_LENGTH: int = 10000

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

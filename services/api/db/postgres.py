"""
Sentinel — PostgreSQL + PostGIS Connection
"""
from log import get_logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from config import settings

logger = get_logger()

engine = None
async_session_factory = None


async def init_postgres():
    """Initialize PostgreSQL connection pool."""
    global engine, async_session_factory
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.APP_DEBUG,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    async_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("postgres.connected", host=settings.POSTGRES_HOST)


async def close_postgres():
    """Close PostgreSQL connection pool."""
    global engine
    if engine:
        await engine.dispose()
        logger.info("postgres.disconnected")


async def get_db() -> AsyncSession:
    """Dependency: get a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


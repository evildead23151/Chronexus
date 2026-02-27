"""
Sentinel — Neo4j Graph Database Connection
"""
from log import get_logger
from neo4j import AsyncGraphDatabase
from config import settings

logger = get_logger()

driver = None


async def init_neo4j():
    """Initialize Neo4j driver."""
    global driver
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    # Verify connectivity
    try:
        await driver.verify_connectivity()
        logger.info("neo4j.connected", uri=settings.NEO4J_URI)
    except Exception as e:
        logger.warning("neo4j.connection_failed", error=str(e))


async def close_neo4j():
    """Close Neo4j driver."""
    global driver
    if driver:
        await driver.close()
        logger.info("neo4j.disconnected")


async def get_neo4j_session():
    """Dependency: get a Neo4j session."""
    async with driver.session() as session:
        yield session


async def run_cypher(query: str, parameters: dict = None):
    """Execute a Cypher query and return results."""
    async with driver.session() as session:
        result = await session.run(query, parameters or {})
        records = [record.data() async for record in result]
        return records


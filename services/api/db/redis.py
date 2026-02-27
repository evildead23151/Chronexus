"""
Sentinel — Redis Connection (Streams)
"""
from log import get_logger
import redis.asyncio as aioredis
from config import settings

logger = get_logger()

redis_client = None


async def init_redis():
    """Initialize Redis connection."""
    global redis_client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        max_connections=20,
    )
    try:
        await redis_client.ping()
        logger.info("redis.connected", host=settings.REDIS_HOST)
    except Exception as e:
        logger.warning("redis.connection_failed", error=str(e))


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("redis.disconnected")


def get_redis():
    """Dependency: get the Redis client."""
    return redis_client


async def publish_to_stream(stream: str, data: dict, maxlen: int = None):
    """Publish a message to a Redis Stream."""
    maxlen = maxlen or settings.MAX_QUEUE_LENGTH
    await redis_client.xadd(stream, data, maxlen=maxlen)


async def read_from_stream(stream: str, group: str, consumer: str, count: int = 10):
    """Read messages from a Redis Stream consumer group."""
    try:
        messages = await redis_client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=count,
            block=5000,
        )
        return messages
    except aioredis.ResponseError:
        # Group doesn't exist, create it
        try:
            await redis_client.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError:
            pass
        return []


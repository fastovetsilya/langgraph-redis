from typing import Any, Optional, TypeVar, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

RedisClientType = TypeVar(
    "RedisClientType", bound=Union[Redis, AsyncRedis, RedisCluster, AsyncRedisCluster]
)
IndexType = TypeVar("IndexType", bound=Union[None, type(None)])
MetadataInput = Optional[dict[str, Any]]

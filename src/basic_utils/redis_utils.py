from __future__ import annotations

from typing import Any

from redis.asyncio import Redis

from basic_utils.config import settings


def get_redis_client() -> Redis:
    """Возвращает асинхронный Redis-клиент."""
    if settings.redis_url:
        return Redis.from_url(settings.redis_url, decode_responses=True)

    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


class RedisUtils:
    """Обертка над Redis с поддержкой префикса ключей."""

    def __init__(self, prefix: str = "") -> None:
        self._redis = get_redis_client()
        self._prefix = prefix

    def _build_key(self, key: str) -> str:
        if not self._prefix:
            return key
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> str | None:
        return await self._redis.get(self._build_key(key))

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        result = await self._redis.set(self._build_key(key), value, ex=ex)
        return bool(result)

    async def close(self) -> None:
        await self._redis.aclose()


def get_redis_utils(prefix: str = "") -> RedisUtils:
    """Возвращает Redis-обертку с заданным префиксом."""
    return RedisUtils(prefix=prefix)

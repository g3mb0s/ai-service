from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp
import jwt
from fastapi import Depends, HTTPException, status
from redis.exceptions import RedisError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from basic_utils.config import settings
from basic_utils.jwt_utils import decode_jwt
from basic_utils.redis_utils import get_redis_utils


PublicKeyGetter = Callable[[str | None], str | Awaitable[str]]

_bearer_scheme = HTTPBearer(auto_error=True)
_public_key_cache: dict[str, tuple[float, str]] = {}
_public_key_lock = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    id: UUID
    email: str | None
    role: str | None

async def get_public_key_from_text(_: str | None = None) -> str:
    """Возвращает публичный ключ из конфигурации."""
    return settings.jwt_public_key


async def get_public_key_from_server(key_id: str | None = None) -> str:
    """Возвращает публичный ключ через локальный/Redis-кэш и auth service."""

    cache_key = key_id or settings.jwt_default_key_id
    local_cached = _public_key_cache.get(cache_key)
    if local_cached and local_cached[0] > time.monotonic():
        return local_cached[1]

    async with _public_key_lock:
        local_cached = _public_key_cache.get(cache_key)
        if local_cached and local_cached[0] > time.monotonic():
            return local_cached[1]

        redis = (
            get_redis_utils(prefix=settings.jwt_public_key_cache_prefix)
            if settings.redis_url
            else None
        )
        if redis is not None:
            try:
                cached_key = await redis.get(cache_key)
                if cached_key:
                    _cache_public_key(cache_key, cached_key)
                    return cached_key
            except RedisError:
                pass

        try:
            timeout = aiohttp.ClientTimeout(
                total=settings.jwt_public_key_request_timeout
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(settings.jwt_public_key_url) as response:
                    response.raise_for_status()
                    payload = await response.json()

            public_key = payload.get(settings.jwt_public_key_response_field)
            if not isinstance(public_key, str) or not public_key.strip():
                raise ValueError(
                    "Public key is missing in JWT public key server response."
                )

            _cache_public_key(cache_key, public_key)
            if redis is not None:
                try:
                    await redis.set(
                        cache_key,
                        public_key,
                        ex=settings.jwt_public_key_cache_ttl,
                    )
                except RedisError:
                    pass
            return public_key
        finally:
            if redis is not None:
                await redis.close()


def _cache_public_key(cache_key: str, public_key: str) -> None:
    expires_at = time.monotonic() + settings.jwt_public_key_cache_ttl
    _public_key_cache[cache_key] = (expires_at, public_key)


def get_decoded_jwt_dependency(
    public_key_getter: PublicKeyGetter,
    *,
    algorithms: list[str] | None = None,
    audience: str | None = None,
    issuer: str | None = None,
    options: dict[str, Any] | None = None,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Создает dependency, которая возвращает декодированный JWT."""

    async def _dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    ) -> dict[str, Any]:
        try:
            return await decode_jwt(
                credentials.credentials,
                public_key_getter,
                algorithms=algorithms,
                audience=audience,
                issuer=issuer,
                options=options,
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired access token",
            ) from exc
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service is unavailable",
            ) from exc

    return _dependency


_get_decoded_jwt = get_decoded_jwt_dependency(get_public_key_from_server)


async def get_current_user(
    payload: dict[str, Any] = Depends(_get_decoded_jwt),
) -> AuthenticatedUser:
    """Преобразует JWT payload в пользователя текущего запроса."""

    try:
        user_id = UUID(str(payload["sub"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT does not contain a valid subject",
        ) from exc

    email = payload.get("email")
    role = payload.get("role")
    return AuthenticatedUser(
        id=user_id,
        email=email if isinstance(email, str) else None,
        role=role if isinstance(role, str) else None,
    )


async def get_manager_or_admin(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Разрешает доступ только контент-менеджерам и администраторам."""

    if user.role not in {"manager", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or admin role is required",
        )
    return user


async def get_admin(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """Разрешает доступ только администраторам."""

    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is required",
        )
    return user

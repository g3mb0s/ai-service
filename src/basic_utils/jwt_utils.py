from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import jwt

from basic_utils.config import settings


PublicKeyGetter = Callable[[str | None], str | Awaitable[str]]


async def decode_jwt(
    token: str,
    public_key_getter: PublicKeyGetter,
    *,
    algorithms: list[str] | None = None,
    audience: str | None = None,
    issuer: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Декодирует и валидирует JWT с внешним источником публичного ключа."""
    unverified_header = jwt.get_unverified_header(token)
    key_id = unverified_header.get("kid")
    public_key = await public_key_getter(key_id)

    decode_options = {"verify_signature": True, "verify_exp": settings.jwt_verify_exp}
    if options:
        decode_options.update(options)

    return jwt.decode(
        token,
        public_key,
        algorithms=algorithms or settings.jwt_algorithms,
        audience=audience or settings.jwt_audience,
        issuer=issuer or settings.jwt_issuer,
        options=decode_options,
    )

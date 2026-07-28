from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from urllib.parse import quote, urlparse

from minio import Minio

from basic_utils.config import settings


class ObjectStorage:
    def __init__(self) -> None:
        endpoint = urlparse(settings.s3_endpoint_url)
        if not endpoint.hostname:
            raise ValueError("S3_ENDPOINT_URL must include a hostname")
        host = endpoint.hostname
        if endpoint.port:
            host = f"{host}:{endpoint.port}"
        self._client = Minio(
            host,
            access_key=settings.s3_access_key_id,
            secret_key=settings.s3_secret_access_key,
            secure=endpoint.scheme == "https",
            region=settings.s3_region_name,
        )
        self.bucket = settings.s3_bucket_name

    def put(self, object_key: str, content: bytes, content_type: str) -> str:
        self._client.put_object(
            self.bucket,
            object_key,
            BytesIO(content),
            length=len(content),
            content_type=content_type,
        )
        encoded_key = quote(object_key, safe="/")
        return (
            f"{settings.s3_public_url.rstrip('/')}/"
            f"{quote(self.bucket, safe='')}/{encoded_key}"
        )

    def remove(self, object_key: str) -> None:
        self._client.remove_object(self.bucket, object_key)


@lru_cache
def get_object_storage() -> ObjectStorage:
    return ObjectStorage()

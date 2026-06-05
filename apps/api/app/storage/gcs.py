import asyncio
from datetime import datetime
from typing import BinaryIO

from app.storage.base import ObjectStorage


class GCSStorage(ObjectStorage):
    """Google Cloud Storage provider adapter."""

    def __init__(self, bucket_name: str, project_id: str | None = None) -> None:
        self.bucket_name = bucket_name
        self.project_id = project_id
        self._client = None
        self._bucket = None

    def _get_bucket(self):
        if self._bucket is None:
            try:
                from google.cloud import storage
            except ImportError as err:
                raise RuntimeError(
                    "google-cloud-storage package is required for GCSStorage"
                ) from err
            self._client = storage.Client(project=self.project_id)
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        def _upload() -> str:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            blob.upload_from_string(data, content_type=content_type)
            return f"gs://{self.bucket_name}/{key}"

        return await asyncio.to_thread(_upload)

    async def put_stream(
        self, key: str, source: BinaryIO, content_type: str = "application/pdf"
    ) -> str:
        def _upload() -> str:
            source.seek(0)
            blob = self._get_bucket().blob(key)
            blob.upload_from_file(source, rewind=True, content_type=content_type)
            return f"gs://{self.bucket_name}/{key}"

        return await asyncio.to_thread(_upload)

    async def get(self, key: str) -> bytes:
        def _download() -> bytes:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            if not blob.exists():
                raise FileNotFoundError(f"Object {key} not found in GCS bucket {self.bucket_name}")
            return blob.download_as_bytes()

        return await asyncio.to_thread(_download)

    async def exists(self, key: str) -> bool:
        def _check() -> bool:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            return blob.exists()

        return await asyncio.to_thread(_check)

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            if blob.exists():
                blob.delete()

        await asyncio.to_thread(_delete)

    async def open_stream(self, key: str, chunk_size: int = 64 * 1024):
        blob = self._get_bucket().blob(key)
        reader = await asyncio.to_thread(blob.open, "rb", chunk_size=chunk_size)
        try:
            while chunk := await asyncio.to_thread(reader.read, chunk_size):
                yield chunk
        finally:
            await asyncio.to_thread(reader.close)

    async def list_objects(
        self, prefix: str = "", limit: int = 500
    ) -> list[tuple[str, datetime | None]]:
        def _list() -> list[tuple[str, datetime | None]]:
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=prefix if prefix else None, max_results=limit)
            return [(b.name, b.updated) for b in blobs]

        return await asyncio.to_thread(_list)

    async def list_keys(self, prefix: str = "", limit: int = 500) -> list[str]:
        objs = await self.list_objects(prefix=prefix, limit=limit)
        return [k for k, _ in objs]

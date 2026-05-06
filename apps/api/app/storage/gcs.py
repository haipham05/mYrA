import asyncio

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

    async def get(self, key: str) -> bytes:
        def _download() -> bytes:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            if not blob.exists():
                raise FileNotFoundError(f"Object {key} not found in GCS bucket {self.bucket_name}")
            return blob.download_as_bytes()

        return await asyncio.to_thread(_download)

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            bucket = self._get_bucket()
            blob = bucket.blob(key)
            if blob.exists():
                blob.delete()

        await asyncio.to_thread(_delete)

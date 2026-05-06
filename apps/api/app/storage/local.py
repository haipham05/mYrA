import os
from pathlib import Path

from app.storage.base import ObjectStorage


class LocalStorage(ObjectStorage):
    """Local filesystem storage implementation."""

    def __init__(self, base_dir: str = "data/storage") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        clean_key = key.lstrip("/")
        return self.base_dir / clean_key

    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        file_path = self._resolve_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return str(file_path)

    async def get(self, key: str) -> bytes:
        file_path = self._resolve_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Key not found in local storage: {key}")
        return file_path.read_bytes()

    async def delete(self, key: str) -> None:
        file_path = self._resolve_path(key)
        if file_path.exists():
            os.remove(file_path)


class MemoryStorage(ObjectStorage):
    """In-memory storage for unit testing without disk I/O."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        self._store[key] = data
        return f"memory://{key}"

    async def get(self, key: str) -> bytes:
        if key not in self._store:
            raise FileNotFoundError(f"Key not found in memory storage: {key}")
        return self._store[key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

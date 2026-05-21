import asyncio
import os
from pathlib import Path
from typing import BinaryIO

from app.storage.base import ObjectStorage


class LocalStorage(ObjectStorage):
    """Local filesystem storage implementation."""

    def __init__(self, base_dir: str = "data/storage") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        base = self.base_dir.resolve()
        supplied = Path(key)
        if supplied.is_absolute():
            try:
                supplied = supplied.relative_to(base)
            except ValueError as err:
                raise ValueError("Storage key is outside the local storage directory") from err
        elif supplied.parts[: len(self.base_dir.parts)] == self.base_dir.parts:
            # Read legacy rows whose storage_path included the relative base dir.
            supplied = Path(*supplied.parts[len(self.base_dir.parts) :])
        resolved = (base / supplied).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError("Storage key is outside the local storage directory")
        return resolved

    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        file_path = self._resolve_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return key

    async def put_stream(
        self, key: str, source: BinaryIO, content_type: str = "application/pdf"
    ) -> str:
        def _copy() -> None:
            file_path = self._resolve_path(key)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)

        await asyncio.to_thread(_copy)
        return key

    async def get(self, key: str) -> bytes:
        file_path = self._resolve_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Key not found in local storage: {key}")
        return file_path.read_bytes()

    async def delete(self, key: str) -> None:
        file_path = self._resolve_path(key)
        if file_path.exists():
            os.remove(file_path)

    async def exists(self, key: str) -> bool:
        return self._resolve_path(key).exists()

    async def open_stream(self, key: str, chunk_size: int = 64 * 1024):
        file_path = self._resolve_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Key not found in local storage: {key}")
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk


class MemoryStorage(ObjectStorage):
    """In-memory storage for unit testing without disk I/O."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        self._store[key] = data
        return f"memory://{key}"

    async def get(self, key: str) -> bytes:
        if key not in self._store:
            raise FileNotFoundError(f"Key not found in memory storage: {key}")
        return self._store[key]

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

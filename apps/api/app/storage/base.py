from abc import ABC, abstractmethod


class ObjectStorage(ABC):
    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        """Store bytes under key and return storage path / identifier."""
        pass

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Retrieve bytes for key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete object under key."""
        pass

    async def exists(self, key: str) -> bool:
        """Check if an object exists under key."""
        try:
            await self.get(key)
            return True
        except FileNotFoundError:
            return False

    async def open_stream(self, key: str, chunk_size: int = 64 * 1024):
        """Stream chunks of bytes for key."""
        data = await self.get(key)
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

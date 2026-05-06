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

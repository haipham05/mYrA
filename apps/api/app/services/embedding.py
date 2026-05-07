import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic 1024-dimensional embedding provider for testing and offline development."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    @property
    def model_name(self) -> str:
        return "deterministic-fake"

    @property
    def model_version(self) -> str:
        return "v1"

    def _hash_to_vector(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dimension

        # Generate pseudo-random vector from SHA-256 seed
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self.dimension):
            byte_val = seed_bytes[i % len(seed_bytes)]
            val = ((byte_val ^ (i % 256)) / 128.0) - 1.0
            vec.append(val)

        # Normalize to unit length (cosine similarity ready)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """BGE-M3 local embedding provider."""

    def __init__(self, model_name: str = "BAAI/bge-m3", model_version: str = "v1") -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as err:
                raise RuntimeError(
                    f"Production embedding provider {self.model_name} requested "
                    f"but could not be loaded: {err}"
                ) from err
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


_default_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _default_embedding_provider
    if _default_embedding_provider is None:
        import os

        provider_type = os.getenv("MYRA_EMBEDDING_PROVIDER", "deterministic").lower()
        if provider_type in ("bge-m3", "bge", "production"):
            _default_embedding_provider = BGEM3EmbeddingProvider()
        else:
            _default_embedding_provider = DeterministicEmbeddingProvider()
    return _default_embedding_provider


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    global _default_embedding_provider
    _default_embedding_provider = provider

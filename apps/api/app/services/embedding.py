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

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model_version: str = "5617a9f61b028005a4858fdac845db406aefb181",
    ) -> None:
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

                self._model = SentenceTransformer(
                    self.model_name, revision=self.model_version, local_files_only=True
                )
            except Exception as err:
                raise RuntimeError(
                    f"Pinned embedding model {self.model_name}@{self.model_version} is not "
                    "available locally. Provision the approved model cache before indexing."
                ) from err
        return self._model

    @staticmethod
    def _validated_vector(value) -> list[float]:
        vector = value.tolist()
        if len(vector) != 1024:
            raise RuntimeError(f"BGE-M3 returned {len(vector)} dimensions; expected 1024")
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [self._validated_vector(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return self._validated_vector(embedding)


_default_embedding_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _default_embedding_provider
    if _default_embedding_provider is None:
        import os

        provider_type = os.getenv("MYRA_EMBEDDING_PROVIDER", "bge-m3").lower()
        if provider_type == "bge-m3":
            _default_embedding_provider = BGEM3EmbeddingProvider()
        elif provider_type in ("test", "demo", "deterministic"):
            _default_embedding_provider = DeterministicEmbeddingProvider()
        else:
            raise ValueError(f"Unknown MYRA_EMBEDDING_PROVIDER: {provider_type}")
    return _default_embedding_provider


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    global _default_embedding_provider
    _default_embedding_provider = provider

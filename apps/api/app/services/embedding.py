import hashlib
import json
import math
from abc import ABC, abstractmethod

from app.services.cache import get_cache

EMBEDDING_CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_key(provider: "EmbeddingProvider", operation: str, text: str) -> str:
    """Build a key from model identity and an exact-input digest, never the text itself."""
    identity = json.dumps(
        [operation, provider.model_name, provider.model_version],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    model_digest = hashlib.sha256(identity).hexdigest()
    input_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embedding:v1:{model_digest}:{input_digest}"


def _expected_dimension(provider: "EmbeddingProvider") -> int | None:
    dimension = getattr(provider, "dimension", None)
    if dimension is None and isinstance(provider, BGEM3EmbeddingProvider):
        return 1024
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def _validate_vector(value: object, *, expected_dimension: int | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("embedding must be a non-empty vector")
    if expected_dimension is not None and len(value) != expected_dimension:
        raise ValueError("embedding dimension does not match model")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("embedding contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("embedding contains a non-finite value")
        vector.append(number)
    return vector


def _cached_query(provider: "EmbeddingProvider", text: str, compute) -> list[float]:
    key = _cache_key(provider, "query", text)
    cache = get_cache()
    expected = _expected_dimension(provider)

    def validate(value: object) -> list[float]:
        return _validate_vector(value, expected_dimension=expected)

    cached = cache.get(key, validate)
    if cached is not None:
        return cached
    vector = validate(compute())
    cache.set(key, vector, ttl_seconds=EMBEDDING_CACHE_TTL_SECONDS)
    return vector


def _cached_documents(
    provider: "EmbeddingProvider", texts: list[str], compute
) -> list[list[float]]:
    if not texts:
        return []
    cache = get_cache()
    expected = _expected_dimension(provider)
    keys = [_cache_key(provider, "document", text) for text in texts]

    def validate(value: object) -> list[float]:
        return _validate_vector(value, expected_dimension=expected)

    vectors: list[list[float] | None] = [cache.get(key, validate) for key in keys]
    missing_positions = [index for index, vector in enumerate(vectors) if vector is None]
    if missing_positions:
        fresh = compute([texts[index] for index in missing_positions])
        if not isinstance(fresh, (list, tuple)) or len(fresh) != len(missing_positions):
            raise RuntimeError("embedding provider returned an unexpected batch size")
        validated = [validate(vector) for vector in fresh]
        dimensions = {len(vector) for vector in validated}
        if len(dimensions) != 1:
            raise RuntimeError("embedding provider returned inconsistent dimensions")
        for position, vector in zip(missing_positions, validated, strict=True):
            vectors[position] = vector
            cache.set(keys[position], vector, ttl_seconds=EMBEDDING_CACHE_TTL_SECONDS)
    return [vector for vector in vectors if vector is not None]


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
        return _cached_documents(
            self, texts, lambda missing: [self._hash_to_vector(t) for t in missing]
        )

    def embed_query(self, text: str) -> list[float]:
        return _cached_query(self, text, lambda: self._hash_to_vector(text))


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
        def compute(missing: list[str]) -> list[list[float]]:
            model = self._load_model()
            embeddings = model.encode(missing, normalize_embeddings=True)
            return [self._validated_vector(e) for e in embeddings]

        return _cached_documents(self, texts, compute)

    def embed_query(self, text: str) -> list[float]:
        def compute() -> list[float]:
            model = self._load_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return self._validated_vector(embedding)

        return _cached_query(self, text, compute)


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

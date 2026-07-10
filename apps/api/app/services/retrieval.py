import hashlib
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import Literal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.crud.corpus import has_pending_corpus_revision, read_corpus_revision
from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, PaperPage
from app.ingestion.parser import find_verbatim_span
from app.observability.policy import sanitize_text
from app.observability.telemetry import get_telemetry
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)
from app.services.cache import get_cache
from app.services.embedding import get_embedding_provider

_TRACE_PREFIX_CHARS = 240


class _RetrievalObservation:
    def __init__(self, observation, metadata: dict) -> None:
        self._observation = observation
        self.metadata = dict(metadata)
        self.input: dict | None = None
        self.output: dict | None = None

    def update(
        self,
        *,
        metadata: dict | None = None,
        input: dict | None = None,
        output: dict | None = None,
    ) -> None:
        if metadata is not None:
            self.metadata.update(metadata)
        if input is not None:
            self.input = input
        if output is not None:
            self.output = output


@contextmanager
def _retrieval_observation(telemetry, name: str, metadata: dict):
    """Record bounded metadata while ensuring tracing can never change retrieval."""
    started = time.perf_counter()
    manager = nullcontext(None)
    observation = None
    try:
        manager = telemetry.stage(name, metadata=metadata)
        observation = manager.__enter__()
    except Exception:
        manager = nullcontext(None)
        observation = None

    recorder = _RetrievalObservation(observation, metadata) if observation is not None else None
    error = None
    try:
        yield recorder
    except BaseException as exc:
        error = exc
        raise
    finally:
        if recorder is not None:
            try:
                observation.update(
                    **{
                        "metadata": {
                            **recorder.metadata,
                            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                            "outcome": "error" if error is not None else "success",
                            **({"error_code": type(error).__name__} if error else {}),
                        },
                        **({"input": recorder.input} if recorder.input is not None else {}),
                        **({"output": recorder.output} if recorder.output is not None else {}),
                    }
                )
            except Exception:
                pass
        try:
            manager.__exit__(
                type(error) if error is not None else None,
                error,
                error.__traceback__ if error is not None else None,
            )
        except Exception:
            # Export and SDK teardown are outside the retrieval result contract.
            pass


def _trace_prefix(value: str) -> tuple[str, bool]:
    sanitized = sanitize_text(value)
    return sanitized[:_TRACE_PREFIX_CHARS], len(sanitized) > _TRACE_PREFIX_CHARS


def _candidate_trace_item(
    *,
    chunk_id: UUID | str,
    paper_id: UUID | str,
    text_value: str,
    rank: int,
    score_name: str,
    score: float | None,
) -> dict:
    prefix, truncated = _trace_prefix(text_value)
    return {
        "chunk_id": str(chunk_id),
        "paper_id": str(paper_id),
        "rank": rank,
        "score": {"metric": score_name, "value": score},
        "text_prefix": prefix,
        "text_truncated": truncated,
    }


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _rerank_with_cache(
    query: str,
    documents: list[str],
    reranker,
    *,
    observation: _RetrievalObservation | None = None,
) -> list[tuple[int, float]]:
    """Cache ordered reranker scores without persisting query/document text."""
    content_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in documents]
    cache_key = "rerank:" + json.dumps(
        {
            "query": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "documents": content_hashes,
            "model": reranker.model_name,
            "revision": getattr(reranker, "model_version", "unversioned"),
            "policy": "top-evidence-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def validate(value: object) -> list[tuple[int, float]]:
        if not isinstance(value, list):
            raise ValueError("invalid reranker cache value")
        results: list[tuple[int, float]] = []
        seen: set[int] = set()
        for item in value:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or isinstance(item[0], bool)
                or not isinstance(item[0], int)
                or item[0] < 0
                or item[0] >= len(documents)
                or item[0] in seen
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, float))
                or not math.isfinite(item[1])
            ):
                raise ValueError("invalid reranker cache value")
            seen.add(item[0])
            results.append((item[0], float(item[1])))
        return results

    cache = get_cache()
    cached = cache.get(cache_key, validate)
    if cached is not None:
        if observation is not None:
            observation.update(metadata={"cache_status": "hit"})
        return cached
    if observation is not None:
        observation.update(metadata={"cache_status": "miss"})
    result = reranker.rerank(query, documents)
    cache.set(cache_key, [[index, score] for index, score in result], ttl_seconds=3600)
    return result


def _candidate_cache_key(
    *,
    project_id: UUID,
    corpus_revision: int,
    query: str,
    backend: str,
    embedding_model: str,
    embedding_version: str,
    strategy: str,
    candidate_limit: int,
    rrf_k: int,
) -> str:
    """Build a content-free key for the ordered pre-rerank candidate IDs."""
    return "retrieval-candidates:v1:" + json.dumps(
        {
            "project_id": str(project_id),
            "corpus_revision": corpus_revision,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "backend": backend,
            "embedding_model": embedding_model,
            "embedding_revision": embedding_version,
            "policy_revision": "hybrid-rrf-v1",
            "strategy": strategy,
            "candidate_limit": candidate_limit,
            "fusion": {"method": "reciprocal_rank_fusion", "rrf_k": rrf_k},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_candidate_ids(value: object, *, limit: int) -> list[UUID]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("invalid retrieval candidate cache value")
    result: list[UUID] = []
    seen: set[UUID] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("invalid retrieval candidate cache value")
        parsed = UUID(item)
        if parsed in seen:
            raise ValueError("invalid retrieval candidate cache value")
        seen.add(parsed)
        result.append(parsed)
    return result


class RerankerProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return list of (document_index, score) sorted in descending order of relevance."""
        pass


class SimpleLexicalReranker(RerankerProvider):
    """Lightweight lexical reranker for the explicit test/demo profile."""

    model_version = "builtin-v1"

    _stop_words = frozenset(
        {
            "a",
            "about",
            "an",
            "and",
            "are",
            "as",
            "by",
            "can",
            "could",
            "describe",
            "do",
            "does",
            "explain",
            "for",
            "from",
            "give",
            "how",
            "i",
            "in",
            "is",
            "me",
            "my",
            "of",
            "on",
            "please",
            "tell",
            "the",
            "to",
            "was",
            "we",
            "what",
            "why",
            "with",
            "would",
            "you",
            "your",
        }
    )

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", value.casefold())
        return {
            word[:-1] if word.endswith("s") and len(word) > 4 else word
            for word in words
            if word not in cls._stop_words
        }

    @property
    def model_name(self) -> str:
        return "simple-lexical"

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        query_words = self._tokens(query)
        document_words = [self._tokens(doc) for doc in documents]
        document_frequency = {
            word: sum(word in words for words in document_words) for word in query_words
        }
        scored: list[tuple[int, float]] = []
        for idx, doc_words in enumerate(document_words):
            if not query_words or not doc_words:
                scored.append((idx, 0.0))
                continue
            score = sum(
                math.log((len(documents) + 1) / (document_frequency[word] + 1)) + 1
                for word in query_words.intersection(doc_words)
            ) / math.sqrt(len(query_words) * len(doc_words))
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class BGERerankerProvider(RerankerProvider):
    """BGE multilingual cross-encoder reranker with lazy initialization."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model_version: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    ) -> None:
        self._model_name = model_name
        self.model_version = model_version
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name, revision=self.model_version, local_files_only=True
                )
            except Exception as err:
                raise RuntimeError(
                    f"Pinned reranker {self._model_name}@{self.model_version} is not "
                    "available locally. Provision the approved model cache before retrieval."
                ) from err
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        if not documents:
            return []
        model = self._load_model()
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores


_reranker_instance: RerankerProvider | None = None


def get_reranker() -> RerankerProvider:
    global _reranker_instance
    if _reranker_instance is None:
        provider_type = os.getenv("MYRA_RERANKER_PROVIDER", "bge").lower()
        if provider_type == "bge":
            _reranker_instance = BGERerankerProvider()
        elif provider_type in ("test", "demo", "simple-lexical"):
            _reranker_instance = SimpleLexicalReranker()
        else:
            raise ValueError(f"Unknown MYRA_RERANKER_PROVIDER: {provider_type}")
    return _reranker_instance


def set_reranker(reranker: RerankerProvider | None) -> None:
    global _reranker_instance
    _reranker_instance = reranker


class HybridRetriever:
    """Production database-native hybrid retriever with pgvector, FTS, RRF, and reranking."""

    def __init__(
        self,
        top_candidates: int = 40,
        top_evidence: int = 6,
        rrf_k: int = 60,
    ) -> None:
        self.top_candidates = top_candidates
        self.top_evidence = top_evidence
        self.rrf_k = rrf_k

    @staticmethod
    def _hydrate_candidate_ids(
        db: Session,
        project_id: UUID,
        candidate_ids: list[UUID],
    ) -> list[PaperChunk] | None:
        """Hydrate only still-eligible IDs; ``None`` means invalidate the whole hit."""
        if not candidate_ids:
            return []
        chunks = (
            db.query(PaperChunk)
            .join(Paper, PaperChunk.paper_id == Paper.id)
            .filter(
                PaperChunk.id.in_(candidate_ids),
                PaperChunk.chunk_type == "child",
                Paper.project_id == project_id,
                Paper.status == "READY",
            )
            .populate_existing()
            .all()
        )
        chunk_map = {chunk.id: chunk for chunk in chunks}
        if len(chunk_map) != len(candidate_ids):
            return None
        return [chunk_map[chunk_id] for chunk_id in candidate_ids]

    def _retrieve_postgres(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        query_vec: list[float],
        embedding_model: str,
        embedding_version: str,
        telemetry=None,
        strategy: Literal["dense-only", "hybrid-unreranked", "hybrid-reranked"] = "hybrid-reranked",
    ) -> list[PaperChunk]:
        """Execute database-native pgvector cosine distance and PostgreSQL full-text search."""
        vec_str = "[" + ",".join(str(float(x)) for x in query_vec) + "]"

        # 1. Native pgvector cosine distance query
        dense_sql = text("""
            SELECT pc.id, pc.paper_id, pc.text,
                   1.0 - (pc.embedding_vec <=> :query_vec) AS cosine_similarity
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE p.project_id = :project_id
              AND p.status = 'READY'
              AND pc.chunk_type = 'child'
              AND pc.embedding_vec IS NOT NULL
              AND pc.embedding_model = :embedding_model
              AND pc.embedding_version = :embedding_version
            ORDER BY pc.embedding_vec <=> :query_vec ASC
            LIMIT :limit;
        """)
        with _retrieval_observation(
            telemetry,
            "retrieval.dense_search",
            {
                "backend": "postgres_pgvector",
                "embedding_model": embedding_model,
                "embedding_revision": embedding_version,
                "candidate_limit": self.top_candidates,
            },
        ) as observation:
            dense_rows = db.execute(
                dense_sql,
                {
                    "project_id": project_id,
                    "query_vec": vec_str,
                    "embedding_model": embedding_model,
                    "embedding_version": embedding_version,
                    "limit": self.top_candidates,
                },
            ).fetchall()
            dense_trace = [
                _candidate_trace_item(
                    chunk_id=row[0],
                    paper_id=row[1],
                    text_value=row[2],
                    rank=rank,
                    score_name="cosine_similarity",
                    score=float(row[3]),
                )
                for rank, row in enumerate(dense_rows, 1)
            ]
            if observation is not None:
                observation.update(
                    metadata={"candidate_count": len(dense_rows)},
                    output={"candidates": dense_trace},
                )
        dense_cids = [row[0] for row in dense_rows]

        if strategy == "dense-only":
            dense_cids = dense_cids[: self.top_candidates]
            chunks = db.query(PaperChunk).filter(PaperChunk.id.in_(dense_cids)).all()
            chunk_map = {chunk.id: chunk for chunk in chunks}
            return [chunk_map[cid] for cid in dense_cids if cid in chunk_map]

        # 2. PostgreSQL Full-Text Search with plainto_tsquery and ts_rank
        fts_tokens = SimpleLexicalReranker._tokens(query)
        fts_query = " ".join(fts_tokens) if fts_tokens else query
        fts_sql = text("""
            SELECT pc.id, pc.paper_id, pc.text,
                   ts_rank(pc.tsv_content, plainto_tsquery('english', :query)) AS fts_score
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE p.project_id = :project_id
              AND p.status = 'READY'
              AND pc.chunk_type = 'child'
              AND pc.tsv_content @@ plainto_tsquery('english', :query)
            ORDER BY ts_rank(pc.tsv_content, plainto_tsquery('english', :query)) DESC
            LIMIT :limit;
        """)
        with _retrieval_observation(
            telemetry,
            "retrieval.fts_search",
            {"backend": "postgres_fts", "candidate_limit": self.top_candidates},
        ) as observation:
            fts_rows = db.execute(
                fts_sql,
                {
                    "project_id": project_id,
                    "query": fts_query,
                    "limit": self.top_candidates,
                },
            ).fetchall()
            fts_trace = [
                _candidate_trace_item(
                    chunk_id=row[0],
                    paper_id=row[1],
                    text_value=row[2],
                    rank=rank,
                    score_name="postgres_fts_rank",
                    score=float(row[3]),
                )
                for rank, row in enumerate(fts_rows, 1)
            ]
            if observation is not None:
                observation.update(
                    metadata={"candidate_count": len(fts_rows)},
                    output={"candidates": fts_trace},
                )
        fts_cids = [row[0] for row in fts_rows]

        # 3. Reciprocal Rank Fusion (RRF)
        with _retrieval_observation(
            telemetry,
            "retrieval.fusion",
            {
                "policy_revision": "rrf-v1",
                "fusion_method": "reciprocal_rank_fusion",
                "rrf_k": self.rrf_k,
                "dense_candidate_count": len(dense_cids),
                "fts_candidate_count": len(fts_cids),
            },
        ) as observation:
            rrf_scores: dict[UUID, float] = {}
            for rank, cid in enumerate(dense_cids):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

            for rank, cid in enumerate(fts_cids):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

            sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
                : self.top_candidates
            ]
            if observation is not None:
                dense_by_id = {item["chunk_id"]: item for item in dense_trace}
                fts_by_id = {item["chunk_id"]: item for item in fts_trace}
                fused_trace = []
                for rank, cid in enumerate(sorted_cids, 1):
                    dense_item = dense_by_id.get(str(cid))
                    fts_item = fts_by_id.get(str(cid))
                    candidate = dense_item or fts_item
                    if candidate is None:
                        continue
                    fused_trace.append(
                        {
                            "chunk_id": str(cid),
                            "paper_id": candidate["paper_id"],
                            "rank": rank,
                            "score": {
                                "metric": "reciprocal_rank_fusion",
                                "value": rrf_scores[cid],
                            },
                            "dense_rank": dense_item["rank"] if dense_item else None,
                            "dense_score": dense_item["score"] if dense_item else None,
                            "fts_rank": fts_item["rank"] if fts_item else None,
                            "fts_score": fts_item["score"] if fts_item else None,
                            "text_prefix": candidate["text_prefix"],
                            "text_truncated": candidate["text_truncated"],
                        }
                    )
                observation.update(
                    metadata={"candidate_count": len(sorted_cids)},
                    input={"dense_candidates": dense_trace, "fts_candidates": fts_trace},
                    output={"candidates": fused_trace},
                )

        if not sorted_cids:
            return []

        # Fetch candidate chunks and preserve RRF ordering
        chunks = db.query(PaperChunk).filter(PaperChunk.id.in_(sorted_cids)).all()
        chunk_map = {c.id: c for c in chunks}
        hydrated = [chunk_map[cid] for cid in sorted_cids if cid in chunk_map]
        return hydrated

    def _retrieve_fallback(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        query_vec: list[float],
        embedding_model: str,
        embedding_version: str,
        telemetry=None,
        strategy: Literal["dense-only", "hybrid-unreranked", "hybrid-reranked"] = "hybrid-reranked",
    ) -> list[PaperChunk]:
        """In-memory scoring fallback for SQLite / test environments."""
        chunks = (
            db.query(PaperChunk)
            .join(Paper, PaperChunk.paper_id == Paper.id)
            .filter(
                Paper.project_id == project_id,
                Paper.status == "READY",
                PaperChunk.chunk_type == "child",
            )
            .all()
        )
        if not chunks:
            return []

        with _retrieval_observation(
            telemetry,
            "retrieval.dense_search",
            {
                "backend": "in_memory_cosine",
                "embedding_model": embedding_model,
                "embedding_revision": embedding_version,
                "candidate_limit": self.top_candidates,
            },
        ) as observation:
            dense_scores = []
            for chunk in chunks:
                if (
                    chunk.embedding_model != embedding_model
                    or chunk.embedding_version != embedding_version
                ):
                    continue
                vec = chunk.embedding_vec or chunk.embedding or []
                sim = cosine_similarity(query_vec, vec)
                dense_scores.append((chunk, sim))
            dense_scores.sort(key=lambda x: x[1], reverse=True)
            dense_trace = [
                _candidate_trace_item(
                    chunk_id=chunk.id,
                    paper_id=chunk.paper_id,
                    text_value=chunk.text,
                    rank=rank,
                    score_name="cosine_similarity",
                    score=float(score),
                )
                for rank, (chunk, score) in enumerate(dense_scores[: self.top_candidates], 1)
            ]
            if observation is not None:
                observation.update(
                    metadata={"candidate_count": len(dense_scores)},
                    output={"candidates": dense_trace},
                )

        if strategy == "dense-only":
            return [chunk for chunk, _ in dense_scores[: self.top_candidates]]

        with _retrieval_observation(
            telemetry,
            "retrieval.fts_search",
            {"backend": "in_memory_lexical", "candidate_limit": self.top_candidates},
        ) as observation:
            query_terms = set(query.lower().split())
            lexical_scores = []
            for chunk in chunks:
                chunk_lower = chunk.text.lower()
                matches = sum(1 for term in query_terms if term in chunk_lower)
                lexical_scores.append((chunk, matches))
            lexical_scores.sort(key=lambda x: x[1], reverse=True)
            fts_trace = [
                _candidate_trace_item(
                    chunk_id=chunk.id,
                    paper_id=chunk.paper_id,
                    text_value=chunk.text,
                    rank=rank,
                    score_name="term_match_count",
                    score=float(matches),
                )
                for rank, (chunk, matches) in enumerate(lexical_scores[: self.top_candidates], 1)
            ]
            if observation is not None:
                observation.update(
                    metadata={"candidate_count": len(lexical_scores)},
                    output={"candidates": fts_trace},
                )

        with _retrieval_observation(
            telemetry,
            "retrieval.fusion",
            {
                "policy_revision": "rrf-v1",
                "fusion_method": "reciprocal_rank_fusion",
                "rrf_k": self.rrf_k,
                "dense_candidate_count": len(dense_scores[: self.top_candidates]),
                "fts_candidate_count": len(lexical_scores[: self.top_candidates]),
            },
        ) as observation:
            rrf_scores: dict[UUID, float] = {}
            for rank, (chunk, _) in enumerate(dense_scores[: self.top_candidates]):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (
                    1.0 / (self.rrf_k + rank + 1)
                )

            for rank, (chunk, _) in enumerate(lexical_scores[: self.top_candidates]):
                rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (
                    1.0 / (self.rrf_k + rank + 1)
                )

            chunk_lookup = {c.id: c for c in chunks}
            sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
                : self.top_candidates
            ]
            if observation is not None:
                dense_by_id = {item["chunk_id"]: item for item in dense_trace}
                fts_by_id = {item["chunk_id"]: item for item in fts_trace}
                fused_trace = []
                for rank, cid in enumerate(sorted_cids, 1):
                    dense_item = dense_by_id.get(str(cid))
                    fts_item = fts_by_id.get(str(cid))
                    candidate = dense_item or fts_item
                    if candidate is None:
                        continue
                    fused_trace.append(
                        {
                            "chunk_id": str(cid),
                            "paper_id": candidate["paper_id"],
                            "rank": rank,
                            "score": {
                                "metric": "reciprocal_rank_fusion",
                                "value": rrf_scores[cid],
                            },
                            "dense_rank": dense_item["rank"] if dense_item else None,
                            "dense_score": dense_item["score"] if dense_item else None,
                            "fts_rank": fts_item["rank"] if fts_item else None,
                            "fts_score": fts_item["score"] if fts_item else None,
                            "text_prefix": candidate["text_prefix"],
                            "text_truncated": candidate["text_truncated"],
                        }
                    )
                observation.update(
                    metadata={"candidate_count": len(sorted_cids)},
                    input={"dense_candidates": dense_trace, "fts_candidates": fts_trace},
                    output={"candidates": fused_trace},
                )
        return [chunk_lookup[cid] for cid in sorted_cids]

    def retrieve(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        *,
        strategy: Literal["dense-only", "hybrid-unreranked", "hybrid-reranked"] = "hybrid-reranked",
    ) -> list[EvidenceItem]:
        """Retrieve evidence; ``strategy`` is an opt-in evaluation ablation seam.

        Production callers keep the historical hybrid + reranker behavior by
        default. The other strategies are intended for controlled evaluation.
        """
        if strategy not in {"dense-only", "hybrid-unreranked", "hybrid-reranked"}:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        try:
            telemetry = get_telemetry()
        except Exception:
            telemetry = None
        with _retrieval_observation(
            telemetry,
            "retrieval",
            {
                "policy_revision": "hybrid-rrf-v1",
                "candidate_limit": self.top_candidates,
                "evidence_limit": self.top_evidence,
                "rrf_k": self.rrf_k,
                "strategy": strategy,
            },
        ):
            return self._retrieve_impl(db, project_id, query, telemetry, strategy)

    def _retrieve_impl(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        telemetry,
        strategy: Literal["dense-only", "hybrid-unreranked", "hybrid-reranked"] = "hybrid-reranked",
    ) -> list[EvidenceItem]:
        embed_provider = get_embedding_provider()
        with _retrieval_observation(
            telemetry,
            "retrieval.query_embedding",
            {
                "embedding_model": embed_provider.model_name,
                "embedding_revision": embed_provider.model_version,
            },
        ) as observation:
            query_vec = embed_provider.embed_query(query)
            if observation is not None:
                observation.update(
                    metadata={"vector_dimensions": len(query_vec)},
                    input={"question": query},
                )

        # Check if database dialect is PostgreSQL
        is_postgres = False
        try:
            bind = db.get_bind()
            is_postgres = bind.dialect.name == "postgresql"
        except Exception:
            pass

        backend = "postgres" if is_postgres else "fallback"

        def retrieve_candidates() -> list[PaperChunk]:
            if is_postgres:
                return self._retrieve_postgres(
                    db=db,
                    project_id=project_id,
                    query=query,
                    query_vec=query_vec,
                    embedding_model=embed_provider.model_name,
                    embedding_version=embed_provider.model_version,
                    telemetry=telemetry,
                    strategy=strategy,
                )
            return self._retrieve_fallback(
                db=db,
                project_id=project_id,
                query=query,
                query_vec=query_vec,
                embedding_model=embed_provider.model_name,
                embedding_version=embed_provider.model_version,
                telemetry=telemetry,
                strategy=strategy,
            )

        cache = get_cache()
        caching_enabled = bool(getattr(cache, "enabled", True))
        pending_mutation = has_pending_corpus_revision(db)
        candidate_cache_hit = False
        if not caching_enabled or pending_mutation:
            candidate_chunks = retrieve_candidates()
        else:
            # If revision metadata is unavailable, preserve retrieval behavior
            # and simply bypass this disposable cache for the request.
            try:
                starting_revision = read_corpus_revision(db, project_id)
            except Exception:
                candidate_chunks = retrieve_candidates()
            else:
                cache_key = _candidate_cache_key(
                    project_id=project_id,
                    corpus_revision=starting_revision,
                    query=query,
                    backend=backend,
                    embedding_model=embed_provider.model_name,
                    embedding_version=embed_provider.model_version,
                    strategy=strategy,
                    candidate_limit=self.top_candidates,
                    rrf_k=self.rrf_k,
                )
                candidate_chunks = []
                stable_revision = False
                for attempt in range(2):
                    if attempt:
                        try:
                            starting_revision = read_corpus_revision(db, project_id)
                        except Exception:
                            candidate_chunks = retrieve_candidates()
                            stable_revision = True
                            break
                        cache_key = _candidate_cache_key(
                            project_id=project_id,
                            corpus_revision=starting_revision,
                            query=query,
                            backend=backend,
                            embedding_model=embed_provider.model_name,
                            embedding_version=embed_provider.model_version,
                            strategy=strategy,
                            candidate_limit=self.top_candidates,
                            rrf_k=self.rrf_k,
                        )

                    cached_ids = cache.get(
                        cache_key,
                        lambda value: _validate_candidate_ids(value, limit=self.top_candidates),
                    )
                    hydrated = (
                        self._hydrate_candidate_ids(db, project_id, cached_ids)
                        if cached_ids is not None
                        else None
                    )
                    from_cache = cached_ids is not None and hydrated is not None
                    candidate_chunks = hydrated if from_cache else retrieve_candidates()

                    # The caller owns the transaction. An in-flight corpus
                    # mutation must never publish or consume revision-keyed IDs.
                    if has_pending_corpus_revision(db):
                        stable_revision = True
                        break

                    try:
                        ending_revision = read_corpus_revision(db, project_id)
                    except Exception:
                        stable_revision = True
                        break
                    if ending_revision == starting_revision:
                        stable_revision = True
                        candidate_cache_hit = from_cache
                        if not from_cache and not has_pending_corpus_revision(db):
                            cache.set(
                                cache_key,
                                [str(chunk.id) for chunk in candidate_chunks],
                                ttl_seconds=300,
                            )
                        break
                    if attempt == 1:
                        candidate_chunks = []
                        try:
                            telemetry.event(
                                "retrieval.candidates",
                                metadata={
                                    "outcome": "CORPUS_CHANGED",
                                    "backend": backend,
                                    "retry_count": 1,
                                },
                            )
                        except Exception:
                            pass
                if not stable_revision:
                    candidate_chunks = []

        if candidate_cache_hit:
            with _retrieval_observation(
                telemetry,
                "retrieval.candidate_cache",
                {
                    "cache_status": "hit",
                    "skipped_stages": [
                        "retrieval.dense_search",
                        "retrieval.fts_search",
                        "retrieval.fusion",
                    ],
                    "candidate_count": len(candidate_chunks),
                },
            ) as observation:
                if observation is not None:
                    observation.update(
                        output={
                            "candidates": [
                                _candidate_trace_item(
                                    chunk_id=chunk.id,
                                    paper_id=chunk.paper_id,
                                    text_value=chunk.text,
                                    rank=rank,
                                    score_name="unavailable_from_candidate_cache",
                                    score=None,
                                )
                                for rank, chunk in enumerate(candidate_chunks, 1)
                            ]
                        }
                    )

        if not candidate_chunks:
            return []

        # Ablations skip reranking unless the strategy explicitly includes it;
        # candidate retrieval and evidence/provenance construction stay shared.
        if strategy != "hybrid-reranked":
            top_chunks = candidate_chunks[: self.top_evidence]
        else:
            top_chunks = None

        # Rerank candidates for the production strategy.
        if top_chunks is None:
            reranker = get_reranker()
            docs = [c.text for c in candidate_chunks]
            with _retrieval_observation(
                telemetry,
                "retrieval.reranking",
                {
                    "reranker_model": reranker.model_name,
                    "reranker_revision": getattr(reranker, "model_version", "unversioned"),
                    "policy_revision": "top-evidence-v1",
                    "input_candidate_count": len(candidate_chunks),
                    "evidence_limit": self.top_evidence,
                },
            ) as observation:
                reranked_order = _rerank_with_cache(query, docs, reranker, observation=observation)
                if observation is not None:
                    reranked_trace = []
                    selected_indices = {
                        index for index, _score in reranked_order[: self.top_evidence]
                    }
                    for rank, (index, score) in enumerate(reranked_order, 1):
                        chunk = candidate_chunks[index]
                        prefix, truncated = _trace_prefix(chunk.text)
                        reranked_trace.append(
                            {
                                "chunk_id": str(chunk.id),
                                "paper_id": str(chunk.paper_id),
                                "rank": rank,
                                "score": {
                                    "metric": "reranker_score",
                                    "value": float(score),
                                },
                                "text_prefix": prefix,
                                "text_truncated": truncated,
                                "selected_for_evidence": index in selected_indices,
                            }
                        )
                    observation.update(
                        metadata={"ranked_candidate_count": len(reranked_order)},
                        input={
                            "question": query,
                            "candidates": [
                                _candidate_trace_item(
                                    chunk_id=chunk.id,
                                    paper_id=chunk.paper_id,
                                    text_value=chunk.text,
                                    rank=rank,
                                    score_name="not_scored_before_reranking",
                                    score=None,
                                )
                                for rank, chunk in enumerate(candidate_chunks, 1)
                            ],
                        },
                        output={"candidates": reranked_trace},
                    )

            top_chunks = [candidate_chunks[idx] for idx, _ in reranked_order[: self.top_evidence]]

        # Build EvidenceItems with exact provenance mapping and parent expansion
        evidence_items: list[EvidenceItem] = []
        evidence_started = time.perf_counter()
        for i, chunk in enumerate(top_chunks):
            evidence_id = f"E{i + 1}"
            paper = db.query(Paper).filter(Paper.id == chunk.paper_id).first()

            # Retrieve ordered source elements
            chunk_elements = (
                db.query(ChunkElement)
                .filter(ChunkElement.chunk_id == chunk.id)
                .order_by(ChunkElement.order_index.asc())
                .all()
            )
            elem_ids = [ce.element_id for ce in chunk_elements]
            source_elements = (
                db.query(PaperElement)
                .filter(PaperElement.id.in_(elem_ids))
                .order_by(PaperElement.element_index.asc())
                .all()
                if elem_ids
                else []
            )

            # Parent context expansion: fetch parent chunk if available
            parent_chunk = (
                db.query(PaperChunk)
                .join(ChunkElement, ChunkElement.chunk_id == PaperChunk.id)
                .filter(
                    PaperChunk.paper_id == chunk.paper_id,
                    PaperChunk.chunk_type == "parent",
                    ChunkElement.element_id.in_(elem_ids),
                )
                .first()
                if elem_ids
                else None
            )
            parent_context = parent_chunk.text if parent_chunk else chunk.text

            # Map page-specific bboxes and anchors with verbatim text span verification
            bboxes: list[BoundingBox] = []
            anchors: list[CitationAnchor] = []
            page_number = 1
            exact_quote = chunk.text[:250].strip()
            parser_ver = None

            if source_elements:
                page_text_cache: dict[int, str | None] = {}
                evaluated_elements: list[
                    tuple[PaperElement, CitationAnchor, list[BoundingBox]]
                ] = []

                for elem in source_elements:
                    elem_boxes: list[BoundingBox] = []
                    if elem.bbox_x_min is not None and elem.page_width and elem.page_height:
                        box = BoundingBox(
                            x_min=elem.bbox_x_min,
                            y_min=elem.bbox_y_min or 0.0,
                            x_max=elem.bbox_x_max or 0.0,
                            y_max=elem.bbox_y_max or 0.0,
                            page_width=elem.page_width,
                            page_height=elem.page_height,
                            origin=CoordinateOrigin(elem.coordinate_origin),
                            rotation=elem.rotation,
                        )
                        elem_boxes.append(box)

                    # Retrieve canonical page text to verify verbatim span
                    if elem.page_number not in page_text_cache:
                        page_record = (
                            db.query(PaperPage)
                            .filter(
                                PaperPage.paper_id == chunk.paper_id,
                                PaperPage.page_number == elem.page_number,
                            )
                            .first()
                        )
                        if page_record is None:
                            page_text_cache[elem.page_number] = None
                        else:
                            p_text = page_record.raw_text
                            if not p_text:
                                # Fallback to ordered elements if raw_text was not saved
                                page_elems = (
                                    db.query(PaperElement)
                                    .filter(
                                        PaperElement.paper_id == chunk.paper_id,
                                        PaperElement.page_number == elem.page_number,
                                    )
                                    .order_by(PaperElement.element_index)
                                    .all()
                                )
                                if page_elems:
                                    p_text = "\n\n".join(e.text for e in page_elems if e.text)
                            page_text_cache[elem.page_number] = p_text

                    page_text = page_text_cache[elem.page_number]
                    # Require valid document SHA-256 and parser version
                    elem_provenance = bool(
                        paper and paper.document_sha256 and elem.parser_version and page_text
                    )

                    span = find_verbatim_span(page_text, elem.text) if elem_provenance else None

                    if span is not None and elem.text.strip():
                        start_char, end_char = span
                        anchor_status = AnchorStatus.VERIFIED
                        verified_boxes = elem_boxes
                    elif not elem_provenance:
                        start_char, end_char = None, None
                        anchor_status = (
                            AnchorStatus.LEGACY if not page_text else AnchorStatus.UNRESOLVED
                        )
                        verified_boxes = []
                    else:
                        start_char, end_char = None, None
                        anchor_status = AnchorStatus.UNRESOLVED
                        verified_boxes = []

                    cand_anchor = CitationAnchor(
                        page_number=elem.page_number,
                        source_element_id=elem.id,
                        exact_quote=elem.text,
                        source_char_start=start_char,
                        source_char_end=end_char,
                        document_sha256=paper.document_sha256 if paper else None,
                        parser_version=elem.parser_version,
                        anchor_status=anchor_status,
                        bounding_boxes=verified_boxes,
                    )
                    anchors.append(cand_anchor)
                    evaluated_elements.append((elem, cand_anchor, verified_boxes))

                # Select best element: prioritize verified anchors, lexical match
                # without stopwords, then substantive length
                query_tokens = SimpleLexicalReranker._tokens(query)
                if not query_tokens:
                    query_tokens = {
                        w for w in re.findall(r"[a-z0-9]+", query.casefold()) if len(w) > 2
                    }

                best_elem, best_anchor, _ = max(
                    evaluated_elements,
                    key=lambda item: (
                        (1000.0 if item[1].anchor_status == AnchorStatus.VERIFIED else 0.0)
                        + (
                            len(
                                query_tokens.intersection(
                                    SimpleLexicalReranker._tokens(item[0].text)
                                )
                            )
                            * 50.0
                        )
                        + (min(len(item[0].text.strip()), 300) / 300.0)
                    ),
                )

                page_number = best_elem.page_number
                exact_quote = best_elem.text
                parser_ver = best_elem.parser_version

                for elem, cand_anchor, v_boxes in evaluated_elements:
                    if (
                        cand_anchor.anchor_status == AnchorStatus.VERIFIED
                        and elem.page_number == page_number
                    ):
                        bboxes.extend(v_boxes)

            evidence_items.append(
                EvidenceItem(
                    id=evidence_id,
                    paper_id=chunk.paper_id,
                    paper_title=paper.filename if paper else None,
                    chunk_id=chunk.id,
                    quote=exact_quote,
                    parent_context=parent_context,
                    page_number=page_number,
                    bounding_boxes=bboxes,
                    source_element_ids=elem_ids,
                    document_sha256=paper.document_sha256 if paper else None,
                    parser_version=parser_ver,
                    anchors=anchors,
                )
            )

        try:
            telemetry.event(
                "retrieval.evidence_built",
                metadata={
                    "policy_revision": "evidence-provenance-v1",
                    "candidate_count": len(candidate_chunks),
                    "evidence_count": len(evidence_items),
                    "duration_ms": round((time.perf_counter() - evidence_started) * 1000, 2),
                },
            )
        except Exception:
            pass
        return evidence_items

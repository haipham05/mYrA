import math
import os
from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)
from app.services.embedding import get_embedding_provider


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class RerankerProvider(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """Return list of (document_index, score) sorted in descending order of relevance."""
        pass


class SimpleLexicalReranker(RerankerProvider):
    """Fast lexical overlap and length-normalized reranker for testing and CPU fallback."""

    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        query_words = set(query.lower().split())
        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            if not query_words or not doc_words:
                scored.append((idx, 0.0))
                continue
            overlap = len(query_words.intersection(doc_words))
            score = overlap / math.sqrt(len(query_words) * len(doc_words))
            scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class BGERerankerProvider(RerankerProvider):
    """BGE multilingual cross-encoder reranker with lazy initialization."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as err:
                raise RuntimeError(f"Could not load BGE reranker {self.model_name}: {err}") from err
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
        use_bge = os.getenv("MYRA_USE_BGE_RERANKER", "false").lower() in ("true", "1")
        if use_bge:
            _reranker_instance = BGERerankerProvider()
        else:
            _reranker_instance = SimpleLexicalReranker()
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

    def _retrieve_postgres(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        query_vec: list[float],
    ) -> list[PaperChunk]:
        """Execute database-native pgvector cosine distance and PostgreSQL full-text search."""
        vec_str = "[" + ",".join(str(float(x)) for x in query_vec) + "]"

        # 1. Native pgvector cosine distance query
        dense_sql = text("""
            SELECT pc.id
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE p.project_id = :project_id
              AND p.status = 'READY'
              AND pc.chunk_type = 'child'
              AND pc.embedding_vec IS NOT NULL
            ORDER BY pc.embedding_vec <=> :query_vec ASC
            LIMIT :limit;
        """)
        dense_rows = db.execute(
            dense_sql,
            {
                "project_id": project_id,
                "query_vec": vec_str,
                "limit": self.top_candidates,
            },
        ).fetchall()
        dense_cids = [row[0] for row in dense_rows]

        # 2. PostgreSQL Full-Text Search with plainto_tsquery and ts_rank
        fts_sql = text("""
            SELECT pc.id
            FROM paper_chunks pc
            JOIN papers p ON pc.paper_id = p.id
            WHERE p.project_id = :project_id
              AND p.status = 'READY'
              AND pc.chunk_type = 'child'
              AND pc.tsv_content @@ plainto_tsquery('english', :query)
            ORDER BY ts_rank(pc.tsv_content, plainto_tsquery('english', :query)) DESC
            LIMIT :limit;
        """)
        fts_rows = db.execute(
            fts_sql,
            {
                "project_id": project_id,
                "query": query,
                "limit": self.top_candidates,
            },
        ).fetchall()
        fts_cids = [row[0] for row in fts_rows]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[UUID, float] = {}
        for rank, cid in enumerate(dense_cids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, cid in enumerate(fts_cids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
            : self.top_candidates
        ]

        if not sorted_cids:
            return []

        # Fetch candidate chunks and preserve RRF ordering
        chunks = db.query(PaperChunk).filter(PaperChunk.id.in_(sorted_cids)).all()
        chunk_map = {c.id: c for c in chunks}
        return [chunk_map[cid] for cid in sorted_cids if cid in chunk_map]

    def _retrieve_fallback(
        self,
        db: Session,
        project_id: UUID,
        query: str,
        query_vec: list[float],
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

        dense_scores = []
        for chunk in chunks:
            vec = chunk.embedding_vec or chunk.embedding or []
            sim = cosine_similarity(query_vec, vec)
            dense_scores.append((chunk, sim))
        dense_scores.sort(key=lambda x: x[1], reverse=True)

        query_terms = set(query.lower().split())
        lexical_scores = []
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            matches = sum(1 for term in query_terms if term in chunk_lower)
            lexical_scores.append((chunk, matches))
        lexical_scores.sort(key=lambda x: x[1], reverse=True)

        rrf_scores: dict[UUID, float] = {}
        for rank, (chunk, _) in enumerate(dense_scores[: self.top_candidates]):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (chunk, _) in enumerate(lexical_scores[: self.top_candidates]):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        chunk_lookup = {c.id: c for c in chunks}
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
            : self.top_candidates
        ]
        return [chunk_lookup[cid] for cid in sorted_cids]

    def retrieve(
        self,
        db: Session,
        project_id: UUID,
        query: str,
    ) -> list[EvidenceItem]:
        embed_provider = get_embedding_provider()
        query_vec = embed_provider.embed_query(query)

        # Check if database dialect is PostgreSQL
        is_postgres = False
        try:
            bind = db.get_bind()
            is_postgres = bind.dialect.name == "postgresql"
        except Exception:
            pass

        if is_postgres:
            candidate_chunks = self._retrieve_postgres(
                db=db, project_id=project_id, query=query, query_vec=query_vec
            )
        else:
            candidate_chunks = self._retrieve_fallback(
                db=db, project_id=project_id, query=query, query_vec=query_vec
            )

        if not candidate_chunks:
            return []

        # Rerank candidates
        reranker = get_reranker()
        docs = [c.text for c in candidate_chunks]
        reranked_order = reranker.rerank(query, docs)

        top_chunks = [candidate_chunks[idx] for idx, _ in reranked_order[: self.top_evidence]]

        # Build EvidenceItems with exact provenance mapping and parent expansion
        evidence_items: list[EvidenceItem] = []
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
                db.query(PaperElement).filter(PaperElement.id.in_(elem_ids)).all()
                if elem_ids
                else []
            )

            # Map page-specific bboxes and anchors
            bboxes: list[BoundingBox] = []
            anchors: list[CitationAnchor] = []
            page_number = 1
            exact_quote = chunk.text[:250].strip()
            parser_ver = None
            if source_elements:
                page_number = source_elements[0].page_number
                exact_quote = source_elements[0].text
                parser_ver = source_elements[0].parser_version
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
                        # STRICT: Only include bboxes on main page_number
                        if elem.page_number == page_number:
                            bboxes.append(box)

                    anchors.append(
                        CitationAnchor(
                            page_number=elem.page_number,
                            source_element_id=elem.id,
                            exact_quote=elem.text,
                            document_sha256=paper.document_sha256 if paper else None,
                            parser_version=elem.parser_version,
                            anchor_status=AnchorStatus.VERIFIED
                            if elem_boxes
                            else AnchorStatus.UNRESOLVED,
                            bounding_boxes=elem_boxes,
                        )
                    )

            evidence_items.append(
                EvidenceItem(
                    id=evidence_id,
                    paper_id=chunk.paper_id,
                    paper_title=paper.filename if paper else None,
                    chunk_id=chunk.id,
                    quote=exact_quote,
                    page_number=page_number,
                    bounding_boxes=bboxes,
                    source_element_ids=elem_ids,
                    document_sha256=paper.document_sha256 if paper else None,
                    parser_version=parser_ver,
                    anchors=anchors,
                )
            )

        return evidence_items

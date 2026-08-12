"""Small, per-paper retrieval helper for the comparison workflow."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.evidence import EvidenceItem
from app.services.retrieval import HybridRetriever


@dataclass(frozen=True)
class ComparisonEvidence:
    paper_id: UUID
    evidence: list[EvidenceItem]


class ComparisonEvidenceRetriever:
    """Run one shared-question hybrid search per selected paper, without reranking."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        if retriever is None:
            self._retriever = HybridRetriever(top_candidates=12, top_evidence=4)
        elif isinstance(retriever, HybridRetriever):
            self._retriever = HybridRetriever(
                top_candidates=12,
                top_evidence=4,
                rrf_k=retriever.rrf_k,
            )
        else:
            self._retriever = retriever

    def retrieve(
        self,
        db: Session,
        project_id: UUID,
        paper_ids: list[UUID],
        question: str,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[ComparisonEvidence]:
        results: list[ComparisonEvidence] = []
        for paper_id in paper_ids:
            evidence = self._retriever.retrieve(
                db,
                project_id,
                question,
                query_embedding=query_embedding,
                selected_paper_ids=[paper_id],
                strategy="hybrid-unreranked",
            )
            results.append(
                ComparisonEvidence(
                    paper_id=paper_id,
                    evidence=[item for item in evidence if item.paper_id == paper_id][:4],
                )
            )
        return results

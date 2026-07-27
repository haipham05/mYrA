"""Paper- and dimension-scoped evidence retrieval for comparisons."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.evidence import EvidenceItem
from app.services.retrieval import HybridRetriever

_MAX_QUERY_PART_CHARS = 240


@dataclass(frozen=True)
class ComparisonEvidence:
    """Evidence retrieved for one paper and one comparison dimension."""

    paper_id: UUID
    dimension: str
    query: str
    evidence: list[EvidenceItem]


def _build_query(question: str | None, dimension: str) -> str:
    """Build a stable, bounded retrieval query without an extra model call."""
    bounded_dimension = " ".join(dimension.split())[:_MAX_QUERY_PART_CHARS]
    if question is None or not question.strip():
        return bounded_dimension
    bounded_question = " ".join(question.split())[:_MAX_QUERY_PART_CHARS]
    return f"{bounded_question}\nComparison dimension: {bounded_dimension}"[
        : _MAX_QUERY_PART_CHARS * 2 + len("\nComparison dimension: ")
    ]


class ComparisonEvidenceRetriever:
    """Retrieve each paper/dimension independently through the existing retriever."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self._retriever = retriever or HybridRetriever()

    def retrieve(
        self,
        db: Session,
        project_id: UUID,
        paper_ids: list[UUID],
        dimensions: list[str],
        *,
        comparison_question: str | None = None,
    ) -> list[ComparisonEvidence]:
        """Return separate evidence lists; evidence for another paper is discarded."""
        results: list[ComparisonEvidence] = []
        for paper_id in paper_ids:
            for dimension in dimensions:
                query = _build_query(comparison_question, dimension)
                retrieved = self._retriever.retrieve(
                    db,
                    project_id,
                    query,
                    selected_paper_ids=[paper_id],
                )
                scoped = [item for item in retrieved if item.paper_id == paper_id]
                results.append(
                    ComparisonEvidence(
                        paper_id=paper_id,
                        dimension=dimension,
                        query=query,
                        evidence=scoped,
                    )
                )
        return results

from uuid import uuid4

from app.schemas.evidence import EvidenceItem
from app.services.comparison_retrieval import ComparisonEvidenceRetriever
from app.services.retrieval import HybridRetriever


def _evidence(paper_id):
    return EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote="evidence quote",
        page_number=1,
    )


class FakeRetriever:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def retrieve(self, db, project_id, query, *, query_embedding, selected_paper_ids, strategy):
        self.calls.append((db, project_id, query, query_embedding, selected_paper_ids, strategy))
        return self.responses.get(selected_paper_ids[0], [])


def test_comparison_uses_one_question_and_unreranked_search_per_paper():
    paper_a, paper_b = uuid4(), uuid4()
    evidence_a, evidence_b = _evidence(paper_a), _evidence(paper_b)
    retriever = FakeRetriever({paper_a: [evidence_a], paper_b: [evidence_b]})
    db, project_id, embedding = object(), uuid4(), [0.25, 0.75]

    results = ComparisonEvidenceRetriever(retriever).retrieve(
        db,
        project_id,
        [paper_a, paper_b],
        "How do the methods compare?",
        query_embedding=embedding,
    )

    assert [result.evidence for result in results] == [[evidence_a], [evidence_b]]
    assert len(retriever.calls) == 2
    assert [call[2] for call in retriever.calls] == [
        "How do the methods compare?",
        "How do the methods compare?",
    ]
    assert all(call[3] is embedding for call in retriever.calls)
    assert [call[4] for call in retriever.calls] == [[paper_a], [paper_b]]
    assert all(call[5] == "hybrid-unreranked" for call in retriever.calls)


def test_comparison_search_defaults_are_bounded_to_twelve_and_four():
    retriever = ComparisonEvidenceRetriever()

    assert isinstance(retriever._retriever, HybridRetriever)
    assert retriever._retriever.top_candidates == 12
    assert retriever._retriever.top_evidence == 4


def test_foreign_evidence_is_dropped_at_service_boundary():
    requested_paper, foreign_paper = uuid4(), uuid4()
    retriever = FakeRetriever({requested_paper: [_evidence(foreign_paper)]})

    result = ComparisonEvidenceRetriever(retriever).retrieve(
        object(), uuid4(), [requested_paper], "What is the method?"
    )

    assert result[0].paper_id == requested_paper
    assert result[0].evidence == []

from uuid import uuid4

from app.schemas.evidence import EvidenceItem
from app.services.comparison_retrieval import ComparisonEvidenceRetriever


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

    def retrieve(self, db, project_id, query, *, selected_paper_ids):
        self.calls.append((db, project_id, query, selected_paper_ids))
        return self.responses.get(selected_paper_ids[0], [])


def test_retrieves_each_paper_and_dimension_with_exact_single_paper_scope():
    paper_a, paper_b = uuid4(), uuid4()
    evidence_a, evidence_b = _evidence(paper_a), _evidence(paper_b)
    retriever = FakeRetriever({paper_a: [evidence_a], paper_b: [evidence_b]})
    db, project_id = object(), uuid4()

    results = ComparisonEvidenceRetriever(retriever).retrieve(
        db,
        project_id,
        [paper_a, paper_b],
        ["method", "results"],
        comparison_question="Compare their performance",
    )

    assert [(item.paper_id, item.dimension) for item in results] == [
        (paper_a, "method"),
        (paper_a, "results"),
        (paper_b, "method"),
        (paper_b, "results"),
    ]
    assert all(call[0] is db and call[1] == project_id for call in retriever.calls)
    assert [call[3] for call in retriever.calls] == [
        [paper_a],
        [paper_a],
        [paper_b],
        [paper_b],
    ]
    assert results[0].evidence == [evidence_a]
    assert results[2].evidence == [evidence_b]
    assert all("Compare their performance" in item.query for item in results)


def test_empty_retrieval_is_kept_as_an_explicit_empty_cell():
    paper_id = uuid4()
    result = ComparisonEvidenceRetriever(FakeRetriever()).retrieve(
        object(), uuid4(), [paper_id], ["limitations"]
    )

    assert len(result) == 1
    assert result[0].paper_id == paper_id
    assert result[0].dimension == "limitations"
    assert result[0].evidence == []


def test_foreign_evidence_is_dropped_at_service_boundary():
    requested_paper, foreign_paper = uuid4(), uuid4()
    retriever = FakeRetriever({requested_paper: [_evidence(foreign_paper)]})

    result = ComparisonEvidenceRetriever(retriever).retrieve(
        object(), uuid4(), [requested_paper], ["dataset"]
    )

    assert result[0].evidence == []


def test_query_is_deterministic_whitespace_normalized_and_bounded():
    paper_id = uuid4()
    retriever = FakeRetriever()
    result = ComparisonEvidenceRetriever(retriever).retrieve(
        object(),
        uuid4(),
        [paper_id],
        ["  method   and architecture  "],
        comparison_question="  How   does it work?  ",
    )

    assert result[0].query == ("How does it work?\nComparison dimension: method and architecture")
    assert len(result[0].query) <= 2 * 240 + len("\nComparison dimension: ")

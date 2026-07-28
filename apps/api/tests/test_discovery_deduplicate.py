import pytest

from app.schemas.discovery import CatalogCandidate
from app.services.discovery.deduplicate import deduplicate_candidates


def _candidate(**updates) -> CatalogCandidate:
    values = {
        "catalog": "openalex",
        "catalog_id": "https://openalex.org/W1",
        "title": "Attention Is All You Need",
        "source_url": "https://openalex.org/W1",
    }
    values.update(updates)
    return CatalogCandidate(**values)


def test_identical_doi_coalesces_and_fills_missing_metadata():
    first = _candidate(doi="https://doi.org/10.1234/ABC")
    second = _candidate(
        catalog="arxiv",
        catalog_id="1706.03762",
        doi="10.1234/abc.",
        arxiv_id="1706.03762v7",
        abstract="A useful abstract",
        authors=["Author"],
        source_url="https://arxiv.org/abs/1706.03762v7",
    )

    results = deduplicate_candidates([first, second])

    assert len(results) == 1
    assert results[0].arxiv_id == "1706.03762v7"
    assert results[0].abstract == "A useful abstract"
    assert results[0].authors == ["Author"]


def test_arxiv_versions_and_distinct_published_version_remain_inspectable():
    preprint = _candidate(catalog="arxiv", catalog_id="2401.11111v1", arxiv_id="2401.11111v1")
    later_preprint = _candidate(catalog="arxiv", catalog_id="2401.11111v2", arxiv_id="2401.11111v2")
    published = _candidate(
        catalog="openalex", catalog_id="https://openalex.org/W2", doi="10.1234/published"
    )

    results = deduplicate_candidates([preprint, later_preprint, published])

    assert len(results) == 3


def test_title_similarity_only_marks_possible_duplicate():
    first = _candidate(doi="10.1234/first")
    second = _candidate(
        catalog_id="https://openalex.org/W2",
        title="Attention Is All You Need!",
        doi="10.1234/second",
    )

    results = deduplicate_candidates([first, second])

    assert len(results) == 2
    assert all(candidate.possible_duplicate for candidate in results)


def test_same_title_without_identifier_does_not_remove_candidate():
    first = _candidate()
    second = _candidate(catalog_id="https://openalex.org/W2")

    results = deduplicate_candidates([first, second])

    assert len(results) == 2
    assert all(candidate.possible_duplicate for candidate in results)


def test_candidate_count_is_bounded():
    with pytest.raises(ValueError, match="at most 50"):
        deduplicate_candidates([_candidate(catalog_id=str(index)) for index in range(51)])

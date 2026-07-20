from app.services.claim_validation import (
    claim_clauses_for_citations,
    is_explicit_comparison,
)


def test_claim_clauses_split_at_citations_and_keep_trailing_text():
    sentence = "Paper A reports 2.1% ECE [E1], whereas Paper B reports 8.5% [E2]."
    citations = [
        (sentence.index("[E1]"), sentence.index("[E1]") + 4),
        (sentence.index("[E2]"), sentence.index("[E2]") + 4),
    ]

    clauses = claim_clauses_for_citations(
        sentence,
        citations,
        clean_sentence="Paper A reports 2.1% ECE, whereas Paper B reports 8.5%.",
    )

    assert clauses == ["Paper A reports 2.1% ECE", "Paper B reports 8.5%."]


def test_single_citation_keeps_full_claim_and_comparison_detection_is_explicit():
    sentence = "The method improved accuracy [E1]."
    citation = (sentence.index("[E1]"), sentence.index("[E1]") + 4)

    assert claim_clauses_for_citations(
        sentence, [citation], clean_sentence="The method improved accuracy."
    ) == ["The method improved accuracy."]
    assert is_explicit_comparison("Paper A is lower than Paper B")
    assert not is_explicit_comparison("Paper A reports an accuracy result")


def test_uncited_tail_remains_attached_to_last_cited_clause():
    sentence = "Evidence from A [E1], and this proves the method is universally best."
    citation = (sentence.index("[E1]"), sentence.index("[E1]") + 4)

    assert claim_clauses_for_citations(
        sentence,
        [citation],
        clean_sentence="Evidence from A, and this proves the method is universally best.",
    ) == ["Evidence from A, and this proves the method is universally best."]

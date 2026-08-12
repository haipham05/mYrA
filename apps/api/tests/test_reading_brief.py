from app.services.reading_brief import parse_reading_brief_sections


def test_reading_brief_sections_keep_citations_and_mark_missing_evidence():
    sections = parse_reading_brief_sections(
        "## Research question\nThe authors study sequence modeling [1].\n"
        "## Method\nThey use self-attention layers [2].\n"
        "## Limitations\n"
    )

    by_key = {section["key"]: section for section in sections}
    assert by_key["research_question"]["content"] == ("The authors study sequence modeling [1].")
    assert by_key["research_question"]["citation_indexes"] == [1]
    assert by_key["method"]["citation_indexes"] == [2]
    assert by_key["limitations"]["content"] == "Not found in retrieved evidence."
    assert by_key["evaluation_setup"]["content"] == ("Not returned in the reading-brief format.")


def test_reading_brief_accepts_bold_heading_style_without_inventing_sections():
    sections = parse_reading_brief_sections("**Results**\nThe model improves by 2 points [1].")

    results = next(section for section in sections if section["key"] == "results")
    assert results["content"] == "The model improves by 2 points [1]."
    assert results["citation_indexes"] == [1]


def test_reading_brief_does_not_treat_source_bibliography_as_our_citation():
    sections = parse_reading_brief_sections(
        "## Results\nThe paper states “We used learned positional embeddings [9].” [5]. "
        "The BLEU score was 41.8 [2]."
    )

    results = next(section for section in sections if section["key"] == "results")
    assert results["content"].endswith("[2].")
    assert results["citation_indexes"] == [2, 5]

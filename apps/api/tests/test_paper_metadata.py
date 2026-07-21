from app.ingestion.parser import ParsedElement, _extract_docling_title


def test_docling_title_extraction_uses_only_explicit_bounded_title_elements():
    elements = [
        ParsedElement(0, 1, "paragraph", "A first-page heading that is not labelled as a title"),
        ParsedElement(1, 1, "title", "  Explicit paper title  "),
        ParsedElement(2, 1, "title", "A later title"),
    ]

    assert _extract_docling_title(elements) == "Explicit paper title"


def test_docling_title_extraction_keeps_unknown_for_missing_or_unbounded_values():
    assert _extract_docling_title([ParsedElement(0, 1, "paragraph", "Only a paragraph")]) is None
    assert _extract_docling_title([ParsedElement(0, 1, "title", " " * 2001)]) is None

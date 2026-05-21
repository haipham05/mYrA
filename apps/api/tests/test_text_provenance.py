from pathlib import Path

import pytest
from pypdf import PdfReader

from app.ingestion.parser import (
    DocumentParser,
    find_verbatim_span,
    normalize_text,
)


def test_normalize_text_ligatures_and_quotes():
    # Test fi ligature and curly quotes
    raw = "The \ufb01rst paper uses “attention” and \ufb02exible self\xadattention."
    normalized = normalize_text(raw)
    assert normalized == 'The first paper uses "attention" and flexible selfattention.'


def test_find_verbatim_span_exact_and_normalized():
    page_text = "Attention Is All You Need. We propose the Transformer, a novel neural network."
    quote = "We propose the Transformer"

    span = find_verbatim_span(page_text, quote)
    assert span is not None
    start, end = span
    assert page_text[start:end] == quote


def test_find_verbatim_span_with_differing_whitespace():
    page_text = "Attention Is All You Need.\n\nWe propose the Transformer."
    quote = "We propose the Transformer."

    span = find_verbatim_span(page_text, quote)
    assert span is not None


def test_find_verbatim_span_missing_returns_none():
    page_text = "Attention Is All You Need."
    quote = "Convolutional neural network"

    span = find_verbatim_span(page_text, quote)
    assert span is None


def test_find_verbatim_span_reversible_mapping_ligatures():
    # Page text with ligatures, multiple spaces, and newlines
    page_text = "Here is the \ufb01nal   result.\nIt was verified."
    quote = "final result."

    span = find_verbatim_span(page_text, quote)
    assert span is not None
    start, end = span
    # Slicing the raw text must recover the exact raw characters
    assert page_text[start:end] == "\ufb01nal   result."


def test_find_verbatim_span_repeated_text_ambiguity_returns_none():
    # Repeated sentence on the same page without disambiguating signal
    page_text = (
        "Section 1: The model achieves state-of-the-art results. "
        "Section 5: The model achieves state-of-the-art results."
    )
    quote = "The model achieves state-of-the-art results."

    span = find_verbatim_span(page_text, quote)
    # Must return None (UNRESOLVED) to prevent false exact highlight
    assert span is None


def test_find_verbatim_span_repeated_text_disambiguation_with_preferred_offset():
    page_text = (
        "Section 1: The model achieves state-of-the-art results. "
        "Section 5: The model achieves state-of-the-art results."
    )
    quote = "The model achieves state-of-the-art results."

    # Only an independently known exact offset may identify an occurrence.
    second_start = page_text.rfind(quote)
    span_second = find_verbatim_span(page_text, quote, preferred_char_start=second_start)
    assert span_second is not None
    assert span_second[0] >= 50
    assert page_text[span_second[0] : span_second[1]] == quote

    first_start = page_text.find(quote)
    span_first = find_verbatim_span(page_text, quote, preferred_char_start=first_start)
    assert span_first is not None
    assert span_first[0] < 30
    assert page_text[span_first[0] : span_first[1]] == quote
    assert find_verbatim_span(page_text, quote, preferred_char_start=first_start + 1) is None


def test_parser_never_invents_fallback_50_50_boxes():
    # PDF with plain text
    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 50 >> stream
BT
/F1 12 Tf
72 712 Td
(Deep residual learning for image recognition.) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000345 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
421
%%EOF"""

    parser = DocumentParser()
    result = parser.parse(pdf_bytes)
    assert len(result.pages) == 1
    assert len(result.elements) >= 1

    for elem in result.elements:
        # Assert none of the elements have the old fabricated 50.0, 50.0 coordinates
        if elem.bbox_x_min is not None:
            assert not (elem.bbox_x_min == 50.0 and elem.bbox_y_min == 50.0)


def test_parser_rejects_empty_scanned_pdf():
    # A valid PDF structure but with 0 text content
    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
187
%%EOF"""

    parser = DocumentParser()
    with pytest.raises(ValueError, match="no selectable text layer"):
        parser.parse(pdf_bytes)


def test_docling_failure_does_not_silently_use_pypdf(monkeypatch):
    parser = DocumentParser(use_docling=True)

    def fail_docling(_pdf_bytes):
        raise RuntimeError("Docling assets unavailable")

    def unexpected_fallback(_pdf_bytes):
        raise AssertionError("pypdf fallback must not run")

    monkeypatch.setattr(parser, "_parse_with_docling", fail_docling)
    monkeypatch.setattr(parser, "_parse_with_pypdf", unexpected_fallback)

    with pytest.raises(RuntimeError, match="Docling assets unavailable"):
        parser.parse(b"%PDF")


def test_docling_page_text_is_independent_of_its_elements():
    fixture = (
        Path(__file__).resolve().parents[3] / "tests/retrieval_eval/fixtures/devlin2018_bert.pdf"
    )
    source = fixture.read_bytes()
    result = DocumentParser(use_docling=True).parse(source)
    reader = PdfReader(fixture)

    assert len(result.pages) == len(reader.pages)
    assert result.pages[4].raw_text == reader.pages[4].extract_text()
    assert result.pages[4].raw_text
    assert result.elements[4].element_type == "text"
    assert result.pages[4].crop_box is not None
    assert find_verbatim_span(result.pages[4].raw_text, result.elements[4].text)

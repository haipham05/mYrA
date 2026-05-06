import pytest

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

import io
from dataclasses import dataclass, field

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    """Normalize ligatures, unicode quotes, dashes, soft-hyphens, and whitespace."""
    if not text:
        return ""
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xad": "",  # soft hyphen
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return " ".join(text.split())


def find_verbatim_span(page_text: str, quote: str) -> tuple[int, int] | None:
    """Find exact character span [start, end] of quote within page_text using
    normalized comparison.
    """
    norm_page = normalize_text(page_text)
    norm_quote = normalize_text(quote)
    if not norm_quote or norm_quote not in norm_page:
        return None
    start = norm_page.find(norm_quote)
    end = start + len(norm_quote)
    return (start, end)


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    rotation: int = 0
    crop_box: dict[str, float] | None = None


@dataclass
class ParsedElement:
    element_index: int
    page_number: int
    element_type: str
    text: str
    bbox_x_min: float | None = None
    bbox_y_min: float | None = None
    bbox_x_max: float | None = None
    bbox_y_max: float | None = None
    page_width: float | None = None
    page_height: float | None = None
    coordinate_origin: str = "TOP_LEFT"
    rotation: int = 0
    section_path: list[str] = field(default_factory=list)
    parser_version: str = "pypdf-v2-nofake"


@dataclass
class ParseResult:
    pages: list[ParsedPage]
    elements: list[ParsedElement]


class DocumentParser:
    """PDF parser extracting canonical pages and elements with true bounding coordinates."""

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[ParsedPage] = []
        elements: list[ParsedElement] = []
        element_idx = 0

        for page_num_0, page in enumerate(reader.pages):
            page_number = page_num_0 + 1
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            rotation = int(page.get("/Rotate", 0) or 0)

            crop_box = None
            if page.cropbox:
                crop_box = {
                    "left": float(page.cropbox.left),
                    "bottom": float(page.cropbox.bottom),
                    "right": float(page.cropbox.right),
                    "top": float(page.cropbox.top),
                }

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    width=width,
                    height=height,
                    rotation=rotation,
                    crop_box=crop_box,
                )
            )

            # Extract text blocks / lines with visitor
            extracted_blocks: list[tuple[str, tuple[float, float, float, float]]] = []

            def visitor_body(text, cm, tm, font_dict, font_size):
                if text and text.strip():
                    # Handle transformation matrix
                    if cm and len(cm) >= 6:
                        raw_x = tm[4]
                        raw_y = tm[5]
                        x = raw_x * cm[0] + raw_y * cm[2] + cm[4]
                        y = raw_x * cm[1] + raw_y * cm[3] + cm[5]
                    else:
                        x = tm[4]
                        y = tm[5]

                    approx_w = max(len(text) * (font_size * 0.5), 8.0)
                    approx_h = max(font_size, 8.0)
                    y_top = height - y
                    extracted_blocks.append((text, (x, y_top - approx_h, x + approx_w, y_top)))

            try:
                page.extract_text(visitor_text=visitor_body)
            except Exception:
                extracted_blocks = []

            if not extracted_blocks:
                # Fallback to plain extract_text: NEVER invent fake boxes!
                plain_text = page.extract_text() or ""
                paragraphs = [p.strip() for p in plain_text.split("\n\n") if p.strip()]
                if not paragraphs and plain_text.strip():
                    paragraphs = [plain_text.strip()]

                for para in paragraphs:
                    elements.append(
                        ParsedElement(
                            element_index=element_idx,
                            page_number=page_number,
                            element_type="paragraph",
                            text=normalize_text(para),
                            bbox_x_min=None,
                            bbox_y_min=None,
                            bbox_x_max=None,
                            bbox_y_max=None,
                            page_width=width,
                            page_height=height,
                            coordinate_origin="TOP_LEFT",
                            rotation=rotation,
                            section_path=[],
                        )
                    )
                    element_idx += 1
            else:
                # Group text fragments into line/paragraph elements
                current_text: list[str] = []
                current_boxes: list[tuple[float, float, float, float]] = []

                for text, box in extracted_blocks:
                    current_text.append(text)
                    current_boxes.append(box)
                    if "\n" in text or len(current_text) >= 10:
                        combined_text = normalize_text(" ".join(current_text))
                        if combined_text:
                            x_mins = [b[0] for b in current_boxes]
                            y_mins = [b[1] for b in current_boxes]
                            x_maxs = [b[2] for b in current_boxes]
                            y_maxs = [b[3] for b in current_boxes]
                            elements.append(
                                ParsedElement(
                                    element_index=element_idx,
                                    page_number=page_number,
                                    element_type="paragraph",
                                    text=combined_text,
                                    bbox_x_min=max(0.0, min(x_mins)),
                                    bbox_y_min=max(0.0, min(y_mins)),
                                    bbox_x_max=min(width, max(x_maxs)),
                                    bbox_y_max=min(height, max(y_maxs)),
                                    page_width=width,
                                    page_height=height,
                                    coordinate_origin="TOP_LEFT",
                                    rotation=rotation,
                                    section_path=[],
                                )
                            )
                            element_idx += 1
                        current_text = []
                        current_boxes = []

                if current_text:
                    combined_text = normalize_text(" ".join(current_text))
                    if combined_text:
                        x_mins = [b[0] for b in current_boxes]
                        y_mins = [b[1] for b in current_boxes]
                        x_maxs = [b[2] for b in current_boxes]
                        y_maxs = [b[3] for b in current_boxes]
                        elements.append(
                            ParsedElement(
                                element_index=element_idx,
                                page_number=page_number,
                                element_type="paragraph",
                                text=combined_text,
                                bbox_x_min=max(0.0, min(x_mins)),
                                bbox_y_min=max(0.0, min(y_mins)),
                                bbox_x_max=min(width, max(x_maxs)),
                                bbox_y_max=min(height, max(y_maxs)),
                                page_width=width,
                                page_height=height,
                                coordinate_origin="TOP_LEFT",
                                rotation=rotation,
                                section_path=[],
                            )
                        )
                        element_idx += 1

        # Check for scanned / empty text layer
        total_text_len = sum(len(e.text.strip()) for e in elements)
        if total_text_len == 0 and len(pages) > 0:
            raise ValueError(
                "Document contains no selectable text layer (scanned or image-only PDF). "
                "An OCR text layer is required."
            )

        return ParseResult(pages=pages, elements=elements)

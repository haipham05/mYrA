import io
from dataclasses import dataclass, field

from pypdf import PdfReader


def normalize_text_with_mapping(raw_text: str) -> tuple[str, list[int]]:
    """Normalize ligatures, unicode quotes, dashes, soft-hyphens, and whitespace,
    returning (normalized_text, norm_to_raw_indices).
    norm_to_raw_indices[i] maps character i of normalized_text back to its
    source character index in raw_text.
    """
    if not raw_text:
        return "", []
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

    # Step 1: Character substitution and expansion tracking original raw index
    exp_chars: list[str] = []
    exp_raw: list[int] = []
    for raw_idx, ch in enumerate(raw_text):
        if ch in replacements:
            for rep_ch in replacements[ch]:
                exp_chars.append(rep_ch)
                exp_raw.append(raw_idx)
        else:
            exp_chars.append(ch)
            exp_raw.append(raw_idx)

    # Step 2: Strip leading whitespace
    first = 0
    while first < len(exp_chars) and exp_chars[first].isspace():
        first += 1

    # Strip trailing whitespace
    last = len(exp_chars) - 1
    while last >= 0 and exp_chars[last].isspace():
        last -= 1

    # Step 3: Collapse internal whitespace runs
    norm_chars: list[str] = []
    norm_to_raw: list[int] = []
    in_ws = False

    for i in range(first, last + 1):
        ch = exp_chars[i]
        raw_idx = exp_raw[i]
        if ch.isspace():
            if not in_ws:
                norm_chars.append(" ")
                norm_to_raw.append(raw_idx)
                in_ws = True
        else:
            norm_chars.append(ch)
            norm_to_raw.append(raw_idx)
            in_ws = False

    return "".join(norm_chars), norm_to_raw


def normalize_text(text: str) -> str:
    """Normalize ligatures, unicode quotes, dashes, soft-hyphens, and whitespace."""
    norm_str, _ = normalize_text_with_mapping(text)
    return norm_str


def find_verbatim_span(
    page_text: str,
    quote: str,
    preferred_char_start: int | None = None,
) -> tuple[int, int] | None:
    """Find exact character span [start, end] of quote within page_text using
    reversible normalized-to-raw offset mapping.

    Repeated text is ambiguous unless the caller supplies an independently
    established exact source offset. A nearby offset is not sufficient.
    """
    if not page_text or not quote:
        return None

    norm_page, mapping = normalize_text_with_mapping(page_text)
    norm_quote, _ = normalize_text_with_mapping(quote)

    if not norm_quote or not norm_page or len(mapping) != len(norm_page):
        return None

    candidates: list[tuple[int, int]] = []
    idx = norm_page.find(norm_quote)
    while idx != -1:
        candidates.append((idx, idx + len(norm_quote)))
        idx = norm_page.find(norm_quote, idx + 1)

    if not candidates:
        return None

    if len(candidates) == 1:
        s, e = candidates[0]
        return (mapping[s], mapping[e - 1] + 1)

    # Only an exact, independent source offset can identify a repeated occurrence.
    if preferred_char_start is not None:
        exact = [(s, e) for s, e in candidates if mapping[s] == preferred_char_start]
        if len(exact) == 1:
            s, e = exact[0]
            return (mapping[s], mapping[e - 1] + 1)

    # Ambiguous repeated text without disambiguating context: return None
    return None


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    rotation: int = 0
    crop_box: dict[str, float] | None = None
    raw_text: str = ""


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
    """PDF parser extracting canonical pages and elements with true bounding coordinates.

    Docling is required by default. pypdf is an explicitly selected demo/test parser.
    """

    def __init__(self, use_docling: bool | None = None) -> None:
        if use_docling is None:
            import os

            self.use_docling = os.getenv("MYRA_USE_DOCLING", "true").lower() in ("true", "1")
        else:
            self.use_docling = use_docling

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        if self.use_docling:
            return self._parse_with_docling(pdf_bytes)
        return self._parse_with_pypdf(pdf_bytes)

    def _parse_with_docling(self, pdf_bytes: bytes) -> ParseResult:
        import docling
        from docling.datamodel.base_models import DocumentStream, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        converter = DocumentConverter(format_options=format_options)

        doc_stream = DocumentStream(name="document.pdf", stream=io.BytesIO(pdf_bytes))
        res = converter.convert(doc_stream)
        doc = res.document
        source_reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(source_reader.pages) != len(doc.pages):
            raise ValueError("Docling page count does not match the source PDF")

        elements: list[ParsedElement] = []
        element_idx = 0
        current_section: list[str] = []

        for item in doc.texts:
            for prov in item.prov:
                start_char, end_char = prov.charspan
                elem_text = normalize_text(item.text[start_char:end_char].strip())
                if not elem_text:
                    continue

                page = doc.pages.get(prov.page_no)
                page_w = float(page.size.width) if page else 612.0
                page_h = float(page.size.height) if page else 792.0

                bbox = prov.bbox
                x_min = float(bbox.l)
                x_max = float(bbox.r)
                y_min = float(page_h - bbox.t)
                y_max = float(page_h - bbox.b)
                element_type = str(item.label)
                if element_type in {"title", "section_header"}:
                    current_section = [elem_text]

                elements.append(
                    ParsedElement(
                        element_index=element_idx,
                        page_number=prov.page_no,
                        element_type=element_type,
                        text=elem_text,
                        bbox_x_min=max(0.0, x_min),
                        bbox_y_min=max(0.0, y_min),
                        bbox_x_max=min(page_w, x_max),
                        bbox_y_max=min(page_h, y_max),
                        page_width=page_w,
                        page_height=page_h,
                        coordinate_origin="TOP_LEFT",
                        rotation=int(source_reader.pages[prov.page_no - 1].get("/Rotate", 0) or 0),
                        section_path=current_section.copy(),
                        parser_version=f"docling-{docling.__version__}",
                    )
                )
                element_idx += 1

        pages: list[ParsedPage] = []
        for page_no, page in doc.pages.items():
            source_page = source_reader.pages[page_no - 1]
            crop = source_page.cropbox
            pages.append(
                ParsedPage(
                    page_number=page_no,
                    width=float(page.size.width),
                    height=float(page.size.height),
                    rotation=int(source_page.get("/Rotate", 0) or 0),
                    crop_box={
                        "left": float(crop.left),
                        "bottom": float(crop.bottom),
                        "right": float(crop.right),
                        "top": float(crop.top),
                    },
                    # This is deliberately independent of Docling's elements.
                    # If it cannot support a quote, the anchor stays unresolved.
                    raw_text=source_page.extract_text() or "",
                )
            )

        total_text_len = sum(len(e.text.strip()) for e in elements)
        if total_text_len == 0 and len(pages) > 0:
            raise ValueError(
                "Document contains no selectable text layer (scanned or image-only PDF). "
                "An OCR text layer is required."
            )

        return ParseResult(pages=pages, elements=elements)

    def _parse_with_pypdf(self, pdf_bytes: bytes) -> ParseResult:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[ParsedPage] = []
        elements: list[ParsedElement] = []
        element_idx = 0

        for page_num_0, page in enumerate(reader.pages):
            page_number = page_num_0 + 1
            width = float(page.mediabox.width)
            height = float(page.mediabox.height)
            rotation = int(page.get("/Rotate", 0) or 0)
            raw_text = page.extract_text() or ""

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
                    raw_text=raw_text,
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

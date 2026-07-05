import math
import re
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.memory import (
    atomic_create_and_supersede,
    create_memory,
    list_memories,
    record_memory_access,
)
from app.db.models import (
    ChunkElement,
    Conversation,
    Memory,
    MemorySource,
    Message,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
)
from app.ingestion.parser import find_verbatim_span
from app.observability.telemetry import get_telemetry
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)
from app.schemas.memory import (
    MemoryCreate,
    MemorySourceCreate,
    MemorySourceType,
    MemoryStatus,
    MemoryType,
)
from app.services.error_sanitizer import redact_secrets

# Regex patterns for extracting decisions, preferences, and terminology from utterances
DECISION_PATTERNS = [
    re.compile(
        r"(?:we\s+(?:have\s+)?decided\s+(?:to\s+|that\s+)|"
        r"let's\s+(?:decide\s+to\s+|go\s+with\s+|choose\s+|use\s+)|"
        r"decision:\s*|"
        r"we\s+will\s+(?:use|choose|adopt|focus\s+on)\s+)(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(r"choose\s+([A-Za-z0-9_-]+)\s+over\s+([A-Za-z0-9_-]+)", re.IGNORECASE),
]

PREFERENCE_PATTERNS = [
    re.compile(
        r"(?:i\s+prefer\s+|my\s+preference\s+is\s+|i\s+like\s+|"
        r"preference:\s*|please\s+always\s+)(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]

TERMINOLOGY_PATTERNS = [
    re.compile(
        r"(?:we\s+define\s+|definition:\s*|"
        r"([A-Za-z0-9_-]+)\s+(?:stands\s+for|means|is\s+defined\s+as)\s+)(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]

COMMON_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "because",
    "as",
    "what",
    "which",
    "this",
    "that",
    "these",
    "those",
    "then",
    "just",
    "so",
    "than",
    "such",
    "both",
    "through",
    "about",
    "for",
    "is",
    "of",
    "while",
    "during",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "only",
    "own",
    "same",
    "too",
    "very",
    "can",
    "will",
    "should",
    "now",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "could",
    "shall",
    "our",
    "their",
    "its",
    "we",
    "they",
    "it",
    "i",
    "you",
    "he",
    "she",
    "him",
    "her",
    "us",
    "them",
    "model",
    "paper",
    "papers",
    "author",
    "authors",
    "study",
    "work",
    "article",
    "proves",
    "prove",
    "shows",
    "show",
    "demonstrates",
    "demonstrate",
    "presents",
    "present",
    "finds",
    "find",
    "proposes",
    "propose",
    "use",
    "uses",
    "used",
    "using",
    "across",
    "between",
    "with",
    "into",
}

CONTRAST_OR_NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "fails",
    "fail",
    "failed",
    "failing",
    "cannot",
    "can't",
    "won't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "without",
    "unlikely",
    "disprove",
    "disproved",
    "contrary",
    "unlike",
    "whereas",
    "while",
}


def _check_claim_supported_by_text(claim: str, evidence: str) -> tuple[bool, str]:
    clean_claim = claim.strip()
    clean_evidence = evidence.strip()
    if not clean_claim:
        return False, "Claim is empty."
    if not clean_evidence:
        return False, "Evidence is empty."

    token_pattern = r"\d+(?:\.\d+)?%?|[a-zA-Z0-9_-]+"
    claim_tokens = re.findall(token_pattern, clean_claim.lower())
    evidence_tokens = re.findall(token_pattern, clean_evidence.lower())

    if not claim_tokens:
        return False, "Claim contains no substantive factual terms."
    if not evidence_tokens:
        return False, "Evidence contains no valid tokens."

    # 1. Negation and contrasting consistency
    negation_patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bneither\b",
        r"\bnor\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdid\s+not\b",
        r"\bdidn't\b",
        r"\bdoes\s+not\b",
        r"\bdoesn't\b",
        r"\bwas\s+not\b",
        r"\bwasn't\b",
        r"\bwithout\b",
        r"\bfail\b",
        r"\bfails\b",
        r"\bfailed\b",
        r"\bfailing\b",
        r"\bfailure\b",
        r"\bwhile\b",
        r"\bwhereas\b",
        r"\bunlike\b",
        r"\bcontrary\b",
        r"\bdisprove\b",
        r"\bdisproved\b",
    ]

    def has_negation(text: str) -> bool:
        t = text.lower()
        return any(re.search(pat, t) is not None for pat in negation_patterns)

    claim_neg = has_negation(clean_claim)
    evidence_neg = has_negation(clean_evidence)
    if evidence_neg and not claim_neg:
        return (
            False,
            "Quote contains negation or contrasting clause that contradicts positive claim.",
        )
    if claim_neg and not evidence_neg:
        return (
            False,
            "Claim asserts negation not supported by cited quote.",
        )

    # 2. Directional / Antonym opposition
    opposites = [
        (
            {
                "increase",
                "increased",
                "increasing",
                "higher",
                "gain",
                "gains",
                "gained",
                "grow",
                "growth",
                "grew",
            },
            {
                "decrease",
                "decreased",
                "decreasing",
                "lower",
                "loss",
                "losses",
                "lost",
                "drop",
                "dropped",
                "dropping",
                "reduce",
                "reduced",
                "reduction",
            },
        ),
        (
            {
                "improve",
                "improved",
                "improving",
                "better",
                "superior",
                "outperform",
                "outperforms",
                "outperformed",
            },
            {
                "worsen",
                "worsened",
                "worsening",
                "worse",
                "inferior",
                "underperform",
                "underperforms",
                "underperformed",
            },
        ),
        ({"positive", "positives"}, {"negative", "negatives"}),
        (
            {"above", "exceed", "exceeds", "exceeded", "greater"},
            {"below", "under", "less", "fewer"},
        ),
    ]
    claim_token_set = set(claim_tokens)
    evidence_token_set = set(evidence_tokens)
    for group_a, group_b in opposites:
        if (claim_token_set & group_a and evidence_token_set & group_b) or (
            claim_token_set & group_b and evidence_token_set & group_a
        ):
            return (
                False,
                "Claim asserts opposite directional or comparative relationship to quote.",
            )

    # 3. Numeric values preservation and monotonic sequence
    number_pattern = r"^\d+(?:\.\d+)?%?$"
    claim_numbers = [t for t in claim_tokens if re.match(number_pattern, t)]
    evidence_numbers = [t for t in evidence_tokens if re.match(number_pattern, t)]

    if claim_numbers:
        for num in claim_numbers:
            if num not in evidence_numbers:
                return False, f"Claim asserts numeric value '{num}' not present in quote."

        num_pos = 0
        for num in claim_numbers:
            while num_pos < len(evidence_numbers) and evidence_numbers[num_pos] != num:
                num_pos += 1
            if num_pos == len(evidence_numbers):
                return (
                    False,
                    "Claim inverts or alters the order of numerical values: "
                    f"'{' '.join(claim_numbers)}' vs quote numbers.",
                )
            num_pos += 1

    # 4. Substantive tokens extraction and grounding check
    substantive_claim_tokens = [w for w in claim_tokens if w not in COMMON_STOP_WORDS]
    if not substantive_claim_tokens:
        return False, "Claim contains no substantive factual terms."

    matched_tokens = [w for w in substantive_claim_tokens if w in evidence_token_set]
    unmatched_tokens = [w for w in substantive_claim_tokens if w not in evidence_token_set]

    # Foreign concept check
    if unmatched_tokens:
        unsupported_key_terms = [w for w in unmatched_tokens if len(w) >= 3]
        if unsupported_key_terms:
            return False, (
                f"Claim introduces unsupported concepts not present in quote: "
                f"{', '.join(unsupported_key_terms)}"
            )

    grounding_ratio = len(matched_tokens) / len(substantive_claim_tokens)
    if grounding_ratio < 0.70:
        return False, (
            f"Insufficient grounding ratio ({grounding_ratio:.1%}): "
            "claim concepts are not adequately supported by cited quote."
        )

    # 5. Monotonic token ordering to prevent entity role / relation reversal
    order_check_tokens = [
        t
        for t in claim_tokens
        if t in evidence_token_set
        and (
            t
            not in {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "in",
                "of",
                "to",
                "for",
                "on",
                "at",
            }
        )
    ]

    position = 0
    for token in order_check_tokens:
        while position < len(evidence_tokens) and evidence_tokens[position] != token:
            position += 1
        if position == len(evidence_tokens):
            return (
                False,
                "Claim reverses entity roles, relations, or ordering of concepts: "
                f"'{token}' not in monotonic sequence.",
            )
        position += 1

    return True, "Claim is supported by quote."


def verify_claim_supported_by_quote(
    claim: str,
    quote: str,
    page_text: str | None = None,
) -> tuple[bool, str]:
    """Verify that a paper fact claim is textually grounded in and supported by the cited quote.

    Enforces:
    - Non-empty claim and quote
    - Negation and contrast consistency
    - Directional / antonym opposition prevention (increase vs decrease, better vs worse)
    - Strict numeric preservation and monotonic sequence (e.g., 80 to 90 cannot be 90 to 80)
    - Substantive concept grounding directly in the cited quote (>= 70%)
    - Strict monotonic token ordering to prevent entity role reversal (e.g.,
      'Method B is better than Method A' vs 'Method A is better than Method B')
    - Rejection of foreign/unsupported concepts not present in quote
    - Strict quote binding: the cited quote itself must support the claim. Surrounding
      or adjacent page text cannot substitute for a mismatched or contradictory cited quote.

    Returns (is_supported: bool, reason: str).
    """
    return _check_claim_supported_by_text(claim, quote)


def extract_candidates_from_text(
    text: str,
    message_id: UUID | None = None,
) -> list[MemoryCreate]:
    """Extract candidate memories from raw text with automatic secret redaction."""
    safe_text = redact_secrets(text.strip())
    candidates: list[MemoryCreate] = []

    # 1. Check decisions
    for pat in DECISION_PATTERNS:
        matches = pat.finditer(safe_text)
        for m in matches:
            matched_str = m.group(0).strip()
            # If "choose X over Y"
            if len(m.groups()) == 2 and m.group(1) and m.group(2):
                x_val, y_val = m.group(1).strip(), m.group(2).strip()
                title = f"Decision: Choose {x_val} over {y_val}"
                content = f"Project decision: Selected {x_val} over {y_val}."
            else:
                body = m.group(1).strip() if m.group(1) else matched_str
                # Capitalize first letter
                body = body[0].upper() + body[1:] if body else body
                title = f"Decision: {body[:60]}"
                content = f"Project decision: {body}."

            source = (
                [MemorySourceCreate(source_type=MemorySourceType.MESSAGE, message_id=message_id)]
                if message_id
                else []
            )
            candidates.append(
                MemoryCreate(
                    memory_type=MemoryType.DECISION,
                    title=title,
                    content=content,
                    confidence=0.9,
                    importance=0.8,
                    is_pinned=False,
                    sources=source,
                )
            )

    # 2. Check preferences
    for pat in PREFERENCE_PATTERNS:
        matches = pat.finditer(safe_text)
        for m in matches:
            body = m.group(1).strip() if m.group(1) else m.group(0).strip()
            title = f"Preference: {body[:60]}"
            content = f"User preference: {body}."
            source = (
                [MemorySourceCreate(source_type=MemorySourceType.MESSAGE, message_id=message_id)]
                if message_id
                else []
            )
            candidates.append(
                MemoryCreate(
                    memory_type=MemoryType.PREFERENCE,
                    title=title,
                    content=content,
                    confidence=0.85,
                    importance=0.7,
                    is_pinned=False,
                    sources=source,
                )
            )

    # 3. Check terminology
    for pat in TERMINOLOGY_PATTERNS:
        matches = pat.finditer(safe_text)
        for m in matches:
            matched_str = m.group(0).strip()
            title = f"Terminology: {matched_str[:60]}"
            content = matched_str
            source = (
                [MemorySourceCreate(source_type=MemorySourceType.MESSAGE, message_id=message_id)]
                if message_id
                else []
            )
            candidates.append(
                MemoryCreate(
                    memory_type=MemoryType.TERMINOLOGY,
                    title=title,
                    content=content,
                    confidence=0.95,
                    importance=0.6,
                    is_pinned=False,
                    sources=source,
                )
            )

    return candidates


def _extract_decision_keywords(text: str) -> set[str]:
    """Extract normalized tokens from text for topic conflict detection."""
    words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", text.lower())
    stop_words = {
        "project",
        "decision",
        "user",
        "preference",
        "decided",
        "choose",
        "over",
        "selected",
        "will",
        "with",
        "from",
        "that",
        "this",
    }
    return {w for w in words if w not in stop_words}


def resolve_paper_memory_source(
    db: Session,
    project_id: UUID,
    source: MemorySource | MemorySourceCreate,
) -> tuple[EvidenceItem | None, CitationAnchor | None, AnchorStatus]:
    """Resolve a paper source and record its verification outcome without ORM serialization."""
    telemetry = get_telemetry()
    with telemetry.stage(
        "memory.source_resolution",
        metadata={"source_type": str(source.source_type)},
    ) as observation:
        result = _resolve_paper_memory_source(db, project_id, source)
        if observation is not None:
            evidence, anchor, status = result
            observation.update(
                output={"outcome": status.value, "verified": bool(evidence and anchor)}
            )
        return result


def _resolve_paper_memory_source(
    db: Session,
    project_id: UUID,
    source: MemorySource | MemorySourceCreate,
) -> tuple[EvidenceItem | None, CitationAnchor | None, AnchorStatus]:
    """Deterministically resolve a paper memory source to an M1-compatible verified EvidenceItem
    and CitationAnchor.

    Returns (EvidenceItem, CitationAnchor, AnchorStatus.VERIFIED) if unambiguous and verified,
    or (None, None, AnchorStatus.UNRESOLVED) otherwise.
    """
    if (
        not source.paper_id
        or not source.page_number
        or not source.quote_text
        or not source.quote_text.strip()
    ):
        return None, None, AnchorStatus.UNRESOLVED

    # 1. Verify paper ownership, status, and document SHA-256
    paper = (
        db.query(Paper).filter(Paper.id == source.paper_id, Paper.project_id == project_id).first()
    )
    if not paper or paper.status != "READY" or not paper.document_sha256:
        return None, None, AnchorStatus.UNRESOLVED

    if source.document_sha256 and source.document_sha256 != paper.document_sha256:
        return None, None, AnchorStatus.UNRESOLVED

    # 2. Locate matching PaperElement on this page
    elements = (
        db.query(PaperElement)
        .filter(
            PaperElement.paper_id == paper.id,
            PaperElement.page_number == source.page_number,
        )
        .order_by(PaperElement.element_index)
        .all()
    )
    if not elements:
        return None, None, AnchorStatus.UNRESOLVED

    # 3. Verify page text and locate unambiguous verbatim character span
    page = (
        db.query(PaperPage)
        .filter(PaperPage.paper_id == paper.id, PaperPage.page_number == source.page_number)
        .first()
    )
    page_text = page.raw_text if (page and page.raw_text) else ""
    if not page_text and elements:
        page_text = "\n\n".join(elem.text for elem in elements if elem.text)

    if not page_text:
        return None, None, AnchorStatus.UNRESOLVED

    span = find_verbatim_span(page_text, source.quote_text.strip())
    if span is None:
        # Quote missing or ambiguous (multiple occurrences without offset)
        return None, None, AnchorStatus.UNRESOLVED

    start_char, end_char = span

    best_elem: PaperElement | None = None
    normalized_quote = source.quote_text.strip().lower()
    for elem in elements:
        if elem.text and normalized_quote in elem.text.strip().lower():
            best_elem = elem
            break

    if best_elem is None:
        # Match element by highest word overlap on this page
        quote_words = set(re.findall(r"\w+", normalized_quote))
        best_score = 0
        for elem in elements:
            if not elem.text:
                continue
            elem_words = set(re.findall(r"\w+", elem.text.lower()))
            overlap = len(quote_words & elem_words)
            if overlap > best_score:
                best_score = overlap
                best_elem = elem

    if best_elem is None or not best_elem.parser_version:
        return None, None, AnchorStatus.UNRESOLVED

    # 4. Locate linked PaperChunk (child chunk)
    chunk: PaperChunk | None = None
    if getattr(source, "chunk_id", None):
        chunk = (
            db.query(PaperChunk)
            .filter(PaperChunk.id == source.chunk_id, PaperChunk.paper_id == paper.id)
            .first()
        )
        if not chunk:
            return None, None, AnchorStatus.UNRESOLVED

    if chunk is None:
        chunk_elem = db.query(ChunkElement).filter(ChunkElement.element_id == best_elem.id).first()
        if chunk_elem:
            chunk = (
                db.query(PaperChunk)
                .filter(PaperChunk.id == chunk_elem.chunk_id, PaperChunk.paper_id == paper.id)
                .first()
            )

    if chunk is None:
        candidate_chunks = (
            db.query(PaperChunk)
            .filter(PaperChunk.paper_id == paper.id, PaperChunk.chunk_type == "child")
            .all()
        )
        for c in candidate_chunks:
            if c.text and normalized_quote in c.text.lower():
                chunk = c
                break

    if chunk is None:
        # Missing or deleted chunk
        return None, None, AnchorStatus.UNRESOLVED

    # 5. Extract bounding boxes from best_elem
    bboxes: list[BoundingBox] = []
    if best_elem.bbox_x_min is not None and best_elem.page_width and best_elem.page_height:
        coord_origin = (
            CoordinateOrigin(best_elem.coordinate_origin)
            if best_elem.coordinate_origin
            else CoordinateOrigin.TOP_LEFT
        )
        bboxes.append(
            BoundingBox(
                x_min=best_elem.bbox_x_min,
                y_min=best_elem.bbox_y_min or 0.0,
                x_max=best_elem.bbox_x_max or 0.0,
                y_max=best_elem.bbox_y_max or 0.0,
                page_width=best_elem.page_width,
                page_height=best_elem.page_height,
                origin=coord_origin,
                rotation=best_elem.rotation or 0,
            )
        )

    # 6. Construct verified CitationAnchor and EvidenceItem
    anchor = CitationAnchor(
        page_number=source.page_number,
        source_element_id=best_elem.id,
        exact_quote=source.quote_text.strip(),
        source_char_start=start_char,
        source_char_end=end_char,
        document_sha256=paper.document_sha256,
        parser_version=best_elem.parser_version,
        anchor_status=AnchorStatus.VERIFIED,
        bounding_boxes=bboxes,
    )

    source_id_str = str(getattr(source, "id", "mem"))
    evidence_item = EvidenceItem(
        id=f"mem-{source_id_str}",
        paper_id=paper.id,
        paper_title=paper.filename,
        chunk_id=chunk.id,
        quote=source.quote_text.strip(),
        parent_context=chunk.text,
        page_number=source.page_number,
        bounding_boxes=bboxes,
        source_element_ids=[best_elem.id],
        document_sha256=paper.document_sha256,
        parser_version=best_elem.parser_version,
        anchors=[anchor],
    )

    return evidence_item, anchor, AnchorStatus.VERIFIED


def validate_memory_candidate(
    db: Session,
    project_id: UUID,
    candidate: MemoryCreate,
) -> None:
    """Validate memory candidate against project authority and paper/message sources.

    Enforces:
    - Secret redaction.
    - Message sources must exist and belong to a conversation in the project.
    - Paper facts require:
      - Paper belongs to project.
      - Paper status == 'READY'.
      - Non-empty quote text and valid page_number.
      - Matching document_sha256 if present.
      - Exact quote text has an unambiguous verified text span on the page
        (resolve_paper_memory_source).
      - Monotonic claim support strictly on the verified quote text.
    """
    candidate.title = redact_secrets(candidate.title)
    candidate.content = redact_secrets(candidate.content)

    if candidate.memory_type == MemoryType.PAPER_FACT and not candidate.sources:
        raise ValueError("Paper fact requires at least one paper source.")

    for s in candidate.sources:
        if s.source_type == MemorySourceType.MESSAGE and s.message_id:
            msg = (
                db.query(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .filter(Message.id == s.message_id, Conversation.project_id == project_id)
                .first()
            )
            if not msg:
                raise ValueError(
                    f"Message {s.message_id} does not exist in project {project_id}; "
                    "cannot forge message source memory."
                )

        if (
            s.source_type == MemorySourceType.PAPER_CHUNK
            or candidate.memory_type == MemoryType.PAPER_FACT
        ):
            if not s.paper_id:
                raise ValueError("Paper source requires a valid paper_id.")

            paper = (
                db.query(Paper)
                .filter(Paper.id == s.paper_id, Paper.project_id == project_id)
                .first()
            )
            if not paper:
                raise ValueError(
                    f"Paper {s.paper_id} does not exist in project {project_id}; "
                    "cannot forge paper fact memory."
                )
            if paper.status != "READY":
                raise ValueError(
                    f"Paper {s.paper_id} status is '{paper.status}'; "
                    "must be 'READY' to record paper facts."
                )
            if not s.quote_text or not s.quote_text.strip():
                raise ValueError("Paper fact requires valid non-empty quote text.")

            if s.document_sha256 and paper.document_sha256:
                if s.document_sha256 != paper.document_sha256:
                    raise ValueError(
                        f"Document SHA-256 mismatch for paper {s.paper_id}: "
                        f"expected {paper.document_sha256}, got {s.document_sha256}."
                    )
            elif paper.document_sha256 and not s.document_sha256:
                s.document_sha256 = paper.document_sha256

            normalized_quote = s.quote_text.strip().lower()

            # Infer page_number if omitted, or validate presence
            if s.page_number is None:
                pages = db.query(PaperPage).filter(PaperPage.paper_id == paper.id).all()
                matching_pages = [
                    p for p in pages if p.raw_text and normalized_quote in p.raw_text.lower()
                ]
                if len(matching_pages) == 1:
                    s.page_number = matching_pages[0].page_number
                elif len(matching_pages) == 0:
                    raise ValueError(f"Quote text was not found in paper {s.paper_id}.")
                else:
                    raise ValueError(
                        f"Quote text is ambiguous across multiple pages in paper {s.paper_id}; "
                        "specify page_number."
                    )
            elif s.page_number < 1:
                raise ValueError("Paper source requires a valid page_number >= 1.")

            if candidate.memory_type == MemoryType.PAPER_FACT:
                is_supported, support_reason = verify_claim_supported_by_quote(
                    candidate.content, s.quote_text.strip(), page_text=None
                )
                if not is_supported:
                    raise ValueError(
                        f"Paper fact claim is not supported by cited quote: {support_reason}"
                    )

            ev, anchor, status = resolve_paper_memory_source(db, project_id, s)
            if status != AnchorStatus.VERIFIED or not ev or not anchor:
                raise ValueError(
                    f"Paper fact cited quote has no unambiguous verified text span "
                    f"on page {s.page_number} of paper {s.paper_id}."
                )

            if not s.document_sha256 and anchor.document_sha256:
                s.document_sha256 = anchor.document_sha256
            if not getattr(s, "chunk_id", None) and ev.chunk_id:
                s.chunk_id = ev.chunk_id


def consolidate_memory_candidate(
    db: Session,
    project_id: UUID,
    candidate: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    """Validate, consolidate, and trace a deliberately selected memory candidate."""
    telemetry = get_telemetry()
    selected_candidate = {
        "memory_type": candidate.memory_type.value,
        "title": candidate.title,
        "content": candidate.content,
    }
    with telemetry.stage(
        "memory.consolidation",
        input={"candidate": selected_candidate},
        metadata={"source_count": len(candidate.sources)},
    ) as observation:
        try:
            memory = _consolidate_memory_candidate(db, project_id, candidate, embedding)
        except ValueError:
            if observation is not None:
                observation.update(output={"outcome": "rejected"})
            raise
        if observation is not None:
            observation.update(
                output={"outcome": "consolidated", "memory_type": memory.memory_type}
            )
        return memory


def _consolidate_memory_candidate(
    db: Session,
    project_id: UUID,
    candidate: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    """Idempotently insert or supersede memory in the target project.

    Enforces:
    - Secret redaction on all fields.
    - Verified paper/message sources.
    - Idempotency: exact content matches return existing active memory.
    - Conflict resolution: decisions on the same topic supersede older active decisions.
    - Semantic vector embedding generation.
    """
    validate_memory_candidate(db, project_id=project_id, candidate=candidate)
    safe_content = candidate.content

    # 1. Check exact content duplicate among active memories in this project
    existing_active, _ = list_memories(
        db,
        project_id=project_id,
        status=MemoryStatus.ACTIVE,
        memory_type=candidate.memory_type,
    )
    for mem in existing_active:
        if mem.content.strip().lower() == safe_content.strip().lower():
            # Exact duplicate: return existing memory (idempotent, skips embedding call)
            return mem

    if embedding is None:
        try:
            from app.services.embedding import get_embedding_provider

            provider = get_embedding_provider()
            embedding = provider.embed_query(safe_content)
        except Exception:
            embedding = None

    # 2. Check conflict / supersession for DECISION and PREFERENCE types
    if candidate.memory_type in (MemoryType.DECISION, MemoryType.PREFERENCE):
        cand_keywords = _extract_decision_keywords(safe_content)
        if cand_keywords:
            for mem in existing_active:
                mem_keywords = _extract_decision_keywords(mem.content)
                overlap = cand_keywords.intersection(mem_keywords)
                # If there is substantial topic overlap (e.g. "AURC", "ECE" or shared metric/model)
                if overlap and (len(overlap) >= 2 or any(len(w) >= 4 for w in overlap)):
                    # New memory atomically supersedes old memory
                    _old_mem, new_mem = atomic_create_and_supersede(
                        db,
                        project_id=project_id,
                        old_memory_id=mem.id,
                        expected_version=mem.version,
                        new_memory_in=candidate,
                        embedding=embedding,
                        reason=(
                            "New decision supersedes prior choice on topic: "
                            f"{', '.join(sorted(overlap))}"
                        ),
                    )
                    return new_mem

    # 3. Create fresh active memory
    return create_memory(db, project_id=project_id, memory_in=candidate, embedding=embedding)


def capture_conversation_memories(
    db: Session,
    project_id: UUID,
    conversation_id: UUID,
) -> list[Memory]:
    """Capture selected memories while tracing aggregate outcomes only."""
    telemetry = get_telemetry()
    with telemetry.stage("memory.capture") as observation:
        memories = _capture_conversation_memories(db, project_id, conversation_id)
        if observation is not None:
            observation.update(output={"captured_count": len(memories), "outcome": "completed"})
        return memories


def _capture_conversation_memories(
    db: Session,
    project_id: UUID,
    conversation_id: UUID,
) -> list[Memory]:
    """Extract and consolidate candidate memories from committed user messages of a conversation."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.project_id == project_id)
        .first()
    )
    if not conv:
        raise ValueError(f"Conversation {conversation_id} not found in project {project_id}.")

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    if not messages:
        return []

    created_memories: list[Memory] = []
    for msg in messages:
        # Only capture decisions and preferences from USER messages
        if msg.role.upper() != "USER":
            continue
        candidates = extract_candidates_from_text(msg.content, message_id=msg.id)
        for cand in candidates:
            mem = consolidate_memory_candidate(db, project_id=project_id, candidate=cand)
            created_memories.append(mem)

    return created_memories


def score_memory_relevance(
    memory: Memory,
    query_tokens: set[str],
    now: datetime,
    query_embedding: list[float] | None = None,
) -> float:
    """Compute ranking score for a memory given a query."""
    score = 0.0

    # 1. Pinning boost
    if memory.is_pinned:
        score += 5.0

    # 2. Importance score
    score += memory.importance * 3.0

    # 3. Recency decay (half-life of 30 days)
    age_days = max(0.0, (now - memory.created_at.replace(tzinfo=UTC)).total_seconds() / 86400.0)
    recency_factor = math.exp(-age_days / 30.0)
    score += recency_factor * 1.5

    # 4. Lexical token overlap with title & content
    if query_tokens:
        mem_tokens = set(
            re.findall(r"\b[A-Za-z0-9_-]{3,}\b", (memory.title + " " + memory.content).lower())
        )
        overlap = query_tokens.intersection(mem_tokens)
        score += len(overlap) * 2.5

    # 5. Semantic vector similarity if query and memory embeddings exist
    if query_embedding and memory.embedding:
        try:
            mem_emb = memory.embedding if isinstance(memory.embedding, list) else []
            if len(mem_emb) == len(query_embedding):
                dot_product = sum(a * b for a, b in zip(mem_emb, query_embedding, strict=False))
                norm_a = math.sqrt(sum(a * a for a in mem_emb))
                norm_b = math.sqrt(sum(b * b for b in query_embedding))
                if norm_a > 0 and norm_b > 0:
                    cosine = dot_product / (norm_a * norm_b)
                    score += max(0.0, cosine) * 4.0
        except Exception:
            pass

    return score


def retrieve_project_memories(
    db: Session,
    project_id: UUID,
    query: str,
    limit: int = 5,
    record_access: bool = True,
    query_embedding: list[float] | None = None,
) -> list[Memory]:
    """Look up project memories and trace selected query/results under retention policy."""
    telemetry = get_telemetry()
    with telemetry.stage(
        "memory.lookup",
        input={"query": query},
        metadata={"requested_limit": limit, "record_access": record_access},
    ) as observation:
        memories = _retrieve_project_memories(
            db,
            project_id,
            query,
            limit=limit,
            record_access=record_access,
            query_embedding=query_embedding,
        )
        if observation is not None:
            observation.update(
                output={
                    "memory_count": len(memories),
                    "memory_types": sorted({memory.memory_type for memory in memories}),
                    "memories": [
                        {"title": memory.title, "content": memory.content} for memory in memories
                    ],
                }
            )
        return memories


def _retrieve_project_memories(
    db: Session,
    project_id: UUID,
    query: str,
    limit: int = 5,
    record_access: bool = True,
    query_embedding: list[float] | None = None,
) -> list[Memory]:
    """Retrieve top-k active memories strictly scoped to the specified project."""
    active_memories, _ = list_memories(
        db,
        project_id=project_id,
        status=MemoryStatus.ACTIVE,
        limit=100,  # pull candidate pool for re-ranking
    )
    if not active_memories:
        return []

    now = datetime.now(UTC)
    query_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query.lower()))

    scored = [
        (score_memory_relevance(m, query_tokens, now, query_embedding=query_embedding), m)
        for m in active_memories
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    top_memories = [m for _, m in scored[:limit]]

    if record_access and top_memories:
        record_memory_access(db, [m.id for m in top_memories])

    return top_memories


def format_memories_for_prompt(
    memories: list[Memory],
    db: Session | None = None,
    include_paper_facts: bool = True,
) -> str:
    """Format memories into structured system prompt sections.

    Separates USER DECISIONS & PREFERENCES from VERIFIED PAPER EVIDENCE
    to prevent memory from masquerading as a paper citation.
    Paper facts whose source paper is unready, deleted, or missing are excluded.
    If include_paper_facts is False, paper facts are omitted so they can be routed
    as verified EvidenceItems in the evidence pipeline.
    """
    if not memories:
        return ""

    decision_prefs: list[str] = []
    terminology: list[str] = []
    paper_facts: list[str] = []
    general: list[str] = []

    for m in memories:
        if m.memory_type in (MemoryType.DECISION, MemoryType.PREFERENCE):
            pin_badge = "[PINNED] " if m.is_pinned else ""
            decision_prefs.append(f"- {pin_badge}{m.title}: {m.content}")
        elif m.memory_type == MemoryType.TERMINOLOGY:
            terminology.append(f"- {m.title}: {m.content}")
        elif m.memory_type == MemoryType.PAPER_FACT and include_paper_facts:
            valid_sources = []
            for s in m.sources:
                if s.source_type == MemorySourceType.PAPER_CHUNK.value and s.paper_id:
                    if db is not None:
                        ev, anchor, status = resolve_paper_memory_source(db, m.project_id, s)
                        if status == AnchorStatus.VERIFIED and ev and anchor:
                            is_supp, _ = verify_claim_supported_by_quote(
                                m.content, s.quote_text.strip(), page_text=None
                            )
                            if is_supp:
                                valid_sources.append(s)
                    elif s.paper is not None and s.paper.status == "READY":
                        is_supp, _ = verify_claim_supported_by_quote(
                            m.content, s.quote_text.strip(), page_text=None
                        )
                        if is_supp:
                            valid_sources.append(s)

            if valid_sources:
                paper_source_info = ""
                for s in valid_sources:
                    if s.quote_text:
                        paper_source_info = f' (Source quote: "{s.quote_text[:100]}...")'
                        break
                paper_facts.append(f"- {m.title}: {m.content}{paper_source_info}")

    sections = []
    if decision_prefs:
        sections.append(
            "PROJECT DECISIONS & USER PREFERENCES (Authority: User):\n" + "\n".join(decision_prefs)
        )
    if terminology:
        sections.append("PROJECT TERMINOLOGY & CONVENTIONS:\n" + "\n".join(terminology))
    if paper_facts:
        sections.append("ESTABLISHED PAPER FACTS:\n" + "\n".join(paper_facts))
    if general:
        sections.append("ADDITIONAL PROJECT KNOWLEDGE:\n" + "\n".join(general))

    return "\n\n".join(sections)

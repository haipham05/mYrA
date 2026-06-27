import logging
import re
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.chat import add_message, get_conversation
from app.db.models import Memory, Message, PaperElement, PaperPage
from app.ingestion.parser import find_verbatim_span
from app.schemas.chat import MessageResponse, MessageRole
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    Citation,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)
from app.schemas.memory import MemoryStatus, MemoryType
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.router import (
    GraphIntent,
    retrieve_graph_evidence,
    route_query_intent,
)
from app.services.llm import generate_with_metadata, get_llm_provider
from app.services.memory_service import (
    capture_conversation_memories,
    format_memories_for_prompt,
    resolve_paper_memory_source,
    retrieve_project_memories,
)
from app.services.retrieval import HybridRetriever

logger = logging.getLogger("myra.chat")

STOPWORDS = {
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
    "i",
    "we",
    "our",
    "ours",
    "you",
    "your",
    "yours",
    "he",
    "she",
    "it",
    "they",
    "them",
    "their",
    "theirs",
    "me",
    "him",
    "her",
    "us",
    "are",
    "was",
    "were",
    "be",
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
    "project",
    "decision",
    "preference",
}

DISCOURSE_PREFIX_TOKENS = {
    "according",
    "to",
    "our",
    "the",
    "project",
    "decision",
    "decisions",
    "preference",
    "preferences",
    "memory",
    "memories",
    "as",
    "per",
    "based",
    "on",
    "in",
    "for",
    "this",
    "we",
    "have",
    "decided",
    "chosen",
    "agreed",
    "preferred",
    "is",
    "was",
    "that",
    "note",
    "recall",
    "states",
    "stated",
    "says",
    "said",
}

NEGATION_PATTERNS = [
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
    r"\bfailed\b",
    r"\bfails\b",
    r"\bfailure\b",
]


def _has_negation(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) is not None for pat in NEGATION_PATTERNS)


def is_attributed_to_memories(
    sentence: str,
    memories: list[Memory],
) -> bool:
    """Verify that an uncited sentence is directly supported by a decision/preference memory.

    Rejects sentences that:
    - Merely share an entity or subject (e.g. 'AURC cures cancer')
    - Contain unsupported trailing/leading clauses or modifiers (e.g. '...mistakenly')
    - Invert or negate the memory polarity (e.g. 'We did not choose AURC')
    - Contain ungrounded foreign tokens outside standard discourse framing
    """
    valid_memories = [
        m
        for m in memories
        if (getattr(m, "status", None) == MemoryStatus.ACTIVE.value or not hasattr(m, "status"))
        and getattr(m, "memory_type", None) != MemoryType.PAPER_FACT.value
    ]
    if not valid_memories:
        return False

    sentence_clean = sentence.strip()
    if not sentence_clean:
        return False

    sentence_neg = _has_negation(sentence_clean)

    prefix_pattern = (
        r"^(?:"
        r"(?:according to|as per|per|based on|in)\s+"
        r"(?:(?:our|the)\s+)?(?:project\s+)?(?:decisions?|preferences?|memory)|"
        r"(?:according to|as per|per|based on|in)\s+(?:(?:our|the)\s+)?project|"
        r"for\s+(?:this|our|the)\s+project|"
        r"in\s+(?:our|the)\s+project|"
        r"project\s+(?:decision|preference|note)|"
        r"decision|"
        r"we\s+have\s+(?:decided|chosen|agreed|preferred)|"
        r"our\s+(?:project\s+)?(?:decision|preference)\s+(?:is|was)(?:\s+that)?|"
        r"the\s+project\s+decision\s+is\s+that|"
        r"as\s+decided|"
        r"note\s+that|"
        r"recall\s+that)[,:]?\s*"
    )
    stripped = re.sub(prefix_pattern, "", sentence_clean, flags=re.IGNORECASE).strip()

    for m in valid_memories:
        mem_content = (m.content or "").strip()
        mem_title = (m.title or "").strip()
        if not mem_content:
            continue

        clean_mem_content = (
            re.sub(prefix_pattern, "", mem_content, flags=re.IGNORECASE).strip() or mem_content
        )

        # 1. Negation polarity check
        mem_neg = _has_negation(mem_content)
        if sentence_neg != mem_neg:
            continue

        # 2. Check foreign substantive tokens in the whole sentence
        # Every substantive token in the sentence MUST be covered by memory content/title
        # or recognized discourse framing. Zero foreign substantive tokens permitted!
        sentence_tokens = [
            t
            for t in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", stripped.lower())
            if t not in DISCOURSE_PREFIX_TOKENS and t not in STOPWORDS
        ]
        mem_tokens = set(
            re.findall(
                r"\b[a-zA-Z0-9_-]{2,}\b",
                f"{mem_title} {mem_content} {clean_mem_content}".lower(),
            )
        )
        foreign_tokens = [t for t in sentence_tokens if t not in mem_tokens]
        if foreign_tokens:
            continue

        # 3. Exact equality of normalized content (ignoring case, spaces, and punctuation)
        norm_content = re.sub(r"\s+", " ", clean_mem_content.lower()).rstrip(".?!")
        norm_stripped = re.sub(r"\s+", " ", stripped.lower()).rstrip(".?!")
        if norm_content and (
            norm_content == norm_stripped
            or norm_content == re.sub(r"^we\s+", "", norm_stripped)
            or re.sub(r"^we\s+", "", norm_content) == norm_stripped
        ):
            return True

        # 4. Check quoted clause: if sentence wraps memory in quotes,
        # ensure the quote is supported and the unquoted remainder has zero foreign tokens
        quote_matches = list(re.finditer(r'["“]([^"”]+)["”]', sentence_clean))
        if quote_matches:
            for match in quote_matches:
                quoted = match.group(1).strip()
                if len(quoted.split()) >= 3 and (
                    check_claim_support(quoted, mem_content)
                    or check_claim_support(quoted, clean_mem_content)
                    or (mem_title and check_claim_support(quoted, f"{mem_title}: {mem_content}"))
                ):
                    remainder = (
                        sentence_clean[: match.start()] + " " + sentence_clean[match.end() :]
                    )
                    rem_tokens = [
                        t
                        for t in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", remainder.lower())
                        if t not in DISCOURSE_PREFIX_TOKENS and t not in STOPWORDS
                    ]
                    if not [t for t in rem_tokens if t not in mem_tokens]:
                        return True

        # 5. Check claim support on stripped sentence
        if stripped and (
            check_claim_support(stripped, mem_content)
            or check_claim_support(stripped, clean_mem_content)
            or check_claim_support(re.sub(r"^we\s+", "", stripped), clean_mem_content)
            or (mem_title and check_claim_support(stripped, f"{mem_title}: {mem_content}"))
        ):
            return True

    return False


def check_claim_support(claim_text: str, evidence_quote: str) -> bool:
    """Accept only extractive claims whose tokens retain the quote's order.

    Word-set overlap is not evidence of a relationship: it accepts reversed actors
    and swapped measurements. This intentionally rejects many valid paraphrases;
    uncertain model output must abstain rather than invent a supported claim.
    """
    if not claim_text.strip() or not evidence_quote.strip():
        return False

    # Normalize decimal points separated by spaces (common in PDF extraction like "41 . 0")
    claim_text = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", claim_text)
    evidence_quote = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", evidence_quote)

    # Keep number/unit tokens intact: 90%, 90, and 90 mg are not interchangeable.
    token_pattern = r"\d+(?:\.\d+)?%?|[a-zA-Z]+"
    claim_tokens = re.findall(token_pattern, claim_text.lower())
    quote_tokens = re.findall(token_pattern, evidence_quote.lower())
    if len(claim_tokens) < 3 or not quote_tokens:
        return False

    # 2. Negation consistency
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
        r"\bfailed\b",
        r"\bfails\b",
        r"\bfailure\b",
    ]

    def has_negation(text: str) -> bool:
        t = text.lower()
        return any(re.search(pat, t) is not None for pat in negation_patterns)

    claim_neg = has_negation(claim_text)
    quote_neg = has_negation(evidence_quote)
    if claim_neg != quote_neg:
        return False

    # 3. Directional / antonym opposition
    opposites = [
        (
            {"increase", "increased", "increasing", "higher", "gain"},
            {"decrease", "decreased", "decreasing", "lower", "loss"},
        ),
        (
            {"improve", "improved", "improving", "better"},
            {"worsen", "worsened", "worsening", "worse"},
        ),
        ({"positive"}, {"negative"}),
        ({"above", "exceed", "exceeds"}, {"below", "under"}),
    ]
    claim_words = set(re.findall(r"\w+", claim_text.lower()))
    quote_words = set(re.findall(r"\w+", evidence_quote.lower()))

    for group_a, group_b in opposites:
        if (claim_words & group_a and quote_words & group_b) or (
            claim_words & group_b and quote_words & group_a
        ):
            return False

    # A monotonic subsequence allows small descriptive insertions in the source,
    # but cannot swap actors, values, negation scope, or comparison direction.
    position = 0
    for token in claim_tokens:
        while position < len(quote_tokens) and quote_tokens[position] != token:
            position += 1
        if position == len(quote_tokens):
            return False
        position += 1
    return True


def extract_verbatim_quoted_phrase(claim_text: str, evidence_quote: str) -> str | None:
    """Keep a model-quoted or bolded source phrase, never its unsupported surrounding prose."""
    norm_evidence = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", evidence_quote)
    for match in re.finditer(r'(?:["“]([^"”]+)["”]|(?:\*\*|__)([^*_]+)(?:\*\*|__))', claim_text):
        candidate = (match.group(1) or match.group(2) or "").strip()
        if not candidate:
            continue
        norm_candidate = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", candidate)
        if len(re.findall(r"\b\w+\b", norm_candidate)) < 3:
            continue
        candidate_tokens = re.findall(r"\w+|[^\w\s]", norm_candidate)
        if not candidate_tokens:
            continue
        flex_pattern = r"\s*".join(re.escape(tok) for tok in candidate_tokens)
        source_match = re.search(flex_pattern, norm_evidence, flags=re.IGNORECASE)
        if source_match is None:
            continue
        source_phrase = source_match.group(0)
        if check_claim_support(source_phrase, evidence_quote):
            return source_phrase
    return None


def find_substantive_span(claim_text: str, evidence_quote: str, min_words: int = 4) -> str | None:
    norm_evidence = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", evidence_quote)
    norm_claim = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", claim_text)
    claim_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", norm_claim)
    quote_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", norm_evidence))
    for num in claim_numbers:
        if num not in quote_numbers:
            return None

    words = re.findall(r"\b\w+\b", norm_claim)
    max_search_len = min(len(words), 30)
    for length in range(max_search_len, min_words - 1, -1):
        for start in range(len(words) - length + 1):
            sub_words = words[start : start + length]
            pattern = r"\s*".join(re.escape(w) for w in sub_words)
            m = re.search(pattern, norm_evidence, flags=re.IGNORECASE)
            if m:
                matched_span = m.group(0)
                if check_claim_support(matched_span, evidence_quote):
                    return matched_span
    return None


def matching_verified_anchor(evidence: EvidenceItem) -> CitationAnchor | None:
    if not evidence.document_sha256 or not evidence.parser_version:
        return None
    return next(
        (
            anchor
            for anchor in evidence.anchors
            if anchor.anchor_status == AnchorStatus.VERIFIED
            and anchor.page_number == evidence.page_number
            and anchor.exact_quote == evidence.quote
            and anchor.document_sha256 == evidence.document_sha256
            and anchor.parser_version == evidence.parser_version
            and anchor.source_element_id is not None
            and anchor.source_char_start is not None
            and anchor.source_char_end is not None
            and anchor.source_char_end > anchor.source_char_start
        ),
        None,
    )


def resolve_claim_anchor(
    db: Session,
    evidence: EvidenceItem,
    clean_claim: str,
    cite_count: int,
) -> tuple[CitationAnchor | None, str | None]:
    """Resolve a verified CitationAnchor supporting clean_claim from evidence.

    Returns (matched_anchor, extracted_verbatim_phrase).
    """
    if not evidence.document_sha256 or not evidence.parser_version:
        return None, None

    verified_anchors = [
        a
        for a in evidence.anchors
        if a.anchor_status == AnchorStatus.VERIFIED
        and a.source_char_start is not None
        and a.source_char_end is not None
        and a.source_char_end > a.source_char_start
    ]

    primary = matching_verified_anchor(evidence)
    if primary and primary in verified_anchors:
        verified_anchors.remove(primary)
        verified_anchors.insert(0, primary)

    # 1. Direct monotonic claim support against verified anchors
    for cand in verified_anchors:
        if check_claim_support(clean_claim, cand.exact_quote):
            return cand, None

    # 2. Extract verbatim quoted phrases (e.g. model output wrapped in quotes)
    if cite_count == 1:
        # Check if candidate quotes match any verified anchor
        for cand in verified_anchors:
            phrase = extract_verbatim_quoted_phrase(clean_claim, cand.exact_quote)
            if phrase is not None:
                return cand, phrase

        # Check if candidate quotes match across parent_context / canonical page text
        # (Quotes that span across multiple elements or lines in the PDF)
        for match in re.finditer(
            r'(?:["“]([^"”]+)["”]|(?:\*\*|__)([^*_]+)(?:\*\*|__))', clean_claim
        ):
            candidate_raw = (match.group(1) or match.group(2) or "").strip()
            if not candidate_raw:
                continue
            candidate_variants = [candidate_raw]
            stripped = candidate_raw.rstrip(".,;:!? ")
            if stripped != candidate_raw and len(stripped) > 0:
                candidate_variants.append(stripped)

            for cand_phrase in candidate_variants:
                norm_candidate = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", cand_phrase)
                if len(re.findall(r"\b\w+\b", norm_candidate)) < 3:
                    continue

                candidate_pages = sorted(
                    list({a.page_number for a in verified_anchors} | {evidence.page_number})
                )
                for p_num in candidate_pages:
                    page_rec = (
                        db.query(PaperPage)
                        .filter(
                            PaperPage.paper_id == evidence.paper_id,
                            PaperPage.page_number == p_num,
                        )
                        .first()
                    )
                    if not page_rec or not page_rec.raw_text:
                        continue

                    span = find_verbatim_span(page_rec.raw_text, cand_phrase)
                    if span is None:
                        continue

                    start_char, end_char = span
                    elems = (
                        db.query(PaperElement)
                        .filter(
                            PaperElement.paper_id == evidence.paper_id,
                            PaperElement.page_number == p_num,
                        )
                        .order_by(PaperElement.element_index)
                        .all()
                    )
                    overlapping_boxes: list[BoundingBox] = []
                    first_elem_id = None
                    for elem in elems:
                        e_span = find_verbatim_span(page_rec.raw_text, elem.text)
                        if e_span and e_span[0] < end_char and e_span[1] > start_char:
                            if first_elem_id is None:
                                first_elem_id = elem.id
                            if elem.bbox_x_min is not None and elem.page_width and elem.page_height:
                                overlapping_boxes.append(
                                    BoundingBox(
                                        x_min=elem.bbox_x_min,
                                        y_min=elem.bbox_y_min or 0.0,
                                        x_max=elem.bbox_x_max or 0.0,
                                        y_max=elem.bbox_y_max or 0.0,
                                        page_width=elem.page_width,
                                        page_height=elem.page_height,
                                        origin=CoordinateOrigin(elem.coordinate_origin),
                                        rotation=elem.rotation,
                                    )
                                )

                    new_anchor = CitationAnchor(
                        page_number=p_num,
                        source_element_id=first_elem_id
                        or (verified_anchors[0].source_element_id if verified_anchors else None),
                        exact_quote=cand_phrase,
                        source_char_start=start_char,
                        source_char_end=end_char,
                        document_sha256=evidence.document_sha256,
                        parser_version=evidence.parser_version,
                        anchor_status=AnchorStatus.VERIFIED,
                        bounding_boxes=overlapping_boxes,
                    )
                    return new_anchor, cand_phrase

    # 3. Check combined multi-element support for unquoted monotonic claims
    for i in range(len(verified_anchors) - 1):
        a1 = verified_anchors[i]
        a2 = verified_anchors[i + 1]
        if a1.page_number == a2.page_number:
            combined = f"{a1.exact_quote} {a2.exact_quote}"
            if check_claim_support(clean_claim, combined):
                return a1, None

    # 4. Check substantive contiguous span for unquoted natural sentences
    if cite_count == 1:
        for cand in verified_anchors:
            span = find_substantive_span(clean_claim, cand.exact_quote)
            if span is not None:
                return cand, span

    return None, None


class ChatService:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        graph_repo: Neo4jRepository | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        if graph_repo is not None:
            self.graph_repo = graph_repo
        else:
            settings = Settings.from_environment()
            if settings.graphrag_enabled:
                try:
                    self.graph_repo = Neo4jRepository.from_settings(settings)
                except Exception as exc:
                    logger.warning("Failed to initialize Neo4jRepository: %s", exc)
                    self.graph_repo = None
            else:
                self.graph_repo = None

    async def answer_question(
        self,
        db: Session,
        conversation_id: UUID,
        question: str,
    ) -> MessageResponse:
        start_time = time.perf_counter()
        conv = get_conversation(db, conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")

        project_id = conv.project_id
        intent = route_query_intent(question)

        # 1. Save user message if not an immediate duplicate of prior unanswered message
        last_msg = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .first()
        )
        if not (last_msg and last_msg.role == MessageRole.USER and last_msg.content == question):
            add_message(
                db=db,
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=question,
                citations=[],
                evidence=[],
            )

        # 2a. Retrieve evidence
        evidence_items: list[EvidenceItem] = self.retriever.retrieve(
            db=db,
            project_id=project_id,
            query=question,
        )

        # 2b. Retrieve active project memories with semantic scoring if available
        query_embedding = None
        try:
            from app.services.embedding import get_embedding_provider

            embed_provider = get_embedding_provider()
            query_embedding = embed_provider.embed_query(question)
        except Exception:
            query_embedding = None

        raw_project_memories = retrieve_project_memories(
            db=db,
            project_id=project_id,
            query=question,
            limit=5,
            record_access=True,
            query_embedding=query_embedding,
        )

        decision_preference_memories: list[Memory] = []
        paper_fact_memories: list[Memory] = []

        for mem in raw_project_memories:
            # Foreign-project memories or non-active memories must never supply
            # evidence or prompt context
            if mem.project_id != project_id or mem.status != MemoryStatus.ACTIVE.value:
                continue
            if mem.memory_type == MemoryType.PAPER_FACT.value:
                paper_fact_memories.append(mem)
            else:
                decision_preference_memories.append(mem)

        # Resolve PAPER_FACT memory sources to verified EvidenceItem and CitationAnchor
        existing_evidence_quotes = {e.quote.strip().lower() for e in evidence_items}
        for p_mem in paper_fact_memories:
            for src in p_mem.sources:
                ev_item, anchor, status = resolve_paper_memory_source(db, project_id, src)
                if status == AnchorStatus.VERIFIED and ev_item and anchor:
                    if ev_item.quote.strip().lower() in existing_evidence_quotes:
                        continue
                    existing_evidence_quotes.add(ev_item.quote.strip().lower())
                    new_id = f"E{len(evidence_items) + 1}"
                    resolved_evidence = EvidenceItem(
                        id=new_id,
                        paper_id=ev_item.paper_id,
                        paper_title=ev_item.paper_title,
                        chunk_id=ev_item.chunk_id,
                        quote=ev_item.quote,
                        parent_context=ev_item.parent_context,
                        page_number=ev_item.page_number,
                        bounding_boxes=ev_item.bounding_boxes,
                        source_element_ids=ev_item.source_element_ids,
                        document_sha256=ev_item.document_sha256,
                        parser_version=ev_item.parser_version,
                        anchors=[anchor],
                    )
                    evidence_items.append(resolved_evidence)

        # 2c. Retrieve graph evidence
        graph_evidence_items, graph_notice = retrieve_graph_evidence(
            db=db,
            repo=self.graph_repo,
            project_id=project_id,
            query=question,
            intent=intent,
        )
        for g_item in graph_evidence_items:
            norm_q = g_item.quote.strip().lower()
            if norm_q in existing_evidence_quotes:
                continue
            existing_evidence_quotes.add(norm_q)
            new_id = f"E{len(evidence_items) + 1}"
            g_item.id = new_id
            evidence_items.append(g_item)

        evidence_map: dict[str, EvidenceItem] = {e.id: e for e in evidence_items}
        memory_block = format_memories_for_prompt(
            decision_preference_memories, db=db, include_paper_facts=False
        )

        # 3. Retrieve prior conversation history (bounded to last 6 messages)
        history_msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        history_turns = []
        for msg in history_msgs:
            if msg.role == MessageRole.USER and msg.content == question and msg == history_msgs[-1]:
                continue
            history_turns.append(f"{msg.role}: {msg.content}")

        history_block = "\n".join(history_turns[-6:]) if history_turns else ""

        system_prompt = (
            "You are mYrA, an academic research assistant. "
            "Answer the QUESTION using the provided EVIDENCE quotes and PROJECT MEMORY, "
            "maintaining continuity with CONVERSATION HISTORY when relevant.\n\n"
            "ANSWER DIRECTNESS & FLUENCY RULES:\n"
            "- Answer directly, precisely, and concisely to the specific QUESTION asked.\n"
            "- Always explicitly name the subject at the beginning (e.g., "
            "'The Transformer is...', 'BERT is...'). "
            "NEVER start answers with vague pronouns like "
            "'It is...', 'This is...', or 'It was...'.\n"
            "- Write fluent, natural, grammatically complete sentences. "
            "NEVER use artificial bracketed "
            "inflections inside quotes (e.g., do NOT write 'eschew[es]' or 'rel[ies]'). "
            "If quoting, "
            "quote clean, verbatim grammatical phrases that fit seamlessly into "
            "the sentence, or state "
            "the facts directly in well-formed prose.\n"
            "- Avoid redundancy: do NOT generate multiple sentences stating the same "
            "definition in different ways. "
            "Provide one clear, authoritative definition and its key architectural "
            "principle without repetition.\n"
            "- Rely strictly on the provided EVIDENCE quotes for factual claims, but use ONLY "
            "the evidence necessary to answer the user's specific query.\n"
            "- Do NOT summarize or dump all provided evidence chunks. Do not add unrequested "
            "tangential details (such as training hardware, GPU hours, benchmark scores, "
            "dataset names, or hyperparameter layer counts) unless the question "
            "explicitly asks for them.\n\n"
            "CITATION & PROVENANCE RULES:\n"
            "- For any factual claim from papers, support it with a citation ID such as [E1]. "
            "You may quote key phrases in quotation marks or integrate facts naturally into "
            "clear, complete sentences.\n"
            "- Preserve exact technical terms, numbers, and definitions from the cited evidence.\n"
            "- Never invent a citation ID. Never use [E...] brackets for project decisions "
            "or user preferences.\n"
            "- For questions regarding project decisions, user preferences, terminology, or "
            "choices, answer accurately using the PROJECT MEMORY section without adding "
            "paper citation brackets.\n"
            "- If neither the evidence nor the project memory contains enough information "
            "to answer, state that evidence is insufficient."
        )

        if intent == GraphIntent.CONTRADICTION:
            system_prompt += (
                "\n\nCONTRADICTION ANALYSIS:\n"
                "- When differing results or conflicting claims are present in the evidence, "
                "present each side as its own source-supported statement with its own citation "
                "(e.g. 'Paper A reports ... [E1], whereas Paper B observes ... [E2]').\n"
                "- Do not merge conflicting claims into a single synthetic sentence without "
                "individual citations.\n"
                "- If the evidence does not contain conflicting or opposing claims on the "
                "specified topic, clearly state that no direct contradictions were found in "
                "the current evidence."
            )

        if graph_notice and intent != GraphIntent.FACTUAL:
            system_prompt += f"\n\n[NOTE: {graph_notice}]"

        evidence_text_parts = []
        for e in evidence_items:
            title = e.paper_title or "Paper"
            # Pass parent context to LLM for comprehension while atomic quote stays child
            context = e.parent_context if e.parent_context else e.quote
            evidence_text_parts.append(
                f"[{e.id}] (From: {title}, Page {e.page_number}):\n"
                f'Evidence quote: "{e.quote}"\n'
                f'Context: "{context}"'
            )

        evidence_block = (
            "\n\n".join(evidence_text_parts)
            if evidence_text_parts
            else "No relevant evidence found."
        )
        if graph_notice and intent != GraphIntent.FACTUAL:
            evidence_block = f"[NOTE: {graph_notice}]\n\n{evidence_block}"

        prompt_parts = []
        if memory_block:
            prompt_parts.append(f"PROJECT MEMORY:\n{memory_block}")
        prompt_parts.append(f"EVIDENCE:\n{evidence_block}")
        if history_block:
            prompt_parts.append(f"CONVERSATION HISTORY:\n{history_block}")
        prompt_parts.append(
            f"QUESTION:\n{question}\n\n"
            f"INSTRUCTION: Answer directly and concisely to the question above. "
            f"Explicitly name the subject (e.g. 'The Transformer is...'), write fluent "
            f"English without bracketed words like 'eschew[es]', "
            f"avoid repeating the definition across multiple sentences, and include "
            f"only the necessary facts from the evidence."
        )
        user_prompt = "\n\n".join(prompt_parts)

        # 4. Generate answer with LLM
        llm = get_llm_provider()
        generation = await generate_with_metadata(
            llm, system_prompt=system_prompt, user_prompt=user_prompt
        )
        raw_answer = generation.content

        # 5. Sentence-level citation validation and claim support checking
        protected_answer = re.sub(
            r"\b(et\s+al|vs|e\.g|i\.e|fig|tab|no|vol|sec)\.",
            r"\1<DOT>",
            raw_answer,
            flags=re.IGNORECASE,
        )
        validated_citations: list[Citation] = []
        citation_to_display_index: dict[str, int] = {}
        display_idx = 1
        retained_paragraphs: list[str] = []

        paragraphs = [p for p in protected_answer.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_lines = [line for line in para.split("\n") if line.strip()]
            retained_lines: list[str] = []
            for line in para_lines:
                clean_line = line.strip()
                cite_matches_in_line = list(re.finditer(r"\[E(\d+)\]", clean_line))

                # Retain section headers and transition/introductory lines
                is_structure = (
                    clean_line.endswith(":")
                    or (
                        clean_line.startswith(("#", "**"))
                        and not clean_line.endswith((".", "!", "?"))
                    )
                ) and len(clean_line) < 100

                if is_structure and not cite_matches_in_line:
                    retained_lines.append(line)
                    continue

                line_sentences = [
                    s.replace("<DOT>", ".").strip()
                    for s in re.split(r"(?<=[.!?])\s+(?!\[E\d+\])", clean_line)
                    if s.strip()
                ]

                retained_line_sentences: list[str] = []
                for sentence in line_sentences:
                    cite_matches = list(re.finditer(r"\[E(\d+)\]", sentence))
                    if not cite_matches:
                        if is_attributed_to_memories(
                            sentence,
                            memories=decision_preference_memories,
                        ):
                            retained_line_sentences.append(sentence)
                        continue

                    # Verify all citation IDs exist in evidence
                    all_valid_ids = True
                    for m in cite_matches:
                        e_id = f"E{m.group(1)}"
                        if e_id not in evidence_map:
                            all_valid_ids = False
                            break

                    if not all_valid_ids:
                        # Hallucinated unknown citation: discard entire sentence
                        continue

                    # Verify claim support against cited evidence
                    sentence_supported = True
                    resolved_anchors_for_sentence: dict[str, tuple[CitationAnchor, str | None]] = {}
                    clean_claim = re.sub(r"\[E\d+\]", "", sentence).strip()

                    for m in cite_matches:
                        e_id = f"E{m.group(1)}"
                        evidence = evidence_map[e_id]
                        anchor, source_phrase = resolve_claim_anchor(
                            db=db,
                            evidence=evidence,
                            clean_claim=clean_claim,
                            cite_count=len(cite_matches),
                        )
                        if anchor is None:
                            sentence_supported = False
                            break
                        resolved_anchors_for_sentence[e_id] = (anchor, source_phrase)

                    if not sentence_supported:
                        # Unsupported claim: discard entire sentence
                        continue

                    # Register verified citations for supported sentence
                    for m in cite_matches:
                        e_id = f"E{m.group(1)}"
                        if e_id not in citation_to_display_index:
                            evidence = evidence_map[e_id]
                            citation_to_display_index[e_id] = display_idx
                            matched_anchor, _ = resolved_anchors_for_sentence[e_id]

                            anchors_list = list(evidence.anchors)
                            if not any(
                                a.page_number == matched_anchor.page_number
                                and a.exact_quote == matched_anchor.exact_quote
                                and a.source_char_start == matched_anchor.source_char_start
                                for a in anchors_list
                            ):
                                anchors_list.append(matched_anchor)

                            validated_citations.append(
                                Citation(
                                    citation_index=display_idx,
                                    evidence_id=e_id,
                                    paper_id=evidence.paper_id,
                                    page_number=matched_anchor.page_number,
                                    bounding_boxes=(
                                        matched_anchor.bounding_boxes or evidence.bounding_boxes
                                    ),
                                    quote=matched_anchor.exact_quote,
                                    document_sha256=evidence.document_sha256,
                                    parser_version=evidence.parser_version,
                                    anchor_status=AnchorStatus.VERIFIED,
                                    anchors=anchors_list,
                                )
                            )
                            display_idx += 1

                    def replace_cite(m: re.Match) -> str:
                        eid = f"E{m.group(1)}"
                        return f"[{citation_to_display_index[eid]}]"

                    retained_line_sentences.append(re.sub(r"\[E(\d+)\]", replace_cite, sentence))

                if retained_line_sentences:
                    bullet_prefix = ""
                    stripped_l = line.lstrip()
                    for prefix in ("- ", "* ", "+ ", "• "):
                        if stripped_l.startswith(prefix):
                            bullet_prefix = line[: len(line) - len(stripped_l)] + prefix
                            break
                    joined_sent = " ".join(retained_line_sentences)
                    if bullet_prefix and not joined_sent.startswith(bullet_prefix):
                        if not any(joined_sent.startswith(p) for p in ("- ", "* ", "+ ", "• ")):
                            joined_sent = f"{bullet_prefix}{joined_sent}"
                    retained_lines.append(joined_sent)

            if retained_lines:
                retained_paragraphs.append("\n".join(retained_lines))

        if retained_paragraphs and (validated_citations or decision_preference_memories):
            formatted_answer = "\n\n".join(retained_paragraphs)
        else:
            formatted_answer = (
                "Insufficient evidence available in the uploaded papers to answer this question."
            )
            validated_citations = []

        # 6. Save assistant message
        model_name = (
            generation.reported_model
            or generation.requested_model
            or getattr(llm, "model_name", None)
            or getattr(llm, "provider_name", "unknown")
        )
        token_count_estimate = max(1, len(formatted_answer.split()))
        assistant_msg: Message = add_message(
            db=db,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=formatted_answer,
            citations=[c.model_dump(mode="json") for c in validated_citations],
            evidence=[e.model_dump(mode="json") for e in evidence_items],
            model_name=model_name,
            token_count=token_count_estimate,
        )

        # 7. Post-turn memory capture (extract decisions/preferences from turns)
        try:
            capture_conversation_memories(
                db=db, project_id=project_id, conversation_id=conversation_id
            )
        except Exception as e:
            logger.warning("Failed to capture conversation memories: %s", e)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "answer_generated",
            extra={
                "conversation_id": str(conversation_id),
                "latency_ms": round(latency_ms, 2),
                "evidence_count": len(evidence_items),
                "citations_count": len(validated_citations),
                "token_count_estimate": token_count_estimate,
                "provider_usage": (
                    {
                        "prompt_tokens": generation.usage.prompt_tokens,
                        "completion_tokens": generation.usage.completion_tokens,
                        "total_tokens": generation.usage.total_tokens,
                        "prompt_cache_hit_tokens": generation.usage.prompt_cache_hit_tokens,
                        "prompt_cache_miss_tokens": generation.usage.prompt_cache_miss_tokens,
                    }
                    if generation.usage is not None
                    else None
                ),
                "provider_response_id": generation.response_id,
                "provider_model": generation.reported_model,
            },
        )

        return MessageResponse(
            id=assistant_msg.id,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=formatted_answer,
            citations=validated_citations,
            evidence=evidence_items,
            model_name=assistant_msg.model_name,
            token_count=assistant_msg.token_count,
            created_at=assistant_msg.created_at,
        )

    answer = answer_question

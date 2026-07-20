import logging
import re
import time
from collections import Counter
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.chat import add_message, get_conversation
from app.db.models import Memory, Message, PaperPage
from app.observability.telemetry import Observation, TelemetryAdapter, get_telemetry
from app.schemas.chat import MessageResponse, MessageRole
from app.schemas.evidence import (
    AnchorStatus,
    Citation,
    CitationAnchor,
    ClaimEvidenceSupport,
    EvidenceItem,
)
from app.schemas.memory import MemoryStatus, MemoryType
from app.services.chat_prompts import build_chat_system_prompt, build_chat_user_prompt
from app.services.claim_validation import claim_clauses_for_citations, is_explicit_comparison
from app.services.evidence_assembly import assemble_evidence_items
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
from app.services.source_resolution import resolve_exact_source_anchor

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
    project_id: UUID,
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

                    new_anchor = resolve_exact_source_anchor(
                        db,
                        project_id=project_id,
                        paper_id=evidence.paper_id,
                        page_number=p_num,
                        exact_quote=cand_phrase,
                        document_sha256=evidence.document_sha256,
                        parser_version=evidence.parser_version,
                    )
                    if new_anchor is None:
                        continue
                    if not new_anchor.source_element_id and verified_anchors:
                        new_anchor = new_anchor.model_copy(
                            update={"source_element_id": verified_anchors[0].source_element_id}
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
        telemetry = get_telemetry()
        with telemetry.operation(
            "chat.answer",
            input={"question": question},
            metadata={"conversation_id": str(conversation_id)},
        ) as observation:
            return await self._answer_question(
                db, conversation_id, question, telemetry, observation
            )

    async def _answer_question(
        self,
        db: Session,
        conversation_id: UUID,
        question: str,
        telemetry: TelemetryAdapter,
        observation: Observation | None,
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

        # Share one query embedding between retrieval and semantic memory lookup.
        query_embedding = None
        try:
            from app.services.embedding import get_embedding_provider

            embed_provider = get_embedding_provider()
            query_embedding = embed_provider.embed_query(question)
        except Exception:
            # Retrieval retains its existing behavior and computes the vector itself
            # if a memory-specific embedding attempt is unavailable.
            query_embedding = None

        # 2a. Retrieve evidence
        evidence_items: list[EvidenceItem] = self.retriever.retrieve(
            db=db,
            project_id=project_id,
            query=question,
            query_embedding=query_embedding,
        )
        supplementary_evidence: list[EvidenceItem] = []

        # 2b. Retrieve active project memories with semantic scoring if available
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
        for p_mem in paper_fact_memories:
            for src in p_mem.sources:
                ev_item, anchor, status = resolve_paper_memory_source(db, project_id, src)
                if status == AnchorStatus.VERIFIED and ev_item and anchor:
                    supplementary_evidence.append(ev_item.model_copy(update={"anchors": [anchor]}))

        # 2c. Retrieve graph evidence
        graph_evidence_items, graph_notice = retrieve_graph_evidence(
            db=db,
            repo=self.graph_repo,
            project_id=project_id,
            query=question,
            intent=intent,
        )
        for g_item in graph_evidence_items:
            supplementary_evidence.append(g_item)

        evidence_items = assemble_evidence_items(evidence_items + supplementary_evidence)

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

        system_prompt = build_chat_system_prompt(
            intent, graph_notice if intent != GraphIntent.FACTUAL else None
        )

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

        user_prompt = build_chat_user_prompt(
            question=question,
            evidence_block=evidence_block,
            memory_block=memory_block,
            history_block=history_block,
        )

        # 4. Generate answer with LLM
        llm = get_llm_provider()
        selected_evidence = [
            {
                "evidence_id": item.id,
                "paper_title": item.paper_title,
                "page_number": item.page_number,
                "quote": item.quote,
                "context": item.parent_context,
            }
            for item in evidence_items
        ]
        with telemetry.stage(
            "chat.generation",
            generation=True,
            input={
                "question": question,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "selected_evidence": selected_evidence,
            },
            metadata={"attempt": 1},
        ) as generation_observation:
            generation = await generate_with_metadata(
                llm, system_prompt=system_prompt, user_prompt=user_prompt
            )
            generation_usage = (
                {
                    key: value
                    for key, value in {
                        "prompt_tokens": generation.usage.prompt_tokens,
                        "completion_tokens": generation.usage.completion_tokens,
                        "total_tokens": generation.usage.total_tokens,
                        "prompt_cache_hit_tokens": generation.usage.prompt_cache_hit_tokens,
                        "prompt_cache_miss_tokens": generation.usage.prompt_cache_miss_tokens,
                    }.items()
                    if value is not None
                }
                if generation.usage is not None
                else None
            )
            telemetry.generation_metadata(
                generation_observation,
                model=generation.reported_model or generation.requested_model,
                usage=generation_usage,
                output=generation.content,
            )
            if generation_observation is not None:
                generation_observation.update(
                    metadata={"attempt": 1, "response_id": generation.response_id}
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
        validated_claim_supports: list[ClaimEvidenceSupport] = []
        citation_validation = {
            "citation_markers": 0,
            "accepted_citations": 0,
            "rejected_unknown_id": 0,
            "rejected_unsupported_claim": 0,
        }

        paragraphs = [p for p in protected_answer.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_lines = [line for line in para.split("\n") if line.strip()]
            retained_lines: list[str] = []
            for line in para_lines:
                clean_line = line.strip()
                cite_matches_in_line = list(re.finditer(r"\[E(\d+)\]", clean_line))
                citation_validation["citation_markers"] += len(cite_matches_in_line)

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

                # Sentence punctuation inside a verbatim quote is part of the
                # source span, not a boundary between generated claims.
                quote_punctuation = {".": "<QPERIOD>", "!": "<QEXCLAMATION>", "?": "<QQUESTION>"}

                def protect_quoted_punctuation(match: re.Match) -> str:
                    return "".join(quote_punctuation.get(char, char) for char in match.group(0))

                split_ready_line = re.sub(
                    r'(?:["“][^"”]+["”]|(?:\*\*|__)[^*_]+(?:\*\*|__))',
                    protect_quoted_punctuation,
                    clean_line,
                )
                line_sentences = []
                for sentence_part in re.split(r"(?<=[.!?])\s+(?!\[E\d+\])", split_ready_line):
                    restored = sentence_part.replace("<DOT>", ".")
                    for punctuation, placeholder in quote_punctuation.items():
                        restored = restored.replace(placeholder, punctuation)
                    if restored.strip():
                        line_sentences.append(restored.strip())

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
                        citation_validation["rejected_unknown_id"] += len(cite_matches)
                        continue

                    # Verify claim support against cited evidence
                    sentence_supported = True
                    resolved_anchors_for_sentence: dict[str, tuple[CitationAnchor, str | None]] = {}
                    clean_claim = re.sub(r"\[E\d+\]", "", sentence).strip()
                    claim_segments = claim_clauses_for_citations(
                        sentence,
                        [(match.start(), match.end()) for match in cite_matches],
                        clean_sentence=clean_claim,
                    )
                    claim_counts = Counter(claim_segments)
                    claim_text_by_evidence_id: dict[str, str] = {}

                    for m, claim_segment in zip(cite_matches, claim_segments, strict=True):
                        e_id = f"E{m.group(1)}"
                        evidence = evidence_map[e_id]
                        anchor, source_phrase = resolve_claim_anchor(
                            db=db,
                            evidence=evidence,
                            clean_claim=claim_segment,
                            cite_count=claim_counts[claim_segment],
                            project_id=project_id,
                        )
                        if anchor is None:
                            sentence_supported = False
                            break
                        resolved_anchors_for_sentence[e_id] = (anchor, source_phrase)
                        claim_text_by_evidence_id[e_id] = claim_segment

                    if not sentence_supported:
                        # Unsupported claim: discard entire sentence
                        citation_validation["rejected_unsupported_claim"] += len(cite_matches)
                        continue

                    # A source-verified quotation does not validate its generated
                    # wrapper. Keep the single-citation extraction behavior below,
                    # but reject a multi-source sentence rather than publishing
                    # unsupported wrapper prose as part of a comparison.
                    if len(cite_matches) > 1 and any(
                        phrase is not None for _, phrase in resolved_anchors_for_sentence.values()
                    ):
                        citation_validation["rejected_unsupported_claim"] += len(cite_matches)
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
                            citation_validation["accepted_citations"] += 1
                            display_idx += 1

                    def replace_cite(m: re.Match) -> str:
                        eid = f"E{m.group(1)}"
                        return f"[{citation_to_display_index[eid]}]"

                    # A verbatim fragment can be verified even when the prose
                    # around it is not. In that case, publish only the verified
                    # source text with its citation; never keep the model's
                    # unsupported wrapper or trailing claim.
                    verified_phrases = {
                        phrase
                        for _, phrase in resolved_anchors_for_sentence.values()
                        if phrase is not None
                    }
                    supported_claim_text = (
                        next(iter(verified_phrases))
                        if len(cite_matches) == 1 and verified_phrases
                        else None
                    )
                    support_groups: dict[str, list[str]] = {}
                    for match in cite_matches:
                        evidence_id = f"E{match.group(1)}"
                        claim_text = claim_text_by_evidence_id[evidence_id]
                        if len(cite_matches) == 1 and supported_claim_text:
                            claim_text = supported_claim_text
                        support_groups.setdefault(claim_text, []).append(evidence_id)
                    for claim_text, evidence_ids in support_groups.items():
                        validated_claim_supports.append(
                            ClaimEvidenceSupport(
                                claim_text=claim_text,
                                evidence_ids=list(dict.fromkeys(evidence_ids)),
                            )
                        )
                    distinct_papers = {
                        evidence_map[evidence_id].paper_id
                        for match in cite_matches
                        if (evidence_id := f"E{match.group(1)}") in evidence_map
                    }
                    if (
                        len(support_groups) > 1
                        and len(distinct_papers) > 1
                        and is_explicit_comparison(clean_claim)
                    ):
                        validated_claim_supports.append(
                            ClaimEvidenceSupport(
                                claim_text=clean_claim,
                                evidence_ids=list(
                                    dict.fromkeys(f"E{match.group(1)}" for match in cite_matches)
                                ),
                                support_kind="derived",
                            )
                        )
                    if len(cite_matches) == 1 and verified_phrases:
                        phrase = supported_claim_text
                        e_id = f"E{cite_matches[0].group(1)}"
                        display_index = citation_to_display_index[e_id]
                        retained_line_sentences.append(f"“{phrase}” [{display_index}]")
                    else:
                        retained_line_sentences.append(
                            re.sub(r"\[E(\d+)\]", replace_cite, sentence)
                        )

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

        if observation is not None:
            observation.update(
                output=formatted_answer,
                metadata={
                    "citation_validation": citation_validation,
                    "citation_count": len(validated_citations),
                    "abstained": not bool(validated_citations or decision_preference_memories),
                    "generation_attempts": [
                        {
                            "attempt": 1,
                            "requested_model": generation.requested_model,
                            "reported_model": generation.reported_model,
                            "response_id": generation.response_id,
                            "usage": generation_usage,
                        }
                    ],
                    "outcome": "success",
                },
            )
        telemetry.event(
            "chat.citation_validation",
            metadata={
                **citation_validation,
                "citation_count": len(validated_citations),
                "abstained": not bool(validated_citations or decision_preference_memories),
            },
        )

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
            claim_supports=validated_claim_supports,
            evidence=evidence_items,
            model_name=assistant_msg.model_name,
            token_count=assistant_msg.token_count,
            created_at=assistant_msg.created_at,
        )

    answer = answer_question

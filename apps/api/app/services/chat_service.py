import logging
import re
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.chat import add_message, get_conversation
from app.db.models import Memory, Message
from app.schemas.chat import MessageResponse, MessageRole
from app.schemas.evidence import AnchorStatus, Citation, CitationAnchor, EvidenceItem
from app.schemas.memory import MemoryStatus, MemoryType
from app.services.llm import get_llm_provider
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
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "too",
    "very",
    "can",
    "will",
    "don",
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


def is_attributed_to_memories(
    sentence: str,
    memories: list[Memory],
) -> bool:
    """Verify that an uncited sentence is directly supported by a decision/preference memory.

    Rejects sentences that merely share a subject or entity (e.g. 'AURC cures cancer')
    while having no support for the predicate or relationship in the stored decision.
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

    norm_sentence = re.sub(r"\s+", " ", sentence_clean.lower())

    for m in valid_memories:
        mem_content = (m.content or "").strip()
        mem_title = (m.title or "").strip()
        if not mem_content:
            continue

        # 1. Exact or near-exact containment of the memory content in the sentence
        norm_content = re.sub(r"\s+", " ", mem_content.lower()).rstrip(".?!")
        if norm_content and norm_content in norm_sentence:
            return True

        # 2. Extract quotes from sentence (e.g., 'We chose "AURC over ECE"')
        for match in re.finditer(r'["“]([^"”]+)["”]', sentence_clean):
            quoted = match.group(1).strip()
            if len(quoted.split()) >= 3 and (
                check_claim_support(quoted, mem_content)
                or (mem_title and check_claim_support(quoted, f"{mem_title}: {mem_content}"))
            ):
                return True

        # 3. Strip conversational/discourse prefixes and check claim support against memory content
        prefix_pattern = (
            r"^(?:according to (?:our|the )?(?:project )?"
            r"(?:decision|decisions|preference|preferences|memory)|"
            r"as (?:per )?(?:the )?(?:project )?(?:decision|decisions|preference|preferences)|"
            r"per (?:our|the) (?:decision|decisions|preference|preferences)|"
            r"based on (?:our|the )?(?:project )?(?:decision|decisions|preference|preferences)|"
            r"in (?:the )?(?:project )?(?:decision|decisions|preference|preferences)|"
            r"for this project[,:]?|"
            r"we have (?:decided|chosen|agreed|preferred)[,:]?|"
            r"the project decision is that|"
            r"as decided[,:]?|"
            r"note that[,:]?|"
            r"recall that[,:]?)\s*"
        )
        stripped = re.sub(prefix_pattern, "", sentence_clean, flags=re.IGNORECASE).strip()
        if stripped and (
            check_claim_support(stripped, mem_content)
            or (mem_title and check_claim_support(stripped, f"{mem_title}: {mem_content}"))
        ):
            return True

        # 4. Monotonic subsequence check with substantive coverage:
        words = (stripped or sentence_clean).split()
        for w_len in range(len(words), 2, -1):
            for start_idx in range(len(words) - w_len + 1):
                subphrase = " ".join(words[start_idx : start_idx + w_len])
                sub_tokens = [
                    t
                    for t in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", subphrase.lower())
                    if t not in STOPWORDS
                ]
                if len(sub_tokens) >= 3 and (
                    check_claim_support(subphrase, mem_content)
                    or (mem_title and check_claim_support(subphrase, f"{mem_title}: {mem_content}"))
                ):
                    remainder = " ".join(words[:start_idx] + words[start_idx + w_len :])
                    rem_tokens = [
                        t
                        for t in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", remainder.lower())
                        if t not in STOPWORDS
                    ]
                    mem_tokens = set(
                        re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", f"{mem_title} {mem_content}".lower())
                    )
                    foreign_tokens = [t for t in rem_tokens if t not in mem_tokens]
                    if len(foreign_tokens) <= 1:
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
    """Keep a model-quoted source phrase, never its unsupported surrounding prose."""
    norm_evidence = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", evidence_quote)
    for match in re.finditer(r'["“]([^"”]+)["”]', claim_text):
        candidate = match.group(1).strip()
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


class ChatService:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()

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
            "CITATION & PROVENANCE RULES:\n"
            "- For any factual claim from papers, copy a short relevant sentence or phrase "
            "verbatim from ONE Evidence quote, preserving its words, numbers, and order, "
            "then append that quote's citation ID such as [E1].\n"
            "- Never invent a citation ID. Never use [E...] brackets for project decisions "
            "or user preferences.\n"
            "- For questions regarding project decisions, user preferences, terminology, or "
            "choices, answer accurately using the PROJECT MEMORY section without adding "
            "paper citation brackets.\n"
            "- If neither the evidence nor the project memory contains enough information "
            "to answer, state that evidence is insufficient."
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

        prompt_parts = []
        if memory_block:
            prompt_parts.append(f"PROJECT MEMORY:\n{memory_block}")
        prompt_parts.append(f"EVIDENCE:\n{evidence_block}")
        if history_block:
            prompt_parts.append(f"CONVERSATION HISTORY:\n{history_block}")
        prompt_parts.append(f"QUESTION:\n{question}")
        user_prompt = "\n\n".join(prompt_parts)

        # 4. Generate answer with LLM
        llm = get_llm_provider()
        raw_answer = await llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)

        # 5. Sentence-level citation validation and claim support checking
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_answer) if s.strip()]

        validated_citations: list[Citation] = []
        citation_to_display_index: dict[str, int] = {}
        display_idx = 1
        retained_sentences: list[str] = []

        for sentence in raw_sentences:
            cite_matches = list(re.finditer(r"\[E(\d+)\]", sentence))
            if not cite_matches:
                if is_attributed_to_memories(
                    sentence,
                    memories=decision_preference_memories,
                ):
                    retained_sentences.append(sentence)
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
            source_phrase: str | None = None
            for m in cite_matches:
                e_id = f"E{m.group(1)}"
                evidence = evidence_map[e_id]
                clean_claim = re.sub(r"\[E\d+\]", "", sentence).strip()
                if not matching_verified_anchor(evidence):
                    sentence_supported = False
                    break
                if not check_claim_support(clean_claim, evidence.quote):
                    # Some otherwise useful responses wrap a verbatim excerpt in
                    # unverified prose. Publish only the exact excerpt, and only
                    # when one verified source is cited by this sentence.
                    source_phrase = (
                        extract_verbatim_quoted_phrase(clean_claim, evidence.quote)
                        if len(cite_matches) == 1
                        else None
                    )
                    if source_phrase is None:
                        sentence_supported = False
                        break

            if not sentence_supported:
                # Unsupported claim: discard entire sentence
                continue

            if source_phrase is not None:
                sentence = f'"{source_phrase}" {cite_matches[0].group()}.'

            # Register verified citations for supported sentence
            for m in cite_matches:
                e_id = f"E{m.group(1)}"
                if e_id not in citation_to_display_index:
                    evidence = evidence_map[e_id]
                    citation_to_display_index[e_id] = display_idx
                    # Determine anchor status specifically for the displayed quote and page
                    matching_anchor = matching_verified_anchor(evidence)
                    anchor_status = (
                        matching_anchor.anchor_status
                        if matching_anchor
                        else AnchorStatus.UNRESOLVED
                    )
                    validated_citations.append(
                        Citation(
                            citation_index=display_idx,
                            evidence_id=e_id,
                            paper_id=evidence.paper_id,
                            page_number=evidence.page_number,
                            bounding_boxes=evidence.bounding_boxes,
                            quote=evidence.quote,
                            document_sha256=evidence.document_sha256,
                            parser_version=evidence.parser_version,
                            anchor_status=anchor_status,
                            anchors=evidence.anchors,
                        )
                    )
                    display_idx += 1

            def replace_cite(m: re.Match) -> str:
                eid = f"E{m.group(1)}"
                return f"[{citation_to_display_index[eid]}]"

            retained_sentences.append(re.sub(r"\[E(\d+)\]", replace_cite, sentence))

        if retained_sentences and (
            validated_citations or decision_preference_memories
        ):
            formatted_answer = " ".join(retained_sentences)
        else:
            formatted_answer = (
                "Insufficient evidence available in the uploaded papers to answer this question."
            )
            validated_citations = []

        # 6. Save assistant message
        model_name = getattr(llm, "model_name", llm.provider_name)
        token_count = max(1, len(formatted_answer.split()))
        assistant_msg: Message = add_message(
            db=db,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=formatted_answer,
            citations=[c.model_dump(mode="json") for c in validated_citations],
            evidence=[e.model_dump(mode="json") for e in evidence_items],
            model_name=model_name,
            token_count=token_count,
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
                "token_count": token_count,
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

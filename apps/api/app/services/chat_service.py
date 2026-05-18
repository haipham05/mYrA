import logging
import re
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.chat import add_message, get_conversation
from app.db.models import Message
from app.schemas.chat import MessageResponse, MessageRole
from app.schemas.evidence import AnchorStatus, Citation, CitationAnchor, EvidenceItem
from app.services.llm import get_llm_provider
from app.services.retrieval import HybridRetriever

logger = logging.getLogger("myra.chat")


def check_claim_support(claim_text: str, evidence_quote: str) -> bool:
    """Accept only extractive claims whose tokens retain the quote's order.

    Word-set overlap is not evidence of a relationship: it accepts reversed actors
    and swapped measurements. This intentionally rejects many valid paraphrases;
    uncertain model output must abstain rather than invent a supported claim.
    """
    if not claim_text.strip() or not evidence_quote.strip():
        return False

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

        # 1. Save user message
        add_message(
            db=db,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=question,
            citations=[],
            evidence=[],
        )

        # 2. Retrieve evidence
        evidence_items: list[EvidenceItem] = self.retriever.retrieve(
            db=db,
            project_id=project_id,
            query=question,
        )

        evidence_map: dict[str, EvidenceItem] = {e.id: e for e in evidence_items}

        system_prompt = (
            "You are mYrA, an academic research assistant. "
            "Answer the user's question using ONLY the provided evidence. "
            "Every statement derived from the papers MUST cite the evidence using [E1], [E2]. "
            "If evidence is insufficient to answer the question, clearly state that. "
            "Do not make up citations."
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
        user_prompt = f"EVIDENCE:\n{evidence_block}\n\nQUESTION:\n{question}"

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
                if not evidence_items:
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
            for m in cite_matches:
                e_id = f"E{m.group(1)}"
                evidence = evidence_map[e_id]
                clean_claim = re.sub(r"\[E\d+\]", "", sentence).strip()
                if not matching_verified_anchor(evidence) or not check_claim_support(
                    clean_claim, evidence.quote
                ):
                    sentence_supported = False
                    break

            if not sentence_supported:
                # Unsupported claim: discard entire sentence
                continue

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

        if retained_sentences and validated_citations:
            formatted_answer = " ".join(retained_sentences)
        else:
            formatted_answer = (
                "Insufficient evidence available in the uploaded papers to answer this question."
            )
            validated_citations = []

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "answer_generated",
            extra={
                "conversation_id": str(conversation_id),
                "latency_ms": round(latency_ms, 2),
                "evidence_count": len(evidence_items),
                "citations_count": len(validated_citations),
            },
        )

        # 6. Save assistant message
        assistant_msg: Message = add_message(
            db=db,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=formatted_answer,
            citations=[c.model_dump(mode="json") for c in validated_citations],
            evidence=[e.model_dump(mode="json") for e in evidence_items],
        )

        return MessageResponse(
            id=assistant_msg.id,
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=formatted_answer,
            citations=validated_citations,
            evidence=evidence_items,
            created_at=assistant_msg.created_at,
        )

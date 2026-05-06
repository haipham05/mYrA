import logging
import re
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.chat import add_message, get_conversation
from app.db.models import Message
from app.schemas.chat import MessageResponse, MessageRole
from app.schemas.evidence import AnchorStatus, Citation, EvidenceItem
from app.services.llm import get_llm_provider
from app.services.retrieval import HybridRetriever

logger = logging.getLogger("myra.chat")


def check_claim_support(claim_text: str, evidence_quote: str) -> bool:
    """Validate claim-to-cite support via lexical keyword overlap."""
    if not claim_text.strip() or not evidence_quote.strip():
        return False

    # Normalized content words (excluding common stop words)
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "of",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "their",
        "we",
        "our",
        "you",
        "your",
        "based",
        "according",
    }
    claim_words = {
        w for w in re.findall(r"\w+", claim_text.lower()) if len(w) > 2 and w not in stop_words
    }
    quote_words = {
        w for w in re.findall(r"\w+", evidence_quote.lower()) if len(w) > 2 and w not in stop_words
    }

    if not claim_words or not quote_words:
        return True  # Bounded fallback if no distinctive words

    overlap = claim_words.intersection(quote_words)
    # Require at least one non-trivial keyword overlap
    return len(overlap) > 0


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
            evidence_text_parts.append(
                f'[{e.id}] (From: {title}, Page {e.page_number}):\n"{e.quote}"'
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

        # 5. Citation validation and claim support checking
        # Match pattern [E1], [E2], etc.
        citation_matches = list(re.finditer(r"\[E(\d+)\]", raw_answer))

        validated_citations: list[Citation] = []
        citation_to_display_index: dict[str, int] = {}
        display_idx = 1

        for match in citation_matches:
            e_id = f"E{match.group(1)}"
            if e_id in evidence_map and e_id not in citation_to_display_index:
                evidence = evidence_map[e_id]

                # Extract surrounding sentence as the claim
                match_start = match.start()
                sentence_start = max(0, raw_answer.rfind(".", 0, match_start) + 1)
                sentence_end = raw_answer.find(".", match.end())
                if sentence_end == -1:
                    sentence_end = len(raw_answer)
                claim_text = raw_answer[sentence_start:sentence_end].strip()

                is_supported = check_claim_support(claim_text, evidence.quote)

                # Status reflects whether claim is supported and anchor has bounding boxes
                if not is_supported:
                    status = AnchorStatus.UNRESOLVED
                elif evidence.bounding_boxes or evidence.anchors:
                    status = AnchorStatus.VERIFIED
                else:
                    status = AnchorStatus.UNRESOLVED

                citation_to_display_index[e_id] = display_idx
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
                        anchor_status=status,
                        anchors=evidence.anchors,
                    )
                )
                display_idx += 1

        # Replace [E1] with [1], and strip invalid hallucinated citations [EX]
        def replace_citation(m: re.Match) -> str:
            e_id = f"E{m.group(1)}"
            if e_id in citation_to_display_index:
                return f"[{citation_to_display_index[e_id]}]"
            # Unsupported / hallucinated citation removed
            return ""

        formatted_answer = re.sub(r"\[E(\d+)\]", replace_citation, raw_answer).strip()

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

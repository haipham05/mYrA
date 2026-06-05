import math
import re
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.memory import (
    create_memory,
    list_memories,
    record_memory_access,
    supersede_memory,
)
from app.db.models import Memory, Message, Paper
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


def consolidate_memory_candidate(
    db: Session,
    project_id: UUID,
    candidate: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    """Idempotently insert or supersede memory in the target project.

    Enforces:
    - Secret redaction on all fields.
    - Paper facts require verified paper in the project.
    - Idempotency: exact content matches return existing active memory.
    - Conflict resolution: decisions on the same topic supersede older active decisions.
    """
    safe_title = redact_secrets(candidate.title)
    safe_content = redact_secrets(candidate.content)
    candidate.title = safe_title
    candidate.content = safe_content

    # If paper fact, verify paper exists and belongs to this project
    if candidate.memory_type == MemoryType.PAPER_FACT:
        for s in candidate.sources:
            if s.source_type == MemorySourceType.PAPER_CHUNK and s.paper_id:
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
                if not s.quote_text or not s.quote_text.strip():
                    raise ValueError("Paper fact requires valid non-empty quote text.")

    # 1. Check exact content duplicate among active memories in this project
    existing_active, _ = list_memories(
        db,
        project_id=project_id,
        status=MemoryStatus.ACTIVE,
        memory_type=candidate.memory_type,
    )
    for mem in existing_active:
        if mem.content.strip().lower() == safe_content.strip().lower():
            # Exact duplicate: return existing memory (idempotent)
            return mem

    # 2. Check conflict / supersession for DECISION and PREFERENCE types
    if candidate.memory_type in (MemoryType.DECISION, MemoryType.PREFERENCE):
        cand_keywords = _extract_decision_keywords(safe_content)
        if cand_keywords:
            for mem in existing_active:
                mem_keywords = _extract_decision_keywords(mem.content)
                overlap = cand_keywords.intersection(mem_keywords)
                # If there is substantial topic overlap (e.g. "AURC", "ECE" or shared metric/model)
                if overlap and (len(overlap) >= 2 or any(len(w) >= 4 for w in overlap)):
                    # New memory supersedes old memory
                    new_mem = create_memory(
                        db, project_id=project_id, memory_in=candidate, embedding=embedding
                    )
                    supersede_memory(
                        db,
                        old_memory=mem,
                        new_memory=new_mem,
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
    """Extract and consolidate candidate memories from recent messages of a conversation."""
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
        # We only extract decisions/preferences from user messages or assistant confirmations
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


def format_memories_for_prompt(memories: list[Memory]) -> str:
    """Format memories into structured system prompt sections.

    Separates USER DECISIONS & PREFERENCES from VERIFIED PAPER EVIDENCE
    to prevent memory from masquerading as a paper citation.
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
        elif m.memory_type == MemoryType.PAPER_FACT:
            paper_source_info = ""
            for s in m.sources:
                if s.source_type == MemorySourceType.PAPER_CHUNK.value and s.quote_text:
                    paper_source_info = f' (Source quote: "{s.quote_text[:100]}...")'
                    break
            paper_facts.append(f"- {m.title}: {m.content}{paper_source_info}")
        else:
            general.append(f"- {m.title}: {m.content}")

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

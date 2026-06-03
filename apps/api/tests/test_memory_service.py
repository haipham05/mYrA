from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.memory import (
    MemoryVersionConflictError,
    create_memory,
    delete_memory,
    get_memory,
    update_memory,
)
from app.db.base import Base
from app.db.models import Conversation, Message, Paper, Project
from app.schemas.memory import (
    MemoryCreate,
    MemorySourceCreate,
    MemorySourceType,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
)
from app.services.memory_service import (
    capture_conversation_memories,
    consolidate_memory_candidate,
    extract_candidates_from_text,
    format_memories_for_prompt,
    retrieve_project_memories,
)


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_memory.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_memory_crud_and_optimistic_concurrency(db: Session) -> None:
    project = Project(name="Memory CRUD Project")
    db.add(project)
    db.commit()

    # 1. Create
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Metric Decision",
        content="We chose AURC as our primary calibration metric.",
        importance=0.8,
        confidence=0.95,
        is_pinned=True,
    )
    mem = create_memory(db, project_id=project.id, memory_in=mem_in)
    assert mem.id is not None
    assert mem.version == 1
    assert mem.status == MemoryStatus.ACTIVE.value
    assert mem.is_pinned is True
    assert len(mem.history) == 1
    assert mem.history[0].action == "CREATED"

    # 2. Get
    fetched = get_memory(db, mem.id, project_id=project.id)
    assert fetched is not None
    assert fetched.title == "Metric Decision"

    # Cross-project get should return None
    assert get_memory(db, mem.id, project_id=uuid4()) is None

    # 3. Update with matching version
    update_in = MemoryUpdate(
        title="Refined Metric Decision",
        importance=0.9,
        version=1,
    )
    updated = update_memory(db, mem, update_in)
    assert updated.title == "Refined Metric Decision"
    assert updated.version == 2
    assert updated.importance == 0.9
    assert len(updated.history) == 2

    # 4. Optimistic concurrency conflict with stale version
    stale_update = MemoryUpdate(
        title="Stale Update Attempt",
        version=1,  # Stale, current is 2
    )
    with pytest.raises(MemoryVersionConflictError):
        update_memory(db, updated, stale_update)

    # 5. Soft delete (Archive)
    delete_memory(db, updated, hard_delete=False)
    archived = get_memory(db, mem.id, project_id=project.id)
    assert archived is not None
    assert archived.status == MemoryStatus.ARCHIVED.value
    assert archived.version == 3

    # 6. Hard delete
    delete_memory(db, archived, hard_delete=True)
    assert get_memory(db, mem.id, project_id=project.id) is None


def test_secret_redaction_and_extraction() -> None:
    sensitive_text = (
        "We decided to use DeepSeek with sk-1234567890abcdef1234567890 "
        "and password=mysecretpass for our tests."
    )
    candidates = extract_candidates_from_text(sensitive_text)
    assert len(candidates) > 0
    decision = candidates[0]
    assert "sk-" not in decision.content
    assert "mysecretpass" not in decision.content
    assert "[REDACTED]" in decision.content


def test_idempotency_and_supersession(db: Session) -> None:
    project = Project(name="Supersession Project")
    db.add(project)
    db.commit()

    # Decision 1: Initial decision on ECE
    cand1 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Metric Decision: ECE",
        content="Project decision: We decided to use ECE for calibration evaluation.",
        importance=0.7,
        confidence=0.9,
    )
    mem1 = consolidate_memory_candidate(db, project_id=project.id, candidate=cand1)
    assert mem1.status == MemoryStatus.ACTIVE.value

    # Re-running same candidate is idempotent
    mem1_repeat = consolidate_memory_candidate(db, project_id=project.id, candidate=cand1)
    assert mem1_repeat.id == mem1.id

    # Decision 2: Contradictory / Updated decision on AURC vs ECE
    cand2 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Decision: Choose AURC over ECE",
        content="Project decision: Selected AURC over ECE for calibration evaluation.",
        importance=0.9,
        confidence=0.95,
    )
    mem2 = consolidate_memory_candidate(db, project_id=project.id, candidate=cand2)
    assert mem2.id != mem1.id
    assert mem2.status == MemoryStatus.ACTIVE.value

    # Verify old memory is superseded, not erased
    db.refresh(mem1)
    assert mem1.status == MemoryStatus.SUPERSEDED.value
    assert mem1.superseded_by_id == mem2.id
    assert mem1.version == 2
    assert any(h.action == "SUPERSEDED" for h in mem1.history)


def test_paper_fact_validation_and_forgery_prevention(db: Session) -> None:
    project1 = Project(name="Project 1")
    project2 = Project(name="Project 2")
    db.add_all([project1, project2])
    db.commit()

    paper1 = Paper(
        project_id=project1.id,
        filename="attention.pdf",
        storage_path="/fake/path",
        document_sha256="abc123",
        status="PROCESSED",
    )
    db.add(paper1)
    db.commit()

    # 1. Paper fact referencing a paper in a DIFFERENT project should be rejected
    forged_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Attention Paper Fact",
        content="Transformer model uses multi-head attention.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=paper1.id,
                quote_text="multi-head attention mechanism",
            )
        ],
    )
    with pytest.raises(ValueError, match="does not exist in project"):
        consolidate_memory_candidate(db, project_id=project2.id, candidate=forged_cand)

    # 2. Legitimate paper fact in same project succeeds
    valid_mem = consolidate_memory_candidate(
        db, project_id=project1.id, candidate=forged_cand
    )
    assert valid_mem.id is not None
    assert valid_mem.memory_type == MemoryType.PAPER_FACT.value


def test_conversation_capture_and_scoped_retrieval(db: Session) -> None:
    project_a = Project(name="Project A")
    project_b = Project(name="Project B")
    db.add_all([project_a, project_b])
    db.commit()

    conv_a = Conversation(project_id=project_a.id, title="Chat A")
    conv_b = Conversation(project_id=project_b.id, title="Chat B")
    db.add_all([conv_a, conv_b])
    db.commit()

    # User message in Project A with decision and preference
    msg_a1 = Message(
        conversation_id=conv_a.id,
        role="user",
        content="Let's decide to use ViT over ResNet50. Also, please always provide bullet points.",
    )
    db.add(msg_a1)
    db.commit()

    # User message in Project B with different decision
    msg_b1 = Message(
        conversation_id=conv_b.id,
        role="user",
        content="We decided to use ResNet50 for image processing.",
    )
    db.add(msg_b1)
    db.commit()

    # Capture memories for both
    mems_a = capture_conversation_memories(db, project_a.id, conv_a.id)
    mems_b = capture_conversation_memories(db, project_b.id, conv_b.id)

    assert len(mems_a) >= 2  # Decision on ViT and Preference on bullet points
    assert len(mems_b) >= 1  # Decision on ResNet50

    # Scoped retrieval for Project A
    retrieved_a = retrieve_project_memories(
        db, project_id=project_a.id, query="What model architecture are we using?"
    )
    assert len(retrieved_a) > 0
    # Must NOT contain Project B memories
    project_ids = {m.project_id for m in retrieved_a}
    assert project_ids == {project_a.id}
    # First item should be the ViT decision
    assert any("vit" in m.content.lower() for m in retrieved_a)
    assert not any("resnet50 for image processing" in m.content.lower() for m in retrieved_a)

    # Verify last_accessed_at was updated
    for m in retrieved_a:
        assert m.last_accessed_at is not None

    # Context formatting for prompt
    formatted = format_memories_for_prompt(retrieved_a)
    assert "PROJECT DECISIONS & USER PREFERENCES" in formatted
    assert "ViT" in formatted

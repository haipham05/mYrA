from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.memory import (
    MemoryVersionConflictError,
    atomic_create_and_supersede,
    create_memory,
    delete_memory,
    get_memory,
    supersede_memory,
    update_memory,
)
from app.db.base import Base
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
    Project,
)
from app.schemas.evidence import AnchorStatus
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
    resolve_paper_memory_source,
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

    # 3. Guard against self-supersession
    with pytest.raises(ValueError, match="Cannot supersede memory with itself"):
        supersede_memory(db, old_memory=mem2, new_memory=mem2)

    # 4. Guard against same-content supersession
    mem2_dup = create_memory(db, project_id=project.id, memory_in=cand2)
    with pytest.raises(ValueError, match="Cannot supersede memory with identical content"):
        supersede_memory(db, old_memory=mem2, new_memory=mem2_dup)

    # 5. Guard against superseding already superseded/inactive memory
    cand3 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Third Decision",
        content="New active decision.",
    )
    mem3 = create_memory(db, project_id=project.id, memory_in=cand3)
    with pytest.raises(ValueError, match="must be ACTIVE"):
        supersede_memory(db, old_memory=mem1, new_memory=mem3)


def test_paper_fact_validation_and_forgery_prevention(db: Session) -> None:
    project1 = Project(name="Project 1")
    project2 = Project(name="Project 2")
    db.add_all([project1, project2])
    db.commit()

    # Unready paper
    unready_paper = Paper(
        project_id=project1.id,
        filename="unready.pdf",
        storage_path="/papers/unready.pdf",
        document_sha256="hash_unready",
        status="PROCESSING",
    )
    # Ready paper with verified page text
    ready_paper = Paper(
        project_id=project1.id,
        filename="attention.pdf",
        storage_path="/papers/attention.pdf",
        document_sha256="hash_ready_123",
        status="READY",
    )
    db.add_all([unready_paper, ready_paper])
    db.commit()

    def add_page_element_chunk(paper_id, page_no: int, text: str) -> PaperPage:
        page = PaperPage(
            paper_id=paper_id,
            page_number=page_no,
            width=612.0,
            height=792.0,
            raw_text=text,
        )
        elem = PaperElement(
            paper_id=paper_id,
            element_index=page_no * 10,
            element_type="paragraph",
            text=text,
            page_number=page_no,
            bbox_x_min=50.0,
            bbox_y_min=100.0,
            bbox_x_max=550.0,
            bbox_y_max=200.0,
            page_width=612.0,
            page_height=792.0,
            coordinate_origin="TOP_LEFT",
            rotation=0,
            parser_version="docling-test",
        )
        chunk = PaperChunk(
            paper_id=paper_id,
            chunk_type="child",
            chunk_index=page_no,
            text=text,
        )
        db.add_all([page, elem, chunk])
        db.flush()
        db.add(ChunkElement(chunk_id=chunk.id, element_id=elem.id, order_index=0))
        db.commit()
        return page

    add_page_element_chunk(
        ready_paper.id,
        1,
        "The Transformer model uses multi-head attention mechanism across sub-layers.",
    )

    # 1. Non-READY paper must be rejected
    unready_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Unready Fact",
        content="Fact from unready paper.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=unready_paper.id,
                quote_text="some quote",
            )
        ],
    )
    with pytest.raises(ValueError, match="must be 'READY'"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=unready_cand)

    # 2. Paper fact referencing a paper in a DIFFERENT project must be rejected
    foreign_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Foreign Project Fact",
        content="Transformer model uses multi-head attention.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                quote_text="multi-head attention mechanism",
            )
        ],
    )
    with pytest.raises(ValueError, match="does not exist in project"):
        consolidate_memory_candidate(db, project_id=project2.id, candidate=foreign_cand)

    # 3. Invented quote not present in paper must be rejected
    forged_quote_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Forged Quote Fact",
        content="Transformer model uses quantum superposition.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                quote_text="quantum superposition in attention layers",
            )
        ],
    )
    with pytest.raises(ValueError, match="Quote text was not found"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=forged_quote_cand)

    # 4. Document SHA-256 mismatch must be rejected
    mismatched_hash_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Hash Mismatch Fact",
        content="Transformer model uses multi-head attention.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                quote_text="multi-head attention mechanism",
                document_sha256="wrong_hash_xyz",
            )
        ],
    )
    with pytest.raises(ValueError, match="Document SHA-256 mismatch"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=mismatched_hash_cand)

    # 5. Legitimate paper fact with verified quote succeeds
    valid_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Attention Paper Fact",
        content="Transformer model uses multi-head attention mechanism across sub-layers.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=1,
                quote_text=(
                    "The Transformer model uses multi-head attention mechanism across sub-layers."
                ),
                document_sha256="hash_ready_123",
            )
        ],
    )
    valid_mem = consolidate_memory_candidate(db, project_id=project1.id, candidate=valid_cand)
    assert valid_mem.id is not None
    assert valid_mem.memory_type == MemoryType.PAPER_FACT.value

    # 6. Prompt formatting revalidation: if paper becomes unready, fact is excluded
    formatted_ready = format_memories_for_prompt([valid_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" in formatted_ready
    assert "multi-head attention" in formatted_ready

    ready_paper.status = "FAILED"
    db.commit()
    formatted_failed = format_memories_for_prompt([valid_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" not in formatted_failed

    # 7. Real quote in paper, but claim asserts unsupported concepts
    # (quantum teleportation counterexample)
    ready_paper.status = "READY"
    db.commit()
    unsupported_claim_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Quantum Teleportation Claim",
        content="The paper proves quantum teleportation.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=1,
                quote_text="multi-head attention mechanism",
                document_sha256="hash_ready_123",
            )
        ],
    )
    with pytest.raises(ValueError, match="not supported by cited quote"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=unsupported_claim_cand)

    # 8. Misleading-but-real quote containing negation/contrast
    misleading_text = (
        "While quantum teleportation is fascinating, our work focuses strictly on transformers."
    )
    add_page_element_chunk(ready_paper.id, 2, misleading_text)

    misleading_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Misleading Quote Fact",
        content="The paper proves quantum teleportation.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=2,
                quote_text=misleading_text,
                document_sha256="hash_ready_123",
            )
        ],
    )
    with pytest.raises(ValueError, match="not supported by cited quote"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=misleading_cand)

    # 9. PATCH turning valid paper fact into unsupported claim is rejected
    with pytest.raises(ValueError, match="not supported by cited source quote"):
        update_memory(
            db,
            memory=valid_mem,
            memory_update=MemoryUpdate(
                version=1,
                content="The paper proves quantum teleportation.",
            ),
        )

    # 10. Reversed relation claim: "Method B is better than method A."
    # vs quote "Method A is better than method B."
    add_page_element_chunk(
        ready_paper.id,
        3,
        "In our empirical evaluation, Method A is better than method B across all benchmarks.",
    )

    reversed_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Reversed Method Relation",
        content="Method B is better than method A.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=3,
                quote_text="Method A is better than method B",
                document_sha256="hash_ready_123",
            )
        ],
    )
    with pytest.raises(ValueError, match="reverses entity roles, relations, or ordering"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=reversed_cand)

    # 11. Swapped numbers claim: "accuracy decreased from 90% to 80%"
    # vs quote "accuracy increased from 80% to 90%"
    page4 = add_page_element_chunk(
        ready_paper.id, 4, "The accuracy increased from 80% to 90% after fine-tuning."
    )

    swapped_numbers_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Swapped Numbers Fact",
        content="The accuracy decreased from 90% to 80% after fine-tuning.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=4,
                quote_text="The accuracy increased from 80% to 90% after fine-tuning.",
                document_sha256="hash_ready_123",
            )
        ],
    )
    with pytest.raises(
        ValueError, match="asserts opposite directional|inverts or alters the order"
    ):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=swapped_numbers_cand)

    # 12. Swapped numbers without antonym: "from 90 to 80" vs quote "from 80 to 90"
    swapped_plain_nums = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Plain Swapped Numbers",
        content="The score was from 90 to 80.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=4,
                quote_text="The score was from 80 to 90.",
            )
        ],
    )
    page4.raw_text += " The score was from 80 to 90."
    db.commit()
    with pytest.raises(ValueError, match="inverts or alters the order of numerical values"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=swapped_plain_nums)

    # 13. Genuinely supported relation claim succeeds and appears in prompt
    supported_relation_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Verified Method Superiority",
        content="Method A is better than method B.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=3,
                quote_text="Method A is better than method B",
                document_sha256="hash_ready_123",
            )
        ],
    )
    supported_mem = consolidate_memory_candidate(
        db, project_id=project1.id, candidate=supported_relation_cand
    )
    assert supported_mem.id is not None
    formatted_supported = format_memories_for_prompt([supported_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" in formatted_supported
    assert "Method A is better than method B" in formatted_supported

    # And verify that a reversed fake memory object created directly in DB never enters prompt
    reversed_fake_mem = Memory(
        project_id=project1.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Reversed Fake Memory",
        content="Method B is better than method A.",
        sources=[
            MemorySource(
                source_type=MemorySourceType.PAPER_CHUNK.value,
                paper_id=ready_paper.id,
                page_number=3,
                quote_text="Method A is better than method B",
                document_sha256="hash_ready_123",
            )
        ],
    )
    db.add(reversed_fake_mem)
    db.commit()
    formatted_reversed = format_memories_for_prompt([reversed_fake_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" not in formatted_reversed
    assert "Method B is better than method A" not in formatted_reversed
    db.delete(reversed_fake_mem)
    db.commit()

    # 14. Two opposite sentences on one page: citing X > Y must not establish Y > X,
    # even if Y > X appears nearby on that same page.
    add_page_element_chunk(
        ready_paper.id, 5, "Method X is superior to method Y. Method Y is superior to method X."
    )

    # Mismatched citation: claiming Y > X while citing X > Y must fail
    mismatched_quote_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Contradictory Citation Fact",
        content="Method Y is superior to method X.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=5,
                quote_text="Method X is superior to method Y",
                document_sha256="hash_ready_123",
            )
        ],
    )
    with pytest.raises(ValueError, match="reverses entity roles, relations, or ordering"):
        consolidate_memory_candidate(db, project_id=project1.id, candidate=mismatched_quote_cand)

    # Directly inserted mismatched record must be excluded from prompt formatting
    mismatched_db_mem = Memory(
        project_id=project1.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Mismatched DB Memory",
        content="Method Y is superior to method X.",
        sources=[
            MemorySource(
                source_type=MemorySourceType.PAPER_CHUNK.value,
                paper_id=ready_paper.id,
                page_number=5,
                quote_text="Method X is superior to method Y",
                document_sha256="hash_ready_123",
            )
        ],
    )
    db.add(mismatched_db_mem)
    db.commit()
    formatted_mismatched = format_memories_for_prompt([mismatched_db_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" not in formatted_mismatched
    assert "Method Y is superior to method X" not in formatted_mismatched
    db.delete(mismatched_db_mem)
    db.commit()

    # Accurately cited fact: claiming Y > X while citing Y > X succeeds and formats
    accurately_cited_cand = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Accurately Cited Fact",
        content="Method Y is superior to method X.",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=ready_paper.id,
                page_number=5,
                quote_text="Method Y is superior to method X",
                document_sha256="hash_ready_123",
            )
        ],
    )
    accurate_mem = consolidate_memory_candidate(
        db, project_id=project1.id, candidate=accurately_cited_cand
    )
    assert accurate_mem.id is not None
    formatted_accurate = format_memories_for_prompt([accurate_mem], db=db)
    assert "ESTABLISHED PAPER FACTS" in formatted_accurate
    assert "Method Y is superior to method X" in formatted_accurate
    assert 'Source quote: "Method Y is superior to method X' in formatted_accurate


def test_atomic_supersession_transactional_integrity(db: Session) -> None:
    project = Project(name="Atomic Project")
    db.add(project)
    db.commit()

    cand1 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Initial Choice",
        content="Project decision: Use Adam optimizer.",
    )
    mem1 = create_memory(db, project_id=project.id, memory_in=cand1)
    assert mem1.status == MemoryStatus.ACTIVE.value
    assert mem1.version == 1

    # 1. Version conflict on atomic_create_and_supersede
    cand2 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Second Choice",
        content="Project decision: Use SGD optimizer.",
    )
    with pytest.raises(MemoryVersionConflictError, match="Version conflict"):
        atomic_create_and_supersede(
            db,
            project_id=project.id,
            old_memory_id=mem1.id,
            expected_version=99,
            new_memory_in=cand2,
        )

    # Ensure no partial rows or status change occurred
    db.refresh(mem1)
    assert mem1.status == MemoryStatus.ACTIVE.value
    assert mem1.version == 1
    mems = db.query(Memory).filter(Memory.project_id == project.id).all()
    assert len(mems) == 1

    # 2. Failure injection during atomic_create_and_supersede
    original_commit = db.commit

    def failing_commit():
        raise RuntimeError("Simulated DB commit error")

    db.commit = failing_commit
    try:
        with pytest.raises(RuntimeError, match="Simulated DB commit error"):
            atomic_create_and_supersede(
                db,
                project_id=project.id,
                old_memory_id=mem1.id,
                expected_version=1,
                new_memory_in=cand2,
            )
    finally:
        db.commit = original_commit

    # Verify rollback: old memory remains ACTIVE, exactly 1 memory exists!
    db.refresh(mem1)
    assert mem1.status == MemoryStatus.ACTIVE.value
    assert mem1.version == 1
    active_mems = (
        db.query(Memory)
        .filter(
            Memory.project_id == project.id,
            Memory.status == MemoryStatus.ACTIVE.value,
        )
        .all()
    )
    assert len(active_mems) == 1
    total_mems = db.query(Memory).filter(Memory.project_id == project.id).all()
    assert len(total_mems) == 1

    # 3. Successful atomic supersession
    old_res, new_res = atomic_create_and_supersede(
        db,
        project_id=project.id,
        old_memory_id=mem1.id,
        expected_version=1,
        new_memory_in=cand2,
    )
    assert old_res.status == MemoryStatus.SUPERSEDED.value
    assert old_res.version == 2
    assert new_res.status == MemoryStatus.ACTIVE.value
    assert new_res.version == 1
    assert old_res.superseded_by_id == new_res.id


def test_conversation_capture_and_scoped_retrieval(db: Session) -> None:
    project_a = Project(name="Project A")
    project_b = Project(name="Project B")
    db.add_all([project_a, project_b])
    db.commit()

    conv_a = Conversation(project_id=project_a.id, title="Chat A")
    conv_b = Conversation(project_id=project_b.id, title="Chat B")
    db.add_all([conv_a, conv_b])
    db.commit()

    # Foreign conversation capture must be rejected
    with pytest.raises(ValueError, match="not found in project"):
        capture_conversation_memories(db, project_id=project_a.id, conversation_id=conv_b.id)

    # User message in Project A with decision and preference
    msg_a1 = Message(
        conversation_id=conv_a.id,
        role="user",
        content="Let's decide to use ViT over ResNet50. Also, please always provide bullet points.",
    )
    # Assistant message should NOT create a decision memory
    msg_a_assistant = Message(
        conversation_id=conv_a.id,
        role="assistant",
        content="We decided to choose ResNet101 over ViT as a recommendation.",
    )
    db.add_all([msg_a1, msg_a_assistant])
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

    assert (
        len(mems_a) == 2
    )  # Decision on ViT and Preference on bullet points (NOT assistant ResNet101!)
    assert not any("resnet101" in m.content.lower() for m in mems_a)
    assert len(mems_b) == 1  # Decision on ResNet50

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


def test_paper_memory_source_resolution(db: Session) -> None:
    """Test 4.F1: Deterministic M1-compatible paper memory source resolution and edge cases."""
    project1 = Project(name="Resolution Project 1")
    project2 = Project(name="Resolution Project 2")
    db.add_all([project1, project2])
    db.commit()

    paper = Paper(
        project_id=project1.id,
        filename="transformer.pdf",
        storage_path="/papers/transformer.pdf",
        document_sha256="doc_hash_abc123",
        status="READY",
    )
    db.add(paper)
    db.commit()

    page_text = "The Transformer relies entirely on an attention mechanism to model dependencies."
    page1 = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text=page_text,
    )
    elem1 = PaperElement(
        paper_id=paper.id,
        element_index=0,
        element_type="paragraph",
        text=page_text,
        page_number=1,
        bbox_x_min=72.0,
        bbox_y_min=100.0,
        bbox_x_max=540.0,
        bbox_y_max=150.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="docling-2.0",
    )
    chunk1 = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text=page_text,
    )
    db.add_all([page1, elem1, chunk1])
    db.flush()
    chunk_elem1 = ChunkElement(chunk_id=chunk1.id, element_id=elem1.id, order_index=0)
    db.add(chunk_elem1)
    db.commit()

    # 1. Valid quote resolves to expected chunk, element, character start/end, and VERIFIED anchor
    valid_source = MemorySourceCreate(
        source_type=MemorySourceType.PAPER_CHUNK,
        paper_id=paper.id,
        page_number=1,
        quote_text="attention mechanism",
        document_sha256="doc_hash_abc123",
    )
    ev, anchor, status = resolve_paper_memory_source(db, project1.id, valid_source)
    assert status == AnchorStatus.VERIFIED
    assert ev is not None
    assert anchor is not None
    assert ev.chunk_id == chunk1.id
    assert ev.paper_id == paper.id
    assert anchor.source_element_id == elem1.id
    assert anchor.page_number == 1
    assert anchor.parser_version == "docling-2.0"
    assert anchor.document_sha256 == "doc_hash_abc123"
    assert anchor.source_char_start == page_text.index("attention mechanism")
    assert anchor.source_char_end == anchor.source_char_start + len("attention mechanism")
    assert len(anchor.bounding_boxes) == 1

    # 2. Wrong project must NOT resolve
    ev_wrong_proj, anchor_wrong_proj, status_wrong_proj = resolve_paper_memory_source(
        db, project2.id, valid_source
    )
    assert status_wrong_proj == AnchorStatus.UNRESOLVED
    assert ev_wrong_proj is None
    assert anchor_wrong_proj is None

    # 3. Changed document hash must NOT resolve
    stale_hash_source = MemorySourceCreate(
        source_type=MemorySourceType.PAPER_CHUNK,
        paper_id=paper.id,
        page_number=1,
        quote_text="attention mechanism",
        document_sha256="stale_outdated_hash_456",
    )
    ev_stale, _, status_stale = resolve_paper_memory_source(db, project1.id, stale_hash_source)
    assert status_stale == AnchorStatus.UNRESOLVED
    assert ev_stale is None

    # 4. Wrong page must NOT resolve
    wrong_page_source = MemorySourceCreate(
        source_type=MemorySourceType.PAPER_CHUNK,
        paper_id=paper.id,
        page_number=99,
        quote_text="attention mechanism",
        document_sha256="doc_hash_abc123",
    )
    ev_wrong_page, _, status_wrong_page = resolve_paper_memory_source(
        db, project1.id, wrong_page_source
    )
    assert status_wrong_page == AnchorStatus.UNRESOLVED
    assert ev_wrong_page is None

    # 5. Ambiguous / duplicate quote on page must NOT resolve
    page2 = PaperPage(
        paper_id=paper.id,
        page_number=2,
        width=612.0,
        height=792.0,
        raw_text="Attention is key. Attention is all we need. Attention is crucial.",
    )
    elem2 = PaperElement(
        paper_id=paper.id,
        element_index=1,
        element_type="paragraph",
        text="Attention is key. Attention is all we need. Attention is crucial.",
        page_number=2,
        parser_version="docling-2.0",
    )
    chunk2 = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=1,
        text="Attention is key. Attention is all we need. Attention is crucial.",
    )
    db.add_all([page2, elem2, chunk2])
    db.flush()
    db.add(ChunkElement(chunk_id=chunk2.id, element_id=elem2.id, order_index=0))
    db.commit()

    ambig_source = MemorySourceCreate(
        source_type=MemorySourceType.PAPER_CHUNK,
        paper_id=paper.id,
        page_number=2,
        quote_text="Attention",
        document_sha256="doc_hash_abc123",
    )
    ev_ambig, _, status_ambig = resolve_paper_memory_source(db, project1.id, ambig_source)
    assert status_ambig == AnchorStatus.UNRESOLVED
    assert ev_ambig is None

    # 6. Unready paper must NOT resolve
    paper.status = "PROCESSING"
    db.commit()
    ev_unready, _, status_unready = resolve_paper_memory_source(db, project1.id, valid_source)
    assert status_unready == AnchorStatus.UNRESOLVED
    assert ev_unready is None

    # Restore ready status for remaining checks
    paper.status = "READY"
    db.commit()

    # 7. Deleted chunk must NOT resolve
    db.delete(chunk_elem1)
    db.delete(chunk1)
    db.commit()
    ev_no_chunk, _, status_no_chunk = resolve_paper_memory_source(db, project1.id, valid_source)
    assert status_no_chunk == AnchorStatus.UNRESOLVED
    assert ev_no_chunk is None

    # 8. Deleted element must NOT resolve
    db.delete(elem1)
    db.commit()
    ev_no_elem, _, status_no_elem = resolve_paper_memory_source(db, project1.id, valid_source)
    assert status_no_elem == AnchorStatus.UNRESOLVED
    assert ev_no_elem is None

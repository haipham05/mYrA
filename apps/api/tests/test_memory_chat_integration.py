from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.chat import create_conversation
from app.crud.memory import create_memory, list_memories
from app.db.base import Base
from app.db.models import Project
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    EvidenceItem,
)
from app.schemas.memory import (
    MemoryCreate,
    MemoryStatus,
    MemoryType,
)
from app.services.chat_service import ChatService


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_chat_memory.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.mark.anyio
async def test_chat_answers_from_memory_without_paper_citations(db: Session) -> None:
    project = Project(name="Memory Chat Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Decision Query")

    # Store a project decision
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Decision: Choose AURC over ECE",
        content="Project decision: Selected AURC over ECE for calibration evaluation.",
        importance=0.9,
        confidence=0.95,
        is_pinned=True,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        "Based on our project decision, we selected AURC over ECE for calibration evaluation."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="Why did we choose AURC over ECE?",
        )

    assert resp.content == (
        "Based on our project decision, we selected AURC over ECE for calibration evaluation."
    )
    # Decisions must never fabricate paper citations
    assert len(resp.citations) == 0

    # Ensure memory prompt block contained the decision
    _, kwargs = mock_llm.generate.call_args
    assert "PROJECT MEMORY:" in kwargs["user_prompt"]
    assert "AURC over ECE" in kwargs["user_prompt"]


@pytest.mark.anyio
async def test_chat_distinguishes_paper_evidence_from_user_decision(db: Session) -> None:
    project = Project(name="Hybrid Chat Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Hybrid Query")

    # Store a user decision
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Metric Decision",
        content="Project decision: We prioritize calibration metrics over raw accuracy.",
        importance=0.8,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    # Mock paper evidence with verified anchor
    paper_id = uuid4()
    anchor = CitationAnchor(
        page_number=1,
        exact_quote="Transformer architectures rely on multi-head self-attention mechanisms.",
        document_sha256="docsha123",
        parser_version="1.0.0",
        source_element_id=uuid4(),
        source_char_start=0,
        source_char_end=70,
        anchor_status=AnchorStatus.VERIFIED,
    )
    evidence = EvidenceItem(
        id="E1",
        chunk_id=uuid4(),
        paper_id=paper_id,
        paper_title="Attention Paper",
        page_number=1,
        quote="Transformer architectures rely on multi-head self-attention mechanisms.",
        confidence=0.95,
        bounding_boxes=[
            BoundingBox(
                x_min=10.0,
                y_min=10.0,
                x_max=90.0,
                y_max=30.0,
                page_width=100.0,
                page_height=100.0,
            )
        ],
        document_sha256="docsha123",
        parser_version="1.0.0",
        anchors=[anchor],
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [evidence]

    mock_llm = AsyncMock()
    # LLM cites E1 for paper fact, but cites nothing for project decision
    mock_llm.generate.return_value = (
        "Transformer architectures rely on multi-head self-attention mechanisms [E1]. "
        "In our project, we prioritize calibration metrics over raw accuracy."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What architecture do we use and what is our metric priority?",
        )

    # Paper fact must have citation [1]
    assert "[1]" in resp.content
    assert len(resp.citations) == 1
    assert resp.citations[0].paper_id == paper_id
    # Project decision must remain in response without citation tag
    assert "we prioritize calibration metrics over raw accuracy" in resp.content


@pytest.mark.anyio
async def test_post_turn_memory_capture_and_supersession_in_chat(db: Session) -> None:
    project = Project(name="Auto Capture Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Auto Capture Conv")

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Understood, noted your decision to use ViT over ResNet50."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        # Turn 1: User declares a decision
        await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="We decided to use ViT over ResNet50 for image representation.",
        )

    # Check that a memory was captured post-turn
    mems, count = list_memories(db, project_id=project.id, status=MemoryStatus.ACTIVE)
    assert count >= 1
    vit_mem = next(m for m in mems if "vit" in m.content.lower())
    assert vit_mem.status == MemoryStatus.ACTIVE.value

    # Turn 2: User supersedes the decision
    mock_llm.generate.return_value = "Updated, switching our decision to Swin Transformer."
    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="We decided to use Swin over ViT for hierarchical feature representations.",
        )

    # ViT memory should now be superseded
    db.refresh(vit_mem)
    assert vit_mem.status == MemoryStatus.SUPERSEDED.value
    assert vit_mem.superseded_by_id is not None

    active_mems, _ = list_memories(db, project_id=project.id, status=MemoryStatus.ACTIVE)
    assert any("swin" in m.content.lower() for m in active_mems)
    assert not any(m.id == vit_mem.id for m in active_mems)

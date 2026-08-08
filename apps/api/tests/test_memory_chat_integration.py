from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.chat import create_conversation, list_messages
from app.crud.memory import create_memory, list_memories
from app.db.base import Base
from app.db.models import (
    AssistantRun,
    ChunkElement,
    Memory,
    MemorySource,
    Message,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
    Project,
)
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    EvidenceItem,
)
from app.schemas.memory import (
    MemoryCreate,
    MemorySourceType,
    MemoryStatus,
    MemoryType,
)
from app.services.chat_service import (
    AssistantRunCancelled,
    ChatService,
    _recent_conversation_messages,
)


@pytest.mark.anyio
async def test_cancelled_run_does_not_dispatch_paid_generation(db: Session) -> None:
    project = Project(name="Cancellation project")
    db.add(project)
    db.commit()
    conversation = create_conversation(db, project_id=project.id, title="Cancellation")
    run = AssistantRun(
        project_id=project.id,
        conversation_id=conversation.id,
        idempotency_key="cancel-before-generation",
        request_hash="a" * 64,
        request_payload={},
        status="RUNNING",
        cancel_requested=True,
        attempt_count=1,
        lease_owner="worker-cancel",
    )
    db.add(run)
    db.commit()

    class EmptyRetriever:
        def retrieve(self, *_args, **_kwargs):
            return []

    class FakeEmbeddingProvider:
        def embed_query(self, _query):
            return [0.1]

    with (
        patch(
            "app.services.embedding.get_embedding_provider", return_value=FakeEmbeddingProvider()
        ),
        patch("app.services.chat_service.retrieve_project_memories", return_value=[]),
        patch(
            "app.services.chat_service.get_llm_provider",
            side_effect=AssertionError("cancelled run must not dispatch the provider"),
        ),
    ):
        with pytest.raises(AssistantRunCancelled):
            await ChatService(retriever=EmptyRetriever()).answer_question(
                db,
                conversation.id,
                "Explain the selected evidence",
                assistant_run_id=run.id,
                run_worker_id="worker-cancel",
                run_attempt_count=1,
            )


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
async def test_chat_forwards_persisted_paper_scope_to_all_evidence_sources(db: Session) -> None:
    project = Project(name="Persisted chat scope")
    db.add(project)
    db.flush()
    paper = Paper(
        project_id=project.id,
        filename="selected.pdf",
        storage_path="selected.pdf",
        status="READY",
    )
    db.add(paper)
    db.commit()
    conv = create_conversation(
        db,
        project_id=project.id,
        title="Only one paper",
        paper_scope="paper",
        selected_paper_ids=[str(paper.id)],
    )

    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = AsyncMock()
    llm.generate.return_value = "I do not have a supported answer in the selected evidence."
    llm.model_name = "test-deepseek"
    selected_ids = [paper.id]

    with (
        patch("app.services.chat_service.get_llm_provider", return_value=llm),
        patch("app.services.chat_service.retrieve_project_memories", return_value=[]) as memories,
        patch(
            "app.services.chat_service.retrieve_graph_evidence", return_value=([], None)
        ) as graph,
        patch(
            "app.services.embedding.get_embedding_provider",
            return_value=MagicMock(embed_query=MagicMock(return_value=[0.1, 0.2, 0.3])),
        ),
    ):
        await ChatService(retriever=retriever).answer_question(
            db=db,
            conversation_id=conv.id,
            question="What does the selected paper say?",
        )

    assert retriever.retrieve.call_args.kwargs["selected_paper_ids"] == selected_ids
    assert memories.call_args.kwargs["selected_paper_ids"] == selected_ids
    assert graph.call_args.kwargs["selected_paper_ids"] == selected_ids


@pytest.mark.anyio
async def test_short_follow_up_retrieves_against_single_selected_paper(db: Session) -> None:
    project = Project(name="Follow-up scope")
    db.add(project)
    db.flush()
    paper = Paper(
        project_id=project.id,
        filename="attention.pdf",
        title="Attention Is All You Need",
        storage_path="attention.pdf",
        status="READY",
    )
    db.add(paper)
    db.commit()
    conv = create_conversation(
        db,
        project_id=project.id,
        title="Follow-up",
        paper_scope="paper",
        selected_paper_ids=[str(paper.id)],
    )
    db.add_all(
        [
            Message(
                conversation_id=conv.id,
                role="USER",
                content="Explain the attention mechanism.",
                citations=[],
                evidence=[],
            ),
            Message(
                conversation_id=conv.id,
                role="ASSISTANT",
                content="It uses multi-head self-attention.",
                citations=[],
                evidence=[],
            ),
        ]
    )
    db.commit()

    retriever = MagicMock()
    retriever.retrieve.return_value = []
    llm = AsyncMock()
    llm.generate.return_value = (
        "The selected paper's limitations are not established in the retrieved evidence."
    )
    llm.model_name = "test-deepseek"
    with (
        patch("app.services.chat_service.get_llm_provider", return_value=llm),
        patch("app.services.chat_service.retrieve_project_memories", return_value=[]),
        patch("app.services.chat_service.retrieve_graph_evidence", return_value=([], None)),
        patch(
            "app.services.embedding.get_embedding_provider",
            return_value=MagicMock(embed_query=MagicMock(return_value=[0.1, 0.2, 0.3])),
        ),
    ):
        await ChatService(retriever=retriever).answer_question(
            db=db, conversation_id=conv.id, question="What are its limitations?"
        )

    resolved = "What limitations does Attention Is All You Need report?"
    assert retriever.retrieve.call_args.kwargs["query"] == resolved
    assert (
        llm.generate.call_args.kwargs["user_prompt"].find(
            "RESOLVED RETRIEVAL QUESTION:\n" + resolved
        )
        >= 0
    )


@pytest.mark.anyio
async def test_ambiguous_follow_up_asks_instead_of_searching(db: Session) -> None:
    project = Project(name="Ambiguous follow-up")
    db.add(project)
    db.flush()
    papers = [
        Paper(
            project_id=project.id,
            filename=f"paper-{index}.pdf",
            title=f"Paper {index}",
            storage_path=f"paper-{index}.pdf",
            status="READY",
        )
        for index in (1, 2)
    ]
    db.add_all(papers)
    db.commit()
    conv = create_conversation(db, project_id=project.id, title="Ambiguous")
    db.add(
        Message(
            conversation_id=conv.id,
            role="ASSISTANT",
            content="Both papers were discussed.",
            citations=[],
            evidence=[{"paper_id": str(paper.id)} for paper in papers],
        )
    )
    db.commit()

    retriever = MagicMock()
    response = await ChatService(retriever=retriever).answer_question(
        db=db, conversation_id=conv.id, question="their limitations"
    )

    assert "more than one paper" in response.content
    retriever.retrieve.assert_not_called()


def test_chat_history_query_is_bounded_to_recent_messages(db: Session) -> None:
    project = Project(name="Bounded history")
    db.add(project)
    db.commit()
    conv = create_conversation(db, project_id=project.id, title="History")
    db.add_all(
        [
            Message(
                conversation_id=conv.id,
                role="USER" if index % 2 == 0 else "ASSISTANT",
                content=f"turn-{index}",
                citations=[],
                evidence=[],
            )
            for index in range(12)
        ]
    )
    db.commit()

    history = _recent_conversation_messages(db, conv.id, question="new question")

    assert len(history) == 6
    assert [message.content for message in history] == [f"turn-{index}" for index in range(6, 12)]


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


def setup_ingested_paper(
    db: Session,
    project_id: UUID,
    doc_sha256: str = "hash_doc_123",
    status: str = "READY",
    page_text: str = "The Transformer model uses multi-head attention mechanism across sub-layers.",
    page_number: int = 1,
) -> tuple[Paper, PaperPage, PaperElement, PaperChunk]:
    paper = Paper(
        project_id=project_id,
        filename="transformer.pdf",
        storage_path="papers/transformer.pdf",
        document_sha256=doc_sha256,
        status=status,
    )
    db.add(paper)
    db.flush()

    page = PaperPage(
        paper_id=paper.id,
        page_number=page_number,
        width=612.0,
        height=792.0,
        raw_text=page_text,
    )
    elem = PaperElement(
        paper_id=paper.id,
        page_number=page_number,
        element_index=0,
        element_type="paragraph",
        text=page_text,
        bbox_x_min=0.1,
        bbox_y_min=0.1,
        bbox_x_max=0.9,
        bbox_y_max=0.2,
        page_width=612.0,
        page_height=792.0,
        parser_version="docling_test",
    )
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text=page_text,
    )
    db.add_all([page, elem, chunk])
    db.flush()
    db.add(ChunkElement(chunk_id=chunk.id, element_id=elem.id, order_index=0))
    db.commit()
    return paper, page, elem, chunk


@pytest.mark.anyio
async def test_unresolvable_paper_memory_does_not_publish_uncited_answer(db: Session) -> None:
    project = Project(name="Unresolvable Memory Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Unresolvable Query")

    # Paper exists but is FAILED (unready)
    paper, _, _, _ = setup_ingested_paper(db, project_id=project.id, status="FAILED")

    # Paper fact memory points to unready paper
    mem = Memory(
        project_id=project.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Unresolvable Fact",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        confidence=0.9,
        importance=0.8,
        version=1,
    )
    db.add(mem)
    db.flush()

    src = MemorySource(
        memory_id=mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper.id,
        page_number=1,
        quote_text="The Transformer model uses multi-head attention mechanism across sub-layers.",
        document_sha256=paper.document_sha256,
    )
    db.add(src)
    db.commit()

    mock_llm = AsyncMock()
    # Mock LLM generates uncited factual response
    mock_llm.generate.return_value = (
        "The Transformer model uses multi-head attention mechanism across sub-layers."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What mechanism does Transformer use?",
        )

    # Must NOT publish the uncited factual assertion
    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_verified_paper_memory_routes_through_evidence_and_citations(db: Session) -> None:
    project = Project(name="Verified Paper Memory Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Verified Fact Query")
    quote = "The Transformer model uses multi-head attention mechanism across sub-layers."
    paper, _, _, _ = setup_ingested_paper(
        db, project_id=project.id, status="READY", page_text=quote
    )

    mem = Memory(
        project_id=project.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Transformer Multi-Head Attention",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        confidence=1.0,
        importance=0.9,
        version=1,
    )
    db.add(mem)
    db.flush()

    src = MemorySource(
        memory_id=mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper.id,
        page_number=1,
        quote_text=quote,
        document_sha256=paper.document_sha256,
    )
    db.add(src)
    db.commit()

    mock_llm = AsyncMock()
    # LLM accurately cites E1 (routed from the resolved verified paper fact memory)
    mock_llm.generate.return_value = f"{quote.rstrip('.')} [E1]."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What mechanism does Transformer use?",
        )

    # Verified paper fact must publish with citation [1]
    assert "[1]" in resp.content
    assert f"{quote.rstrip('.')} [1]." in resp.content
    assert len(resp.citations) == 1
    assert resp.citations[0].paper_id == paper.id
    assert resp.citations[0].page_number == 1
    assert resp.citations[0].anchor_status == AnchorStatus.VERIFIED
    assert resp.citations[0].quote == quote


@pytest.mark.anyio
@pytest.mark.parametrize(
    "wrapped_claim",
    [
        (
            'This suggests the paper proves "The Transformer model relies entirely on '
            'self-attention" and cures cancer.'
        ),
        (
            "This suggests the paper proves **The Transformer model relies entirely on "
            "self-attention** and cures cancer."
        ),
    ],
)
async def test_chat_publishes_only_verified_quote_from_unsupported_wrapper(
    db: Session, wrapped_claim: str
) -> None:
    project = Project(name="Quoted Claim Project")
    db.add(project)
    db.commit()
    conv = create_conversation(db, project_id=project.id, title="Quoted Claim")
    paper, _, _, chunk = setup_ingested_paper(
        db,
        project_id=project.id,
        page_text="The Transformer model relies entirely on self-attention.",
    )
    quote = "The Transformer model relies entirely on self-attention."
    anchor = CitationAnchor(
        page_number=1,
        exact_quote=quote,
        document_sha256=paper.document_sha256,
        parser_version="docling_test",
        source_element_id=uuid4(),
        source_char_start=0,
        source_char_end=len(quote),
        anchor_status=AnchorStatus.VERIFIED,
    )
    evidence = EvidenceItem(
        id="E1",
        chunk_id=chunk.id,
        paper_id=paper.id,
        paper_title="Transformer paper",
        page_number=1,
        quote=quote,
        document_sha256=paper.document_sha256,
        parser_version="docling_test",
        anchors=[anchor],
    )
    retriever = MagicMock()
    retriever.retrieve.return_value = [evidence]
    llm = AsyncMock()
    llm.generate.return_value = f"{wrapped_claim} [E1]"
    llm.model_name = "test-deepseek"

    with (
        patch("app.services.chat_service.get_llm_provider", return_value=llm),
        patch("app.services.chat_service.retrieve_project_memories", return_value=[]),
        patch("app.services.chat_service.retrieve_graph_evidence", return_value=([], None)),
        patch(
            "app.services.embedding.get_embedding_provider",
            return_value=MagicMock(embed_query=MagicMock(return_value=[0.1, 0.2, 0.3])),
        ),
    ):
        response = await ChatService(retriever=retriever).answer_question(
            db=db,
            conversation_id=conv.id,
            question="What does the paper say about the Transformer?",
        )

    assert "cures cancer" not in response.content
    assert "proves" not in response.content
    assert f"“{quote.removesuffix('.')}” [1]" == response.content
    assert len(response.citations) == 1
    assert retriever.retrieve.call_args.kwargs["query_embedding"] == [0.1, 0.2, 0.3]
    assert len(response.claim_supports) == 1
    assert response.claim_supports[0].claim_text == quote.removesuffix(".")
    assert response.claim_supports[0].evidence_ids == ["E1"]
    assert "cures cancer" not in response.claim_supports[0].claim_text
    assert all(support.support_kind != "interpretation" for support in response.claim_supports)

    messages, _ = list_messages(db, conv.id)
    saved_answer = messages[-1]
    assert saved_answer.content == response.content
    assert "cures cancer" not in saved_answer.content


@pytest.mark.anyio
async def test_chat_preserves_valid_multi_element_verbatim_quote(db: Session) -> None:
    project = Project(name="Multi Element Quote Project")
    db.add(project)
    db.commit()
    conv = create_conversation(db, project_id=project.id, title="Multi Element Quote")
    page_text = "A Transformer uses attention mechanisms. The decoder removes recurrent layers."
    paper = Paper(
        project_id=project.id,
        filename="transformer.pdf",
        storage_path="papers/transformer.pdf",
        document_sha256="multi-element-hash",
        status="READY",
    )
    db.add(paper)
    db.flush()
    page = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text=page_text,
    )
    first_text = "A Transformer uses attention mechanisms."
    second_text = "The decoder removes recurrent layers."
    elements = [
        PaperElement(
            paper_id=paper.id,
            page_number=1,
            element_index=index,
            element_type="paragraph",
            text=text,
            page_width=612.0,
            page_height=792.0,
            parser_version="docling_test",
        )
        for index, text in enumerate((first_text, second_text))
    ]
    chunk = PaperChunk(paper_id=paper.id, chunk_type="child", chunk_index=0, text=page_text)
    db.add_all([page, *elements, chunk])
    db.flush()
    anchors = [
        CitationAnchor(
            page_number=1,
            exact_quote=text,
            document_sha256=paper.document_sha256,
            parser_version="docling_test",
            source_element_id=element.id,
            source_char_start=start,
            source_char_end=start + len(text),
            anchor_status=AnchorStatus.VERIFIED,
        )
        for text, element, start in (
            (first_text, elements[0], 0),
            (second_text, elements[1], len(first_text) + 1),
        )
    ]
    evidence = EvidenceItem(
        id="E1",
        chunk_id=chunk.id,
        paper_id=paper.id,
        paper_title="Transformer paper",
        page_number=1,
        quote=first_text,
        document_sha256=paper.document_sha256,
        parser_version="docling_test",
        anchors=anchors,
    )
    retriever = MagicMock()
    retriever.retrieve.return_value = [evidence]
    quote = f"{first_text} {second_text}"
    llm = AsyncMock()
    llm.generate.return_value = f'The paper states "{quote}" [E1].'
    llm.model_name = "test-deepseek"

    with (
        patch("app.services.chat_service.get_llm_provider", return_value=llm),
        patch("app.services.chat_service.retrieve_project_memories", return_value=[]),
        patch("app.services.chat_service.retrieve_graph_evidence", return_value=([], None)),
        patch(
            "app.services.embedding.get_embedding_provider",
            return_value=MagicMock(embed_query=MagicMock(return_value=None)),
        ),
    ):
        response = await ChatService(retriever=retriever).answer_question(
            db=db,
            conversation_id=conv.id,
            question="What does the paper say about attention and recurrence?",
        )

    assert response.content == f"“{quote}” [1]"
    assert len(response.citations) == 1
    assert response.citations[0].quote == quote
    messages, _ = list_messages(db, conv.id)
    assert messages[-1].content == response.content


@pytest.mark.anyio
async def test_archived_stale_or_foreign_project_memories_cannot_supply_evidence(
    db: Session,
) -> None:
    project_a = Project(name="Project A")
    project_b = Project(name="Project B")
    db.add_all([project_a, project_b])
    db.commit()

    conv = create_conversation(db, project_id=project_a.id, title="Project A Query")
    quote = "The Transformer model uses multi-head attention mechanism across sub-layers."

    # Paper in Project A
    paper_a, _, _, _ = setup_ingested_paper(
        db, project_id=project_a.id, status="READY", page_text=quote
    )
    # Paper in Project B
    paper_b, _, _, _ = setup_ingested_paper(
        db, project_id=project_b.id, status="READY", page_text=quote
    )

    # 1. Archived memory in Project A
    archived_mem = Memory(
        project_id=project_a.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ARCHIVED.value,
        title="Archived Fact",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        version=1,
    )
    # 2. Stale memory in Project A (document_sha256 mismatch)
    stale_mem = Memory(
        project_id=project_a.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Stale Fact",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        version=1,
    )
    # 3. Foreign memory in Project B
    foreign_mem = Memory(
        project_id=project_b.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Foreign Fact",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        version=1,
    )
    db.add_all([archived_mem, stale_mem, foreign_mem])
    db.flush()

    src_archived = MemorySource(
        memory_id=archived_mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper_a.id,
        page_number=1,
        quote_text=quote,
        document_sha256=paper_a.document_sha256,
    )
    src_stale = MemorySource(
        memory_id=stale_mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper_a.id,
        page_number=1,
        quote_text=quote,
        document_sha256="stale_outdated_hash_456",
    )
    src_foreign = MemorySource(
        memory_id=foreign_mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper_b.id,
        page_number=1,
        quote_text=quote,
        document_sha256=paper_b.document_sha256,
    )
    db.add_all([src_archived, src_stale, src_foreign])
    db.commit()

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = f"{quote} [E1]."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        chat_service = ChatService(retriever=mock_retriever)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What mechanism does Transformer use?",
        )

    # Archived, stale, and foreign memories must NOT supply evidence; E1 was not registered
    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_zero_citation_paper_answer_regression_rejected(db: Session) -> None:
    project = Project(name="Zero Citation Regression Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Zero Citation Query")
    quote = "The Transformer model uses multi-head attention mechanism across sub-layers."
    paper, _, _, _ = setup_ingested_paper(
        db, project_id=project.id, status="READY", page_text=quote
    )

    mem = Memory(
        project_id=project.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Transformer Fact",
        content="The Transformer model uses multi-head attention mechanism across sub-layers.",
        confidence=1.0,
        importance=0.9,
        version=1,
    )
    db.add(mem)
    db.flush()

    src = MemorySource(
        memory_id=mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper.id,
        page_number=1,
        quote_text=quote,
        document_sha256=paper.document_sha256,
    )
    db.add(src)
    db.commit()

    mock_llm = AsyncMock()
    # LLM hallucinates an answer to the paper question with ZERO citations
    mock_llm.generate.return_value = (
        "The Transformer model uses multi-head attention mechanism across sub-layers."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What mechanism does Transformer use?",
        )

    # Must be rejected because paper fact has 0 citations!
    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_false_decision_claim_sharing_subject_is_rejected(db: Session) -> None:
    """Ensure arbitrary claims sharing one subject with a decision
    (e.g. 'AURC cures cancer') are rejected.
    """
    project = Project(name="False Claim Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="False Claim Conv")

    # Stored active decision
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Decision",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "AURC cures cancer."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What does AURC do?",
        )

    # False claim sharing 'AURC' must be abstained/rejected!
    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_legitimate_decision_claim_retained_without_fake_citation(
    db: Session,
) -> None:
    """Ensure legitimate decision sentences are retained without paper citations."""
    project = Project(name="Legit Decision Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Legit Decision Conv")

    # Stored active decision
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Choice",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        "According to our project decisions, we chose AURC over ECE for calibration."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What metric did we choose for calibration?",
        )

    assert resp.content == (
        "According to our project decisions, we chose AURC over ECE for calibration."
    )
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_uncommitted_turn_candidate_does_not_authorize_uncited_answer(
    db: Session,
) -> None:
    """Ensure uncommitted turn text does not authorize uncited answers when no memory is saved."""
    project = Project(name="Turn Candidate Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Turn Candidate Conv")

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "We chose AURC over ECE for calibration."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="We decided to use AURC over ECE for calibration.",
        )

    # No saved decision exists in database yet, so uncited answer must NOT be authorized!
    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_unsupported_suffix_clause_is_rejected(db: Session) -> None:
    """Ensure sentences with supported prefix but unsupported suffix clause are rejected."""
    project = Project(name="Suffix Test Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Suffix Test Conv")
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Decision",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        "We chose AURC over ECE for calibration, and AURC cures cancer."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What did we decide about AURC?",
        )

    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_quoted_wrapper_with_unsupported_prose_is_rejected(db: Session) -> None:
    """Ensure quoted memories surrounded by unsupported substantive prose are rejected."""
    project = Project(name="Quote Wrapper Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Quote Wrapper Conv")
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Decision",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        'Our note says "We chose AURC over ECE for calibration" and AURC cures cancer.'
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What does our note say?",
        )

    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_negation_inversion_of_decision_is_rejected(db: Session) -> None:
    """Ensure direct negation of a project decision is rejected and abstained."""
    project = Project(name="Negation Test Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Negation Test Conv")
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Decision",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "We did not choose AURC over ECE for calibration."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="Did we choose AURC for calibration?",
        )

    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0


@pytest.mark.anyio
async def test_single_foreign_modifier_is_rejected(db: Session) -> None:
    """Ensure arbitrary single foreign token modifiers (e.g. 'mistakenly') are rejected."""
    project = Project(name="Foreign Modifier Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Modifier Test Conv")
    mem_in = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Calibration Metric Decision",
        content="We chose AURC over ECE for calibration.",
        importance=0.9,
        confidence=1.0,
    )
    create_memory(db, project_id=project.id, memory_in=mem_in)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "We chose AURC over ECE for calibration mistakenly."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService()
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="How did we choose AURC?",
        )

    assert "Insufficient evidence available in the uploaded papers" in resp.content
    assert len(resp.citations) == 0

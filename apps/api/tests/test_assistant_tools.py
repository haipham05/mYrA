from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Memory, Paper, PaperPage, Project
from app.schemas.assistant import AssistantIntent, AssistantRunRequest, RouteDecision
from app.schemas.evidence import AnchorStatus, Citation, EvidenceItem
from app.services import assistant_tools
from app.services.assistant_tools import (
    AssistantToolResult,
    CompareArguments,
    NotesArguments,
    ToolContext,
    ToolStatus,
    build_tool_registry,
    execute_tool,
    tool_requires_approval,
    validate_tool_input,
)
from app.services.llm import GenerationResult, GenerationUsage, LLMProvider


def _request(*, scope: str = "selection", selected_paper_ids=None) -> AssistantRunRequest:
    selected_ids = [uuid4(), uuid4()] if selected_paper_ids is None else selected_paper_ids
    return AssistantRunRequest.model_validate(
        {
            "message": "compare these papers",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": scope,
            "selected_paper_ids": selected_ids,
            "idempotency_key": "tool-request-123",
        }
    )


def test_registry_contains_only_fixed_intents_and_typed_callable_contracts() -> None:
    registry = build_tool_registry()

    assert set(registry) == set(AssistantIntent)
    assert all(callable(item.handler) for item in registry.values())
    assert all(item.input_model and item.output_model for item in registry.values())
    with pytest.raises(KeyError):
        registry["run_shell"]  # type: ignore[index]


def test_tool_validation_rejects_arbitrary_fields_and_checks_paper_cardinality() -> None:
    registry = build_tool_registry()
    request = _request()
    decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Compare the selected papers",
        arguments={"dimensions": ["method_architecture", "results"]},
    )
    validated = validate_tool_input(registry[AssistantIntent.COMPARE], request, decision)
    assert isinstance(validated.arguments, CompareArguments)
    assert validated.arguments.dimensions == ["method_architecture", "results"]

    bad_decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        action_summary="Compare papers",
        arguments={"dimensions": [], "shell": "rm -rf"},
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_tool_input(registry[AssistantIntent.COMPARE], request, bad_decision)

    one_paper = _request(selected_paper_ids=[uuid4()])
    one_paper_decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        action_summary="Compare papers",
    )
    with pytest.raises(ValueError, match="select more papers"):
        validate_tool_input(registry[AssistantIntent.COMPARE], one_paper, one_paper_decision)


@pytest.mark.anyio
async def test_compare_tool_returns_scoped_citations_and_validated_synthesis(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'compare.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project = Project(name="Comparison project")
        db.add(project)
        db.flush()
        papers = [
            Paper(
                project_id=project.id,
                filename=f"paper-{index}.pdf",
                storage_path=f"local/paper-{index}.pdf",
                status="READY",
            )
            for index in range(2)
        ]
        db.add_all(papers)
        db.commit()

        quote = (
            "Image classification on ImageNet validation reports 90 percent top-1 accuracy "
            "under single crop 224px."
        )

        class FakeRetriever:
            def retrieve(self, _db, _project_id, _query, *, selected_paper_ids):
                assert len(selected_paper_ids) == 1
                return [
                    EvidenceItem(
                        id="E1",
                        paper_id=selected_paper_ids[0],
                        chunk_id=uuid4(),
                        quote=quote,
                        page_number=1,
                    )
                ]

        class FakeChatService:
            retriever = FakeRetriever()

        class FakeProvider(LLMProvider):
            @property
            def provider_name(self):
                return "test"

            async def generate(self, _system_prompt, _user_prompt):
                return (
                    '{"findings":[{"text":"' + quote + '","kind":"direct","evidence_ids":["C1"]}],'
                    '"benchmark_comparisons":[{"left_paper_id":"'
                    + str(papers[0].id)
                    + '","right_paper_id":"'
                    + str(papers[1].id)
                    + '","left_context":{"task":"image classification","dataset":"ImageNet",'
                    '"split":"validation","metric":"top-1 accuracy","unit":"percent",'
                    '"comparison_condition":"single crop 224px"},"right_context":{"task":"image '
                    'classification","dataset":"ImageNet","split":"validation",'
                    '"metric":"top-1 accuracy","unit":"percent","comparison_condition":"single '
                    'crop 224px"},"left_result":"90 percent","right_result":"90 percent",'
                    '"left_context_quote":"' + quote + '","right_context_quote":"' + quote + '",'
                    '"left_evidence_ids":["C1"],"right_evidence_ids":["C2"]}]}'
                )

            async def generate_result(self, system_prompt, user_prompt):
                return GenerationResult(
                    content=await self.generate(system_prompt, user_prompt),
                    usage=GenerationUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16),
                )

        telemetry_events = []

        class FakeObservation:
            def update(self, **kwargs):
                telemetry_events.append(kwargs)

        class FakeTelemetry:
            @contextmanager
            def stage(self, name, **_kwargs):
                telemetry_events.append({"stage": name})
                yield FakeObservation()

        monkeypatch.setattr(assistant_tools, "get_llm_provider", lambda: FakeProvider())
        monkeypatch.setattr(assistant_tools, "get_telemetry", lambda: FakeTelemetry())
        request = AssistantRunRequest(
            message="Compare the selected methods",
            conversation_id=uuid4(),
            project_id=project.id,
            scope="selection",
            selected_paper_ids=[paper.id for paper in papers],
            idempotency_key="compare-tool-123",
        )
        decision = RouteDecision(
            intent=AssistantIntent.COMPARE,
            resolved_paper_ids=request.selected_paper_ids,
            action_summary="Compare selected papers",
            arguments={"dimensions": ["method_architecture"]},
        )
        definition = build_tool_registry()[AssistantIntent.COMPARE]
        result = await execute_tool(
            definition,
            ToolContext(db=db, chat_service=FakeChatService()),  # type: ignore[arg-type]
            validate_tool_input(definition, request, decision),
        )

        matrix = result.structured_payload["matrix"]
        assert result.status is ToolStatus.SUCCEEDED
        assert result.result_type == "comparison"
        assert len(matrix["cells"]) == 2
        synthesis = result.structured_payload["synthesis"]
        assert synthesis["benchmark_comparisons"][0]["comparability"]["status"] == (
            "directly_comparable"
        )
        assert "Reported results:" in result.display_text
        assert {event.get("stage") for event in telemetry_events} >= {
            "comparison.synthesize",
            "comparison.compatibility",
        }
        assert [citation.paper_id for citation in result.citations] == [
            papers[0].id,
            papers[1].id,
        ]
        assert result.usage == {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_cache_hit_tokens": None,
            "prompt_cache_miss_tokens": None,
        }
        assert any(
            event.get("metadata", {}).get("provider_usage", {}).get("total_tokens") == 16
            for event in telemetry_events
        )
        assert "[C1]" in result.display_text
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.anyio
async def test_compare_tool_rejects_papers_outside_project_before_retrieval(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'compare-scope.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project = Project(name="Current project")
        other_project = Project(name="Other project")
        db.add_all([project, other_project])
        db.flush()
        papers = [
            Paper(
                project_id=other_project.id,
                filename=f"foreign-{index}.pdf",
                storage_path=f"local/foreign-{index}.pdf",
                status="READY",
            )
            for index in range(2)
        ]
        db.add_all(papers)
        db.commit()

        class FakeRetriever:
            calls = 0

            def retrieve(self, *_args, **_kwargs):
                self.calls += 1
                return []

        retriever = FakeRetriever()

        class FakeChatService:
            pass

        chat_service = FakeChatService()
        chat_service.retriever = retriever
        request = AssistantRunRequest(
            message="Compare these papers",
            conversation_id=uuid4(),
            project_id=project.id,
            scope="selection",
            selected_paper_ids=[paper.id for paper in papers],
            idempotency_key="compare-scope-123",
        )
        decision = RouteDecision(
            intent=AssistantIntent.COMPARE,
            resolved_paper_ids=request.selected_paper_ids,
            action_summary="Compare selected papers",
        )
        definition = build_tool_registry()[AssistantIntent.COMPARE]
        result = await execute_tool(
            definition,
            ToolContext(db=db, chat_service=chat_service),  # type: ignore[arg-type]
            validate_tool_input(definition, request, decision),
        )

        assert result.status is ToolStatus.NEEDS_INPUT
        assert retriever.calls == 0
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_note_reads_do_not_require_approval_but_mutations_do() -> None:
    definition = build_tool_registry()[AssistantIntent.NOTES]
    request = _request()
    read_decision = RouteDecision(
        intent=AssistantIntent.NOTES,
        action_summary="List notes",
        arguments={"action": "list"},
    )
    update_decision = RouteDecision(
        intent=AssistantIntent.NOTES,
        action_summary="Propose note update",
        arguments={
            "action": "propose_update",
            "content": "new preference",
            "memory_id": str(uuid4()),
            "expected_version": 1,
        },
    )
    assert not tool_requires_approval(
        definition, validate_tool_input(definition, request, read_decision)
    )
    assert tool_requires_approval(
        definition, validate_tool_input(definition, request, update_decision)
    )


@pytest.mark.anyio
async def test_report_draft_uses_existing_grounded_chat_and_records_source_manifest() -> None:
    paper_id = uuid4()
    chunk_id = uuid4()
    digest = "a" * 64

    class FakeChatService:
        def __init__(self) -> None:
            self.call = None

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.call = (conversation_id, question, kwargs)
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "test-model",
                    "content": "## Findings\nThe paper uses attention [1].",
                    "citations": [
                        Citation(
                            citation_index=1,
                            evidence_id="E1",
                            paper_id=paper_id,
                            page_number=3,
                            quote="The model uses attention.",
                            document_sha256=digest,
                            anchor_status=AnchorStatus.VERIFIED,
                        )
                    ],
                    "evidence": [
                        EvidenceItem(
                            id="E1",
                            paper_id=paper_id,
                            paper_title="Attention Is All You Need",
                            chunk_id=chunk_id,
                            quote="The model uses attention.",
                            page_number=3,
                            document_sha256=digest,
                        ),
                        EvidenceItem(
                            id="E2",
                            paper_id=paper_id,
                            paper_title="Attention Is All You Need",
                            chunk_id=uuid4(),
                            quote="Uncited retrieved text.",
                            page_number=4,
                            document_sha256=digest,
                        ),
                    ],
                    "provider_usage": {"total_tokens": 20},
                },
            )()

    request = _request(scope="paper", selected_paper_ids=[paper_id])
    decision = RouteDecision(
        intent=AssistantIntent.REPORT,
        resolved_paper_ids=[paper_id],
        action_summary="Draft a paper report",
        arguments={"question": "Summarize this paper's findings."},
    )
    registry = build_tool_registry()
    chat = FakeChatService()
    result = await execute_tool(
        registry[AssistantIntent.REPORT],
        ToolContext(db=None, chat_service=chat),  # type: ignore[arg-type]
        validate_tool_input(registry[AssistantIntent.REPORT], request, decision),
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result_type == "research_report"
    assert result.structured_payload["saved"] is False
    assert result.structured_payload["source_manifest"] == [
        {
            "evidence_id": "1",
            "source_evidence_id": "E1",
            "citation_index": 1,
            "paper_id": str(paper_id),
            "paper_title": "Attention Is All You Need",
            "page_number": 3,
            "chunk_id": str(chunk_id),
            "document_sha256": digest,
            "quote": "The model uses attention.",
        }
    ]
    assert "[1]" in result.display_text
    assert "E2" not in str(result.structured_payload["source_manifest"])
    assert chat.call[2]["paper_scope"] == "selection"
    assert chat.call[2]["selected_paper_ids"] == [paper_id]
    assert "cite every factual sentence" in chat.call[2]["response_guidance"].lower()


@pytest.mark.anyio
async def test_note_actions_save_update_archive_and_reject_stale_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'note-actions.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project = Project(name="Note actions")
        db.add(project)
        db.commit()
        request = AssistantRunRequest.model_validate(
            {
                "message": "Save a research note",
                "conversation_id": str(uuid4()),
                "project_id": str(project.id),
                "scope": "project",
                "selected_paper_ids": [],
                "idempotency_key": "note-action-123",
            }
        )
        definition = build_tool_registry()[AssistantIntent.NOTES]
        context = ToolContext(db=db, chat_service=object())  # type: ignore[arg-type]

        def tool_input(arguments: NotesArguments):
            decision = RouteDecision(
                intent=AssistantIntent.NOTES,
                action_summary="Manage a research note",
                arguments=arguments.model_dump(mode="json", exclude_none=True),
            )
            return validate_tool_input(definition, request, decision)

        saved = await execute_tool(
            definition,
            context,
            tool_input(
                NotesArguments(
                    action="propose_save", title="Reading preference", content="Keep short notes."
                )
            ),
        )
        memory_id = saved.structured_payload["item"]["id"]
        assert saved.status is ToolStatus.SUCCEEDED
        assert db.query(Memory).filter(Memory.project_id == project.id).count() == 1

        updated = await execute_tool(
            definition,
            context,
            tool_input(
                NotesArguments(
                    action="propose_update",
                    memory_id=memory_id,
                    expected_version=1,
                    content="Keep concise research notes.",
                )
            ),
        )
        assert updated.structured_payload["item"]["version"] == 2

        stale = await execute_tool(
            definition,
            context,
            tool_input(
                NotesArguments(
                    action="propose_update",
                    memory_id=memory_id,
                    expected_version=1,
                    content="Overwrite with stale content.",
                )
            ),
        )
        assert stale.status is ToolStatus.NEEDS_INPUT
        assert (
            db.query(Memory).filter(Memory.id == UUID(memory_id)).one().content
            == "Keep concise research notes."
        )

        archived = await execute_tool(
            definition,
            context,
            tool_input(
                NotesArguments(
                    action="propose_archive",
                    memory_id=memory_id,
                    expected_version=2,
                )
            ),
        )
        assert archived.structured_payload["item"]["status"] == "ARCHIVED"
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
    assert tool_requires_approval(
        build_tool_registry()[AssistantIntent.TRANSLATE],
        validate_tool_input(
            build_tool_registry()[AssistantIntent.TRANSLATE],
            _request(scope="paper", selected_paper_ids=[uuid4()]),
            RouteDecision(
                intent=AssistantIntent.TRANSLATE,
                action_summary="Translate the selected paper",
                arguments={"target_language": "vi"},
            ),
        ),
    )


@pytest.mark.anyio
async def test_claim_tool_verifies_current_source_and_keeps_refutation_model_assessed(
    tmp_path, monkeypatch
):
    engine = create_engine(f"sqlite:///{tmp_path / 'claim-tool.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        project = Project(name="Claim project")
        db.add(project)
        db.flush()
        quote = "The method did not improve accuracy in the trial."
        digest = "a" * 64
        paper = Paper(
            project_id=project.id,
            filename="claim.pdf",
            storage_path="local/claim.pdf",
            document_sha256=digest,
            status="READY",
        )
        db.add(paper)
        db.flush()
        db.add(
            PaperPage(
                paper_id=paper.id,
                page_number=1,
                width=612,
                height=792,
                raw_text=quote,
            )
        )
        db.commit()

        class FakeRetriever:
            def retrieve(self, _db, _project_id, _query, *, selected_paper_ids):
                assert selected_paper_ids == [paper.id]
                return [
                    EvidenceItem(
                        id="E1",
                        paper_id=paper.id,
                        chunk_id=uuid4(),
                        quote=quote,
                        page_number=1,
                        document_sha256=digest,
                    )
                ]

        class FakeChatService:
            retriever = FakeRetriever()

        class FakeProvider(LLMProvider):
            responses = [
                '{"refuting_evidence_ids":[]}',
                '{"refuting_evidence_ids":["S1"]}',
            ]

            @property
            def provider_name(self):
                return "test"

            async def generate(self, _system_prompt, _user_prompt):
                return self.responses.pop(0)

        monkeypatch.setattr(assistant_tools, "get_llm_provider", lambda: FakeProvider())
        definition = build_tool_registry()[AssistantIntent.VERIFY_CLAIM]

        async def run_claim(claim, key):
            request = AssistantRunRequest(
                message="Verify the selected claim",
                conversation_id=uuid4(),
                project_id=project.id,
                scope="paper",
                selected_paper_ids=[paper.id],
                idempotency_key=key,
            )
            decision = RouteDecision(
                intent=AssistantIntent.VERIFY_CLAIM,
                resolved_paper_ids=[paper.id],
                action_summary="Verify the claim",
                arguments={"claim": claim},
            )
            return await execute_tool(
                definition,
                ToolContext(db=db, chat_service=FakeChatService()),  # type: ignore[arg-type]
                validate_tool_input(definition, request, decision),
            )

        supported = await run_claim(quote, "claim-check-123")
        contradicted = await run_claim(
            "The method improved accuracy in the trial.", "claim-check-456"
        )

        assert supported.structured_payload["verdict"] == "supported"
        assert supported.citations[0].anchor_status.value == "verified"
        assert contradicted.structured_payload["verdict"] == "contradicted"
        assert contradicted.structured_payload["semantic_contradiction"] == "model_assessed"
        assert contradicted.citations[0].quote == quote
        assert "model-assessed" in contradicted.display_text
    finally:
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.mark.anyio
async def test_qa_tool_reuses_existing_chat_service_once() -> None:
    class FakeChatService:
        def __init__(self) -> None:
            self.calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((db, conversation_id, question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "deepseek-flash",
                    "content": "Grounded result [1].",
                    "citations": [],
                    "evidence": [],
                    "provider_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )()

    service = FakeChatService()
    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.QA,
        standalone_question="What are its limitations?",
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Answer from the selected paper",
    )
    registry = build_tool_registry()
    definition = registry[AssistantIntent.QA]
    tool_input = validate_tool_input(definition, request, decision)
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=service),  # type: ignore[arg-type]
        tool_input,
    )

    assert isinstance(result, AssistantToolResult)
    assert result.status is ToolStatus.SUCCEEDED
    assert result.display_text == "Grounded result [1]."
    assert len(service.calls) == 1
    assert service.calls[0][2] == request.message
    assert service.calls[0][3]["retrieval_question"] == decision.standalone_question


@pytest.mark.anyio
async def test_read_paper_tool_reuses_scoped_chat_and_returns_brief_sections() -> None:
    class FakeChatService:
        def __init__(self) -> None:
            self.calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((conversation_id, question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "test-model",
                    "content": (
                        "## Research question\nThe paper studies attention [1].\n## Limitations\n"
                    ),
                    "citations": [],
                    "evidence": [],
                    "provider_usage": None,
                },
            )()

    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.READ_PAPER,
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Read the selected paper",
    )
    registry = build_tool_registry()
    definition = registry[AssistantIntent.READ_PAPER]
    validated = validate_tool_input(definition, request, decision)
    service = FakeChatService()
    result = await execute_tool(
        definition,
        ToolContext(
            db=None,
            chat_service=service,  # type: ignore[arg-type]
            assistant_run_id=uuid4(),
            worker_id="worker-read",
            attempt_count=2,
        ),
        validated,
    )

    assert result.result_type == "reading_brief"
    sections = {section["key"]: section for section in result.structured_payload["sections"]}
    assert sections["research_question"]["citation_indexes"] == [1]
    assert sections["limitations"]["content"] == "Not found in retrieved evidence."
    assert len(service.calls) == 1
    assert service.calls[0][2]["paper_scope"] == "selection"
    assert service.calls[0][2]["selected_paper_ids"] == request.selected_paper_ids
    assert service.calls[0][2]["response_guidance"]
    assert service.calls[0][2]["run_worker_id"] == "worker-read"


def test_read_paper_tool_requires_one_paper():
    registry = build_tool_registry()
    project_request = _request(scope="project", selected_paper_ids=[])
    decision = RouteDecision(
        intent=AssistantIntent.READ_PAPER,
        action_summary="Read the selected paper",
    )

    with pytest.raises(ValueError, match="select more papers"):
        validate_tool_input(registry[AssistantIntent.READ_PAPER], project_request, decision)


@pytest.mark.anyio
async def test_qa_tool_verifies_selected_passage_before_using_existing_chat(monkeypatch) -> None:
    paper_id = uuid4()
    project_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": project_id,
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Attention connects all positions.",
                "document_sha256": "a" * 64,
            },
            "idempotency_key": "passage-request-123",
        }
    )
    decision = RouteDecision(
        intent=AssistantIntent.QA,
        resolved_paper_ids=[uuid4()],
        action_summary="Explain the passage",
    )

    class FakeChatService:
        calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "test-model",
                    "content": "The passage says this [1].",
                    "citations": [],
                    "evidence": [],
                    "provider_usage": None,
                },
            )()

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor",
        lambda *args, **kwargs: type(
            "Anchor", (), {"page_number": 3, "source_char_start": 5, "source_char_end": 36}
        )(),
    )
    selected_evidence = EvidenceItem(
        id="selected-passage",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote="Attention connects all positions.",
        page_number=3,
    )
    monkeypatch.setattr(
        "app.services.assistant_tools.build_selected_passage_evidence",
        lambda *args, **kwargs: selected_evidence,
    )
    service = FakeChatService()
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(
            db=None,
            chat_service=service,  # type: ignore[arg-type]
            assistant_run_id=uuid4(),
        ),
        validate_tool_input(definition, request, decision),
    )

    assert result.result_type == "answer"
    assert len(service.calls) == 1
    assert service.calls[0][1]["selected_paper_ids"] == [paper_id]
    assert service.calls[0][1]["paper_scope"] == "selection"
    assert "Attention connects all positions." in service.calls[0][1]["retrieval_question"]
    assert (
        "define symbols only when the surrounding source supports"
        in service.calls[0][1]["response_guidance"]
    )
    assert service.calls[0][1]["additional_evidence"] == [selected_evidence]


@pytest.mark.anyio
async def test_qa_tool_does_not_call_chat_when_selected_passage_is_unverified(monkeypatch) -> None:
    paper_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Repeated phrase",
                "document_sha256": "b" * 64,
            },
            "idempotency_key": "passage-request-456",
        }
    )
    decision = RouteDecision(intent=AssistantIntent.QA, action_summary="Explain the passage")

    class FakeChatService:
        async def answer_question(self, *args, **kwargs):
            raise AssertionError("unverified selections must not reach generation")

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor", lambda *args, **kwargs: None
    )
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=FakeChatService()),  # type: ignore[arg-type]
        validate_tool_input(definition, request, decision),
    )

    assert result.status.value == "NEEDS_INPUT"
    assert result.result_type == "source_selection_unavailable"


@pytest.mark.anyio
async def test_qa_tool_does_not_generate_when_verified_passage_is_not_index_linked(
    monkeypatch,
) -> None:
    paper_id = uuid4()
    project_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": project_id,
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Verified but not indexed.",
                "document_sha256": "c" * 64,
            },
            "idempotency_key": "passage-request-789",
        }
    )

    class FakeChatService:
        async def answer_question(self, *args, **kwargs):
            raise AssertionError("unlinked selection must not reach generation")

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor",
        lambda *args, **kwargs: type("Anchor", (), {})(),
    )
    monkeypatch.setattr(
        "app.services.assistant_tools.build_selected_passage_evidence",
        lambda *args, **kwargs: None,
    )
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=FakeChatService()),  # type: ignore[arg-type]
        validate_tool_input(
            definition,
            request,
            RouteDecision(
                intent=AssistantIntent.QA,
                resolved_paper_ids=[paper_id],
                action_summary="Explain the passage",
            ),
        ),
    )

    assert result.status is ToolStatus.NEEDS_INPUT
    assert result.result_type == "source_selection_unavailable"


def test_source_selection_requires_exact_single_paper_scope() -> None:
    paper_id = uuid4()
    payload = {
        "message": "Explain this passage",
        "conversation_id": uuid4(),
        "project_id": uuid4(),
        "scope": "selection",
        "selected_paper_ids": [paper_id],
        "source_selection": {
            "paper_id": paper_id,
            "page_number": 1,
            "quote": "Some source text.",
            "document_sha256": "c" * 64,
        },
        "idempotency_key": "passage-scope-123",
    }

    with pytest.raises(ValueError, match="single paper as the run scope"):
        AssistantRunRequest.model_validate(payload)


@pytest.mark.anyio
async def test_unimplemented_feature_returns_explicit_unavailable_result() -> None:
    registry = build_tool_registry()
    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.VISION,
        action_summary="Analyze the selected figure",
        arguments={"question": "What does the plot show?"},
    )
    definition = registry[AssistantIntent.VISION]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=None),  # type: ignore[arg-type]
        validate_tool_input(definition, request, decision),
    )

    assert result.status is ToolStatus.UNAVAILABLE
    assert "not connected" in result.display_text

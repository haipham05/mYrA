"""Fixed, typed tool registry for assistant intents.

Registry entries are application code, never names or callables supplied by the LLM. Features
without an implemented service return an explicit UNAVAILABLE result.
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.schemas.assistant import AssistantIntent, AssistantRunRequest, RouteDecision
from app.schemas.evidence import Citation
from app.services.chat_service import ChatService


class ToolStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    NEEDS_INPUT = "NEEDS_INPUT"
    UNAVAILABLE = "UNAVAILABLE"


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArguments(ToolArguments):
    pass


class ReadPaperArguments(ToolArguments):
    section: str | None = Field(default=None, max_length=120)


class CompareArguments(ToolArguments):
    dimensions: list[str] = Field(default_factory=list, max_length=8)


class VerifyClaimArguments(ToolArguments):
    claim: str = Field(min_length=1, max_length=2000)


class DiscoverArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=1000)
    year_from: int | None = Field(default=None, ge=1000, le=3000)
    year_to: int | None = Field(default=None, ge=1000, le=3000)


class NotesArguments(ToolArguments):
    action: Literal["list", "propose_save", "propose_update", "propose_archive"]
    content: str | None = Field(default=None, max_length=4000)
    title: str | None = Field(default=None, max_length=200)


class ReportArguments(ToolArguments):
    question: str | None = Field(default=None, max_length=2000)


class ResearchArguments(ToolArguments):
    goal: str = Field(min_length=1, max_length=2000)
    subquestions: list[str] = Field(default_factory=list, max_length=4)


class GapAnalysisArguments(ToolArguments):
    focus: str | None = Field(default=None, max_length=1000)


class ExperimentPlanArguments(ToolArguments):
    objective: str = Field(min_length=1, max_length=2000)


class TranslateArguments(ToolArguments):
    target_language: Literal["vi"]
    external_processing_acknowledged: bool = False


class VisionArguments(ToolArguments):
    question: str = Field(min_length=1, max_length=1000)
    figure_reference: str | None = Field(default=None, max_length=200)


class GraphArguments(ToolArguments):
    question: str = Field(min_length=1, max_length=1000)


class AssistantToolInput(BaseModel):
    request: AssistantRunRequest
    decision: RouteDecision
    arguments: ToolArguments


class AssistantToolResult(BaseModel):
    status: ToolStatus
    result_type: str = Field(min_length=1, max_length=80)
    display_text: str = Field(max_length=20000)
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list, max_length=200)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    warnings: list[str] = Field(default_factory=list, max_length=50)
    usage: dict[str, int | None] | None = None


@dataclass(frozen=True, slots=True)
class ToolContext:
    db: Session
    chat_service: ChatService
    assistant_run_id: UUID | None = None


ToolHandler = Callable[[ToolContext, AssistantToolInput], Awaitable[AssistantToolResult]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    intent: AssistantIntent
    arguments_model: type[ToolArguments]
    input_model: type[AssistantToolInput]
    output_model: type[AssistantToolResult]
    handler: ToolHandler
    requires_approval: bool = False
    min_papers: int = 0
    max_papers: int = 6
    available: bool = False


async def _unavailable(
    _context: ToolContext, tool_input: AssistantToolInput
) -> AssistantToolResult:
    name = tool_input.decision.intent.value.replace("_", " ")
    return AssistantToolResult(
        status=ToolStatus.UNAVAILABLE,
        result_type="unavailable",
        display_text=f"The {name} capability is not connected yet.",
    )


async def _help(_context: ToolContext, _tool_input: AssistantToolInput) -> AssistantToolResult:
    return AssistantToolResult(
        status=ToolStatus.SUCCEEDED,
        result_type="help",
        display_text=(
            "Ask a question about your selected papers to get a source-grounded answer. "
            "Other routed research actions will be enabled as their services are connected."
        ),
    )


async def _clarify(_context: ToolContext, tool_input: AssistantToolInput) -> AssistantToolResult:
    return AssistantToolResult(
        status=ToolStatus.NEEDS_INPUT,
        result_type="clarification",
        display_text=tool_input.decision.clarification or "Please clarify what you want me to do.",
        structured_payload={"missing_information": tool_input.decision.missing_information},
    )


async def _qa(context: ToolContext, tool_input: AssistantToolInput) -> AssistantToolResult:
    response = await context.chat_service.answer_question(
        context.db,
        tool_input.request.conversation_id,
        tool_input.request.message,
        assistant_run_id=context.assistant_run_id,
        retrieval_question=tool_input.decision.standalone_question,
    )
    return AssistantToolResult(
        status=ToolStatus.SUCCEEDED,
        result_type="answer",
        display_text=response.content,
        structured_payload={"message_id": str(response.id), "model_name": response.model_name},
        citations=response.citations,
        evidence=[item.model_dump(mode="json") for item in response.evidence],
        usage=response.provider_usage,
    )


def build_tool_registry() -> Mapping[AssistantIntent, ToolDefinition]:
    definitions: dict[AssistantIntent, ToolDefinition] = {}

    def register(
        intent: AssistantIntent,
        arguments_model: type[ToolArguments],
        handler: ToolHandler = _unavailable,
        *,
        approval: bool = False,
        min_papers: int = 0,
        available: bool = False,
    ) -> None:
        definitions[intent] = ToolDefinition(
            intent=intent,
            arguments_model=arguments_model,
            input_model=AssistantToolInput,
            output_model=AssistantToolResult,
            handler=handler,
            requires_approval=approval,
            min_papers=min_papers,
            available=available,
        )

    register(AssistantIntent.HELP, EmptyArguments, _help, available=True)
    register(AssistantIntent.QA, EmptyArguments, _qa, available=True)
    register(AssistantIntent.CLARIFY, EmptyArguments, _clarify, available=True)
    register(AssistantIntent.READ_PAPER, ReadPaperArguments)
    register(AssistantIntent.COMPARE, CompareArguments, min_papers=2)
    register(AssistantIntent.VERIFY_CLAIM, VerifyClaimArguments, min_papers=1)
    register(AssistantIntent.DISCOVER, DiscoverArguments)
    register(AssistantIntent.NOTES, NotesArguments, approval=True)
    register(AssistantIntent.REPORT, ReportArguments, min_papers=1)
    register(AssistantIntent.RESEARCH, ResearchArguments, min_papers=1)
    register(AssistantIntent.GAP_ANALYSIS, GapAnalysisArguments, min_papers=1)
    register(AssistantIntent.EXPERIMENT_PLAN, ExperimentPlanArguments, min_papers=1)
    register(AssistantIntent.TRANSLATE, TranslateArguments, approval=True, min_papers=1)
    register(AssistantIntent.VISION, VisionArguments, min_papers=1)
    register(AssistantIntent.GRAPH, GraphArguments, min_papers=1)
    # QA's handler receives the already-configured existing chat service in ToolContext.
    definitions[AssistantIntent.QA] = ToolDefinition(
        intent=AssistantIntent.QA,
        arguments_model=EmptyArguments,
        input_model=AssistantToolInput,
        output_model=AssistantToolResult,
        handler=_qa,
        available=True,
    )
    return MappingProxyType(definitions)


def resolve_tool_definition(
    registry: Mapping[AssistantIntent, ToolDefinition], intent: AssistantIntent
) -> ToolDefinition:
    """Lookup a fixed enum entry; string-based/arbitrary dispatch is deliberately unsupported."""
    return registry[intent]


def validate_tool_input(
    definition: ToolDefinition,
    request: AssistantRunRequest,
    decision: RouteDecision,
) -> AssistantToolInput:
    arguments = definition.arguments_model.model_validate(decision.arguments)
    paper_ids = decision.resolved_paper_ids or request.selected_paper_ids
    if len(paper_ids) > definition.max_papers:
        raise ValueError("too many papers for this action")
    if len(paper_ids) < definition.min_papers:
        raise ValueError("select more papers before running this action")
    return definition.input_model(request=request, decision=decision, arguments=arguments)


def tool_requires_approval(definition: ToolDefinition, tool_input: AssistantToolInput) -> bool:
    if definition.intent is AssistantIntent.NOTES:
        arguments = tool_input.arguments
        return isinstance(arguments, NotesArguments) and arguments.action != "list"
    return definition.requires_approval


async def execute_tool(
    definition: ToolDefinition,
    context: ToolContext,
    tool_input: AssistantToolInput,
) -> AssistantToolResult:
    """Run one statically registered handler and validate its result contract."""
    if not definition.available:
        return AssistantToolResult(
            status=ToolStatus.UNAVAILABLE,
            result_type="unavailable",
            display_text=(
                f"The {definition.intent.value.replace('_', ' ')} capability is not connected yet."
            ),
        )
    result = await definition.handler(context, tool_input)
    return definition.output_model.model_validate(result)

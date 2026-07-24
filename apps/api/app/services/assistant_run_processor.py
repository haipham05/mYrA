"""Process one durable assistant run inside a worker-owned database session."""

import hashlib
import json
import logging
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.crud.assistant_run import (
    RunLeaseLost,
    assert_assistant_run_lease,
    assistant_run_cancel_requested,
    create_assistant_approval,
    finish_assistant_run,
    get_valid_approved_assistant_action,
    save_assistant_step,
    start_assistant_step,
)
from app.db.models import AssistantRun, Message, Paper
from app.observability.context import OperationContext, use_operation_context
from app.observability.telemetry import TelemetryAdapter, get_telemetry
from app.schemas.assistant import (
    AssistantIntent,
    AssistantRunRequest,
    AssistantRunResult,
    RouteDecision,
    RouteOutcome,
    RoutePaperContext,
)
from app.services.assistant_router import AssistantRouter
from app.services.assistant_tools import (
    AssistantToolInput,
    ToolContext,
    ToolStatus,
    build_tool_registry,
    execute_tool,
    tool_requires_approval,
    validate_tool_input,
)
from app.services.chat_service import AssistantRunCancelled, ChatService

logger = logging.getLogger("myra.assistant_run")


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AssistantRunProcessor:
    def __init__(
        self,
        *,
        session_factory: sessionmaker,
        router: AssistantRouter | None = None,
        chat_service: ChatService | None = None,
        telemetry: TelemetryAdapter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._router = router or AssistantRouter(telemetry=telemetry)
        self._chat_service = chat_service or ChatService()
        self._telemetry = telemetry or get_telemetry()
        self._tools = build_tool_registry()

    async def process(self, run_id: UUID, *, worker_id: str, attempt_count: int) -> None:
        with self._session_factory() as db:
            run = assert_assistant_run_lease(
                db, run_id, worker_id=worker_id, attempt_count=attempt_count
            )
            request = AssistantRunRequest.model_validate(run.request_payload)
            request_hash = run.request_hash
            context = OperationContext.validated(correlation_id=str(run.id))

            with use_operation_context(context):
                if self._cancelled(db, run_id, worker_id, attempt_count):
                    self._finish_cancelled(db, run_id, worker_id, attempt_count)
                    return

                decision, route_result, route_error = await self._load_or_route(
                    db, run, request, request_hash, worker_id, attempt_count
                )
                if route_error:
                    self._finish_failure(
                        db,
                        run_id,
                        worker_id,
                        attempt_count,
                        code=route_error,
                        result_type="routing_unavailable",
                        message=(
                            "I couldn't safely route that request. Please retry or choose an "
                            "action explicitly."
                        ),
                    )
                    return
                assert decision is not None
                outcome = route_result.get("outcome", RouteOutcome.ROUTED.value)
                if outcome == RouteOutcome.UNAVAILABLE.value:
                    self._finish_failure(
                        db,
                        run_id,
                        worker_id,
                        attempt_count,
                        code="ROUTE_UNAVAILABLE",
                        result_type="routing_unavailable",
                        message=(
                            "I couldn't safely route that request. Please retry or choose an "
                            "action explicitly."
                        ),
                        decision=decision,
                        route_result=route_result,
                    )
                    return
                if outcome == RouteOutcome.NEEDS_CLARIFICATION.value:
                    self._finish_needs_input(
                        db, run_id, worker_id, attempt_count, decision, route_result
                    )
                    return

                if self._cancelled(db, run_id, worker_id, attempt_count):
                    self._finish_cancelled(db, run_id, worker_id, attempt_count)
                    return

                definition = self._tools[decision.intent]
                try:
                    tool_input = validate_tool_input(definition, request, decision)
                except (ValueError, TypeError):
                    self._finish_needs_input(
                        db,
                        run_id,
                        worker_id,
                        attempt_count,
                        RouteDecision(
                            intent=AssistantIntent.CLARIFY,
                            missing_information=["required_papers_or_arguments"],
                            clarification=(
                                "Please select the required papers or clarify the information "
                                "needed for this action."
                            ),
                            action_summary="Clarify the research scope",
                        ),
                        route_result,
                    )
                    return

                if definition.available and tool_requires_approval(definition, tool_input):
                    approved_arguments = {
                        "message": request.message,
                        "scope": request.scope.value,
                        "paper_ids": [
                            str(item)
                            for item in (decision.resolved_paper_ids or request.selected_paper_ids)
                        ],
                        "action_summary": decision.action_summary,
                        "tool_arguments": tool_input.arguments.model_dump(mode="json"),
                    }
                    try:
                        approved_action = get_valid_approved_assistant_action(
                            db,
                            run_id,
                            action_type=decision.intent.value,
                            expected_arguments=approved_arguments,
                            paper_ids=decision.resolved_paper_ids or request.selected_paper_ids,
                        )
                    except ValueError:
                        self._finish_failure(
                            db,
                            run_id,
                            worker_id,
                            attempt_count,
                            code="APPROVAL_NO_LONGER_VALID",
                            result_type="approval_invalidated",
                            message=(
                                "The approved action no longer matches the current request or "
                                "source. Please review and submit it again."
                            ),
                            decision=decision,
                            route_result=route_result,
                        )
                        return
                    if approved_action is None:
                        self._finish_approval_wait(
                            db,
                            run_id,
                            worker_id,
                            attempt_count,
                            request,
                            decision,
                            route_result,
                            tool_input,
                        )
                        return

                tool_fingerprint = _fingerprint(
                    {"request_hash": request_hash, "decision": decision.model_dump(mode="json")}
                )
                step, should_execute = start_assistant_step(
                    db,
                    run_id,
                    worker_id=worker_id,
                    attempt_count=attempt_count,
                    step_key=f"assistant.tool.{decision.intent.value}",
                    ordinal=2,
                    input_fingerprint=tool_fingerprint,
                    tool_name=decision.intent.value,
                )
                if not should_execute:
                    if step.status == "COMPLETED" and step.output_payload is not None:
                        tool_result_payload = step.output_payload
                    elif decision.intent is AssistantIntent.QA:
                        completed_message = (
                            db.query(Message).filter(Message.assistant_run_id == run_id).first()
                        )
                        if completed_message is None:
                            self._finish_failure(
                                db,
                                run_id,
                                worker_id,
                                attempt_count,
                                code="TOOL_OUTCOME_UNKNOWN",
                                result_type="action_outcome_unknown",
                                message=(
                                    "The action stopped before its result could be confirmed. "
                                    "It was not automatically repeated."
                                ),
                                decision=decision,
                                route_result=route_result,
                            )
                            return
                        tool_result_payload = self._tool_result_from_message(completed_message)
                        save_assistant_step(
                            db,
                            run_id,
                            worker_id=worker_id,
                            attempt_count=attempt_count,
                            step_key=f"assistant.tool.{decision.intent.value}",
                            ordinal=2,
                            input_fingerprint=tool_fingerprint,
                            output_payload=tool_result_payload,
                            tool_name=decision.intent.value,
                            external_effect_id=str(completed_message.id),
                        )
                    else:
                        self._finish_failure(
                            db,
                            run_id,
                            worker_id,
                            attempt_count,
                            code="TOOL_OUTCOME_UNKNOWN",
                            result_type="action_outcome_unknown",
                            message=(
                                "The action stopped before its result could be confirmed. "
                                "It was not automatically repeated."
                            ),
                            decision=decision,
                            route_result=route_result,
                        )
                        return
                else:
                    if self._cancelled(db, run_id, worker_id, attempt_count):
                        self._finish_cancelled(db, run_id, worker_id, attempt_count)
                        return
                    with self._telemetry.stage(
                        "assistant.tool",
                        input={"intent": decision.intent.value},
                        metadata={"run_id": str(run_id), "outcome": "started"},
                    ) as observation:
                        try:
                            tool_result = await execute_tool(
                                definition,
                                ToolContext(
                                    db=db,
                                    chat_service=self._chat_service,
                                    assistant_run_id=run_id,
                                    worker_id=worker_id,
                                    attempt_count=attempt_count,
                                ),
                                tool_input,
                            )
                        except RunLeaseLost:
                            raise
                        except AssistantRunCancelled:
                            self._finish_cancelled(db, run_id, worker_id, attempt_count)
                            return
                        except Exception:
                            tool_result = None
                        if self._cancelled(db, run_id, worker_id, attempt_count):
                            self._finish_cancelled(db, run_id, worker_id, attempt_count)
                            return
                        if tool_result is None:
                            tool_result_payload = AssistantRunResult(
                                result_type="tool_error",
                                display_text="The requested action could not be completed safely.",
                                warnings=[
                                    "The action result was not available; retry was withheld."
                                ],
                            ).model_dump(mode="json")
                            tool_status = ToolStatus.UNAVAILABLE
                            safe_tool_error = "TOOL_EXECUTION_FAILED"
                        else:
                            tool_result_payload = tool_result.model_dump(mode="json")
                            tool_status = tool_result.status
                            safe_tool_error = (
                                "FEATURE_UNAVAILABLE"
                                if tool_status is ToolStatus.UNAVAILABLE
                                else None
                            )
                        save_assistant_step(
                            db,
                            run_id,
                            worker_id=worker_id,
                            attempt_count=attempt_count,
                            step_key=f"assistant.tool.{decision.intent.value}",
                            ordinal=2,
                            input_fingerprint=tool_fingerprint,
                            output_payload=tool_result_payload,
                            tool_name=decision.intent.value,
                            external_effect_id=str(run_id)
                            if decision.intent is AssistantIntent.QA
                            else None,
                        )
                        if observation is not None:
                            observation.update(
                                output={"status": tool_status.value},
                                metadata={"outcome": tool_status.value.lower()},
                            )

                tool_result = self._parse_tool_result(tool_result_payload)
                final_status = (
                    "NEEDS_INPUT"
                    if tool_result.status is ToolStatus.NEEDS_INPUT
                    else "FAILED"
                    if tool_result.status is ToolStatus.UNAVAILABLE
                    else "SUCCEEDED"
                )
                self._finish(
                    db,
                    run_id,
                    worker_id,
                    attempt_count,
                    status=final_status,
                    decision=decision,
                    result_payload=tool_result.model_dump(mode="json"),
                    route_result={**route_result, "tool_usage": tool_result.usage},
                    safe_error=safe_tool_error if should_execute else None,
                )

    async def _load_or_route(
        self,
        db: Session,
        run: AssistantRun,
        request: AssistantRunRequest,
        request_hash: str,
        worker_id: str,
        attempt_count: int,
    ) -> tuple[RouteDecision | None, dict[str, object], str | None]:
        route_step_key = (
            "assistant.route"
            if run.resume_count == 0
            else f"assistant.route.resume.{run.resume_count}"
        )
        step, should_execute = start_assistant_step(
            db,
            run.id,
            worker_id=worker_id,
            attempt_count=attempt_count,
            step_key=route_step_key,
            ordinal=1,
            input_fingerprint=request_hash,
            tool_name="route",
        )
        if not should_execute:
            if step.status == "COMPLETED" and step.output_payload:
                route_output = step.output_payload
                decision = RouteDecision.model_validate(route_output.get("decision", {}))
                run.intent = decision.intent.value
                run.route_decision = decision.model_dump(mode="json")
                run.action_summary = decision.action_summary
                db.commit()
                return decision, route_output, None
            return None, {}, "ROUTE_OUTCOME_UNKNOWN"

        history = [
            {"role": message.role, "content": message.content}
            for message in reversed(
                db.query(Message)
                .filter(Message.conversation_id == run.conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(6)
                .all()
            )
            if message.role in {"USER", "ASSISTANT"}
        ]
        paper_query = db.query(Paper).filter(
            Paper.project_id == run.project_id, Paper.status == "READY"
        )
        if request.selected_paper_ids:
            paper_query = paper_query.filter(Paper.id.in_(request.selected_paper_ids))
        papers = (
            paper_query.order_by(Paper.updated_at.desc(), Paper.created_at.desc()).limit(6).all()
        )
        paper_context = [
            RoutePaperContext(
                id=paper.id,
                title=paper.title or paper.filename,
                authors=paper.authors or [],
                publication_year=paper.publication_year,
            )
            for paper in papers
        ]
        routed = await self._router.route(
            request,
            recent_history=history,
            available_papers=paper_context,
        )
        decision = routed.decision
        route_output: dict[str, object] = {
            "decision": decision.model_dump(mode="json"),
            "outcome": routed.outcome.value,
            "requested_model": routed.requested_model,
            "reported_model": routed.reported_model,
            "usage": routed.usage,
        }
        save_assistant_step(
            db,
            run.id,
            worker_id=worker_id,
            attempt_count=attempt_count,
            step_key=route_step_key,
            ordinal=1,
            input_fingerprint=request_hash,
            output_payload=route_output,
            tool_name="route",
        )
        run.intent = decision.intent.value
        run.route_decision = decision.model_dump(mode="json")
        run.action_summary = decision.action_summary
        db.commit()
        return decision, route_output, None

    def _cancelled(self, db: Session, run_id: UUID, worker_id: str, attempt: int) -> bool:
        return assistant_run_cancel_requested(
            db, run_id, worker_id=worker_id, attempt_count=attempt
        )

    def _finish_cancelled(self, db: Session, run_id: UUID, worker_id: str, attempt: int) -> None:
        self._finish(
            db,
            run_id,
            worker_id,
            attempt,
            status="CANCELLED",
            result_payload={
                "result_type": "cancelled",
                "display_text": "This research run was cancelled.",
            },
        )

    def _finish_needs_input(
        self,
        db: Session,
        run_id: UUID,
        worker_id: str,
        attempt: int,
        decision: RouteDecision,
        route_result: dict[str, object],
    ) -> None:
        result = AssistantRunResult(
            result_type="clarification",
            display_text=decision.clarification or "Please clarify the research request.",
            structured_payload={"missing_information": decision.missing_information},
        )
        self._finish(
            db,
            run_id,
            worker_id,
            attempt,
            status="NEEDS_INPUT",
            decision=decision,
            result_payload=result.model_dump(mode="json"),
            route_result=route_result,
        )

    def _finish_approval_wait(
        self,
        db: Session,
        run_id: UUID,
        worker_id: str,
        attempt: int,
        request: AssistantRunRequest,
        decision: RouteDecision,
        route_result: dict[str, object],
        tool_input: AssistantToolInput,
    ) -> None:
        action = create_assistant_approval(
            db,
            run_id,
            worker_id=worker_id,
            attempt_count=attempt,
            action_type=decision.intent.value,
            arguments={
                "message": request.message,
                "scope": request.scope.value,
                "paper_ids": [str(item) for item in decision.resolved_paper_ids],
                "action_summary": decision.action_summary,
                "tool_arguments": tool_input.arguments.model_dump(mode="json"),
            },
            paper_ids=decision.resolved_paper_ids or request.selected_paper_ids,
        )
        result = AssistantRunResult(
            result_type="approval_required",
            display_text="This action needs your approval before it can run.",
            available_actions=["approve", "reject"],
            structured_payload={"approval_action_id": str(action.id)},
        )
        self._finish(
            db,
            run_id,
            worker_id,
            attempt,
            status="AWAITING_APPROVAL",
            decision=decision,
            result_payload=result.model_dump(mode="json"),
            route_result=route_result,
        )

    def _finish_failure(
        self,
        db: Session,
        run_id: UUID,
        worker_id: str,
        attempt: int,
        *,
        code: str,
        result_type: str,
        message: str,
        decision: RouteDecision | None = None,
        route_result: dict[str, object] | None = None,
    ) -> None:
        result = AssistantRunResult(result_type=result_type, display_text=message)
        self._finish(
            db,
            run_id,
            worker_id,
            attempt,
            status="FAILED",
            decision=decision,
            result_payload=result.model_dump(mode="json"),
            route_result=route_result,
            safe_error=code,
        )

    def _finish(
        self,
        db: Session,
        run_id: UUID,
        worker_id: str,
        attempt: int,
        *,
        status: str,
        decision: RouteDecision | None = None,
        result_payload: dict[str, object] | None = None,
        route_result: dict[str, object] | None = None,
        safe_error: str | None = None,
    ) -> None:
        usage = None
        if route_result:
            routing_usage = route_result.get("usage")
            tool_usage = route_result.get("tool_usage")
            if routing_usage is not None or tool_usage is not None:
                usage = {"routing": routing_usage, "tool": tool_usage}
        finish_assistant_run(
            db,
            run_id,
            worker_id=worker_id,
            attempt_count=attempt,
            status=status,
            result_payload=result_payload,
            intent=decision.intent.value if decision else None,
            route_decision=decision.model_dump(mode="json") if decision else None,
            action_summary=decision.action_summary if decision else None,
            safe_error=safe_error,
            usage=usage if isinstance(usage, dict) else None,
        )

    @staticmethod
    def _parse_tool_result(value: dict[str, object]):
        from app.services.assistant_tools import AssistantToolResult

        return AssistantToolResult.model_validate(value)

    @staticmethod
    def _tool_result_from_message(message: Message) -> dict[str, object]:
        from app.services.assistant_tools import AssistantToolResult

        return AssistantToolResult(
            status=ToolStatus.SUCCEEDED,
            result_type="answer",
            display_text=message.content,
            structured_payload={"message_id": str(message.id), "model_name": message.model_name},
            citations=message.citations or [],
            evidence=message.evidence or [],
            usage=message.provider_usage,
        ).model_dump(mode="json")

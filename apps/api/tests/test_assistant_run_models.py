from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import (
    AssistantApprovalAction,
    AssistantRun,
    AssistantRunStep,
    Conversation,
    Project,
)


def test_assistant_run_steps_and_approvals_persist_without_account_models() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)

    with Session(engine) as session:
        project = Project(name="Research")
        session.add(project)
        session.flush()
        conversation = Conversation(project_id=project.id)
        session.add(conversation)
        session.flush()
        run = AssistantRun(
            project_id=project.id,
            conversation_id=conversation.id,
            idempotency_key="run-key-123",
            request_hash="a" * 64,
            request_payload={"message": "compare these papers"},
            status="QUEUED",
        )
        run.steps.append(
            AssistantRunStep(
                step_key="route",
                ordinal=1,
                status="COMPLETED",
                output_payload={"intent": "compare"},
            )
        )
        run.approvals.append(
            AssistantApprovalAction(
                action_type="note.update",
                arguments={"note_id": str(uuid4())},
                source_fingerprint="b" * 64,
                idempotency_key="approve-key-123",
                expires_at=now + timedelta(minutes=10),
            )
        )
        session.add(run)
        session.commit()

        loaded = session.query(AssistantRun).one()
        assert loaded.steps[0].output_payload == {"intent": "compare"}
        assert loaded.approvals[0].status == "PENDING"
        assert not any(
            table.name in {"users", "accounts", "memberships"}
            for table in Base.metadata.tables.values()
        )

    engine.dispose()

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import GraphEvent, Paper
from app.observability.context import OperationContext


class LostGraphLeaseError(RuntimeError):
    """This worker no longer owns the graph event lease and must not publish its work."""


def create_or_enqueue_graph_event(
    db: Session,
    project_id: UUID | str,
    paper_id: UUID | str,
    action: str = "UPSERT",
    generation_id: str | None = None,
    ontology_version: str = "1.0.0",
    extractor_version: str = "1.0.0",
    trace_context: OperationContext | None = None,
) -> GraphEvent:
    """Enqueue a GraphEvent in the current transaction without committing.

    If generation_id is not provided, generates a stable generation ID:
    f"gen_{paper_id.hex[:8]}_{uuid4().hex[:12]}".
    """
    if isinstance(project_id, str):
        project_id = UUID(project_id)
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)

    if generation_id is None:
        generation_id = f"gen_{paper_id.hex[:8]}_{uuid4().hex[:12]}"

    # Serialize event creation with the publication boundary. A publisher holds
    # this same row lock while checking for newer events and writing its projection.
    db.query(Paper).filter(Paper.id == paper_id).with_for_update().first()

    event = GraphEvent(
        project_id=project_id,
        paper_id=paper_id,
        action=action,
        generation_id=generation_id,
        ontology_version=ontology_version,
        extractor_version=extractor_version,
        status="PENDING",
        attempts=0,
        max_attempts=3,
        correlation_id=trace_context.correlation_id if trace_context else None,
        trace_id=trace_context.trace_id if trace_context else None,
        parent_span_id=trace_context.span_id if trace_context else None,
        trace_sampled=trace_context.sampled if trace_context else False,
    )
    db.add(event)
    return event


def get_graph_event(db: Session, event_id: UUID | str) -> GraphEvent | None:
    """Retrieve a GraphEvent by ID."""
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    return db.get(GraphEvent, event_id)


def get_active_graph_events_for_paper(db: Session, paper_id: UUID | str) -> list[GraphEvent]:
    """Retrieve all active (PENDING or PROCESSING) GraphEvents for a paper,
    ordered by creation time.
    """
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    stmt = (
        select(GraphEvent)
        .where(
            GraphEvent.paper_id == paper_id,
            GraphEvent.status.in_(["PENDING", "PROCESSING"]),
        )
        .order_by(GraphEvent.created_at.asc())
    )
    return list(db.scalars(stmt).all())


def get_latest_completed_graph_event(db: Session, paper_id: UUID | str) -> GraphEvent | None:
    """Retrieve the completed event with the newest source/event order."""
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    stmt = (
        select(GraphEvent)
        .where(
            GraphEvent.paper_id == paper_id,
            GraphEvent.status == "COMPLETED",
        )
        .order_by(GraphEvent.created_at.desc(), GraphEvent.id.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def lock_graph_publication(db: Session, event_id: UUID | str, worker_id: str) -> bool:
    """Hold a per-paper publication lock and fence expired, lost, or superseded work.

    The Paper row lock is shared with create_or_enqueue_graph_event and is retained
    until the caller's event completion/failure commits. PostgreSQL therefore
    linearizes event creation against the final generation check and graph write;
    the SQL/Neo4j operations are still not one distributed transaction.
    """
    event_id = UUID(event_id) if isinstance(event_id, str) else event_id
    event = (
        db.query(GraphEvent)
        .populate_existing()
        .filter(GraphEvent.id == event_id)
        .with_for_update()
        .first()
    )
    if not event or event.status != "PROCESSING" or event.lease_owner != worker_id:
        db.rollback()
        return False

    db.query(Paper).filter(Paper.id == event.paper_id).with_for_update().first()
    now = datetime.now(tz=UTC)
    expiry = event.lease_expires_at
    if expiry is not None:
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        if expiry < now:
            db.rollback()
            return False

    newer_event = (
        db.query(GraphEvent.id)
        .filter(
            GraphEvent.paper_id == event.paper_id,
            GraphEvent.id != event.id,
            (GraphEvent.created_at > event.created_at)
            | ((GraphEvent.created_at == event.created_at) & (GraphEvent.id > event.id)),
        )
        .first()
    )
    if newer_event:
        fail_graph_event(
            db,
            event.id,
            worker_id,
            error_code="SUPERSEDED",
            error_message="A newer graph event superseded this generation before publication.",
            is_transient=False,
        )
        return False
    return True


def claim_next_graph_event(
    db: Session,
    worker_id: str,
    lease_timeout_seconds: int = 300,
) -> GraphEvent | None:
    """Atomically claim the next pending graph event or recover an expired worker lease."""
    bind = db.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    now = datetime.now(tz=UTC)

    # 1. Prioritize PENDING events (honoring backoff for retried events)
    query = (
        db.query(GraphEvent)
        .filter(GraphEvent.status == "PENDING")
        .order_by(GraphEvent.created_at.asc())
    )
    if not is_sqlite:
        query = query.with_for_update(skip_locked=True)

    pending_events = query.all()
    for candidate in pending_events:
        if candidate.attempts >= candidate.max_attempts:
            continue
        eligible = candidate.attempts == 0
        if not eligible and candidate.updated_at:
            backoff_seconds = min(300, (2**candidate.attempts) * 5)
            cand_time = candidate.updated_at
            if cand_time.tzinfo is None:
                cand_time = cand_time.replace(tzinfo=UTC)
            elapsed = (now - cand_time).total_seconds()
            eligible = elapsed >= backoff_seconds
        elif not candidate.updated_at:
            eligible = True

        if not eligible:
            continue

        lease_expires = now + timedelta(seconds=lease_timeout_seconds)
        changed = (
            db.query(GraphEvent)
            .filter(GraphEvent.id == candidate.id, GraphEvent.status == "PENDING")
            .update(
                {
                    GraphEvent.status: "PROCESSING",
                    GraphEvent.lease_owner: worker_id,
                    GraphEvent.lease_expires_at: lease_expires,
                    GraphEvent.updated_at: now,
                },
                synchronize_session=False,
            )
        )
        if changed == 1:
            db.commit()
            db.refresh(candidate)
            return candidate

    # 2. Recover expired PROCESSING leases where lease_expires_at < now
    expired_query = (
        db.query(GraphEvent)
        .filter(
            GraphEvent.status == "PROCESSING",
            GraphEvent.lease_expires_at < now,
            GraphEvent.attempts < GraphEvent.max_attempts,
        )
        .order_by(GraphEvent.lease_expires_at.asc())
    )
    if not is_sqlite:
        expired_query = expired_query.with_for_update(skip_locked=True)

    for candidate in expired_query.all():
        lease_expires = now + timedelta(seconds=lease_timeout_seconds)
        changed = (
            db.query(GraphEvent)
            .filter(
                GraphEvent.id == candidate.id,
                GraphEvent.status == "PROCESSING",
                GraphEvent.lease_expires_at < now,
                GraphEvent.attempts < GraphEvent.max_attempts,
            )
            .update(
                {
                    GraphEvent.status: "PROCESSING",
                    GraphEvent.lease_owner: worker_id,
                    GraphEvent.lease_expires_at: lease_expires,
                    GraphEvent.attempts: GraphEvent.attempts + 1,
                    GraphEvent.updated_at: now,
                },
                synchronize_session=False,
            )
        )
        if changed == 1:
            db.commit()
            db.refresh(candidate)
            return candidate

    db.rollback()
    return None


def renew_graph_event_lease(
    db: Session,
    event_id: UUID | str,
    worker_id: str,
    extend_seconds: int = 300,
) -> bool:
    """Updates lease_expires_at = now + extend_seconds if id == event_id,
    lease_owner == worker_id, and status == 'PROCESSING'.
    Returns True if renewed, False if lost.
    """
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    now = datetime.now(tz=UTC)
    new_expiry = now + timedelta(seconds=extend_seconds)
    changed = (
        db.query(GraphEvent)
        .filter(
            GraphEvent.id == event_id,
            GraphEvent.status == "PROCESSING",
            GraphEvent.lease_owner == worker_id,
        )
        .update(
            {
                GraphEvent.lease_expires_at: new_expiry,
                GraphEvent.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    if changed == 1:
        db.commit()
        return True
    db.rollback()
    return False


def release_graph_event(
    db: Session,
    event_id: UUID | str,
    worker_id: str,
) -> None:
    """If held by worker, resets status = 'PENDING', lease_owner = None, lease_expires_at = None."""
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    now = datetime.now(tz=UTC)
    changed = (
        db.query(GraphEvent)
        .filter(
            GraphEvent.id == event_id,
            GraphEvent.status == "PROCESSING",
            GraphEvent.lease_owner == worker_id,
        )
        .update(
            {
                GraphEvent.status: "PENDING",
                GraphEvent.lease_owner: None,
                GraphEvent.lease_expires_at: None,
                GraphEvent.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    if changed == 1:
        db.commit()
    else:
        db.rollback()


def complete_graph_event(
    db: Session,
    event_id: UUID | str,
    worker_id: str,
) -> None:
    """Sets status = 'COMPLETED', lease_owner = None,
    lease_expires_at = None, completed_at = now.
    """
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    now = datetime.now(tz=UTC)
    changed = (
        db.query(GraphEvent)
        .filter(
            GraphEvent.id == event_id,
            GraphEvent.status == "PROCESSING",
            GraphEvent.lease_owner == worker_id,
        )
        .update(
            {
                GraphEvent.status: "COMPLETED",
                GraphEvent.lease_owner: None,
                GraphEvent.lease_expires_at: None,
                GraphEvent.completed_at: now,
                GraphEvent.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    if changed == 1:
        db.commit()
    else:
        db.rollback()
        raise LostGraphLeaseError(f"GraphEvent {event_id} is no longer owned by worker {worker_id}")


def fail_graph_event(
    db: Session,
    event_id: UUID | str,
    worker_id: str,
    error_code: str,
    error_message: str,
    is_transient: bool = True,
) -> None:
    """Increments attempts.
    If attempts >= max_attempts or not is_transient:
        Sets status = "FAILED", error_code = error_code, error_message = error_message[:500],
        lease_owner = None, lease_expires_at = None.
    Else:
        Sets status = "PENDING", error_code = error_code, error_message = error_message[:500],
        lease_owner = None, lease_expires_at = None.

    INVARIANT: NEVER mutate or fail the source Paper record.
    """
    if isinstance(event_id, str):
        event_id = UUID(event_id)
    event = (
        db.query(GraphEvent)
        .filter(
            GraphEvent.id == event_id,
            GraphEvent.status == "PROCESSING",
            GraphEvent.lease_owner == worker_id,
        )
        .first()
    )
    if not event:
        db.rollback()
        return

    now = datetime.now(tz=UTC)
    event.attempts += 1
    event.error_code = error_code
    event.error_message = error_message[:500] if error_message else None
    event.lease_owner = None
    event.lease_expires_at = None
    event.updated_at = now

    if event.attempts >= event.max_attempts or not is_transient:
        event.status = "FAILED"
    else:
        event.status = "PENDING"

    db.commit()

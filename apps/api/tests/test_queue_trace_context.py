from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.graph import create_or_enqueue_graph_event
from app.crud.job import create_job
from app.crud.paper import create_paper
from app.db.base import Base
from app.db.models import GraphEvent, Job, Project
from app.observability.context import OperationContext


def test_trace_context_round_trips_through_job_and_graph_event() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    context = OperationContext.validated(
        correlation_id="upload-42",
        trace_id="a" * 32,
        span_id="b" * 16,
        sampled=True,
    )
    try:
        with Session(engine) as db:
            project = Project(name="queue context")
            db.add(project)
            db.commit()
            paper = create_paper(db, project.id, "paper.pdf", "papers/paper.pdf")
            job = create_job(db, paper.id, trace_context=context)
            event = create_or_enqueue_graph_event(db, project.id, paper.id, trace_context=context)
            db.commit()

            stored_job = db.get(Job, job.id)
            stored_event = db.get(GraphEvent, event.id)
            assert stored_job is not None
            assert stored_job.correlation_id == "upload-42"
            assert stored_job.trace_id == "a" * 32
            assert stored_job.parent_span_id == "b" * 16
            assert stored_job.trace_sampled is True
            assert stored_event is not None
            assert stored_event.correlation_id == stored_job.correlation_id
            assert stored_event.trace_id == stored_job.trace_id
            assert stored_event.parent_span_id == stored_job.parent_span_id
            assert stored_event.trace_sampled is True

            legacy_job = Job(paper_id=paper.id, status="PENDING", stage="QUEUED", progress=0)
            db.add(legacy_job)
            db.commit()
            db.refresh(legacy_job)
            assert legacy_job.correlation_id is None
            assert legacy_job.trace_id is None
            assert legacy_job.parent_span_id is None
            assert legacy_job.trace_sampled is False
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()

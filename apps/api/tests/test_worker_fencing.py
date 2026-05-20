import asyncio
import hashlib

import pytest

from app.crud.job import (
    LostJobLeaseError,
    claim_next_job,
    create_job,
    fence_job_for_publish,
    update_job_progress,
)
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import Job, PaperChunk, PaperElement, PaperPage
from app.db.session import SessionLocal, create_tables
from app.ingestion.parser import ParsedElement, ParsedPage, ParseResult
from app.schemas.project import ProjectCreate
from app.services.ingestion import IngestionPipeline
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage
from app.worker import run_worker


class FixedParser:
    def parse(self, _data: bytes) -> ParseResult:
        return ParseResult(
            pages=[
                ParsedPage(page_number=1, width=612, height=792, raw_text="Stable source text.")
            ],
            elements=[
                ParsedElement(
                    element_index=0,
                    page_number=1,
                    element_type="paragraph",
                    text="Stable source text.",
                )
            ],
        )


@pytest.mark.anyio
async def test_stale_worker_cannot_publish_or_fail_reclaimed_job():
    create_tables()
    db = SessionLocal()
    db.query(Job).delete()
    db.commit()
    project = create_project(db, ProjectCreate(name="Stale worker fence"))
    paper = create_paper(db, project.id, "fenced.pdf", "fenced.pdf")
    job = create_job(db, paper.id)
    storage = MemoryStorage()
    await storage.put("fenced.pdf", b"%PDF-fenced")
    set_storage(storage)

    class StealingParser(FixedParser):
        def parse(self, data: bytes) -> ParseResult:
            with SessionLocal() as other:
                other.query(Job).filter(Job.id == job.id).update({Job.worker_id: "worker-2"})
                other.commit()
            return super().parse(data)

    try:
        assert claim_next_job(db, worker_id="worker-1").id == job.id
        await IngestionPipeline(parser=StealingParser()).process_paper(
            db, paper.id, job.id, worker_id="worker-1"
        )
        db.expire_all()
        assert db.get(Job, job.id).worker_id == "worker-2"
        assert db.get(Job, job.id).status == "PROCESSING"
        assert db.get(type(paper), paper.id).status != "READY"
        assert db.query(PaperPage).filter(PaperPage.paper_id == paper.id).count() == 0
        assert db.query(PaperElement).filter(PaperElement.paper_id == paper.id).count() == 0
        assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() == 0
        with pytest.raises(LostJobLeaseError):
            update_job_progress(db, job.id, "INDEXING", 0.9, worker_id="worker-1")
        with pytest.raises(LostJobLeaseError):
            fence_job_for_publish(db, job.id, "worker-1")
    finally:
        db.close()
        set_storage(None)


@pytest.mark.anyio
async def test_reindex_replaces_rows_in_one_transaction():
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Idempotent reindex"))
    data = b"%PDF-stable"
    paper = create_paper(
        db,
        project.id,
        "stable.pdf",
        "stable.pdf",
        document_sha256=hashlib.sha256(data).hexdigest(),
    )
    storage = MemoryStorage()
    await storage.put("stable.pdf", data)
    set_storage(storage)
    try:
        for _ in range(2):
            job = create_job(db, paper.id)
            await IngestionPipeline(parser=FixedParser()).process_paper(db, paper.id, job.id)
            db.expire_all()
            assert db.get(Job, job.id).status == "COMPLETED"
            assert db.query(PaperPage).filter(PaperPage.paper_id == paper.id).count() == 1
            assert db.query(PaperElement).filter(PaperElement.paper_id == paper.id).count() == 1
            assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() > 0
    finally:
        db.close()
        set_storage(None)


@pytest.mark.anyio
async def test_worker_stops_when_heartbeat_loses_lease(monkeypatch):
    create_tables()
    db = SessionLocal()
    db.query(Job).delete()
    db.commit()
    project = create_project(db, ProjectCreate(name="Heartbeat lost lease"))
    paper = create_paper(db, project.id, "slow.pdf", "slow.pdf")
    job = create_job(db, paper.id)

    class SlowStorage(MemoryStorage):
        async def get(self, key: str) -> bytes:
            await asyncio.sleep(1)
            return await super().get(key)

    storage = SlowStorage()
    await storage.put("slow.pdf", b"%PDF-slow")
    set_storage(storage)
    monkeypatch.setattr("app.worker.renew_job_lease", lambda *_args, **_kwargs: False)
    try:
        await run_worker(poll_interval=0.01, once=True, heartbeat_interval=0.01)
        db.expire_all()
        assert db.get(Job, job.id).status == "PROCESSING"
        assert db.get(type(paper), paper.id).status != "READY"
        assert db.query(PaperPage).filter(PaperPage.paper_id == paper.id).count() == 0
    finally:
        db.close()
        set_storage(None)

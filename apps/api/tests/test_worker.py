from contextlib import contextmanager

import pytest

from app.crud.job import create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.session import SessionLocal
from app.schemas.project import ProjectCreate
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage
from app.worker import _persisted_job_outcome, run_worker


@pytest.mark.anyio
async def test_run_worker_once(monkeypatch):
    # Initialize in-memory storage
    mem_storage = MemoryStorage()
    set_storage(mem_storage)

    from app.db.session import create_tables

    create_tables()

    db = SessionLocal()
    from app.db.models import Job

    db.query(Job).delete()
    db.commit()

    proj = create_project(db, ProjectCreate(name="Worker Test Project"))

    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 30 >> stream
BT
/F1 12 Tf
72 712 Td
(Worker test paper.) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000325 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
401
%%EOF"""

    storage_key = "worker_test.pdf"
    await mem_storage.put(storage_key, pdf_bytes)

    paper = create_paper(
        db,
        project_id=proj.id,
        filename="worker_test.pdf",
        storage_path=storage_key,
    )
    job = create_job(db, paper_id=paper.id)
    paper_id = paper.id
    job_id = job.id
    db.close()

    # Run worker once
    captured = []

    class Observation:
        def update(self, **kwargs):
            captured[-1]["updates"].append(kwargs)

    class Telemetry:
        @contextmanager
        def operation(self, name, *, metadata=None):
            captured.append({"name": name, "metadata": metadata, "updates": []})
            yield Observation()

    import app.worker

    monkeypatch.setattr(app.worker, "get_telemetry", lambda: Telemetry())
    await run_worker(poll_interval=0.1, once=True)

    assert len(captured) == 1
    assert captured[0]["name"] == "worker.ingestion_attempt"
    assert captured[0]["metadata"]["job_id"] == str(job_id)
    assert captured[0]["metadata"]["attempt_number"] == 1
    assert captured[0]["metadata"]["queue_age_basis"] == "since_initial_enqueue"
    assert captured[0]["metadata"]["initial_queue_age_seconds"] >= 0
    assert captured[0]["updates"] == [{"metadata": {"outcome": "success"}}]

    # Verify job completed and paper ready
    db2 = SessionLocal()
    from app.crud.job import get_job
    from app.crud.paper import get_paper

    updated_job = get_job(db2, job_id)
    assert updated_job.status == "COMPLETED"
    updated_paper = get_paper(db2, paper_id)
    assert updated_paper.status == "READY"
    db2.close()

    set_storage(None)


@pytest.mark.parametrize(
    ("status", "assigned_worker", "current_retry_count", "claimed_retry_count", "expected"),
    [
        ("COMPLETED", "worker-1", 0, 0, "success"),
        ("COMPLETED", "worker-2", 1, 0, "abandoned"),
        ("PENDING", None, 1, 0, "retry_scheduled"),
        ("FAILED", None, 0, 0, "failed"),
        ("PROCESSING", "worker-2", 1, 0, "abandoned"),
        ("PROCESSING", "worker-1", 0, 0, "incomplete"),
    ],
)
def test_persisted_job_attempt_outcome(
    status, assigned_worker, current_retry_count, claimed_retry_count, expected
):
    from types import SimpleNamespace

    job = SimpleNamespace(status=status, worker_id=assigned_worker, retry_count=current_retry_count)
    assert _persisted_job_outcome(job, "worker-1", claimed_retry_count) == expected

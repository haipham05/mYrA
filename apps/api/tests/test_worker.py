import pytest

from app.crud.job import create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.session import SessionLocal
from app.schemas.project import ProjectCreate
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage
from app.worker import run_worker


@pytest.mark.anyio
async def test_run_worker_once():
    # Initialize in-memory storage
    mem_storage = MemoryStorage()
    set_storage(mem_storage)

    from app.db.session import create_tables

    create_tables()

    db = SessionLocal()
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
    await run_worker(poll_interval=0.1, once=True)

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

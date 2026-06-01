from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.job import create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.schemas.project import ProjectCreate


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_reliability_api.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_correlation_id_generated_and_propagated(client):
    # Request without header receives generated correlation ID
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert "x-correlation-id" in res.headers
    corr_id = res.headers["x-correlation-id"]
    assert len(corr_id) > 0

    # Request with custom X-Correlation-ID preserves it
    custom_id = "test-corr-id-12345"
    res2 = client.get("/api/v1/health", headers={"X-Correlation-ID": custom_id})
    assert res2.status_code == 200
    assert res2.headers["x-correlation-id"] == custom_id
    assert res2.headers["x-request-id"] == custom_id


def test_job_retry_endpoint(client, db):
    proj = create_project(db, ProjectCreate(name="API Retry Test"))
    paper = create_paper(db, proj.id, "retry_api.pdf", "retry_api.pdf")
    job = create_job(db, paper.id)

    # 1. Retry non-existent job returns 404
    res_404 = client.post(f"/api/v1/jobs/{uuid4()}/retry")
    assert res_404.status_code == 404

    # 2. Mark job FAILED
    job.status = "FAILED"
    job.stage = "FAILED"
    job.error_message = "[CORRUPTED_PDF] Broken header"
    job.retry_count = 3
    db.commit()

    # 3. Successful retry
    res_retry = client.post(f"/api/v1/jobs/{job.id}/retry")
    assert res_retry.status_code == 200
    data = res_retry.json()
    assert data["status"] == "PENDING"
    assert data["stage"] == "QUEUED"
    assert data["retry_count"] == 0
    assert data["error_code"] is None

    # 4. Retrying again while PENDING/PROCESSING returns conflict or bad request
    job.status = "PROCESSING"
    db.commit()
    res_conflict = client.post(f"/api/v1/jobs/{job.id}/retry")
    assert res_conflict.status_code == 409


def test_system_reconcile_endpoint(client, db):
    proj = create_project(db, ProjectCreate(name="Reconcile API Test"))
    paper = create_paper(db, proj.id, "rec.pdf", "rec.pdf")
    job = create_job(db, paper.id)
    assert job.id is not None

    # Trigger dry-run reconciliation
    res_dry = client.post("/api/v1/system/reconcile?dry_run=true")
    assert res_dry.status_code == 200
    report_dry = res_dry.json()
    assert report_dry["dry_run"] is True
    assert "stuck_jobs_expired" in report_dry
    assert "stuck_jobs_recovered" in report_dry

    # Trigger live reconciliation
    res_live = client.post("/api/v1/system/reconcile?dry_run=false")
    assert res_live.status_code == 200
    report_live = res_live.json()
    assert report_live["dry_run"] is False

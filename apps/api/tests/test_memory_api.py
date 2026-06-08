from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.chat import create_conversation
from app.db.base import Base
from app.db.models import Memory, MemorySource, Message, Paper, PaperPage, Project
from app.db.session import get_db
from app.main import app
from app.schemas.memory import MemorySourceType, MemoryStatus, MemoryType


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_api_memory.db"
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
def client(db: Session):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_memory_crud_api_flow(client: TestClient, db: Session) -> None:
    # Setup project
    project = Project(name="API Memory Project")
    db.add(project)
    db.commit()

    # 1. Non-existent project returns 404
    non_existent = uuid4()
    resp = client.get(f"/api/v1/projects/{non_existent}/memories")
    assert resp.status_code == 404

    # 2. Create memory via POST
    payload = {
        "memory_type": MemoryType.DECISION.value,
        "title": "Selected AURC",
        "content": "Project decision: Selected AURC over ECE for calibration.",
        "importance": 0.85,
        "confidence": 0.95,
        "is_pinned": True,
        "sources": [],
    }
    resp = client.post(f"/api/v1/projects/{project.id}/memories", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Selected AURC"
    assert data["version"] == 1
    assert data["is_pinned"] is True
    assert data["status"] == MemoryStatus.ACTIVE.value
    memory_id = data["id"]

    # 3. List memories with filter
    resp = client.get(f"/api/v1/projects/{project.id}/memories?is_pinned=true")
    assert resp.status_code == 200
    list_data = resp.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["id"] == memory_id

    # 4. Get specific memory
    resp = client.get(f"/api/v1/projects/{project.id}/memories/{memory_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == memory_id
    assert len(resp.json()["history"]) == 1

    # Cross-project get returns 404
    other_project_id = uuid4()
    resp = client.get(f"/api/v1/projects/{other_project_id}/memories/{memory_id}")
    assert resp.status_code == 404

    # 5. Patch memory with matching version
    patch_payload = {
        "title": "Updated AURC Selection",
        "version": 1,
        "importance": 0.9,
    }
    resp = client.patch(f"/api/v1/projects/{project.id}/memories/{memory_id}", json=patch_payload)
    assert resp.status_code == 200
    updated_data = resp.json()
    assert updated_data["title"] == "Updated AURC Selection"
    assert updated_data["version"] == 2
    assert updated_data["importance"] == 0.9

    # 6a. Patch memory with missing version returns 422 Unprocessable Entity
    resp_no_ver = client.patch(
        f"/api/v1/projects/{project.id}/memories/{memory_id}", json={"title": "No Version"}
    )
    assert resp_no_ver.status_code == 422

    # 6b. Patch memory with stale version returns 409 Conflict
    stale_payload = {
        "title": "Conflict Attempt",
        "version": 1,  # Stale, version is now 2
    }
    resp = client.patch(f"/api/v1/projects/{project.id}/memories/{memory_id}", json=stale_payload)
    assert resp.status_code == 409
    assert "conflict" in resp.json()["detail"].lower()

    # 7. Supersede memory via POST /{memory_id}/supersede
    # 7a. Version conflict on supersede
    supersede_payload = {
        "memory_type": MemoryType.DECISION.value,
        "title": "Switch to Brier Score",
        "content": "Project decision: Switched from AURC to Brier Score.",
        "importance": 0.95,
        "confidence": 0.9,
        "is_pinned": True,
        "sources": [],
    }
    # 7a. Missing expected_version yields 422
    resp_no_ver = client.post(
        f"/api/v1/projects/{project.id}/memories/{memory_id}/supersede",
        json=supersede_payload,
    )
    assert resp_no_ver.status_code == 422

    # Version conflict on supersede (stale expected_version=1 vs current version=2)
    resp_conf = client.post(
        f"/api/v1/projects/{project.id}/memories/{memory_id}/supersede?expected_version=1",
        json=supersede_payload,
    )
    assert resp_conf.status_code == 409

    # 7b. Same content rejection on supersede
    same_content_payload = {
        "memory_type": MemoryType.DECISION.value,
        "title": "Selected AURC Repeat",
        "content": "Project decision: Selected AURC over ECE for calibration.",
        "importance": 0.9,
        "confidence": 0.95,
        "is_pinned": True,
        "sources": [],
    }
    resp_same = client.post(
        f"/api/v1/projects/{project.id}/memories/{memory_id}/supersede?expected_version=2",
        json=same_content_payload,
    )
    assert resp_same.status_code == 400
    assert "identical content" in resp_same.json()["detail"].lower()

    # 7c. Successful supersede with expected_version
    resp = client.post(
        f"/api/v1/projects/{project.id}/memories/{memory_id}/supersede?expected_version=2",
        json=supersede_payload,
    )
    assert resp.status_code == 200
    new_memory_data = resp.json()
    assert new_memory_data["title"] == "Switch to Brier Score"
    new_id = new_memory_data["id"]

    # 7d. Superseding an already superseded memory is rejected
    resp_again = client.post(
        f"/api/v1/projects/{project.id}/memories/{memory_id}/supersede?expected_version=2",
        json=supersede_payload,
    )
    assert resp_again.status_code == 400
    assert "active" in resp_again.json()["detail"].lower()

    # Verify old memory is now SUPERSEDED
    resp_old = client.get(f"/api/v1/projects/{project.id}/memories/{memory_id}")
    assert resp_old.status_code == 200
    assert resp_old.json()["status"] == MemoryStatus.SUPERSEDED.value
    assert resp_old.json()["superseded_by_id"] == new_id

    # 8. Soft Delete (Archive)
    resp = client.delete(f"/api/v1/projects/{project.id}/memories/{new_id}")
    assert resp.status_code == 204

    resp_archived = client.get(f"/api/v1/projects/{project.id}/memories/{new_id}")
    assert resp_archived.json()["status"] == MemoryStatus.ARCHIVED.value

    # 9. Hard Delete
    resp = client.delete(f"/api/v1/projects/{project.id}/memories/{new_id}?hard_delete=true")
    assert resp.status_code == 204
    resp_deleted = client.get(f"/api/v1/projects/{project.id}/memories/{new_id}")
    assert resp_deleted.status_code == 404


def test_memory_consolidation_endpoint(client: TestClient, db: Session) -> None:
    project = Project(name="Consolidation API Project")
    other_project = Project(name="Other Project")
    db.add_all([project, other_project])
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Decision Conversation")
    other_conv = create_conversation(
        db, project_id=other_project.id, title="Other Decision Conversation"
    )
    msg = Message(
        conversation_id=conv.id,
        role="user",
        content="Let's decide to use LoRA fine-tuning for our experiments.",
    )
    db.add(msg)
    db.commit()

    # Foreign conversation consolidation must return 404
    resp_foreign = client.post(
        f"/api/v1/projects/{project.id}/memories/consolidate",
        json={"conversation_id": str(other_conv.id)},
    )
    assert resp_foreign.status_code == 404

    resp = client.post(
        f"/api/v1/projects/{project.id}/memories/consolidate",
        json={"conversation_id": str(conv.id)},
    )
    assert resp.status_code == 200
    memories = resp.json()
    assert len(memories) >= 1
    assert any("lora" in m["content"].lower() for m in memories)


def test_paper_fact_edit_api_rejects_unsupported_content(client: TestClient, db: Session) -> None:
    """Prove that PATCH /memories/{id} with unsupported content returns 400 and preserves row."""
    project = Project(name="Paper Fact API Project")
    db.add(project)
    db.commit()

    paper = Paper(
        project_id=project.id,
        filename="comparison.pdf",
        storage_path="papers/comparison.pdf",
        document_sha256="hash_comp_123",
        status="READY",
    )
    db.add(paper)
    db.commit()

    page = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text="In our benchmark, Method A is better than method B across all evaluations.",
    )
    db.add(page)
    db.commit()

    initial_embedding = [0.1, 0.2, 0.3]
    mem = Memory(
        project_id=project.id,
        memory_type=MemoryType.PAPER_FACT.value,
        status=MemoryStatus.ACTIVE.value,
        title="Valid Comparison Fact",
        content="Method A is better than method B.",
        confidence=1.0,
        importance=0.9,
        version=1,
        embedding=initial_embedding,
        embedding_vec=initial_embedding,
    )
    db.add(mem)
    db.flush()

    source = MemorySource(
        memory_id=mem.id,
        source_type=MemorySourceType.PAPER_CHUNK.value,
        paper_id=paper.id,
        page_number=1,
        quote_text="Method A is better than method B",
        document_sha256="hash_comp_123",
    )
    db.add(source)
    db.commit()
    db.refresh(mem)

    # 1. Attempt to PATCH with reversed relation: "Method B is better than method A."
    patch_payload = {
        "version": 1,
        "content": "Method B is better than method A.",
    }
    resp = client.patch(f"/api/v1/projects/{project.id}/memories/{mem.id}", json=patch_payload)
    # Must return 400 Bad Request, NOT 500!
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"].lower()

    # 2. Verify in DB that memory content, version, and embedding remain strictly unchanged
    resp_get = client.get(f"/api/v1/projects/{project.id}/memories/{mem.id}")
    assert resp_get.status_code == 200
    current_data = resp_get.json()
    assert current_data["version"] == 1
    assert current_data["content"] == "Method A is better than method B."


def test_paper_fact_creation_api_rejects_mismatched_quote_with_opposite_nearby(
    client: TestClient, db: Session
) -> None:
    project = Project(name="Mismatched Quote Project")
    db.add(project)
    db.commit()

    paper = Paper(
        project_id=project.id,
        filename="opposite_sentences.pdf",
        storage_path="papers/opposite_sentences.pdf",
        document_sha256="hash_opp_456",
        status="READY",
    )
    db.add(paper)
    db.commit()

    page = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text="Method A is better than method B. Method B is better than method A.",
    )
    db.add(page)
    db.commit()

    # 1. Attempt to create memory claiming B > A while citing A > B: must return HTTP 400
    mismatched_payload = {
        "memory_type": MemoryType.PAPER_FACT.value,
        "title": "Claim B > A",
        "content": "Method B is better than method A.",
        "sources": [
            {
                "source_type": MemorySourceType.PAPER_CHUNK.value,
                "paper_id": str(paper.id),
                "page_number": 1,
                "quote_text": "Method A is better than method B",
                "document_sha256": "hash_opp_456",
            }
        ],
    }
    resp = client.post(f"/api/v1/projects/{project.id}/memories", json=mismatched_payload)
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"].lower()

    # 2. Accurately citing B > A succeeds with HTTP 201
    accurate_payload = {
        "memory_type": MemoryType.PAPER_FACT.value,
        "title": "Claim B > A Accurately Cited",
        "content": "Method B is better than method A.",
        "sources": [
            {
                "source_type": MemorySourceType.PAPER_CHUNK.value,
                "paper_id": str(paper.id),
                "page_number": 1,
                "quote_text": "Method B is better than method A",
                "document_sha256": "hash_opp_456",
            }
        ],
    }
    resp_ok = client.post(f"/api/v1/projects/{project.id}/memories", json=accurate_payload)
    assert resp_ok.status_code == 201
    data = resp_ok.json()
    assert data["content"] == "Method B is better than method A."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["quote_text"] == "Method B is better than method A"

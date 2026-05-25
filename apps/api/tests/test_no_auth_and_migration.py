from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_no_auth.db"
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


def test_no_auth_endpoints_exist(client):
    """Prove no auth or account surface exists in the application routes."""
    assert client.get("/api/v1/auth").status_code == 404
    assert client.get("/api/v1/auth/status").status_code == 404
    assert client.post("/api/v1/auth/login", json={}).status_code == 404
    assert client.post("/api/v1/auth/setup", json={}).status_code == 404
    assert client.post("/api/v1/auth/logout").status_code == 404
    assert client.get("/api/v1/auth/me").status_code == 404


def test_zero_login_conversation_reopen_and_citations(client, db):
    """Prove the operator can create project, ask questions, reload, and reopen
    conversations without cookies or login.
    """
    # 1. Create project without any auth headers or cookies
    res_proj = client.post("/api/v1/projects", json={"name": "Provenance Project"})
    assert res_proj.status_code == 201
    proj_id = res_proj.json()["id"]

    # 2. Create conversation
    res_conv = client.post(
        f"/api/v1/projects/{proj_id}/conversations",
        json={"title": "Primary Research Chat"},
    )
    assert res_conv.status_code == 201
    conv_id = res_conv.json()["id"]

    # 3. Ask question and receive answer with citation
    fake_citation = {
        "citation_index": 1,
        "evidence_id": "E1",
        "paper_id": str(uuid4()),
        "page_number": 1,
        "bounding_boxes": [],
        "quote": "Attention is all you need.",
        "anchor_status": "verified",
    }
    with patch("app.services.chat_service.ChatService.answer_question") as mock_answer:
        from app.schemas.chat import MessageResponse, MessageRole
        from app.schemas.evidence import Citation

        mock_answer.return_value = MessageResponse(
            id=uuid4(),
            conversation_id=conv_id,
            role=MessageRole.ASSISTANT,
            content="Attention is an architectural component. [1]",
            citations=[Citation.model_validate(fake_citation)],
            evidence=[],
            model_name="deepseek-chat",
            token_count=15,
            created_at=datetime.now(UTC),
        )

        res_msg = client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"content": "What is the key mechanism?"},
        )
        assert res_msg.status_code == 200
        msg_data = res_msg.json()
        assert len(msg_data["citations"]) == 1
        assert msg_data["citations"][0]["quote"] == "Attention is all you need."

    # 4. Reopen conversation after simulated page refresh (no cookies needed)
    res_list = client.get(f"/api/v1/projects/{proj_id}/conversations")
    assert res_list.status_code == 200
    items = res_list.json()["items"]
    assert len(items) == 1
    assert items[0]["id"] == conv_id
    assert items[0]["title"] == "Primary Research Chat"

    # Reopen detail
    res_detail = client.get(f"/api/v1/conversations/{conv_id}")
    assert res_detail.status_code == 200
    assert res_detail.json()["id"] == conv_id

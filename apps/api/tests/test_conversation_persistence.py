from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.chat import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    update_conversation,
)
from app.db.base import Base
from app.db.models import Message, Project
from app.db.session import get_db
from app.main import app
from app.schemas.chat import MessageRole
from app.services.chat_service import ChatService


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_conv.db"
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


def test_conversation_lifecycle_and_project_scoping(db):
    p1 = Project(name="Project Alpha")
    p2 = Project(name="Project Beta")
    db.add_all([p1, p2])
    db.commit()

    # 1. Create conversations
    c1 = create_conversation(db, project_id=p1.id, title="Alpha Discussion")
    c2 = create_conversation(db, project_id=p2.id, title="Beta Discussion")

    # 2. Scoped get_conversation
    assert get_conversation(db, c1.id, project_id=p1.id) is not None
    assert get_conversation(db, c2.id, project_id=p2.id) is not None
    # Looking for c1 in p2 returns None (isolation)
    assert get_conversation(db, c1.id, project_id=p2.id) is None

    # 3. List conversations scoped by project
    items, total = list_conversations(db, project_id=p1.id)
    assert total == 1
    assert items[0].id == c1.id
    assert items[0].title == "Alpha Discussion"

    # 4. Update title and archive
    updated = update_conversation(db, c1.id, title="Alpha Updated", is_archived=True)
    assert updated.title == "Alpha Updated"
    assert updated.is_archived is True

    # 5. List conversations excludes archived by default
    items_active, total_active = list_conversations(db, project_id=p1.id, include_archived=False)
    assert total_active == 0
    assert len(items_active) == 0

    items_all, total_all = list_conversations(db, project_id=p1.id, include_archived=True)
    assert total_all == 1
    assert items_all[0].id == c1.id

    # 6. Delete conversation
    assert delete_conversation(db, c1.id, project_id=p2.id) is False
    assert delete_conversation(db, c1.id, project_id=p1.id) is True
    assert get_conversation(db, c1.id) is None


def test_messages_persistence_and_metadata(db):
    p = Project(name="Project Gamma")
    db.add(p)
    db.commit()

    conv = create_conversation(db, project_id=p.id, title="Gamma Chat")
    original_updated_at = conv.updated_at

    # Add user message
    msg_user = add_message(
        db=db,
        conversation_id=conv.id,
        role="USER",
        content="What is attention?",
        citations=[],
        evidence=[],
    )
    assert msg_user.id is not None
    assert msg_user.role == "USER"

    # Add assistant message with citations and model metadata
    citation_data = [
        {
            "citation_index": 1,
            "evidence_id": "E1",
            "paper_id": str(uuid4()),
            "page_number": 1,
            "bounding_boxes": [],
            "quote": "Attention is all you need.",
            "anchor_status": "VERIFIED",
        }
    ]
    msg_assistant = add_message(
        db=db,
        conversation_id=conv.id,
        role="ASSISTANT",
        content="Attention is an architectural component. [1]",
        citations=citation_data,
        evidence=[],
        model_name="deepseek-chat",
        token_count=42,
    )
    assert msg_assistant.model_name == "deepseek-chat"
    assert msg_assistant.token_count == 42
    assert len(msg_assistant.citations) == 1

    # Check conversation updated_at was refreshed
    db.refresh(conv)
    assert conv.updated_at >= original_updated_at

    # List messages ordered chronologically
    messages, total = list_messages(db, conv.id)
    assert total == 2
    assert messages[0].id == msg_user.id
    assert messages[1].id == msg_assistant.id

    # Delete conversation cascades to messages
    delete_conversation(db, conv.id)
    assert db.get(Message, msg_user.id) is None
    assert db.get(Message, msg_assistant.id) is None


def test_conversation_api_endpoints(client, db):
    p = Project(name="API Project")
    db.add(p)
    db.commit()

    # 1. Create conversation via API
    res = client.post(
        f"/api/v1/projects/{p.id}/conversations",
        json={"title": "Interactive Research"},
    )
    assert res.status_code == 201
    conv_data = res.json()
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Interactive Research"
    assert conv_data["message_count"] == 0

    # 2. List conversations
    res_list = client.get(f"/api/v1/projects/{p.id}/conversations")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] == 1
    assert len(list_data["items"]) == 1
    assert list_data["items"][0]["id"] == conv_id

    # 3. Patch conversation (rename and archive)
    res_patch = client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": "Renamed Chat", "is_archived": True},
    )
    assert res_patch.status_code == 200
    assert res_patch.json()["title"] == "Renamed Chat"
    assert res_patch.json()["is_archived"] is True

    # 4. List without include_archived returns 0 items
    res_active = client.get(f"/api/v1/projects/{p.id}/conversations")
    assert res_active.json()["total"] == 0

    # 5. List with include_archived=true returns 1 item
    res_archived = client.get(f"/api/v1/projects/{p.id}/conversations?include_archived=true")
    assert res_archived.json()["total"] == 1

    # 6. Delete conversation
    res_del = client.delete(f"/api/v1/conversations/{conv_id}")
    assert res_del.status_code == 204

    # 7. Get deleted conversation returns 404
    res_get = client.get(f"/api/v1/conversations/{conv_id}")
    assert res_get.status_code == 404


@pytest.mark.anyio
async def test_chat_service_transactional_consistency_on_error(db):
    p = Project(name="Error Project")
    db.add(p)
    db.commit()

    conv = create_conversation(db, project_id=p.id, title="Error Chat")

    chat_service = ChatService()

    # Mock llm.generate to raise a failure
    with patch("app.services.chat_service.get_llm_provider") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("DeepSeek connection timed out")
        mock_get_llm.return_value = mock_llm

        with pytest.raises(RuntimeError, match="DeepSeek connection timed out"):
            await chat_service.answer_question(db, conv.id, "Will this fail?")

    # Verify write consistency:
    # 1. User message was saved
    # 2. No assistant message was saved
    messages = db.query(Message).filter(Message.conversation_id == conv.id).all()
    assert len(messages) == 1
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Will this fail?"

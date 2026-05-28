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

    # 3. Patch conversation (rename, archive, and summary)
    res_patch = client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={
            "title": "Renamed Chat",
            "summary": "Initial conversation summary.",
            "is_archived": True,
        },
    )
    assert res_patch.status_code == 200
    assert res_patch.json()["title"] == "Renamed Chat"
    assert res_patch.json()["summary"] == "Initial conversation summary."
    assert res_patch.json()["is_archived"] is True

    # 4. List without include_archived returns 0 items
    res_active = client.get(f"/api/v1/projects/{p.id}/conversations")
    assert res_active.json()["total"] == 0

    # 5. List with include_archived=true returns 1 item with summary
    res_archived = client.get(f"/api/v1/projects/{p.id}/conversations?include_archived=true")
    assert res_archived.json()["total"] == 1
    assert res_archived.json()["items"][0]["summary"] == "Initial conversation summary."

    # 6. Project-scoping query parameter check
    other_project_id = uuid4()
    res_other = client.get(f"/api/v1/conversations/{conv_id}?project_id={other_project_id}")
    assert res_other.status_code == 404
    res_other_msgs = client.get(
        f"/api/v1/conversations/{conv_id}/messages?project_id={other_project_id}"
    )
    assert res_other_msgs.status_code == 404
    res_other_patch = client.patch(
        f"/api/v1/conversations/{conv_id}?project_id={other_project_id}", json={"title": "Hacked"}
    )
    assert res_other_patch.status_code == 404
    res_other_del = client.delete(f"/api/v1/conversations/{conv_id}?project_id={other_project_id}")
    assert res_other_del.status_code == 404

    # 7. Delete conversation
    res_del = client.delete(f"/api/v1/conversations/{conv_id}")
    assert res_del.status_code == 204

    # 8. Get deleted conversation returns 404
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

    # 3. Test duplicate submit deduplication:
    # Retrying the exact same question when the last message is an unanswered USER message
    # does NOT insert an unnecessary duplicate user record.
    with patch("app.services.chat_service.get_llm_provider") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RuntimeError("Second timeout")
        mock_get_llm.return_value = mock_llm

        with pytest.raises(RuntimeError, match="Second timeout"):
            await chat_service.answer_question(db, conv.id, "Will this fail?")

    messages_after_retry = db.query(Message).filter(Message.conversation_id == conv.id).all()
    assert len(messages_after_retry) == 1
    assert messages_after_retry[0].content == "Will this fail?"


@pytest.mark.anyio
async def test_chat_service_multiturn_context_and_token_count(db):
    p = Project(name="Multiturn Project")
    db.add(p)
    db.commit()

    conv = create_conversation(db, project_id=p.id, title="Multiturn Chat")

    # Add prior turn
    add_message(
        db=db,
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="What score did BERT obtain on GLUE?",
        citations=[],
        evidence=[],
    )
    add_message(
        db=db,
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="BERT obtained an 80.5% average score.",
        citations=[],
        evidence=[],
    )

    chat_service = ChatService()

    with patch("app.services.chat_service.get_llm_provider") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.provider_name = "test-provider"
        mock_llm.model_name = "test-chat-model"
        mock_llm.generate.return_value = "No direct comparison found."
        mock_get_llm.return_value = mock_llm

        res = await chat_service.answer_question(
            db, conv.id, "How does that compare to the baseline?"
        )

        # 1. Assert multi-turn history was propagated to user_prompt
        call_args = mock_llm.generate.call_args
        assert call_args is not None
        user_prompt_sent = call_args.kwargs.get("user_prompt", "")
        assert "CONVERSATION HISTORY:" in user_prompt_sent
        assert "What score did BERT obtain on GLUE?" in user_prompt_sent
        assert "BERT obtained an 80.5% average score." in user_prompt_sent
        assert "How does that compare to the baseline?" in user_prompt_sent

        # 2. Assert token count is calculated and populated
        assert res.token_count is not None
        assert res.token_count > 0
        assert res.model_name == "test-chat-model"

        # 3. Assert message record in database has token_count
        db_msg = db.get(Message, res.id)
        assert db_msg is not None
        assert db_msg.token_count == res.token_count
        assert db_msg.model_name == "test-chat-model"

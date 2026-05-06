import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import (
    ChunkElement,
    Conversation,
    Job,
    Message,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
    Project,
)


@pytest.fixture
def db_session():
    # Use isolated in-memory SQLite database for tests
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_project_and_paper_relationships(db_session):
    project = Project(name="Research Project", description="A test research project")
    db_session.add(project)
    db_session.commit()

    paper = Paper(
        project_id=project.id,
        filename="attention.pdf",
        storage_path="papers/attention.pdf",
        status="PROCESSING",
    )
    db_session.add(paper)
    db_session.commit()

    page = PaperPage(paper_id=paper.id, page_number=1, width=612.0, height=792.0)
    db_session.add(page)

    elem = PaperElement(
        paper_id=paper.id,
        page_number=1,
        element_index=0,
        element_type="paragraph",
        text="Attention Is All You Need.",
        bbox_x_min=72.0,
        bbox_y_min=100.0,
        bbox_x_max=540.0,
        bbox_y_max=150.0,
    )
    db_session.add(elem)

    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Attention Is All You Need.",
        embedding=[0.1] * 1024,
    )
    db_session.add(chunk)
    db_session.flush()

    chunk_elem = ChunkElement(chunk_id=chunk.id, element_id=elem.id, order_index=0)
    db_session.add(chunk_elem)

    job = Job(paper_id=paper.id, status="PENDING", stage="QUEUED")
    db_session.add(job)

    db_session.commit()

    # Query back
    saved_project = db_session.query(Project).filter_by(id=project.id).first()
    assert len(saved_project.papers) == 1
    assert saved_project.papers[0].filename == "attention.pdf"
    assert len(saved_project.papers[0].elements) == 1
    assert len(saved_project.papers[0].chunks) == 1
    assert len(saved_project.papers[0].chunks[0].elements) == 1
    assert len(saved_project.papers[0].jobs) == 1


def test_unique_constraints(db_session):
    project = Project(name="P1")
    db_session.add(project)
    db_session.commit()

    paper = Paper(project_id=project.id, filename="p.pdf", storage_path="p.pdf")
    db_session.add(paper)
    db_session.commit()

    p1 = PaperPage(paper_id=paper.id, page_number=1, width=100, height=100)
    db_session.add(p1)
    db_session.commit()

    # Duplicate page number for same paper should violate uniqueness
    p2 = PaperPage(paper_id=paper.id, page_number=1, width=200, height=200)
    db_session.add(p2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_conversation_and_message(db_session):
    project = Project(name="P2")
    db_session.add(project)
    db_session.commit()

    conv = Conversation(project_id=project.id, title="Chat 1")
    db_session.add(conv)
    db_session.commit()

    msg = Message(
        conversation_id=conv.id,
        role="USER",
        content="What is self-attention?",
        citations=[],
        evidence=[],
    )
    db_session.add(msg)
    db_session.commit()

    saved_conv = db_session.query(Conversation).filter_by(id=conv.id).first()
    assert len(saved_conv.messages) == 1
    assert saved_conv.messages[0].content == "What is self-attention?"

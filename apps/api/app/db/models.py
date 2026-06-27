from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    FetchedValue,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import PGVector, TSVector


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    corpus_revision: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default="0", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    papers: Mapped[list["Paper"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    memories: Mapped[list["Memory"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    document_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="PROCESSING", nullable=False, index=True
    )
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="papers")
    pages: Mapped[list["PaperPage"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    elements: Mapped[list["PaperElement"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["PaperChunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["Job"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class PaperPage(Base):
    __tablename__ = "paper_pages"
    __table_args__ = (
        UniqueConstraint("paper_id", "page_number", name="uq_paper_pages_paper_page"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    rotation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    crop_box: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="pages")


class PaperElement(Base):
    __tablename__ = "paper_elements"
    __table_args__ = (
        UniqueConstraint("paper_id", "element_index", name="uq_paper_elements_paper_idx"),
        Index("ix_paper_elements_page", "paper_id", "page_number"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    element_index: Mapped[int] = mapped_column(Integer, nullable=False)
    element_type: Mapped[str] = mapped_column(String(50), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_x_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_y_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_width: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_height: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinate_origin: Mapped[str] = mapped_column(String(20), default="TOP_LEFT", nullable=False)
    rotation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    section_path: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="elements")


class PaperChunk(Base):
    __tablename__ = "paper_chunks"
    __table_args__ = (Index("ix_paper_chunks_paper_type", "paper_id", "chunk_type"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "parent" or "child"
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_vec: Mapped[list[float] | None] = mapped_column(PGVector(1024), nullable=True)
    tsv_content: Mapped[str | None] = mapped_column(
        TSVector, server_default=FetchedValue(), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    elements: Mapped[list["ChunkElement"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", order_by="ChunkElement.order_index"
    )


class ChunkElement(Base):
    __tablename__ = "chunk_elements"
    __table_args__ = (
        UniqueConstraint("chunk_id", "element_id", name="uq_chunk_elements_chunk_elem"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey("paper_chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    element_id: Mapped[UUID] = mapped_column(
        ForeignKey("paper_elements.id", ondelete="CASCADE"), nullable=False, index=True
    )
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    chunk: Mapped["PaperChunk"] = relationship(back_populates="elements")
    element: Mapped["PaperElement"] = relationship()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(50), default="QUEUED", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_retryable: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    trace_sampled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="jobs")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="[Message.created_at, Message.id]",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_memories_project_status", "project_id", "status"),
        Index("ix_memories_project_type", "project_id", "memory_type"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE", nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    superseded_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL"), nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_vec: Mapped[list[float] | None] = mapped_column(PGVector(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    project: Mapped["Project"] = relationship(back_populates="memories")
    sources: Mapped[list["MemorySource"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )
    history: Mapped[list["MemoryAudit"]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
        order_by="MemoryAudit.created_at.desc()",
    )
    superseded_by: Mapped["Memory | None"] = relationship(remote_side=[id])


class MemorySource(Base):
    __tablename__ = "memory_sources"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    memory_id: Mapped[UUID] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "MESSAGE" or "PAPER_CHUNK"
    message_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True
    )
    paper_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("papers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    memory: Mapped["Memory"] = relationship(back_populates="sources")
    message: Mapped["Message | None"] = relationship()
    paper: Mapped["Paper | None"] = relationship()

    @property
    def conversation_id(self) -> UUID | None:
        return self.message.conversation_id if self.message else None


class MemoryAudit(Base):
    __tablename__ = "memory_audits"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    memory_id: Mapped[UUID] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    old_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    memory: Mapped["Memory"] = relationship(back_populates="history")


class GraphEvent(Base):
    __tablename__ = "graph_events"
    __table_args__ = (
        UniqueConstraint(
            "paper_id", "generation_id", "action", name="uq_graph_events_paper_generation_action"
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paper_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    action: Mapped[str] = mapped_column(
        String(50), default="UPSERT", server_default="UPSERT", nullable=False
    )
    generation_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    ontology_version: Mapped[str] = mapped_column(
        String(50), default="1.0.0", server_default="1.0.0", nullable=False
    )
    extractor_version: Mapped[str] = mapped_column(
        String(50), default="1.0.0", server_default="1.0.0", nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(50), default="PENDING", server_default="PENDING", nullable=False, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    max_attempts: Mapped[int] = mapped_column(
        Integer, default=3, server_default="3", nullable=False
    )
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    trace_sampled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    lease_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=sa.func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped["Project"] = relationship()
    snapshots: Mapped[list["GraphFactSnapshot"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class GraphFactSnapshot(Base):
    __tablename__ = "graph_fact_snapshots"

    fact_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paper_id: Mapped[UUID] = mapped_column(nullable=False, index=True)
    generation_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    event_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("graph_events.id", ondelete="SET NULL"), nullable=True, index=True
    )
    subject_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    subject_name: Mapped[str] = mapped_column(String(255), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(50), nullable=False)
    predicate: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    object_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    object_name: Mapped[str] = mapped_column(String(255), nullable=False)
    object_type: Mapped[str] = mapped_column(String(50), nullable=False)
    qualifiers: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Provenance fields (verified source fields, NOT raw LLM response or full PDF text)
    chunk_id: Mapped[UUID | None] = mapped_column(nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    element_id: Mapped[UUID | None] = mapped_column(nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    exact_quote: Mapped[str] = mapped_column(Text, nullable=False)
    document_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    validation_version: Mapped[str] = mapped_column(
        String(50), default="1.0.0", server_default="1.0.0", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=sa.func.now(), nullable=False
    )

    event: Mapped["GraphEvent | None"] = relationship(back_populates="snapshots")
    project: Mapped["Project"] = relationship()

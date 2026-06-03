import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.compatibility import check_schema_compatibility, get_schema_revisions
from app.db.models import Memory, MemoryAudit, MemorySource, Project
from app.schemas.memory import (
    MemoryCreate,
    MemorySourceCreate,
    MemorySourceType,
    MemoryType,
)


def get_alembic_config(db_url: str) -> Config:
    ini_path = Path.cwd() / "alembic.ini"
    if not ini_path.exists():
        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_memory_schema_validation_rules():
    # 1. Valid Decision Memory
    decision = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Optimizer Choice",
        content="Use Adam with learning rate 1e-4",
        importance=0.8,
        confidence=1.0,
    )
    assert decision.memory_type == MemoryType.DECISION
    assert decision.importance == 0.8

    # 2. PAPER_FACT without valid paper source must fail validation
    with pytest.raises(ValidationError, match="requires a verified paper source"):
        MemoryCreate(
            memory_type=MemoryType.PAPER_FACT,
            title="Transformer BLEU Score",
            content="Transformer achieves 28.4 BLEU on En-De",
            sources=[],
        )

    # 3. PAPER_FACT with valid paper source succeeds
    paper_id = uuid4()
    paper_fact = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Transformer BLEU Score",
        content="Transformer achieves 28.4 BLEU on En-De",
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=paper_id,
                page_number=5,
                quote_text="The Transformer model achieves a state-of-the-art BLEU score of 28.4",
            )
        ],
    )
    assert len(paper_fact.sources) == 1
    assert paper_fact.sources[0].paper_id == paper_id

    # 4. Out-of-bounds importance fails
    with pytest.raises(ValidationError):
        MemoryCreate(
            memory_type=MemoryType.PREFERENCE,
            title="Formatting",
            content="Prefer bullet points",
            importance=1.5,
        )


def test_memory_migration_and_cascade_lifecycle():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)

        # 1. Run full upgrade to head (which includes memory migration)
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        current_rev, heads = get_schema_revisions(engine)
        assert current_rev == "f1a2b3c4d5e6"
        check_schema_compatibility(engine)

        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with session_factory() as db:
            # 2. Create Project and Memory with Sources and Audit
            proj = Project(name="Memory Lifecycle Project")
            db.add(proj)
            db.commit()
            db.refresh(proj)

            memory = Memory(
                project_id=proj.id,
                memory_type="DECISION",
                status="ACTIVE",
                title="Model Selection",
                content="Selected BGE-M3 for multilingual retrieval",
                confidence=1.0,
                importance=0.9,
            )
            db.add(memory)
            db.commit()
            db.refresh(memory)

            source = MemorySource(
                memory_id=memory.id,
                source_type="MESSAGE",
                quote_text="Let us use BGE-M3",
            )
            audit = MemoryAudit(
                memory_id=memory.id,
                action="CREATED",
                new_content=memory.content,
                reason="Initial decision capture",
            )
            db.add(source)
            db.add(audit)
            db.commit()

            # 3. Query through relationship
            db.refresh(memory)
            assert len(memory.sources) == 1
            assert len(memory.history) == 1
            assert memory.sources[0].quote_text == "Let us use BGE-M3"
            assert memory.history[0].action == "CREATED"

            # 4. Cascade delete: deleting project removes memory, sources, and audits
            db.delete(proj)
            db.commit()

            assert db.query(Memory).filter(Memory.project_id == proj.id).count() == 0
            assert db.query(MemorySource).filter(MemorySource.memory_id == memory.id).count() == 0
            assert db.query(MemoryAudit).filter(MemoryAudit.memory_id == memory.id).count() == 0

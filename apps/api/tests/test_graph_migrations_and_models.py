import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, event, inspect, select
from sqlalchemy import func as sa_func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.compatibility import check_schema_compatibility, get_schema_revisions
from app.db.models import GraphEvent, GraphFactSnapshot, Paper, Project


def get_alembic_config(db_url: str) -> Config:
    ini_path = Path.cwd() / "alembic.ini"
    if not ini_path.exists():
        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def create_sqlite_fk_engine(db_url: str):
    engine = create_engine(db_url)

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def test_migration_lifecycle_upgrade_downgrade():
    """Verify alembic migration g1a2b3c4d5e6 applies cleanly, defines tables/indexes,
    downgrades back to f1a2b3c4d5e6, and upgrades back to head.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)

        # 1. Upgrade to head
        command.upgrade(cfg, "head")

        engine = create_engine(db_url)
        current_rev, heads = get_schema_revisions(engine)
        assert current_rev == "g1a2b3c4d5e6"
        assert current_rev in heads
        check_schema_compatibility(engine)

        # 2. Inspect created tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "graph_events" in tables
        assert "graph_fact_snapshots" in tables

        # Verify graph_events columns
        event_cols = {col["name"]: col for col in inspector.get_columns("graph_events")}
        for expected_col in [
            "id",
            "project_id",
            "paper_id",
            "action",
            "generation_id",
            "ontology_version",
            "extractor_version",
            "status",
            "attempts",
            "max_attempts",
            "lease_owner",
            "lease_expires_at",
            "error_code",
            "error_message",
            "created_at",
            "updated_at",
            "completed_at",
        ]:
            assert expected_col in event_cols, f"Missing column {expected_col} in graph_events"

        # Verify graph_fact_snapshots columns
        fact_cols = {col["name"]: col for col in inspector.get_columns("graph_fact_snapshots")}
        for expected_col in [
            "fact_id",
            "project_id",
            "paper_id",
            "generation_id",
            "event_id",
            "subject_key",
            "subject_name",
            "subject_type",
            "predicate",
            "object_key",
            "object_name",
            "object_type",
            "qualifiers",
            "chunk_id",
            "page_number",
            "element_id",
            "char_start",
            "char_end",
            "exact_quote",
            "document_sha256",
            "validation_version",
            "created_at",
        ]:
            assert expected_col in fact_cols, (
                f"Missing column {expected_col} in graph_fact_snapshots"
            )

        # Verify unique constraint on graph_events
        uqs = inspector.get_unique_constraints("graph_events")
        uq_names = [uq["name"] for uq in uqs]
        assert "uq_graph_events_paper_generation_action" in uq_names
        uq_match = next(uq for uq in uqs if uq["name"] == "uq_graph_events_paper_generation_action")
        assert set(uq_match["column_names"]) == {"paper_id", "generation_id", "action"}

        # 3. Downgrade to predecessor revision f1a2b3c4d5e6
        command.downgrade(cfg, "f1a2b3c4d5e6")

        current_rev, _ = get_schema_revisions(engine)
        assert current_rev == "f1a2b3c4d5e6"

        inspector = inspect(engine)
        tables_after_downgrade = inspector.get_table_names()
        assert "graph_events" not in tables_after_downgrade
        assert "graph_fact_snapshots" not in tables_after_downgrade
        # Existing tables must be preserved
        assert "memories" in tables_after_downgrade
        assert "papers" in tables_after_downgrade
        assert "projects" in tables_after_downgrade

        # 4. Re-upgrade back to head
        command.upgrade(cfg, "head")
        current_rev, heads = get_schema_revisions(engine)
        assert current_rev == "g1a2b3c4d5e6"
        check_schema_compatibility(engine)


def test_graph_event_model_creation_and_defaults():
    """Verify GraphEvent ORM instantiation, field defaults, and relationship to Project."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Graph Model Test Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            paper = Paper(
                project_id=project.id,
                filename="attention.pdf",
                storage_path="papers/attention.pdf",
                status="READY",
            )
            db.add(paper)
            db.commit()
            db.refresh(paper)

            event = GraphEvent(
                project_id=project.id,
                paper_id=paper.id,
                generation_id="gen-001",
            )
            db.add(event)
            db.commit()
            db.refresh(event)

            # Assert default values
            assert event.id is not None
            assert event.action == "UPSERT"
            assert event.ontology_version == "1.0.0"
            assert event.extractor_version == "1.0.0"
            assert event.status == "PENDING"
            assert event.attempts == 0
            assert event.max_attempts == 3
            assert event.lease_owner is None
            assert event.lease_expires_at is None
            assert event.error_code is None
            assert event.error_message is None
            assert event.completed_at is None
            assert event.created_at is not None
            assert event.updated_at is not None

            # Assert project relationship
            assert event.project.id == project.id
            assert event.project.name == "Graph Model Test Project"


def test_graph_fact_snapshot_model_creation_and_provenance():
    """Verify GraphFactSnapshot model creation, provenance attributes,
    and bidirectional event relationship.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Snapshot Test Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            paper_id = uuid4()
            event = GraphEvent(
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-002",
                status="PROCESSING",
            )
            db.add(event)
            db.commit()
            db.refresh(event)

            chunk_id = uuid4()
            element_id = uuid4()
            snapshot = GraphFactSnapshot(
                fact_id="fact-sha256-deterministic-key-1",
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-002",
                event_id=event.id,
                subject_key=f"{project.id}:method:transformer",
                subject_name="Transformer",
                subject_type="Method",
                predicate="EVALUATED_ON",
                object_key=f"{project.id}:dataset:wmt14",
                object_name="WMT 2014 English-to-German",
                object_type="Dataset",
                qualifiers={"metric": "BLEU", "value": 28.4, "split": "test"},
                chunk_id=chunk_id,
                page_number=5,
                element_id=element_id,
                char_start=45,
                char_end=98,
                exact_quote="Transformer model achieves 28.4 BLEU on English-to-German",
                document_sha256="abc123def456" * 5 + "abcd",
            )
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            db.refresh(event)

            # Assert fields and provenance
            assert snapshot.fact_id == "fact-sha256-deterministic-key-1"
            assert snapshot.validation_version == "1.0.0"
            assert snapshot.qualifiers == {"metric": "BLEU", "value": 28.4, "split": "test"}
            assert snapshot.chunk_id == chunk_id
            assert snapshot.element_id == element_id
            assert snapshot.page_number == 5
            assert snapshot.char_start == 45
            assert snapshot.char_end == 98
            assert (
                snapshot.exact_quote == "Transformer model achieves 28.4 BLEU on English-to-German"
            )
            assert snapshot.created_at is not None

            # Assert relationship navigations
            assert snapshot.event is not None
            assert snapshot.event.id == event.id
            assert len(event.snapshots) == 1
            assert event.snapshots[0].fact_id == "fact-sha256-deterministic-key-1"
            assert snapshot.project.id == project.id


def test_graph_event_unique_constraint():
    """Verify unique constraint on (paper_id, generation_id, action) in graph_events."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="UQ Test Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            paper_id = uuid4()
            event1 = GraphEvent(
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-001",
                action="UPSERT",
            )
            db.add(event1)
            db.commit()

            # Duplicate (paper_id, generation_id, action) must raise IntegrityError
            event_dup = GraphEvent(
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-001",
                action="UPSERT",
            )
            db.add(event_dup)
            with pytest.raises(IntegrityError):
                db.commit()
            db.rollback()

            # Different action ("DELETE") for same paper_id & generation_id is allowed
            event_delete = GraphEvent(
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-001",
                action="DELETE",
            )
            db.add(event_delete)
            db.commit()

            # Different generation_id for same paper_id & action is allowed
            event_gen2 = GraphEvent(
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-002",
                action="UPSERT",
            )
            db.add(event_gen2)
            db.commit()

            events = db.scalars(
                select(GraphEvent)
                .where(GraphEvent.paper_id == paper_id)
                .order_by(GraphEvent.created_at)
            ).all()
            assert len(events) == 3


def test_graph_fact_snapshot_primary_key_uniqueness():
    """Verify primary key constraint on fact_id in graph_fact_snapshots."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="PK Fact Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            paper_id = uuid4()
            snap1 = GraphFactSnapshot(
                fact_id="fact-pk-unique-12345",
                project_id=project.id,
                paper_id=paper_id,
                generation_id="gen-001",
                subject_key=f"{project.id}:method:m1",
                subject_name="M1",
                subject_type="Method",
                predicate="USES_DATASET",
                object_key=f"{project.id}:dataset:d1",
                object_name="D1",
                object_type="Dataset",
                page_number=1,
                char_start=0,
                char_end=10,
                exact_quote="Uses D1 here.",
                document_sha256="12345678" * 8,
            )
            db.add(snap1)
            db.commit()

            # Attempt duplicate primary key in a new session to test DB integrity constraint
            with session_factory() as db2:
                snap2 = GraphFactSnapshot(
                    fact_id="fact-pk-unique-12345",
                    project_id=project.id,
                    paper_id=paper_id,
                    generation_id="gen-001",
                    subject_key=f"{project.id}:method:m2",
                    subject_name="M2",
                    subject_type="Method",
                    predicate="USES_DATASET",
                    object_key=f"{project.id}:dataset:d2",
                    object_name="D2",
                    object_type="Dataset",
                    page_number=2,
                    char_start=0,
                    char_end=10,
                    exact_quote="Uses D2 here.",
                    document_sha256="12345678" * 8,
                )
                db2.add(snap2)
                with pytest.raises(IntegrityError):
                    db2.commit()


def test_tombstone_preservation_when_paper_deleted():
    """Verify that deleting a paper row does NOT cascade to graph_events or snapshots,
    ensuring deletion tombstones and audit records are durably preserved.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Tombstone Preservation Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            paper = Paper(
                project_id=project.id,
                filename="target_paper.pdf",
                storage_path="papers/target_paper.pdf",
                status="READY",
            )
            db.add(paper)
            db.commit()
            db.refresh(paper)
            saved_paper_id = paper.id

            # Create an UPSERT event and a DELETE tombstone event for this paper
            upsert_event = GraphEvent(
                project_id=project.id,
                paper_id=saved_paper_id,
                generation_id="gen-001",
                action="UPSERT",
                status="COMPLETED",
            )
            tombstone_event = GraphEvent(
                project_id=project.id,
                paper_id=saved_paper_id,
                generation_id="gen-001",
                action="DELETE",
                status="PENDING",
            )
            db.add_all([upsert_event, tombstone_event])
            db.commit()

            # Create a snapshot linked to the upsert event
            snapshot = GraphFactSnapshot(
                fact_id="fact-tombstone-test-01",
                project_id=project.id,
                paper_id=saved_paper_id,
                generation_id="gen-001",
                event_id=upsert_event.id,
                subject_key=f"{project.id}:method:m",
                subject_name="Method M",
                subject_type="Method",
                predicate="PROPOSES",
                object_key=f"{project.id}:claim:c",
                object_name="Claim C",
                object_type="Claim",
                page_number=1,
                char_start=10,
                char_end=30,
                exact_quote="We propose Method M here",
                document_sha256="abcd" * 16,
            )
            db.add(snapshot)
            db.commit()

            # Now delete the Paper row
            db.delete(paper)
            db.commit()

            # Verify paper is deleted
            assert db.get(Paper, saved_paper_id) is None

            # Verify both events (including the tombstone) STILL EXIST
            events = db.scalars(
                select(GraphEvent).where(GraphEvent.paper_id == saved_paper_id)
            ).all()
            assert len(events) == 2
            actions = {e.action for e in events}
            assert actions == {"UPSERT", "DELETE"}

            # Verify snapshot STILL EXISTS
            retained_snapshot = db.get(GraphFactSnapshot, "fact-tombstone-test-01")
            assert retained_snapshot is not None
            assert retained_snapshot.paper_id == saved_paper_id


def test_project_cascade_delete():
    """Verify that deleting a project cascades to graph_events and graph_fact_snapshots
    via ForeignKey(projects.id, ondelete='CASCADE').
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Project Cascade Test")
            db.add(project)
            db.commit()
            db.refresh(project)
            proj_id = project.id

            paper_id = uuid4()
            event = GraphEvent(
                project_id=proj_id,
                paper_id=paper_id,
                generation_id="gen-cascade-1",
            )
            db.add(event)
            db.commit()
            db.refresh(event)

            snapshot = GraphFactSnapshot(
                fact_id="fact-cascade-001",
                project_id=proj_id,
                paper_id=paper_id,
                generation_id="gen-cascade-1",
                event_id=event.id,
                subject_key=f"{proj_id}:method:m",
                subject_name="M",
                subject_type="Method",
                predicate="TESTED",
                object_key=f"{proj_id}:task:t",
                object_name="T",
                object_type="Task",
                page_number=1,
                char_start=0,
                char_end=5,
                exact_quote="quote",
                document_sha256="sha" * 21 + "s",
            )
            db.add(snapshot)
            db.commit()

            # Execute database-level DELETE on project with foreign keys enabled
            db.execute(delete(Project).where(Project.id == proj_id))
            db.commit()

            # Verify cascading deletion in graph_events and graph_fact_snapshots
            events_count = db.scalar(
                select(sa_func.count())
                .select_from(GraphEvent)
                .where(GraphEvent.project_id == proj_id)
            )
            assert events_count == 0

            snapshots_count = db.scalar(
                select(sa_func.count())
                .select_from(GraphFactSnapshot)
                .where(GraphFactSnapshot.project_id == proj_id)
            )
            assert snapshots_count == 0


def test_event_deletion_cascade_and_set_null_semantics():
    """Verify ORM event deletion cascades to snapshots,
    and DB-level event deletion sets event_id to NULL.
    """
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Event Cascade Test")
            db.add(project)
            db.commit()
            db.refresh(project)

            # 1. Test ORM cascade: deleting event via session deletes child snapshots
            paper_id1 = uuid4()
            event1 = GraphEvent(
                project_id=project.id,
                paper_id=paper_id1,
                generation_id="gen-orm-cascade",
            )
            db.add(event1)
            db.commit()
            db.refresh(event1)

            snap1 = GraphFactSnapshot(
                fact_id="fact-orm-cascade-1",
                project_id=project.id,
                paper_id=paper_id1,
                generation_id="gen-orm-cascade",
                event_id=event1.id,
                subject_key=f"{project.id}:s",
                subject_name="S",
                subject_type="Method",
                predicate="PRED",
                object_key=f"{project.id}:o",
                object_name="O",
                object_type="Task",
                page_number=1,
                char_start=0,
                char_end=5,
                exact_quote="quote",
                document_sha256="sha" * 21 + "s",
            )
            db.add(snap1)
            db.commit()

            db.delete(event1)
            db.commit()

            assert db.get(GraphFactSnapshot, "fact-orm-cascade-1") is None

            # 2. Test DB-level ON DELETE SET NULL on event_id foreign key
            paper_id2 = uuid4()
            event2 = GraphEvent(
                project_id=project.id,
                paper_id=paper_id2,
                generation_id="gen-db-setnull",
            )
            db.add(event2)
            db.commit()
            db.refresh(event2)
            saved_event2_id = event2.id

            snap2 = GraphFactSnapshot(
                fact_id="fact-db-setnull-1",
                project_id=project.id,
                paper_id=paper_id2,
                generation_id="gen-db-setnull",
                event_id=saved_event2_id,
                subject_key=f"{project.id}:s2",
                subject_name="S2",
                subject_type="Method",
                predicate="PRED",
                object_key=f"{project.id}:o2",
                object_name="O2",
                object_type="Task",
                page_number=1,
                char_start=0,
                char_end=5,
                exact_quote="quote2",
                document_sha256="sha" * 21 + "s",
            )
            db.add(snap2)
            db.commit()

            # Direct DB delete of event2 triggers ON DELETE SET NULL
            db.execute(delete(GraphEvent).where(GraphEvent.id == saved_event2_id))
            db.commit()

            db.expire_all()
            reloaded_snap2 = db.get(GraphFactSnapshot, "fact-db-setnull-1")
            assert reloaded_snap2 is not None
            assert reloaded_snap2.event_id is None


def test_lease_fencing_and_status_filtering():
    """Verify lease fields, expiry comparisons, and status filtering queries."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)
        command.upgrade(cfg, "head")

        engine = create_sqlite_fk_engine(db_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        with session_factory() as db:
            project = Project(name="Lease Fencing Project")
            db.add(project)
            db.commit()
            db.refresh(project)

            now = datetime.now(UTC)
            expired_time = now - timedelta(minutes=5)
            active_time = now + timedelta(minutes=5)

            # Event 1: PENDING
            e1 = GraphEvent(
                project_id=project.id,
                paper_id=uuid4(),
                generation_id="gen-pending",
                status="PENDING",
            )
            # Event 2: PROCESSING with expired lease
            e2 = GraphEvent(
                project_id=project.id,
                paper_id=uuid4(),
                generation_id="gen-expired",
                status="PROCESSING",
                lease_owner="worker-old",
                lease_expires_at=expired_time,
                attempts=1,
            )
            # Event 3: PROCESSING with active lease
            e3 = GraphEvent(
                project_id=project.id,
                paper_id=uuid4(),
                generation_id="gen-active",
                status="PROCESSING",
                lease_owner="worker-live",
                lease_expires_at=active_time,
                attempts=1,
            )
            # Event 4: FAILED
            e4 = GraphEvent(
                project_id=project.id,
                paper_id=uuid4(),
                generation_id="gen-failed",
                status="FAILED",
                error_code="LLM_RATE_LIMIT",
                error_message="Rate limit exceeded after retries",
                attempts=3,
            )
            # Event 5: COMPLETED
            e5 = GraphEvent(
                project_id=project.id,
                paper_id=uuid4(),
                generation_id="gen-completed",
                status="COMPLETED",
                completed_at=now,
            )
            db.add_all([e1, e2, e3, e4, e5])
            db.commit()

            # Query claimable events:
            # status == 'PENDING' OR (status == 'PROCESSING' AND lease_expires_at <= now)
            claimable = db.scalars(
                select(GraphEvent).where(
                    (GraphEvent.status == "PENDING")
                    | ((GraphEvent.status == "PROCESSING") & (GraphEvent.lease_expires_at <= now))
                )
            ).all()
            claimable_ids = {e.id for e in claimable}
            assert e1.id in claimable_ids
            assert e2.id in claimable_ids
            assert e3.id not in claimable_ids
            assert e4.id not in claimable_ids
            assert e5.id not in claimable_ids

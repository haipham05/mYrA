from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Paper, PaperPage, Project
from app.db.session import get_db
from app.main import app


def test_artifact_revisions_are_immutable_and_project_scoped(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'artifacts.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    project = Project(name="Research artifacts")
    other_project = Project(name="Other project")
    db.add_all([project, other_project])
    db.commit()
    project_id = project.id
    other_project_id = other_project.id

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            create = client.post(
                f"/api/v1/projects/{project_id}/artifacts",
                json={
                    "artifact_type": "report",
                    "title": "First draft",
                    "payload": {
                        "summary": "Grounded findings",
                        "report_markdown": "Grounded findings [E1].",
                    },
                    "scope_snapshot": {"paper_ids": ["paper-a"]},
                    "source_manifest": [{"paper_id": "paper-a", "sha256": "abc"}],
                    "config_snapshot": {"prompt_version": "v1"},
                    "usage": {"total_tokens": None},
                },
            )
            assert create.status_code == 201, create.text
            artifact = create.json()
            assert artifact["latest_revision"] == 1
            artifact_id = artifact["id"]

            next_revision = client.post(
                f"/api/v1/projects/{project_id}/artifacts/{artifact_id}/revisions",
                json={
                    "title": "Revised draft",
                    "payload": {"summary": "Updated findings"},
                    "scope_snapshot": {"paper_ids": ["paper-a"]},
                    "source_manifest": [{"paper_id": "paper-a", "sha256": "def"}],
                    "config_snapshot": {"prompt_version": "v2"},
                    "usage": {"total_tokens": 25},
                },
            )
            assert next_revision.status_code == 200, next_revision.text
            assert next_revision.json()["latest_revision"] == 2

            detail = client.get(f"/api/v1/projects/{project_id}/artifacts/{artifact_id}")
            assert detail.status_code == 200
            assert [item["title"] for item in detail.json()["revisions"]] == [
                "First draft",
                "Revised draft",
            ]
            assert detail.json()["revisions"][0]["payload"]["summary"] == "Grounded findings"
            assert detail.json()["latest"]["source_manifest"][0]["sha256"] == "def"
            listed = client.get(f"/api/v1/projects/{project_id}/artifacts")
            assert listed.status_code == 200
            assert listed.json()[0]["latest_revision"] == 2
            exported = client.get(
                f"/api/v1/projects/{project_id}/artifacts/{artifact_id}/export?format=markdown&revision=1"
            )
            assert exported.status_code == 200
            assert "Grounded findings" in exported.text
            assert 'attachment; filename="artifact-' in exported.headers["content-disposition"]

            cross_project = client.get(
                f"/api/v1/projects/{other_project_id}/artifacts/{artifact_id}"
            )
            assert cross_project.status_code == 404
    finally:
        app.dependency_overrides.clear()
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_reopen_and_export_mark_changed_sources_without_rewriting_report(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'source-status.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    project = Project(name="Source status")
    db.add(project)
    db.flush()
    digest = "a" * 64
    paper = Paper(
        project_id=project.id,
        filename="source.pdf",
        storage_path="source.pdf",
        title="Source paper",
        document_sha256=digest,
        status="READY",
    )
    db.add(paper)
    db.flush()
    db.add(
        PaperPage(
            paper_id=paper.id, page_number=2, width=612, height=792, raw_text="A current quote."
        )
    )
    db.commit()
    project_id, paper_id = project.id, paper.id

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            created = client.post(
                f"/api/v1/projects/{project_id}/artifacts",
                json={
                    "artifact_type": "report",
                    "title": "Report",
                    "payload": {"report_markdown": "A finding from [E1]."},
                    "source_manifest": [
                        {
                            "evidence_id": "E1",
                            "paper_id": str(paper_id),
                            "paper_title": "Source paper",
                            "page_number": 2,
                            "document_sha256": digest,
                            "quote": "A current quote.",
                        }
                    ],
                },
            )
            artifact_id = created.json()["id"]
            current = client.get(f"/api/v1/projects/{project_id}/artifacts/{artifact_id}")
            source = current.json()["latest"]["source_manifest"][0]
            assert source["availability"] == "current"
            assert source["current_anchor"]["source_char_start"] == 0

            db.query(Paper).filter(Paper.id == paper_id).update({Paper.document_sha256: "b" * 64})
            db.commit()

            changed = client.get(f"/api/v1/projects/{project_id}/artifacts/{artifact_id}")
            changed_source = changed.json()["latest"]["source_manifest"][0]
            assert changed_source["availability"] == "unavailable"
            assert "current_anchor" not in changed_source
            assert changed.json()["latest"]["payload"]["report_markdown"] == "A finding from [E1]."

            exported = client.get(
                f"/api/v1/projects/{project_id}/artifacts/{artifact_id}/export?format=markdown"
            )
            assert exported.status_code == 200
            assert "A finding from [E1]." in exported.text
            assert "source changed or unavailable" in exported.text
    finally:
        app.dependency_overrides.clear()
        db.close()
        Base.metadata.drop_all(engine)
        engine.dispose()

"""Opt-in Milestone 1 smoke using real providers and disposable data.

Run inside the API image with its existing environment, ADC mount, pinned model
cache, and a read-only PDF fixture. This overrides only the process database URL
to a temporary SQLite file; it never changes the repository's .env or Supabase.
"""

import argparse
import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path
from uuid import UUID, uuid4


def require_status(response, expected: int, step: str) -> dict:
    if response.status_code != expected:
        raise RuntimeError(f"{step} returned HTTP {response.status_code}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args()
    if not args.confirm_live:
        parser.error(
            "--confirm-live is required: this uploads one disposable GCS PDF and calls DeepSeek"
        )
    if not args.pdf.is_file():
        parser.error("PDF fixture is missing")
    if not os.getenv("GCS_BUCKET_NAME") or not os.getenv("DEEPSEEK_API_KEY"):
        parser.error("GCS_BUCKET_NAME and DEEPSEEK_API_KEY must be configured")

    with tempfile.TemporaryDirectory(prefix="myra-live-smoke-") as temporary:
        os.chdir(temporary)
        os.environ["DATABASE_URL"] = f"sqlite:///{temporary}/smoke.db"
        os.environ["MYRA_USE_DOCLING"] = "true"
        os.environ["MYRA_EMBEDDING_PROVIDER"] = "bge-m3"
        os.environ["MYRA_RERANKER_PROVIDER"] = "bge"
        os.environ["MYRA_LLM_MODE"] = "deepseek"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Import only after isolating the database: app.db.session creates its engine at import.
        from app.crud.job import claim_next_job, get_job
        from app.db.models import PaperChunk, PaperElement
        from app.db.session import SessionLocal, create_tables, engine
        from app.main import app
        from app.services.chat_service import check_claim_support
        from app.services.ingestion import IngestionPipeline
        from app.services.llm import get_llm_provider
        from app.storage.factory import get_storage
        from fastapi.testclient import TestClient

        create_tables()
        paper_id: UUID | None = None
        project_id: UUID | None = None
        try:
            with TestClient(app) as client:
                project = require_status(
                    client.post(
                        "/api/v1/projects",
                        json={"name": f"m1-live-smoke-{uuid4().hex[:8]}"},
                    ),
                    201,
                    "project creation",
                )
                project_id = UUID(project["id"])
                upload = require_status(
                    client.post(
                        f"/api/v1/projects/{project_id}/papers",
                        files={
                            "file": (
                                args.pdf.name,
                                args.pdf.read_bytes(),
                                "application/pdf",
                            )
                        },
                    ),
                    202,
                    "PDF upload",
                )
                paper_id = UUID(upload["paper_id"])
                job_id = UUID(upload["job_id"])

                with SessionLocal() as db:
                    job = claim_next_job(db, worker_id="m1-live-smoke")
                    if job is None or job.id != job_id:
                        raise RuntimeError(
                            "the disposable ingestion job was not claimed"
                        )
                    asyncio.run(
                        IngestionPipeline().process_paper(
                            db,
                            paper_id=paper_id,
                            job_id=job_id,
                            worker_id="m1-live-smoke",
                        )
                    )
                    db.expire_all()
                    job = get_job(db, job_id)
                    if job is None or job.status != "COMPLETED":
                        raise RuntimeError(
                            f"ingestion did not complete: {job.status if job else 'missing'}"
                        )
                    parser_versions = {
                        row[0]
                        for row in db.query(PaperElement.parser_version)
                        .filter(PaperElement.paper_id == paper_id)
                        .all()
                    }
                    embedding_models = {
                        row[0]
                        for row in db.query(PaperChunk.embedding_model)
                        .filter(PaperChunk.paper_id == paper_id)
                        .all()
                    }
                    if not any(
                        version and version.startswith("docling-")
                        for version in parser_versions
                    ):
                        raise RuntimeError("Docling did not parse the uploaded PDF")
                    if "BAAI/bge-m3" not in embedding_models:
                        raise RuntimeError("BGE-M3 did not index the uploaded PDF")

                conversation = require_status(
                    client.post(
                        f"/api/v1/projects/{project_id}/conversations",
                        json={"title": "Provider smoke"},
                    ),
                    201,
                    "conversation creation",
                )
                llm = get_llm_provider()
                original_generate = llm.generate
                generated: dict[str, str] = {}

                async def capture_generation(
                    system_prompt: str, user_prompt: str
                ) -> str:
                    generated["text"] = await original_generate(
                        system_prompt, user_prompt
                    )
                    return generated["text"]

                llm.generate = capture_generation
                answer = require_status(
                    client.post(
                        f"/api/v1/conversations/{conversation['id']}/messages",
                        json={"content": "What does the BERT acronym stand for?"},
                    ),
                    200,
                    "grounded answer",
                )
                citations = answer["citations"]
                if not citations:
                    evidence_summary = [
                        (
                            item["page_number"],
                            [anchor["anchor_status"] for anchor in item["anchors"]],
                        )
                        for item in answer["evidence"]
                    ]
                    raw = generated.get("text", "")
                    markers = re.findall(r"\[E\d+\]", raw)
                    supported = [
                        check_claim_support(
                            re.sub(r"\[E\d+\]", "", raw).strip(), item["quote"]
                        )
                        for item in answer["evidence"]
                    ]
                    raise RuntimeError(
                        "DeepSeek answer had no accepted citation; "
                        f"markers={markers}, support={supported}, evidence={evidence_summary}"
                    )
                citation = citations[0]
                if (
                    UUID(citation["paper_id"]) != paper_id
                    or citation["page_number"] != 1
                    or citation["anchor_status"] != "verified"
                ):
                    raise RuntimeError(
                        "citation did not resolve to verified source page 1"
                    )
                document = client.get(f"/api/v1/papers/{paper_id}/document")
                if document.status_code != 200:
                    raise RuntimeError("served PDF could not be fetched")
                if (
                    citation["document_sha256"]
                    != hashlib.sha256(document.content).hexdigest()
                ):
                    raise RuntimeError("citation hash did not match the served PDF")
                print("live_vertical_slice PASS")
                print("parser", sorted(parser_versions))
                print("embedding", sorted(embedding_models))
                print("citation_page", citation["page_number"], "verified", True)
        finally:
            if project_id is not None and paper_id is not None:
                key = f"papers/{project_id}/{paper_id}.pdf"
                asyncio.run(get_storage().delete(key))
            engine.dispose()


if __name__ == "__main__":
    main()

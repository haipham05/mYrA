import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

# Add apps/api to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "apps" / "api"))
sys.path.insert(0, str(repo_root / "tests" / "retrieval_eval"))

from app.config import Settings
from app.crud.chat import create_conversation
from app.crud.paper import create_paper_with_job
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import Project
from app.db.session import SessionLocal, create_tables
from app.schemas.evidence import AnchorStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.embedding import get_embedding_provider
from app.services.ingestion import IngestionPipeline
from app.services.llm import get_llm_provider
from app.services.retrieval import HybridRetriever, get_reranker
from app.storage.factory import get_storage, set_storage
from app.storage.local import LocalStorage
from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker


async def setup_gold_corpus_from_real_pdfs(db, fixtures_dir: Path):
    """Ingest real fixture PDFs via the actual IngestionPipeline."""
    project = create_project(
        db, ProjectCreate(name=f"Gold Benchmark Project {uuid4().hex[:6]}")
    )
    storage = get_storage()
    pipeline = IngestionPipeline()

    paper_objs = []
    for p_spec in GOLD_PAPERS:
        pdf_path = fixtures_dir / p_spec["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fixture PDF not found: {pdf_path}")

        pdf_bytes = pdf_path.read_bytes()
        storage_key = f"gold/{project.id}/{p_spec['filename']}"
        await storage.put(storage_key, pdf_bytes)

        paper, job = create_paper_with_job(
            db,
            project_id=project.id,
            filename=p_spec["filename"],
            storage_path=storage_key,
            status=PaperStatus.PROCESSING,
        )
        paper_objs.append(paper)

        # Run actual end-to-end ingestion pipeline on real PDF
        await pipeline.process_paper(db, paper_id=paper.id, job_id=job.id)
        db.refresh(paper)

    return project, paper_objs


async def run_evaluation(allow_live: bool = False):
    settings = Settings.from_environment()
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"

    temp_dir = None
    isolated_engine = None
    db = None
    project = None

    try:
        # 1. Isolate environment unless explicitly allowed
        is_remote_db = any(
            x in settings.database_url.lower()
            for x in ("supabase", "postgres", "amazonaws")
        )
        if is_remote_db and not allow_live:
            print(
                "Notice: Remote database detected. Initializing isolated disposable SQLite and local storage..."
            )
            temp_dir = tempfile.TemporaryDirectory()
            sqlite_path = Path(temp_dir.name) / "eval_isolated.db"
            isolated_engine = create_engine(
                f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=isolated_engine)
            eval_session_factory = sessionmaker(
                autocommit=False, autoflush=False, bind=isolated_engine
            )
            db = eval_session_factory()
            isolated_storage = LocalStorage(
                base_dir=str(Path(temp_dir.name) / "storage")
            )
            set_storage(isolated_storage)
        else:
            create_tables()
            db = SessionLocal()

        embedder = get_embedding_provider()
        reranker = get_reranker()
        llm = get_llm_provider()
        dialect = db.get_bind().dialect.name

        print("================================================================")
        print("      mYrA REAL-PDF RETRIEVAL & CITATION EVALUATION GATE        ")
        print("================================================================")
        print("Runtime Environment:")
        print(f"  - Database:           {dialect}")
        print(
            f"  - Embedding Provider: {embedder.model_name} ({embedder.model_version})"
        )
        print(f"  - Reranker:           {reranker.model_name}")
        print(f"  - LLM Provider:       {llm.provider_name}")
        print(f"Loading & Ingesting {len(GOLD_PAPERS)} real PDF fixture files...")

        project, paper_objs = await setup_gold_corpus_from_real_pdfs(db, fixtures_dir)
        retriever = HybridRetriever(top_candidates=40, top_evidence=10, rrf_k=60)
        chat_service = ChatService(retriever=retriever)

        # Create evaluation conversation
        conv = create_conversation(db, project_id=project.id, title="Gold Benchmark QA")

        recall_at_5 = 0
        recall_at_10 = 0
        reciprocal_ranks = []
        citation_precisions = []
        exact_highlight_successes = 0
        total_citations_evaluated = 0
        questions_with_citations = 0
        latencies = []

        print("\nRunning question evaluation suite against ingested PDFs:")
        for q in GOLD_QUESTIONS:
            target_paper = paper_objs[q["target_paper_idx"]]
            target_page = q["target_page"]
            key_phrase = q["key_phrase"]

            t0 = time.perf_counter()
            results = retriever.retrieve(db, project_id=project.id, query=q["query"])
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000
            latencies.append(latency)

            # 1. Evaluate target chunk retrieval
            found_rank = None
            for rank, item in enumerate(results):
                if (
                    item.paper_id == target_paper.id
                    and item.page_number == target_page
                    and key_phrase.lower() in item.quote.lower()
                ):
                    found_rank = rank + 1
                    break

            if found_rank is not None:
                if found_rank <= 5:
                    recall_at_5 += 1
                if found_rank <= 10:
                    recall_at_10 += 1
                reciprocal_ranks.append(1.0 / found_rank)
                rank_str = f"Rank {found_rank}"
            else:
                reciprocal_ranks.append(0.0)
                rank_str = "NOT in top 10"

            # 2. Run ChatService to generate answer and validate citations
            chat_resp = await chat_service.answer_question(db, conv.id, q["query"])

            # 3. Evaluate Citation Precision & Exact Highlight Resolution
            question_citations_valid = 0
            if chat_resp.citations:
                questions_with_citations += 1
                for cite in chat_resp.citations:
                    total_citations_evaluated += 1
                    quote = cite.quote.strip()
                    if (
                        quote
                        and cite.anchor_status == AnchorStatus.VERIFIED
                        and cite.anchors
                        and any(
                            a.source_char_start is not None
                            and a.source_char_end is not None
                            for a in cite.anchors
                        )
                    ):
                        exact_highlight_successes += 1
                        question_citations_valid += 1

                precision = question_citations_valid / len(chat_resp.citations)
                citation_precisions.append(precision)

            print(
                f"  [{q['id']}] {rank_str:<14} | Citations: {len(chat_resp.citations)} "
                f"| Latency: {latency:.2f}ms"
            )

        n = len(GOLD_QUESTIONS)
        r5 = recall_at_5 / n
        r10 = recall_at_10 / n
        mrr = sum(reciprocal_ranks) / n
        emitted_precision = (
            sum(citation_precisions) / len(citation_precisions)
            if citation_precisions
            else 0.0
        )
        citation_coverage = questions_with_citations / n
        highlight_rate = exact_highlight_successes / max(1, total_citations_evaluated)
        avg_lat = sum(latencies) / n

        print("\n================================================================")
        print("                     BENCHMARK RESULTS                         ")
        print("================================================================")
        print(f"Total Questions Evaluated:     {n}")
        print(f"Recall@5:                      {r5 * 100:.2f}% ({recall_at_5}/{n})")
        print(f"Recall@10:                     {r10 * 100:.2f}% ({recall_at_10}/{n})")
        print(f"Mean Reciprocal Rank (MRR):    {mrr:.4f}")
        print(
            f"Citation Anchor Precision:     {emitted_precision * 100:.2f}% (on emitted citations)"
        )
        print(
            f"Citation Coverage:             {citation_coverage * 100:.2f}% ({questions_with_citations}/{n})"
        )
        print(
            f"Exact Highlight Success Rate:  {highlight_rate * 100:.2f}% "
            f"({exact_highlight_successes}/{total_citations_evaluated})"
        )
        print(f"Average Query Latency:         {avg_lat:.2f} ms")
        print("Active Evaluated Providers:")
        print(f"  - Database:           {dialect}")
        print(f"  - Embedder:           {embedder.model_name}")
        print(f"  - Reranker:           {reranker.model_name}")
        print(f"  - LLM:                {llm.provider_name}")
        print("================================================================")

        assert r5 >= 0.80, f"Recall@5 ({r5:.2f}) must be >= 0.80"
        assert r10 >= 0.90, f"Recall@10 ({r10:.2f}) must be >= 0.90"
        assert mrr >= 0.70, f"MRR ({mrr:.4f}) must be >= 0.70"
        assert emitted_precision >= 0.90, (
            f"Citation Precision ({emitted_precision:.2f}) must be >= 0.90"
        )
        assert highlight_rate >= 0.90, (
            f"Highlight Rate ({highlight_rate:.2f}) must be >= 0.90"
        )
        assert citation_coverage >= 0.80, (
            f"Citation Coverage ({citation_coverage:.2f}) must be >= 0.80"
        )
        print("\nALL RELEASE EVALUATION ACCEPTANCE GATES PASSED!")

    finally:
        # Guarantee cleanup of test project and isolated resources
        if project and db:
            try:
                db.query(Project).filter(Project.id == project.id).delete()
                db.commit()
            except (SQLAlchemyError, OSError) as err:
                print(f"Cleanup warning: {err}")

        if db:
            db.close()
        if isolated_engine:
            isolated_engine.dispose()
        set_storage(None)
        if temp_dir:
            temp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description="mYrA Retrieval Evaluation Benchmark")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow running against configured live database and storage (MYRA_ALLOW_LIVE_EVAL=1)",
    )
    args = parser.parse_args()
    allow_live = args.live or (
        os.getenv("MYRA_ALLOW_LIVE_EVAL", "0").lower() in ("1", "true")
    )
    asyncio.run(run_evaluation(allow_live=allow_live))


if __name__ == "__main__":
    main()

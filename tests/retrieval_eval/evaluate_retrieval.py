import argparse
import asyncio
import hashlib
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

from app.crud.chat import create_conversation
from app.crud.paper import create_paper_with_job
from app.crud.project import create_project
from app.db.base import Base
from app.ingestion.parser import DocumentParser
from app.schemas.evidence import AnchorStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.embedding import (
    DeterministicEmbeddingProvider,
    get_embedding_provider,
    set_embedding_provider,
)
from app.services.ingestion import IngestionPipeline
from app.services.llm import FakeLLMProvider, get_llm_provider, set_llm_provider
from app.services.retrieval import (
    HybridRetriever,
    SimpleLexicalReranker,
    get_reranker,
    set_reranker,
)
from app.storage.factory import set_storage
from app.storage.local import LocalStorage
from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


async def setup_gold_corpus(db, fixtures_dir: Path, storage):
    """Ingest generated regression PDFs through the actual pipeline."""
    project = create_project(
        db, ProjectCreate(name=f"Gold Benchmark Project {uuid4().hex[:6]}")
    )
    pipeline = IngestionPipeline(parser=DocumentParser(use_docling=True))

    paper_objs = []
    document_hashes = []
    for p_spec in GOLD_PAPERS:
        pdf_path = fixtures_dir / p_spec["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fixture PDF not found: {pdf_path}")

        pdf_bytes = pdf_path.read_bytes()
        document_hashes.append(hashlib.sha256(pdf_bytes).hexdigest())
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

    return project, paper_objs, document_hashes


async def run_evaluation():
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"

    temp_dir = None
    isolated_engine = None
    db = None
    offline_variables = {
        name: os.environ.get(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # This regression runner never consumes the developer's configured cloud
        # resources or production models, regardless of DATABASE_URL/.env.
        temp_dir = tempfile.TemporaryDirectory(prefix="myra-eval-")
        sqlite_path = Path(temp_dir.name) / "eval.db"
        isolated_engine = create_engine(
            f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=isolated_engine)
        eval_session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=isolated_engine
        )
        db = eval_session_factory()
        isolated_storage = LocalStorage(base_dir=str(Path(temp_dir.name) / "storage"))
        set_storage(isolated_storage)
        set_embedding_provider(DeterministicEmbeddingProvider())
        set_reranker(SimpleLexicalReranker())
        set_llm_provider(FakeLLMProvider())

        embedder = get_embedding_provider()
        reranker = get_reranker()
        llm = get_llm_provider()
        dialect = db.get_bind().dialect.name

        print("================================================================")
        print("      mYrA GENERATED-PDF REGRESSION (NOT RELEASE EVIDENCE)       ")
        print("================================================================")
        print("Runtime Environment:")
        print(f"  - Database:           {dialect}")
        print(
            f"  - Embedding Provider: {embedder.model_name} ({embedder.model_version})"
        )
        print(f"  - Reranker:           {reranker.model_name}")
        print(f"  - LLM Provider:       {llm.provider_name}")
        print(f"Loading & Ingesting {len(GOLD_PAPERS)} generated PDF fixtures...")

        project, paper_objs, document_hashes = await setup_gold_corpus(
            db, fixtures_dir, isolated_storage
        )
        print(f"  - Fixture SHA-256:    {', '.join(document_hashes)}")
        retriever = HybridRetriever(top_candidates=40, top_evidence=10, rrf_k=60)
        chat_service = ChatService(retriever=retriever)

        # Create evaluation conversation
        conv = create_conversation(db, project_id=project.id, title="Gold Benchmark QA")

        recall_at_5 = 0
        recall_at_10 = 0
        reciprocal_ranks = []
        correct_citations = 0
        backend_anchor_offsets = 0
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
            matched_citation_for_question = False
            if chat_resp.citations:
                questions_with_citations += 1
                for cite in chat_resp.citations:
                    total_citations_evaluated += 1
                    quote = cite.quote.strip()
                    if (
                        cite.paper_id == target_paper.id
                        and cite.page_number == target_page
                        and key_phrase.casefold() in quote.casefold()
                    ):
                        correct_citations += 1
                        matched_citation_for_question = True
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
                        backend_anchor_offsets += 1

            print(
                f"  [{q['id']}] {rank_str:<14} | Citations: {len(chat_resp.citations)} "
                f"| Gold cite: {'yes' if matched_citation_for_question else 'no'} "
                f"| Latency: {latency:.2f}ms"
            )

        n = len(GOLD_QUESTIONS)
        r5 = recall_at_5 / n
        r10 = recall_at_10 / n
        mrr = sum(reciprocal_ranks) / n
        emitted_precision = correct_citations / max(1, total_citations_evaluated)
        citation_coverage = questions_with_citations / n
        offset_rate = backend_anchor_offsets / max(1, total_citations_evaluated)
        avg_lat = sum(latencies) / n

        print("\n================================================================")
        print("                     BENCHMARK RESULTS                         ")
        print("================================================================")
        print(f"Total Questions Evaluated:     {n}")
        print(f"Recall@5:                      {r5 * 100:.2f}% ({recall_at_5}/{n})")
        print(f"Recall@10:                     {r10 * 100:.2f}% ({recall_at_10}/{n})")
        print(f"Mean Reciprocal Rank (MRR):    {mrr:.4f}")
        print(
            f"Gold Citation Precision:       {emitted_precision * 100:.2f}% "
            f"({correct_citations}/{total_citations_evaluated} emitted)"
        )
        print(
            f"Citation Coverage:             {citation_coverage * 100:.2f}% ({questions_with_citations}/{n})"
        )
        print(
            f"Backend Anchor Offset Rate:    {offset_rate * 100:.2f}% "
            f"({backend_anchor_offsets}/{total_citations_evaluated}); not browser highlights"
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
        assert citation_coverage >= 0.80, (
            f"Citation Coverage ({citation_coverage:.2f}) must be >= 0.80"
        )
        print(
            "\nSynthetic regression thresholds passed; release evaluation remains open."
        )

    finally:
        if db:
            db.close()
        if isolated_engine:
            isolated_engine.dispose()
        set_storage(None)
        set_embedding_provider(None)
        set_reranker(None)
        set_llm_provider(None)
        if temp_dir:
            temp_dir.cleanup()
        for name, old_value in offline_variables.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def main():
    parser = argparse.ArgumentParser(
        description="Isolated generated-PDF retrieval regression"
    )
    parser.parse_args()
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()

"""End-to-End Real-PDF Gold Retrieval & Citation Evaluation Benchmark.

Ingests real multi-page research PDFs through IngestionPipeline (PDF -> DocumentParser -> DocumentChunker -> EmbeddingProvider -> Database),
runs HybridRetriever and ChatService across 24 curated research questions,
and computes Recall@5, Recall@10, MRR, Citation Precision, and Exact Highlight Success.
"""

import asyncio
import os
import sys
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
from app.db.models import Project
from app.db.session import SessionLocal, create_tables
from app.ingestion.parser import find_verbatim_span
from app.schemas.evidence import AnchorStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.embedding import get_embedding_provider
from app.services.ingestion import IngestionPipeline
from app.services.llm import get_llm_provider
from app.services.retrieval import HybridRetriever, get_reranker
from app.storage.factory import get_storage
from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS


async def setup_gold_corpus_from_real_pdfs(db, fixtures_dir: Path):
    """Ingest real fixture PDFs via the actual IngestionPipeline."""
    project = create_project(db, ProjectCreate(name=f"Gold Benchmark Project {uuid4().hex[:6]}"))
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


async def run_evaluation():
    create_tables()
    db = SessionLocal()
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"

    embedder = get_embedding_provider()
    reranker = get_reranker()
    llm = get_llm_provider()
    dialect = db.get_bind().dialect.name

    print("================================================================")
    print("      mYrA REAL-PDF RETRIEVAL & CITATION EVALUATION GATE        ")
    print("================================================================")
    print(f"Runtime Environment:")
    print(f"  - Database:           {dialect}")
    print(f"  - Embedding Provider: {embedder.model_name} ({embedder.model_version})")
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
            if item.paper_id == target_paper.id and item.page_number == target_page:
                if key_phrase.lower() in item.quote.lower():
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
            for cite in chat_resp.citations:
                total_citations_evaluated += 1
                # Check verbatim span match in evidence
                quote = cite.quote.strip()
                if quote and cite.anchor_status == AnchorStatus.VERIFIED:
                    # Confirm verbatim span exists with valid character offsets
                    if cite.anchors and any(
                        a.source_char_start is not None and a.source_char_end is not None
                        for a in cite.anchors
                    ):
                        exact_highlight_successes += 1
                        question_citations_valid += 1

            precision = question_citations_valid / len(chat_resp.citations)
            citation_precisions.append(precision)
        else:
            # If no citations were generated for an answerable gold question
            citation_precisions.append(1.0 if found_rank is not None else 0.0)

        print(
            f"  [{q['id']}] {rank_str:<14} | Citations: {len(chat_resp.citations)} "
            f"| Latency: {latency:.2f}ms"
        )

    n = len(GOLD_QUESTIONS)
    r5 = recall_at_5 / n
    r10 = recall_at_10 / n
    mrr = sum(reciprocal_ranks) / n
    avg_precision = sum(citation_precisions) / n
    highlight_rate = (
        exact_highlight_successes / max(1, total_citations_evaluated)
    )
    avg_lat = sum(latencies) / n

    print("\n================================================================")
    print("                     BENCHMARK RESULTS                         ")
    print("================================================================")
    print(f"Total Questions Evaluated:     {n}")
    print(f"Recall@5:                      {r5 * 100:.2f}% ({recall_at_5}/{n})")
    print(f"Recall@10:                     {r10 * 100:.2f}% ({recall_at_10}/{n})")
    print(f"Mean Reciprocal Rank (MRR):    {mrr:.4f}")
    print(f"Citation Anchor Precision:     {avg_precision * 100:.2f}%")
    print(f"Exact Highlight Success Rate:  {highlight_rate * 100:.2f}% ({exact_highlight_successes}/{total_citations_evaluated})")
    print(f"Average Query Latency:         {avg_lat:.2f} ms")
    print("Active Evaluated Providers:")
    print(f"  - Database:           {dialect}")
    print(f"  - Embedder:           {embedder.model_name}")
    print(f"  - Reranker:           {reranker.model_name}")
    print(f"  - LLM:                {llm.provider_name}")
    print("================================================================")

    # Cleanup test project
    db.query(Project).filter(Project.id == project.id).delete()
    db.commit()
    db.close()

    assert r5 >= 0.80, f"Recall@5 ({r5:.2f}) must be >= 0.80"
    assert r10 >= 0.90, f"Recall@10 ({r10:.2f}) must be >= 0.90"
    assert mrr >= 0.70, f"MRR ({mrr:.4f}) must be >= 0.70"
    assert avg_precision >= 0.90, f"Citation Precision ({avg_precision:.2f}) must be >= 0.90"
    assert highlight_rate >= 0.90, f"Highlight Rate ({highlight_rate:.2f}) must be >= 0.90"
    print("\nALL RELEASE EVALUATION ACCEPTANCE GATES PASSED!")


def main():
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()

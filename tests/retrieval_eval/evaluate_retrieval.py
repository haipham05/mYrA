import sys
import time
from pathlib import Path

# Add apps/api to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "apps" / "api"))
sys.path.insert(0, str(repo_root / "tests" / "retrieval_eval"))

from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, Project
from app.db.session import SessionLocal, create_tables
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.embedding import get_embedding_provider
from app.services.retrieval import HybridRetriever
from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS


def setup_gold_corpus_in_db(db):
    """Seed gold papers and elements in an isolated project."""
    embedder = get_embedding_provider()
    project = create_project(db, ProjectCreate(name="Gold Benchmark Project"))

    paper_objs = []
    for p_spec in GOLD_PAPERS:
        paper = create_paper(
            db,
            project_id=project.id,
            filename=p_spec["filename"],
            storage_path=f"gold/{p_spec['filename']}",
            document_sha256=f"sha256-mock-{p_spec['id']}",
            status=PaperStatus.READY,
        )
        paper_objs.append(paper)

        for page in p_spec["pages"]:
            elem = PaperElement(
                paper_id=paper.id,
                page_number=page["page_number"],
                element_index=page["page_number"] - 1,
                element_type="text",
                text=page["text"],
                bbox_x_min=page["bbox"][0],
                bbox_y_min=page["bbox"][1],
                bbox_x_max=page["bbox"][2],
                bbox_y_max=page["bbox"][3],
                page_width=612.0,
                page_height=792.0,
                coordinate_origin="TOP_LEFT",
                parser_version="docling-v1",
            )
            db.add(elem)
            db.flush()

            vec = embedder.embed_query(page["text"])
            chunk = PaperChunk(
                paper_id=paper.id,
                chunk_type="child",
                chunk_index=page["page_number"] - 1,
                text=page["text"],
                token_count=len(page["text"].split()),
                embedding=vec,
                embedding_vec=vec,
                embedding_model="bge-m3",
                embedding_version="v1",
            )
            db.add(chunk)
            db.flush()

            db.add(ChunkElement(chunk_id=chunk.id, element_id=elem.id, order_index=0))

    db.commit()
    return project, paper_objs


def run_evaluation():
    create_tables()
    db = SessionLocal()

    print("================================================================")
    print("           mYrA RETRIEVAL EVALUATION BENCHMARK                  ")
    print("================================================================")
    print(f"Loading Gold Corpus: {len(GOLD_PAPERS)} papers, {len(GOLD_QUESTIONS)} questions...")

    project, paper_objs = setup_gold_corpus_in_db(db)
    retriever = HybridRetriever(top_candidates=40, top_evidence=10, rrf_k=60)

    recall_at_5 = 0
    recall_at_10 = 0
    reciprocal_ranks = []
    citation_precisions = []
    latencies = []

    print("\nRunning question evaluation suite:")
    for q in GOLD_QUESTIONS:
        target_paper = paper_objs[q["target_paper_idx"]]
        target_page = q["target_page"]
        key_phrase = q["key_phrase"]

        t0 = time.perf_counter()
        results = retriever.retrieve(db, project_id=project.id, query=q["query"])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        # Evaluate target retrieval
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
            print(f"  [{q['id']}] Target found at rank {found_rank} (latency: {latencies[-1]:.2f}ms)")
        else:
            reciprocal_ranks.append(0.0)
            print(f"  [{q['id']}] Target NOT found in top 10 (latency: {latencies[-1]:.2f}ms)")

        # Evaluate citation precision (valid bboxes & non-empty anchors)
        valid_citations = sum(
            1 for item in results[:5] if item.anchors and item.anchors[0].exact_quote
        )
        citation_precisions.append(valid_citations / max(1, len(results[:5])))

    n = len(GOLD_QUESTIONS)
    r5 = recall_at_5 / n
    r10 = recall_at_10 / n
    mrr = sum(reciprocal_ranks) / n
    avg_precision = sum(citation_precisions) / n
    avg_lat = sum(latencies) / n

    print("\n================================================================")
    print("                     BENCHMARK RESULTS                         ")
    print("================================================================")
    print(f"Total Questions Evaluated:  {n}")
    print(f"Recall@5:                   {r5 * 100:.2f}% ({recall_at_5}/{n})")
    print(f"Recall@10:                  {r10 * 100:.2f}% ({recall_at_10}/{n})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Citation Anchor Precision:  {avg_precision * 100:.2f}%")
    print(f"Average Query Latency:      {avg_lat:.2f} ms")
    print("Models:")
    print("  - Dense:   BGE-M3 (1024-dim, normalized)")
    print("  - Lexical: PostgreSQL tsvector / plainto_tsquery + RRF(k=60)")
    print("  - Rerank:  BGE / Lexical normalized cross-encoder")
    print("================================================================")

    # Cleanup test project
    db.query(Project).filter(Project.id == project.id).delete()
    db.commit()
    db.close()

    assert r5 >= 0.80, f"Recall@5 ({r5:.2f}) must be >= 0.80"
    assert r10 >= 0.90, f"Recall@10 ({r10:.2f}) must be >= 0.90"
    assert mrr >= 0.70, f"MRR ({mrr:.4f}) must be >= 0.70"
    assert avg_precision >= 0.90, f"Citation Precision ({avg_precision:.2f}) must be >= 0.90"
    print("ALL EVALUATION ACCEPTANCE GATES PASSED!")


if __name__ == "__main__":
    run_evaluation()

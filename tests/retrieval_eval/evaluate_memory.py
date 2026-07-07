import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

# Add apps/api to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "apps" / "api"))
sys.path.insert(0, str(repo_root / "tests" / "retrieval_eval"))

from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import ChunkElement, PaperChunk, PaperElement, PaperPage
from app.schemas.memory import (
    MemoryCreate,
    MemorySourceCreate,
    MemorySourceType,
    MemoryStatus,
    MemoryType,
)
from app.schemas.project import ProjectCreate
from app.services import embedding as embedding_module
from app.services import llm as llm_module
from app.services import retrieval as retrieval_module
from app.services.embedding import (
    DeterministicEmbeddingProvider,
    set_embedding_provider,
)
from app.services.llm import FakeLLMProvider, set_llm_provider
from app.services.memory_service import (
    consolidate_memory_candidate,
    resolve_paper_memory_source,
    retrieve_project_memories,
)
from app.services.retrieval import SimpleLexicalReranker, set_reranker
from app.storage import factory as storage_factory
from app.storage.factory import set_storage
from app.storage.local import LocalStorage
from fixture_manifest import MANIFEST_PATH, write_report
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def run_memory_evaluation():
    """Run the benchmark offline and restore process-global configuration."""
    original_env = {
        key: os.environ.get(key)
        for key in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "MYRA_EMBEDDING_PROVIDER",
            "MYRA_RERANKER_PROVIDER",
            "MYRA_LLM_MODE",
        )
    }
    prior_storage = storage_factory._storage_instance
    prior_embedding = embedding_module._default_embedding_provider
    prior_reranker = retrieval_module._reranker_instance
    prior_llm = llm_module._llm_instance
    try:
        for key, value in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MYRA_EMBEDDING_PROVIDER": "deterministic",
            "MYRA_RERANKER_PROVIDER": "simple-lexical",
            "MYRA_LLM_MODE": "test",
        }.items():
            os.environ[key] = value
        set_embedding_provider(DeterministicEmbeddingProvider())
        set_reranker(SimpleLexicalReranker())
        set_llm_provider(FakeLLMProvider())
        with tempfile.TemporaryDirectory(prefix="myra-memory-eval-") as temp_dir:
            set_storage(LocalStorage(base_dir=str(Path(temp_dir) / "storage")))
            _run_memory_evaluation(Path(temp_dir))
    finally:
        set_storage(prior_storage)
        set_embedding_provider(prior_embedding)
        set_reranker(prior_reranker)
        set_llm_provider(prior_llm)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_memory_evaluation(temp_dir: Path) -> None:
    """Run the isolated memory retrieval and provenance cases."""
    print("=" * 70)
    print("mYrA Long-Term Research Memory Benchmark (Milestone 9 / D1)")
    print("=" * 70)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # 1. Setup disposable SQLite database
    engine = create_engine(f"sqlite:///{temp_dir / 'memory.db'}")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # 2. Setup target project and foreign project
    project = create_project(db, ProjectCreate(name="Benchmark Evaluation Project"))
    foreign_proj = create_project(db, ProjectCreate(name="Foreign Unrelated Project"))

    # Create paper with verified page text for paper fact provenance
    paper = create_paper(
        db,
        project_id=project.id,
        filename="benchmark_paper.pdf",
        storage_path="papers/benchmark_paper.pdf",
        status="READY",
    )
    paper.document_sha256 = hashlib.sha256(
        b"synthetic-memory-evaluation-paper-v1"
    ).hexdigest()
    db.commit()
    page4 = PaperPage(
        paper_id=paper.id,
        page_number=4,
        width=612.0,
        height=792.0,
        raw_text="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
    )
    db.add(page4)
    db.commit()

    element = PaperElement(
        paper_id=paper.id,
        page_number=4,
        element_index=40,
        element_type="paragraph",
        text=page4.raw_text,
        bbox_x_min=40.0,
        bbox_y_min=100.0,
        bbox_x_max=570.0,
        bbox_y_max=200.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="synthetic-eval-v1",
    )
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=4,
        text=page4.raw_text,
    )
    db.add_all([element, chunk])
    db.flush()
    db.add(ChunkElement(chunk_id=chunk.id, element_id=element.id, order_index=0))
    db.commit()

    # 3. Ingest labeled sequence
    # 3.1 Initial Decision 1
    cand_dec1 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Metric Decision: AURC vs ECE",
        content="We choose AURC metric over ECE for primary model calibration evaluation.",
        importance=0.9,
        confidence=1.0,
    )
    mem_dec1 = consolidate_memory_candidate(
        db, project_id=project.id, candidate=cand_dec1
    )

    # 3.2 Decision 2
    cand_dec2 = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Optimizer Selection",
        content="Choose AdamW optimizer with learning rate 1e-4 and cosine decay.",
        importance=0.7,
        confidence=1.0,
    )
    mem_dec2 = consolidate_memory_candidate(
        db, project_id=project.id, candidate=cand_dec2
    )

    # 3.3 Preference
    cand_pref = MemoryCreate(
        memory_type=MemoryType.PREFERENCE,
        title="LaTeX Formatting Preference",
        content="Always format LaTeX tables with booktabs without vertical rules.",
        importance=0.6,
        confidence=1.0,
    )
    mem_pref = consolidate_memory_candidate(
        db, project_id=project.id, candidate=cand_pref
    )

    # 3.4 Valid Paper Fact with verified paper quote
    cand_paper_fact = MemoryCreate(
        memory_type=MemoryType.PAPER_FACT,
        title="Attention Representation Subspaces",
        content="Multi-head attention allows the model to attend to information from different representation subspaces.",
        importance=0.8,
        confidence=1.0,
        sources=[
            MemorySourceCreate(
                source_type=MemorySourceType.PAPER_CHUNK,
                paper_id=paper.id,
                page_number=4,
                quote_text="Multi-head attention allows the model to jointly attend to information from different representation subspaces",
            )
        ],
    )
    paper_fact_source = cand_paper_fact.sources[0]
    verified_evidence, verified_anchor, anchor_status = resolve_paper_memory_source(
        db, project_id=project.id, source=paper_fact_source
    )
    if (
        verified_evidence is None
        or verified_anchor is None
        or verified_anchor.exact_quote != paper_fact_source.quote_text
        or verified_anchor.source_char_start is None
        or verified_anchor.source_char_end is None
    ):
        raise AssertionError(
            "Memory paper-fact fixture failed exact source verification"
        )
    mem_paper_fact = consolidate_memory_candidate(
        db, project_id=project.id, candidate=cand_paper_fact
    )

    # 3.5 Attempted Invalid / Unsupported Paper Fact (must be rejected)
    rejected_unsupported_count = 0
    # 3.5a Non-existent paper
    try:
        cand_forged = MemoryCreate(
            memory_type=MemoryType.PAPER_FACT,
            title="Forged Claim",
            content="Transformer is completely obsolete.",
            sources=[
                MemorySourceCreate(
                    source_type=MemorySourceType.PAPER_CHUNK,
                    paper_id=uuid4(),  # Non-existent paper in project
                    quote_text="fake quote",
                )
            ],
        )
        consolidate_memory_candidate(db, project_id=project.id, candidate=cand_forged)
    except ValueError:
        rejected_unsupported_count += 1

    # 3.5b Invented quote on existing paper
    try:
        cand_invented_quote = MemoryCreate(
            memory_type=MemoryType.PAPER_FACT,
            title="Invented Quote Claim",
            content="Attention layers use quantum superposition.",
            sources=[
                MemorySourceCreate(
                    source_type=MemorySourceType.PAPER_CHUNK,
                    paper_id=paper.id,
                    page_number=4,
                    quote_text="quantum superposition in attention layers",
                )
            ],
        )
        consolidate_memory_candidate(
            db, project_id=project.id, candidate=cand_invented_quote
        )
    except ValueError:
        rejected_unsupported_count += 1

    # 3.5c Real quote, but claim asserts unsupported concepts
    try:
        cand_unsupported_claim = MemoryCreate(
            memory_type=MemoryType.PAPER_FACT,
            title="Unsupported Claim",
            content="Multi-head attention model proves quantum teleportation.",
            sources=[
                MemorySourceCreate(
                    source_type=MemorySourceType.PAPER_CHUNK,
                    paper_id=paper.id,
                    page_number=4,
                    quote_text="Multi-head attention allows the model to jointly attend to information",
                )
            ],
        )
        consolidate_memory_candidate(
            db, project_id=project.id, candidate=cand_unsupported_claim
        )
    except ValueError:
        rejected_unsupported_count += 1

    # 3.6 Conflicting Decision 1b (Supersedes Decision 1)
    cand_dec1_supersede = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Metric Revision: Switch to Brier Score",
        content="We switch our primary calibration metric from AURC to Brier score due to sample efficiency.",
        importance=0.95,
        confidence=1.0,
    )
    mem_dec1_new = consolidate_memory_candidate(
        db, project_id=project.id, candidate=cand_dec1_supersede
    )

    # 3.7 Foreign Project Memory (must NEVER leak into queries)
    cand_foreign = MemoryCreate(
        memory_type=MemoryType.DECISION,
        title="Foreign Decision",
        content="We choose SGD for the foreign benchmark project.",
        importance=0.9,
        confidence=1.0,
    )
    consolidate_memory_candidate(db, project_id=foreign_proj.id, candidate=cand_foreign)

    # 4. Evaluation Queries & Ground Truth
    test_cases = [
        {
            "query": "What evaluation metric did we decide to use for calibration?",
            "expected_top_id": mem_dec1_new.id,
            "forbidden_ids": [mem_dec1.id],  # Old decision must NOT be top active
            "description": "Decision supersession query",
        },
        {
            "query": "Which optimizer and learning rate are we using?",
            "expected_top_id": mem_dec2.id,
            "forbidden_ids": [],
            "description": "Standard decision recall",
        },
        {
            "query": "How should LaTeX tables be formatted?",
            "expected_top_id": mem_pref.id,
            "forbidden_ids": [],
            "description": "User preference recall",
        },
        {
            "query": "How does multi-head attention attend to information from representation subspaces?",
            "expected_top_id": mem_paper_fact.id,
            "forbidden_ids": [],
            "description": "Paper evidence fact recall",
        },
    ]

    total_queries = len(test_cases)
    recall_at_1_hits = 0
    recall_at_3_hits = 0
    forbidden_leak_count = 0
    foreign_leak_count = 0
    per_query_results = []

    print("\nRunning Evaluation Queries:\n")
    for i, tc in enumerate(test_cases, 1):
        retrieved = retrieve_project_memories(
            db, project_id=project.id, query=tc["query"], limit=3
        )
        retrieved_ids = [m.id for m in retrieved]

        hit_1 = len(retrieved_ids) > 0 and retrieved_ids[0] == tc["expected_top_id"]
        hit_3 = tc["expected_top_id"] in retrieved_ids
        has_forbidden = any(fid in retrieved_ids for fid in tc["forbidden_ids"])

        if hit_1:
            recall_at_1_hits += 1
        if hit_3:
            recall_at_3_hits += 1
        if has_forbidden:
            forbidden_leak_count += 1
        per_query_results.append(
            {
                "description": tc["description"],
                "expected_fact": manifest["memory_expected_facts"][i - 1],
                "recall_at_1": hit_1,
                "recall_at_3": hit_3,
                "superseded_memory_leaked": has_forbidden,
            }
        )

        print(f"[{i}/{total_queries}] {tc['description']}")
        print(f'  Query: "{tc["query"]}"')
        print(f'  Top Result: "{retrieved[0].content if retrieved else "None"}"')
        print(
            f"  Recall@1: {'PASS' if hit_1 else 'FAIL'} | Recall@3: {'PASS' if hit_3 else 'FAIL'}"
        )
        if has_forbidden:
            print("  WARNING: Outdated superseded memory retrieved as active!")

    # Check cross-project isolation
    foreign_retrieved = retrieve_project_memories(
        db, project_id=project.id, query="SGD optimizer foreign project", limit=5
    )
    if any(m.project_id != project.id for m in foreign_retrieved):
        foreign_leak_count += 1

    # Verify status of superseded memory in database
    db.refresh(mem_dec1)
    db.refresh(mem_dec1_new)
    supersession_verified = (
        mem_dec1.status == MemoryStatus.SUPERSEDED.value
        and mem_dec1.superseded_by_id == mem_dec1_new.id
        and mem_dec1_new.status == MemoryStatus.ACTIVE.value
    )

    r_at_1 = (recall_at_1_hits / total_queries) * 100.0
    r_at_3 = (recall_at_3_hits / total_queries) * 100.0

    print("\n" + "=" * 70)
    print("Evaluation Results Summary:")
    print("=" * 70)
    print(
        f"  Recall@1:                          {r_at_1:.1f}% ({recall_at_1_hits}/{total_queries})"
    )
    print(
        f"  Recall@3:                          {r_at_3:.1f}% ({recall_at_3_hits}/{total_queries})"
    )
    print(
        f"  Supersession Resolution:           {'100.0% PASS' if supersession_verified else 'FAIL'}"
    )
    print(
        f"  Superseded Invalidation Leak:      {forbidden_leak_count} occurrences (0 expected)"
    )
    print(
        f"  Cross-Project Isolation Leak:      {foreign_leak_count} occurrences (0 expected)"
    )
    print(
        f"  Unsupported Paper Memory Reject:   {'100.0% PASS' if rejected_unsupported_count == 3 else 'FAIL'}"
    )
    print("=" * 70)

    report_path = write_report(
        {
            "manifest_version": manifest["manifest_version"],
            "evaluation": "memory_retrieval_and_provenance",
            "status": "completed",
            "annotation_method": manifest["annotation_method"],
            "configuration": manifest["evaluation_config"],
            "expected_facts": manifest["memory_expected_facts"],
            "results": {
                "query_count": total_queries,
                "recall_at_1": r_at_1 / 100,
                "recall_at_3": r_at_3 / 100,
                "supersession_verified": supersession_verified,
                "superseded_leaks": forbidden_leak_count,
                "cross_project_leaks": foreign_leak_count,
                "unsupported_fact_rejections": rejected_unsupported_count,
                "paper_fact_provenance_status": mem_paper_fact.status,
                "paper_fact_anchor_status": anchor_status.value,
                "paper_fact_anchor": {
                    "paper_id": str(verified_evidence.paper_id),
                    "page_number": verified_evidence.page_number,
                    "chunk_id": str(verified_evidence.chunk_id),
                    "quote": verified_anchor.exact_quote,
                    "source_char_start": verified_anchor.source_char_start,
                    "source_char_end": verified_anchor.source_char_end,
                    "document_sha256": paper.document_sha256,
                },
                "queries": per_query_results,
            },
        },
        prefix="memory-eval",
    )
    print(f"Machine-readable report: {report_path}")
    db.close()
    engine.dispose()

    assert r_at_1 == 100.0, f"Expected Recall@1 100%, got {r_at_1}%"
    assert r_at_3 == 100.0, f"Expected Recall@3 100%, got {r_at_3}%"
    assert supersession_verified, "Conflict supersession failed verification"
    assert forbidden_leak_count == 0, "Superseded memory leaked into active retrieval"
    assert foreign_leak_count == 0, (
        "Foreign project memory leaked into scoped retrieval"
    )
    assert rejected_unsupported_count == 3, (
        f"Expected 3 unsupported paper memory rejections, got {rejected_unsupported_count}"
    )
    print("\nAll memory benchmark assertions PASSED cleanly.")


def test_memory_evaluation_benchmark():
    """Pytest entrypoint for the offline memory evaluation benchmark."""
    run_memory_evaluation()


if __name__ == "__main__":
    run_memory_evaluation()

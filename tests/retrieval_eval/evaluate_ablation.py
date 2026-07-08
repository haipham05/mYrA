"""Offline, paired retrieval-strategy ablation over the versioned gold corpus.

This evaluation intentionally does not call generation providers: the gold
questions annotate retrieval sources, not independent answer/claim outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "apps" / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.crud.paper import create_paper_with_job
from app.crud.project import create_project
from app.db.base import Base
from app.ingestion.parser import DocumentParser
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services import embedding as embedding_module
from app.services import retrieval as retrieval_module
from app.services.embedding import (
    DeterministicEmbeddingProvider,
    set_embedding_provider,
)
from app.services.ingestion import IngestionPipeline
from app.services.retrieval import HybridRetriever, SimpleLexicalReranker, set_reranker
from app.storage import factory as storage_factory
from app.storage.factory import set_storage
from app.storage.local import LocalStorage
from fixture_manifest import load_and_validate_manifest, write_report
from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS
from metrics import evaluate_ranking
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ABLATION_MANIFEST = {
    "ablation_manifest_version": 1,
    "ablation_revision": "retrieval-strategies-v1",
    "strategies": ["dense-only", "hybrid-unreranked", "hybrid-reranked"],
    "label_definition": "one graded-relevance source per question: annotated paper, page, and key phrase",
    "cache_policy": "disabled; no retrieval cache adapter is enabled in this runner",
    "memory_comparison": {
        "status": "UNTESTED",
        "reason": "The 24 retrieval questions do not annotate paired memory-eligible inputs or expected memory effects.",
        "required_fixture": "A versioned paired question set with pre-seeded memory state, expected memory facts, and independent source labels.",
    },
    "graphrag_comparison": {
        "status": "UNTESTED",
        "reason": "The retrieval gold questions do not define graph extraction/publication state or graph-specific relevant evidence.",
        "required_fixture": "A versioned paired question set with graph state/revision, expected graph facts, and independently labeled paper sources.",
    },
}

STRATEGIES = tuple(ABLATION_MANIFEST["strategies"])


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _evidence_key(item: Any) -> str:
    stable = f"{item.paper_id}|{item.page_number}|{item.quote}"
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _matches_gold(item: Any, question: dict[str, Any], target_paper_id: str) -> bool:
    return (
        str(item.paper_id) == target_paper_id
        and item.page_number == question["target_page"]
        and question["key_phrase"].casefold() in item.quote.casefold()
    )


def build_report(
    *,
    manifest: dict[str, Any],
    papers: list[Any],
    question_results: dict[str, dict[str, list[Any]]],
    latencies_ms: dict[str, list[float]],
    providers: dict[str, Any],
) -> dict[str, Any]:
    """Build stable JSON metrics from results aligned by question and strategy."""
    question_by_id = {question["id"]: question for question in GOLD_QUESTIONS}
    if set(question_results) != set(question_by_id):
        raise ValueError(
            "ablation results must include the unchanged complete gold set"
        )
    per_strategy: dict[str, Any] = {}
    per_question: dict[str, Any] = {}
    for strategy in STRATEGIES:
        rankings: list[list[str]] = []
        relevance_by_query: list[dict[str, int]] = []
        source_correct = 0
        for question_id, question in question_by_id.items():
            target_paper_id = str(papers[question["target_paper_idx"]].id)
            results = question_results[question_id][strategy]
            # The gold target is independent of the retrieved rows. Encode a
            # matching result as that target ID and always keep it in the ideal
            # relevance set, even when it was completely missed.
            gold_source_id = f"gold:{question_id}"
            ranked_keys = [
                gold_source_id
                if _matches_gold(item, question, target_paper_id)
                else _evidence_key(item)
                for item in results
            ]
            relevance = {gold_source_id: 3}
            rankings.append(ranked_keys)
            relevance_by_query.append(relevance)
            matching_ranks = [
                rank
                for rank, item in enumerate(results, start=1)
                if _matches_gold(item, question, target_paper_id)
            ]
            source_correct += bool(matching_ranks)
            per_question.setdefault(question_id, {})[strategy] = {
                "result_count": len(results),
                "relevant_source_rank": matching_ranks[0] if matching_ranks else None,
                "source_match": bool(matching_ranks),
                "expected_source_id": gold_source_id,
            }

        scores = evaluate_ranking(rankings, relevance_by_query)
        times = latencies_ms[strategy]
        per_strategy[strategy] = {
            "sample_count": len(question_by_id),
            "ranking": {
                "recall_at_5": scores.recall_at_5,
                "recall_at_10": scores.recall_at_10,
                "mrr": scores.mrr,
                "ndcg_at_10": scores.ndcg_at_10,
                "query_count": scores.query_count,
            },
            "answer_source_checks": {
                "status": "source-only",
                "source_match_count": source_correct,
                "source_match_rate": source_correct / len(question_by_id),
                "answer_claim_metrics": None,
                "reason": "The fixture contains source/page/key-phrase labels, not expected answer claims or citation sets.",
            },
            "latency_ms": {
                "p50": _percentile(times, 0.50),
                "p95": _percentile(times, 0.95),
            },
            "provider_usage": {"requests": 0, "tokens": None, "cost_usd": None},
        }

    report = {
        "report_version": 1,
        "evaluation": "retrieval_ablation",
        "status": "completed",
        "scope": {
            "provider_mode": "local deterministic synthetic evaluation",
            "real_provider_validation": "not performed",
            "cache": "disabled",
        },
        "manifests": {
            "corpus_manifest_version": manifest["manifest_version"],
            "gold_inputs_revision": manifest["gold_inputs_revision"],
            "retrieval_annotation_revision": manifest["annotation_revisions"][
                "retrieval"
            ],
            "ablation": ABLATION_MANIFEST,
            "ablation_manifest_sha256": hashlib.sha256(
                json.dumps(
                    ABLATION_MANIFEST, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        },
        "unchanged_input_hashes": {
            "pdfs": manifest["fixture_hashes"],
            "gold_papers_sha256": manifest["gold_papers_sha256"],
            "gold_questions_sha256": manifest["gold_questions_sha256"],
        },
        "configuration": {
            "database": "temporary SQLite",
            "storage": "temporary local filesystem",
            "embedding_provider": providers["embedding_provider"],
            "embedding_revision": providers["embedding_revision"],
            "reranker_provider": providers["reranker_provider"],
            "reranker_revision": providers["reranker_revision"],
            "parser": providers["parser"],
            "retrieval": {"top_candidates": 40, "top_evidence": 10, "rrf_k": 60},
            "cache": "disabled",
        },
        "sample_count": len(question_by_id),
        "question_ids": list(question_by_id),
        "per_question": per_question,
        "results": per_strategy,
        "comparison_constraints": {
            "paired_inputs": True,
            "same_question_order": True,
            "answer_generation": "not run; source annotations do not define gold answer claims",
            "memory": ABLATION_MANIFEST["memory_comparison"],
            "graphrag": ABLATION_MANIFEST["graphrag_comparison"],
        },
    }
    return report


async def run_evaluation() -> Path:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    manifest = load_and_validate_manifest(fixtures_dir)
    env_names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "MYRA_EMBEDDING_PROVIDER",
        "MYRA_RERANKER_PROVIDER",
    )
    old_env = {name: os.environ.get(name) for name in env_names}
    previous_storage = storage_factory._storage_instance
    previous_embedding = embedding_module._default_embedding_provider
    previous_reranker = retrieval_module._reranker_instance
    try:
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MYRA_EMBEDDING_PROVIDER": "deterministic",
                "MYRA_RERANKER_PROVIDER": "simple-lexical",
            }
        )
        with tempfile.TemporaryDirectory(prefix="myra-ablation-") as temp_dir:
            root = Path(temp_dir)
            engine = create_engine(
                f"sqlite:///{root / 'ablation.db'}",
                connect_args={"check_same_thread": False},
            )
            Base.metadata.create_all(engine)
            db = sessionmaker(bind=engine, autoflush=False)()
            storage = LocalStorage(base_dir=str(root / "storage"))
            set_storage(storage)
            embedder = DeterministicEmbeddingProvider()
            reranker = SimpleLexicalReranker()
            set_embedding_provider(embedder)
            set_reranker(reranker)

            project = create_project(
                db, ProjectCreate(name="Offline Retrieval Ablation")
            )
            pipeline = IngestionPipeline(parser=DocumentParser(use_docling=True))
            papers = []
            for spec in GOLD_PAPERS:
                pdf = fixtures_dir / spec["filename"]
                body = pdf.read_bytes()
                storage_key = f"gold/{project.id}/{spec['filename']}"
                await storage.put(storage_key, body)
                paper, job = create_paper_with_job(
                    db,
                    project_id=project.id,
                    filename=spec["filename"],
                    storage_path=storage_key,
                    status=PaperStatus.PROCESSING,
                )
                await pipeline.process_paper(db, paper_id=paper.id, job_id=job.id)
                db.refresh(paper)
                papers.append(paper)

            retriever = HybridRetriever(top_candidates=40, top_evidence=10, rrf_k=60)
            question_results: dict[str, dict[str, list[Any]]] = {}
            latencies = {strategy: [] for strategy in STRATEGIES}
            for question in GOLD_QUESTIONS:
                question_results[question["id"]] = {}
                for strategy in STRATEGIES:
                    started = time.perf_counter()
                    results = retriever.retrieve(
                        db,
                        project_id=project.id,
                        query=question["query"],
                        strategy=strategy,
                    )
                    latencies[strategy].append((time.perf_counter() - started) * 1000)
                    question_results[question["id"]][strategy] = results

            report = build_report(
                manifest=manifest,
                papers=papers,
                question_results=question_results,
                latencies_ms=latencies,
                providers={
                    "embedding_provider": embedder.model_name,
                    "embedding_revision": embedder.model_version,
                    "reranker_provider": reranker.model_name,
                    "reranker_revision": reranker.model_version,
                    "parser": _installed_version("docling"),
                },
            )
            path = write_report(report, prefix="retrieval-ablation")
            db.close()
            engine.dispose()
            return path
    finally:
        set_storage(previous_storage)
        set_embedding_provider(previous_embedding)
        set_reranker(previous_reranker)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report_path = asyncio.run(run_evaluation())
    print(f"Ablation report: {report_path}")


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


if __name__ == "__main__":
    main()

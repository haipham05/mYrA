"""Bounded cold/warm cache benchmark using the production retriever and offline fixtures.

Run from the repository root with MYRA_CACHE_ENABLED=true and MYRA_REDIS_URL set.
The benchmark creates no Redis keys outside a fresh hashed namespace and never flushes Redis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "apps" / "api"))
sys.path.insert(0, str(repo_root / "tests" / "retrieval_eval"))

from app.db.base import Base
from app.services import embedding as embedding_module
from app.services import retrieval as retrieval_module
from app.services.cache import get_cache, reset_cache_for_tests
from app.services.embedding import (
    DeterministicEmbeddingProvider,
    set_embedding_provider,
)
from app.services.retrieval import HybridRetriever, SimpleLexicalReranker, set_reranker
from app.storage import factory as storage_factory
from app.storage.factory import set_storage
from app.storage.local import LocalStorage
from evaluate_retrieval import setup_gold_corpus
from fixture_manifest import load_and_validate_manifest, write_report
from gold_corpus import GOLD_QUESTIONS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def evidence_identity(items: list[object]) -> list[dict[str, object]]:
    """Stable citation-relevant identity; deliberately excludes rank labels like E1."""
    identities = []
    for item in items:
        identities.append(
            {
                "paper_id": str(item.paper_id),
                "page_number": item.page_number,
                "quote_sha256": hashlib.sha256(item.quote.encode("utf-8")).hexdigest(),
                "document_sha256": item.document_sha256,
                "source_element_ids": sorted(map(str, item.source_element_ids)),
                "anchor_offsets": [
                    {
                        "start": anchor.source_char_start,
                        "end": anchor.source_char_end,
                        "quote_sha256": hashlib.sha256(
                            anchor.exact_quote.encode("utf-8")
                        ).hexdigest(),
                        "status": str(anchor.anchor_status),
                    }
                    for anchor in item.anchors
                ],
            }
        )
    return identities


def parity_summary(
    cold: list[list[dict[str, object]]], warm: list[list[dict[str, object]]]
) -> dict[str, object]:
    matches = cold == warm
    return {
        "matches": matches,
        "question_count": len(cold),
        "matching_questions": sum(
            left == right for left, right in zip(cold, warm, strict=True)
        ),
        "cold_evidence_count": sum(map(len, cold)),
        "warm_evidence_count": sum(map(len, warm)),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return round(ordered[index], 3)


def stats_delta(
    before: dict[str, int | None], after: dict[str, int | None]
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key in ("hits", "misses", "errors", "evicted_keys_delta"):
        left, right = before.get(key), after.get(key)
        result[key] = (
            right - left if isinstance(left, int) and isinstance(right, int) else None
        )
    return result


def measurement_status(
    *,
    cache_enabled: bool,
    parity_matches: bool,
    warm_hits: int | None,
    errors: int,
) -> tuple[str, list[str]]:
    if not cache_enabled:
        return "UNTESTED", ["redis_cache_adapter_not_available"]
    reasons: list[str] = []
    if not parity_matches:
        reasons.append("cold_warm_evidence_parity_failed")
    if warm_hits is None or warm_hits <= 0:
        reasons.append("warm_run_recorded_no_cache_hits")
    if errors > 0:
        reasons.append("cache_operations_reported_errors")
    return ("PASS" if not reasons else "FAIL"), reasons


def stable_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


async def run_benchmark() -> Path:
    if os.getenv("MYRA_CACHE_ENABLED", "").lower() not in {"true", "1", "yes"}:
        raise RuntimeError("Set MYRA_CACHE_ENABLED=true to run this benchmark")
    if not os.getenv("MYRA_REDIS_URL"):
        raise RuntimeError(
            "Set MYRA_REDIS_URL to the dedicated application Redis service"
        )

    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    manifest = load_and_validate_manifest(fixtures_dir)
    pdf_hashes = {
        filename: hashlib.sha256((fixtures_dir / filename).read_bytes()).hexdigest()
        for filename in manifest["fixture_hashes"]
    }
    query_hashes = {
        question["id"]: hashlib.sha256(question["query"].encode("utf-8")).hexdigest()
        for question in GOLD_QUESTIONS
    }

    # Build/ingest in a private SQLite corpus with cache disabled, then initialize a
    # unique cache namespace only for the cold/warm measurement itself.
    env_names = (
        "MYRA_CACHE_ENABLED",
        "MYRA_CACHE_NAMESPACE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    prior_env = {name: os.environ.get(name) for name in env_names}
    prior_storage = storage_factory._storage_instance
    prior_embedding = embedding_module._default_embedding_provider
    prior_reranker = retrieval_module._reranker_instance
    temporary = tempfile.TemporaryDirectory(prefix="myra-cache-benchmark-")
    engine = None
    db = None
    try:
        os.environ["MYRA_CACHE_ENABLED"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        reset_cache_for_tests()
        engine = create_engine(
            f"sqlite:///{Path(temporary.name) / 'benchmark.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        local_storage = LocalStorage(base_dir=str(Path(temporary.name) / "storage"))
        set_storage(local_storage)
        set_embedding_provider(DeterministicEmbeddingProvider())
        set_reranker(SimpleLexicalReranker())
        project, _, _ = await setup_gold_corpus(db, fixtures_dir, local_storage)

        namespace = f"benchmark-{uuid4().hex}"
        os.environ["MYRA_CACHE_ENABLED"] = "true"
        os.environ["MYRA_CACHE_NAMESPACE"] = namespace
        reset_cache_for_tests()
        cache = get_cache()
        retriever = HybridRetriever(top_candidates=40, top_evidence=10, rrf_k=60)

        baseline = cache.stats_snapshot()
        cold_results: list[list[dict[str, object]]] = []
        cold_ms: list[float] = []
        for question in GOLD_QUESTIONS:
            started = time.perf_counter()
            evidence = retriever.retrieve(
                db, project_id=project.id, query=question["query"]
            )
            cold_ms.append((time.perf_counter() - started) * 1000)
            cold_results.append(evidence_identity(evidence))
        cold_stats = cache.stats_snapshot()

        warm_results: list[list[dict[str, object]]] = []
        warm_ms: list[float] = []
        for question in GOLD_QUESTIONS:
            started = time.perf_counter()
            evidence = retriever.retrieve(
                db, project_id=project.id, query=question["query"]
            )
            warm_ms.append((time.perf_counter() - started) * 1000)
            warm_results.append(evidence_identity(evidence))
        warm_stats = cache.stats_snapshot()
        parity = parity_summary(cold_results, warm_results)
        cold_delta = stats_delta(baseline, cold_stats)
        warm_delta = stats_delta(cold_stats, warm_stats)
        status, reasons = measurement_status(
            cache_enabled=cache.enabled,
            parity_matches=bool(parity["matches"]),
            warm_hits=warm_delta["hits"],
            errors=(cold_delta["errors"] or 0) + (warm_delta["errors"] or 0),
        )

        report: dict[str, object] = {
            "report_version": 1,
            "benchmark": "shared-cache-cold-warm-retrieval",
            "fixture_manifest_version": manifest["manifest_version"],
            "fixture_manifest_sha256": hashlib.sha256(
                (Path(__file__).resolve().parent / "manifest.json").read_bytes()
            ).hexdigest(),
            "input_hashes": {"pdf_sha256": pdf_hashes, "query_sha256": query_hashes},
            "provider_scope": {
                "database": "isolated temporary SQLite",
                "storage": "isolated local filesystem",
                "embedding": "deterministic test provider",
                "reranker": "deterministic lexical test provider",
                "cloud_or_paid_calls": 0,
                "cache_namespace": "unique hashed run namespace",
                "cache_namespace_sha256": hashlib.sha256(
                    namespace.encode("utf-8")
                ).hexdigest(),
            },
            "question_count": len(GOLD_QUESTIONS),
            "status": status,
            "status_reasons": reasons,
            "parity": parity,
            "cold": {
                "p50_latency_ms": percentile(cold_ms, 0.50),
                "p95_latency_ms": percentile(cold_ms, 0.95),
                "cache": cold_delta,
            },
            "warm": {
                "p50_latency_ms": percentile(warm_ms, 0.50),
                "p95_latency_ms": percentile(warm_ms, 0.95),
                "cache": warm_delta,
            },
            "cost_usd": None,
        }
        return write_report(json.loads(stable_json(report)), prefix="cache-benchmark")
    finally:
        if db is not None:
            db.close()
        if engine is not None:
            engine.dispose()
        temporary.cleanup()
        reset_cache_for_tests()
        for name, value in prior_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        set_storage(prior_storage)
        if prior_embedding is not None:
            set_embedding_provider(prior_embedding)
        else:
            embedding_module._default_embedding_provider = None
        if prior_reranker is not None:
            set_reranker(prior_reranker)
        else:
            retrieval_module._reranker_instance = None


def main() -> None:
    report_path = asyncio.run(run_benchmark())
    print(f"Cache benchmark report: {report_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from types import SimpleNamespace

import pytest
from evaluate_ablation import (
    ABLATION_MANIFEST,
    STRATEGIES,
    build_report,
)
from gold_corpus import GOLD_QUESTIONS


def _fixture_inputs():
    manifest = {
        "manifest_version": 1,
        "gold_inputs_revision": "generated-corpus-v1",
        "annotation_revisions": {"retrieval": "gold_corpus.py:v1"},
        "fixture_hashes": {"paper.pdf": "a" * 64},
        "gold_papers_sha256": "b" * 64,
        "gold_questions_sha256": "c" * 64,
    }
    papers = [SimpleNamespace(id=f"paper-{index}") for index in range(4)]
    return manifest, papers


def _evidence(paper_id, page, quote):
    return SimpleNamespace(paper_id=paper_id, page_number=page, quote=quote)


def test_report_contains_paired_metrics_hashes_and_unavailable_comparisons():
    manifest, papers = _fixture_inputs()
    results = {}
    for question in GOLD_QUESTIONS:
        target_id = str(papers[question["target_paper_idx"]].id)
        exact = _evidence(target_id, question["target_page"], question["key_phrase"])
        results[question["id"]] = {
            strategy: [exact] if strategy != "dense-only" else []
            for strategy in STRATEGIES
        }
    times = {strategy: [10.0, 30.0, 20.0] for strategy in STRATEGIES}

    report = build_report(
        manifest=manifest,
        papers=papers,
        question_results=results,
        latencies_ms=times,
        providers={
            "embedding_provider": "deterministic",
            "embedding_revision": "v1",
            "reranker_provider": "simple-lexical",
            "reranker_revision": "v1",
            "parser": "Docling offline",
        },
    )

    assert report["sample_count"] == 24
    assert report["manifests"]["ablation_manifest_sha256"]
    assert report["unchanged_input_hashes"]["gold_questions_sha256"] == "c" * 64
    assert report["scope"]["cache"] == "disabled"
    assert set(report["results"]) == set(STRATEGIES)
    assert report["results"]["hybrid-reranked"]["ranking"]["ndcg_at_10"] == 1.0
    assert (
        report["results"]["hybrid-reranked"]["answer_source_checks"][
            "source_match_count"
        ]
        == 24
    )
    assert report["results"]["dense-only"]["ranking"]["query_count"] == 24
    assert report["results"]["hybrid-unreranked"]["latency_ms"] == {
        "p50": 20.0,
        "p95": 29.0,
    }
    assert report["results"]["hybrid-reranked"]["provider_usage"]["cost_usd"] is None
    assert report["comparison_constraints"]["memory"]["status"] == "UNTESTED"
    assert report["comparison_constraints"]["graphrag"]["status"] == "UNTESTED"
    assert (
        report["results"]["hybrid-reranked"]["answer_source_checks"][
            "answer_claim_metrics"
        ]
        is None
    )


def test_report_rejects_incomplete_question_set():
    manifest, papers = _fixture_inputs()
    with pytest.raises(ValueError, match="complete gold set"):
        build_report(
            manifest=manifest,
            papers=papers,
            question_results={},
            latencies_ms={strategy: [] for strategy in STRATEGIES},
            providers={},
        )


def test_memory_and_graph_ablation_are_not_claimed_as_simulated_wins():
    assert ABLATION_MANIFEST["memory_comparison"]["status"] == "UNTESTED"
    assert "paired" in ABLATION_MANIFEST["memory_comparison"]["required_fixture"]
    assert ABLATION_MANIFEST["graphrag_comparison"]["status"] == "UNTESTED"
    assert "graph state" in ABLATION_MANIFEST["graphrag_comparison"]["required_fixture"]

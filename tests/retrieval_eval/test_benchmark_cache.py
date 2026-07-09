from benchmark_cache import (
    measurement_status,
    parity_summary,
    percentile,
    stable_json,
    stats_delta,
)


def test_cold_warm_parity_compares_citation_relevant_evidence() -> None:
    evidence = [
        [{"paper_id": "p1", "page_number": 2, "quote_sha256": "abc"}],
        [],
    ]
    assert parity_summary(evidence, evidence) == {
        "matches": True,
        "question_count": 2,
        "matching_questions": 2,
        "cold_evidence_count": 1,
        "warm_evidence_count": 1,
    }
    mismatch = parity_summary(evidence, [[], []])
    assert mismatch["matches"] is False
    assert mismatch["matching_questions"] == 1


def test_cache_benchmark_report_helpers_are_deterministic_and_handle_missing_stats() -> (
    None
):
    report = {"z": 1, "a": {"value": 2}}
    assert stable_json(report) == '{\n  "a": {\n    "value": 2\n  },\n  "z": 1\n}\n'
    assert stats_delta(
        {"hits": 1, "misses": 3, "errors": 0, "evicted_keys_delta": None},
        {"hits": 4, "misses": 5, "errors": 1, "evicted_keys_delta": None},
    ) == {"hits": 3, "misses": 2, "errors": 1, "evicted_keys_delta": None}


def test_percentiles_use_nearest_rank_and_empty_input_is_null() -> None:
    assert percentile([4, 1, 3, 2], 0.50) == 2
    assert percentile([4, 1, 3, 2], 0.95) == 4
    assert percentile([], 0.95) is None


def test_measurement_pass_requires_parity_warm_hits_and_error_free_cache() -> None:
    assert measurement_status(
        cache_enabled=True,
        parity_matches=True,
        warm_hits=24,
        errors=0,
    ) == ("PASS", [])
    assert measurement_status(
        cache_enabled=True,
        parity_matches=True,
        warm_hits=0,
        errors=2,
    ) == (
        "FAIL",
        ["warm_run_recorded_no_cache_hits", "cache_operations_reported_errors"],
    )
    assert measurement_status(
        cache_enabled=False,
        parity_matches=True,
        warm_hits=0,
        errors=0,
    ) == ("UNTESTED", ["redis_cache_adapter_not_available"])

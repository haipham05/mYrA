import math

import pytest
from metrics import evaluate_ranking, ndcg_at_k, recall_at_k, reciprocal_rank


def test_metrics_match_hand_calculated_example() -> None:
    # Gold grades: a=3, b=2, c=1. Ranking puts b first, a third, c fourth.
    ranked = ["b", "noise", "a", "c"]
    relevance = {"a": 3, "b": 2, "c": 1}

    assert recall_at_k(ranked, relevance, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevance, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked, relevance) == 1.0

    gain = lambda grade: 2**grade - 1
    dcg = gain(2) + gain(3) / math.log2(4) + gain(1) / math.log2(5)
    ideal = gain(3) + gain(2) / math.log2(3) + gain(1) / math.log2(4)
    assert ndcg_at_k(ranked, relevance, 10) == pytest.approx(dcg / ideal)


def test_duplicate_ranked_ids_count_only_at_first_rank() -> None:
    relevance = {"a": 1, "b": 1}
    assert recall_at_k(["a", "a", "noise", "b"], relevance, 2) == 0.5
    assert reciprocal_rank(["noise", "a", "a"], relevance) == 0.5
    assert ndcg_at_k(["a", "a", "b"], relevance, 10) == 1.0


def test_input_order_is_the_stable_tie_order() -> None:
    relevance = {"low": 1, "high": 3}
    # Equal scores are resolved before metrics are called, so input order is rank.
    low_first = ndcg_at_k(["low", "high"], relevance)
    high_first = ndcg_at_k(["high", "low"], relevance)
    assert high_first > low_first
    assert ndcg_at_k(["low", "high"], relevance) == low_first


def test_no_results_and_queries_without_relevant_items_score_zero() -> None:
    assert recall_at_k([], {"gold": 1}, 5) == 0.0
    assert reciprocal_rank([], {"gold": 1}) == 0.0
    assert ndcg_at_k([], {"gold": 1}) == 0.0
    assert recall_at_k(["x"], {"x": 0}, 5) == 0.0
    assert reciprocal_rank(["x"], {}) == 0.0
    assert ndcg_at_k(["x"], {"x": 0}) == 0.0


def test_macro_average_keeps_empty_and_unanswerable_queries_in_denominator() -> None:
    result = evaluate_ranking(
        [["gold"], [], ["miss"]],
        [{"gold": 1}, {"gold": 1}, {}],
    )
    assert result.query_count == 3
    assert result.recall_at_5 == pytest.approx(1 / 3)
    assert result.recall_at_10 == pytest.approx(1 / 3)
    assert result.mrr == pytest.approx(1 / 3)
    assert result.ndcg_at_10 == pytest.approx(1 / 3)


def test_empty_evaluation_returns_zero_metrics() -> None:
    result = evaluate_ranking([], [])
    assert result.query_count == 0
    assert (result.recall_at_5, result.recall_at_10, result.mrr, result.ndcg_at_10) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k([], {}, k)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k([], {}, k)


def test_rejects_noninteger_k() -> None:
    with pytest.raises(TypeError, match="positive integer"):
        recall_at_k([], {}, 1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="positive integer"):
        ndcg_at_k([], {}, True)  # type: ignore[arg-type]


def test_rejects_negative_nonfinite_and_non_numeric_relevance() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        recall_at_k([], {"bad": -1}, 5)
    with pytest.raises(ValueError, match="finite"):
        ndcg_at_k([], {"bad": math.inf})
    with pytest.raises(TypeError, match="finite number"):
        reciprocal_rank([], {"bad": "1"})  # type: ignore[dict-item]


def test_rejects_misaligned_query_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_ranking([["a"]], [])


def test_evaluation_is_deterministic_for_identical_inputs() -> None:
    rankings = [["x", "b", "a"], ["c"], []]
    relevance = [{"a": 2, "b": 1}, {"c": 1}, {}]
    assert evaluate_ranking(rankings, relevance) == evaluate_ranking(
        rankings, relevance
    )

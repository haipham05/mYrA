"""Small, deterministic ranking metrics for offline retrieval evaluations.

Rankings are sequences in their final order; equal-score tie breaking is therefore
the caller's stable input order. Duplicate IDs in a ranking count only at their
first occurrence. Queries with no relevant items score zero and remain in the
macro-average denominator.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TypeVar

ItemId = TypeVar("ItemId", bound=Hashable)


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    """Macro-averaged metrics, with one score per input query."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    query_count: int


def _validate_relevance(relevance: Mapping[ItemId, Real]) -> None:
    for item_id, grade in relevance.items():
        if isinstance(grade, bool) or not isinstance(grade, Real):
            raise TypeError(f"relevance for {item_id!r} must be a finite number")
        if not math.isfinite(float(grade)):
            raise ValueError(f"relevance for {item_id!r} must be finite")
        if grade < 0:
            raise ValueError(f"relevance for {item_id!r} cannot be negative")


def _unique_ranking(ranked_ids: Sequence[ItemId]) -> list[ItemId]:
    """Deduplicate a ranking while preserving the first (best) rank."""
    unique: list[ItemId] = []
    seen: set[ItemId] = set()
    for item_id in ranked_ids:
        try:
            already_seen = item_id in seen
        except TypeError as exc:
            raise TypeError("ranked IDs must be hashable") from exc
        if not already_seen:
            seen.add(item_id)
            unique.append(item_id)
    return unique


def recall_at_k(
    ranked_ids: Sequence[ItemId], relevance: Mapping[ItemId, Real], k: int
) -> float:
    """Return relevant unique items in the first *k* divided by all relevant IDs.

    Relevance grades greater than zero count as relevant. If there are no
    relevant IDs, or the ranking is empty, the score is zero. ``k`` must be a
    positive integer (``bool`` is not accepted as an integer cutoff).
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    _validate_relevance(relevance)
    relevant_ids = {item_id for item_id, grade in relevance.items() if grade > 0}
    if not relevant_ids:
        return 0.0
    ranking = _unique_ranking(ranked_ids)
    hits = sum(item_id in relevant_ids for item_id in ranking[:k])
    return hits / len(relevant_ids)


def reciprocal_rank(
    ranked_ids: Sequence[ItemId], relevance: Mapping[ItemId, Real]
) -> float:
    """Return reciprocal rank of the first positive-grade item, or zero."""
    _validate_relevance(relevance)
    relevant_ids = {item_id for item_id, grade in relevance.items() if grade > 0}
    if not relevant_ids:
        return 0.0
    for rank, item_id in enumerate(_unique_ranking(ranked_ids), start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[ItemId], relevance: Mapping[ItemId, Real], k: int = 10
) -> float:
    """Return graded nDCG using gain ``2**grade - 1`` and log2 rank discount.

    The ideal ranking sorts every labeled item by descending grade. Zero-grade
    labels contribute no gain. No positive ideal gain (including no labels)
    yields zero. Duplicate ranked IDs are considered only at their first rank.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    _validate_relevance(relevance)

    def gain(grade: Real) -> float:
        return math.pow(2.0, float(grade)) - 1.0

    ideal_gains = sorted((gain(grade) for grade in relevance.values()), reverse=True)
    ideal_dcg = sum(
        value / math.log2(rank + 1) for rank, value in enumerate(ideal_gains[:k], 1)
    )
    if ideal_dcg == 0:
        return 0.0

    ranking = _unique_ranking(ranked_ids)
    dcg = sum(
        gain(relevance.get(item_id, 0)) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranking[:k], 1)
    )
    return dcg / ideal_dcg


def evaluate_ranking(
    rankings: Sequence[Sequence[ItemId]],
    relevance_by_query: Sequence[Mapping[ItemId, Real]],
) -> RankingMetrics:
    """Compute macro Recall@5/10, MRR, and nDCG@10 for aligned query inputs.

    Every query is included in the denominator, including queries with no gold
    relevant items or no returned results. An empty evaluation returns zeros
    and ``query_count=0``.
    """
    if len(rankings) != len(relevance_by_query):
        raise ValueError("rankings and relevance_by_query must have equal lengths")
    count = len(rankings)
    if count == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0)

    per_query = [
        (
            recall_at_k(ranking, relevance, 5),
            recall_at_k(ranking, relevance, 10),
            reciprocal_rank(ranking, relevance),
            ndcg_at_k(ranking, relevance, 10),
        )
        for ranking, relevance in zip(rankings, relevance_by_query, strict=True)
    ]
    return RankingMetrics(
        recall_at_5=sum(row[0] for row in per_query) / count,
        recall_at_10=sum(row[1] for row in per_query) / count,
        mrr=sum(row[2] for row in per_query) / count,
        ndcg_at_10=sum(row[3] for row in per_query) / count,
        query_count=count,
    )

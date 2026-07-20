"""Deterministic merge, deduplication, and prompt budgeting for evidence."""

from __future__ import annotations

from collections import defaultdict

from app.schemas.evidence import AnchorStatus, EvidenceItem

DEFAULT_EVIDENCE_CONTEXT_CHAR_BUDGET = 24_000


def _source_identity(item: EvidenceItem) -> tuple:
    for anchor in item.anchors:
        if (
            anchor.anchor_status == AnchorStatus.VERIFIED
            and anchor.document_sha256 == item.document_sha256
            and anchor.page_number == item.page_number
            and anchor.exact_quote == item.quote
            and anchor.source_char_start is not None
            and anchor.source_char_end is not None
            and anchor.source_char_end > anchor.source_char_start
        ):
            return (
                "span",
                str(item.paper_id),
                item.document_sha256,
                anchor.page_number,
                anchor.source_char_start,
                anchor.source_char_end,
            )
    # Without independently verified offsets, only an identical chunk is a
    # canonical duplicate; equal prose elsewhere in a paper remains separate.
    return ("chunk", str(item.paper_id), item.document_sha256, str(item.chunk_id))


def assemble_evidence_items(
    items: list[EvidenceItem],
    *,
    max_items: int = 12,
    context_char_budget: int = DEFAULT_EVIDENCE_CONTEXT_CHAR_BUDGET,
) -> list[EvidenceItem]:
    """Merge evidence, prefer paper coverage, and cap prompt context size.

    Original order is preserved within each paper. The first pass takes one item
    per paper before filling remaining slots in original order, avoiding a single
    paper consuming the entire small evidence budget. Citation quotes are never
    truncated; only parent context is clipped to fit the remaining budget.
    """
    if max_items < 1 or context_char_budget < 1:
        return []

    deduplicated: list[EvidenceItem] = []
    seen: set[tuple] = set()
    for item in items:
        identity = _source_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(item)

    by_paper: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in deduplicated:
        by_paper[str(item.paper_id)].append(item)

    balanced: list[EvidenceItem] = []
    # Round-robin through stable first-seen paper order.
    paper_order = list(by_paper)
    while len(balanced) < len(deduplicated):
        added = False
        for paper_key in paper_order:
            if by_paper[paper_key]:
                balanced.append(by_paper[paper_key].pop(0))
                added = True
        if not added:
            break

    candidates = balanced[:max_items]

    def fixed_size(item: EvidenceItem) -> int:
        return len(item.paper_title or "Paper") + len(item.quote) + 80

    # Reserve space for each selected citation before distributing optional
    # parent context. If the budget is very small, retain at least the first item.
    while (
        len(candidates) > 1 and sum(fixed_size(item) for item in candidates) > context_char_budget
    ):
        candidates.pop()

    remaining_context = max(0, context_char_budget - sum(fixed_size(item) for item in candidates))
    selected: list[EvidenceItem] = []
    for index, item in enumerate(candidates):
        context = item.parent_context or item.quote
        fair_share = remaining_context // (len(candidates) - index)
        context_size = min(len(context), fair_share)
        bounded_context = context[:context_size]
        bounded_item = item.model_copy(
            update={
                "id": f"E{len(selected) + 1}",
                "parent_context": bounded_context if item.parent_context is not None else None,
            }
        )
        selected.append(bounded_item)
        remaining_context -= len(bounded_context)

    return selected

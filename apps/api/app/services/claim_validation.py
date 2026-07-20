"""Helpers for associating cited answer clauses with evidence IDs."""

import re
from collections.abc import Sequence

_LEADING_CONNECTORS = re.compile(
    r"^(?:(?:and|but|whereas|while|however|in contrast|by contrast|conversely|also)\b[\s,;:—-]*)+",
    re.IGNORECASE,
)


def _clean_clause(value: str) -> str:
    clause = value.strip()
    clause = re.sub(r"^[\s.,;:!?—-]+", "", clause)
    clause = _LEADING_CONNECTORS.sub("", clause).strip()
    clause = re.sub(r"\s+([.,;:!?])", r"\1", clause)
    return re.sub(r"[\s,;:!?—-]+$", "", clause).strip()


def claim_clauses_for_citations(
    sentence: str,
    citation_spans: Sequence[tuple[int, int]],
    *,
    clean_sentence: str,
) -> list[str]:
    """Return the claim text attributed to each citation marker.

    A single citation continues to cover its original sentence. For multiple
    citations, text between markers forms the next clause. Any prose after the
    final marker stays attached to the final clause so an uncited tail cannot be
    accidentally discarded during per-source validation.
    """
    if not citation_spans:
        return []
    if len(citation_spans) == 1:
        return [clean_sentence]

    clauses: list[str] = []
    previous_end = 0
    for index, (start, end) in enumerate(citation_spans):
        clause_start = previous_end if index else 0
        clause = _clean_clause(sentence[clause_start:start])
        if index == len(citation_spans) - 1:
            clause = _clean_clause(f"{clause} {sentence[end:]}")
        if not re.search(r"[A-Za-z0-9]", clause):
            clause = clean_sentence
        clauses.append(clause)
        previous_end = end
    return clauses


def is_explicit_comparison(text: str) -> bool:
    return bool(
        re.search(
            r"\b(whereas|while|in contrast|by contrast|compared with|compared to|versus|vs\.?|"
            r"outperform(?:s|ed)?|higher than|lower than|more than|less than)\b",
            text,
            re.IGNORECASE,
        )
    )


def is_explicit_interpretation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(suggests?|implies?|indicates?|may\s+mean|might\s+mean|likely\s+means?|"
            r"we\s+infer|can\s+infer|taken\s+together|this\s+means)\b",
            text,
            re.IGNORECASE,
        )
    )

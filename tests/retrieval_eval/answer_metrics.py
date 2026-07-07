"""Deterministic answer and citation scoring for versioned evaluation fixtures.

Expected sources and claims are annotation inputs, independent of system output.
This module deliberately does not infer semantic entailment: ambiguous human
judgments are represented as unscored rather than guessed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

HumanLabel = Literal[
    "relevant", "not_relevant", "faithful", "not_faithful", "ambiguous"
]


@dataclass(frozen=True)
class ExpectedSource:
    """An independently annotated source required for an answer/fact."""

    source_id: str
    paper_id: str
    page_number: int
    exact_quote: str
    key_phrase: str
    page_text: str


@dataclass(frozen=True)
class Citation:
    """A returned citation plus the viewer's verified text-anchor evidence."""

    citation_id: str
    paper_id: str
    page_number: int
    quote: str
    anchor_verified: bool
    source_char_start: int | None
    source_char_end: int | None


@dataclass(frozen=True)
class ExpectedClaim:
    """A generated claim and the independently expected supporting sources."""

    claim_id: str
    required_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedClaim:
    claim_id: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationCase:
    expected_sources: tuple[ExpectedSource, ...]
    citations: tuple[Citation, ...]
    expected_claims: tuple[ExpectedClaim, ...]
    generated_claims: tuple[GeneratedClaim, ...]
    expected_abstention: bool
    actual_abstention: bool


@dataclass(frozen=True)
class HumanJudgment:
    """Machine-readable human label; ambiguous decisions carry no score."""

    label: HumanLabel
    score: int | None
    status: Literal["scored", "unscored"]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerMetrics:
    citation_precision: float
    correct_source_coverage: float
    unsupported_claim_rate: float | None
    abstention_correct: bool
    correct_citations: int
    citation_count: int
    covered_sources: int
    expected_source_count: int
    unsupported_claims: int
    generated_claim_count: int

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-serializable values for evaluation reports."""
        return asdict(self)


def _normalized(text: str) -> str:
    """Normalize only whitespace and case; punctuation/content stay significant."""
    return " ".join(text.split()).casefold()


def _anchor_matches(citation: Citation, page_text: str) -> bool:
    start = citation.source_char_start
    end = citation.source_char_end
    if not citation.anchor_verified or start is None or end is None:
        return False
    if start < 0 or end <= start or end > len(page_text):
        return False
    return page_text[start:end] == citation.quote


def _citation_source(
    citation: Citation, sources: dict[str, ExpectedSource]
) -> str | None:
    """Return the sole independently expected source matched by this citation."""
    matches = [
        source.source_id
        for source in sources.values()
        if citation.paper_id == source.paper_id
        and citation.page_number == source.page_number
        and citation.quote == source.exact_quote
        and source.key_phrase.strip()
        and _normalized(source.key_phrase) in _normalized(citation.quote)
        and _normalized(source.exact_quote) in _normalized(source.page_text)
        and _anchor_matches(citation, source.page_text)
    ]
    # Ambiguous duplicate annotations should not be awarded as a correct source.
    return matches[0] if len(matches) == 1 else None


def score_answer(case: EvaluationCase) -> AnswerMetrics:
    """Score exact source attribution, claim support, and expected abstention.

    A citation is source-correct only when paper, page, exact annotated quote,
    key phrase, and verified text offsets all agree. Bounding boxes are not an
    input to this scorer and therefore cannot establish correctness.
    """
    source_by_id = {source.source_id: source for source in case.expected_sources}
    citation_by_id = {citation.citation_id: citation for citation in case.citations}
    matched = {
        citation.citation_id: _citation_source(citation, source_by_id)
        for citation in case.citations
    }
    correct = sum(source_id is not None for source_id in matched.values())
    covered = {source_id for source_id in matched.values() if source_id is not None}

    expected_claims = {claim.claim_id: claim for claim in case.expected_claims}
    unsupported = 0
    for claim in case.generated_claims:
        expected = expected_claims.get(claim.claim_id)
        if expected is None or not expected.required_source_ids:
            unsupported += 1
            continue
        supported_ids = {
            matched[citation_id]
            for citation_id in claim.citation_ids
            if citation_id in citation_by_id and matched[citation_id] is not None
        }
        if not set(expected.required_source_ids).issubset(supported_ids):
            unsupported += 1

    return AnswerMetrics(
        citation_precision=correct / len(case.citations) if case.citations else 0.0,
        correct_source_coverage=(
            len(covered) / len(case.expected_sources) if case.expected_sources else 0.0
        ),
        unsupported_claim_rate=(
            unsupported / len(case.generated_claims) if case.generated_claims else None
        ),
        abstention_correct=case.expected_abstention == case.actual_abstention,
        correct_citations=correct,
        citation_count=len(case.citations),
        covered_sources=len(covered),
        expected_source_count=len(case.expected_sources),
        unsupported_claims=unsupported,
        generated_claim_count=len(case.generated_claims),
    )


def score_human_label(label: HumanLabel) -> HumanJudgment:
    """Map explicit human decisions to scores; never coerce ambiguity to failure."""
    if label == "ambiguous":
        return HumanJudgment(
            label=label, score=None, status="unscored", reason="ambiguous"
        )
    return HumanJudgment(
        label=label,
        score=1 if label in {"relevant", "faithful"} else 0,
        status="scored",
    )

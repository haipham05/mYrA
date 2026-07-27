"""Verify literature claims against explicitly selected, current paper sources.

Exact-source resolution proves that a quote exists in the current document. The
deterministic checker is intentionally conservative; semantic contradiction is
accepted only as an explicit assessment supplied by a caller.
"""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.schemas.evidence import AnchorStatus, Citation, CitationAnchor
from app.services.chat_service import check_claim_support
from app.services.source_resolution import resolve_exact_source_anchor


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class ClaimSource(BaseModel):
    """A candidate passage; it is never trusted until resolved against the DB."""

    evidence_id: str = Field(min_length=1, max_length=32)
    paper_id: UUID
    page_number: int = Field(ge=1)
    exact_quote: str = Field(min_length=1, max_length=12_000)
    document_sha256: str | None = None


class ClaimAssessment(BaseModel):
    """Optional caller-provided semantic assessment; no model call is made here."""

    refuting_evidence_ids: list[str] = Field(default_factory=list, max_length=18)


class ClaimVerificationResult(BaseModel):
    claim: str
    verdict: ClaimVerdict
    citations: list[Citation] = Field(default_factory=list)
    explanation: str
    scope_note: str
    semantic_contradiction: str | None = None


_SCOPE_NOTE = "Assessment is limited to the selected papers; it is not a review of all literature."


def verify_claim(
    db: Session,
    *,
    project_id: UUID,
    claim: str,
    selected_paper_ids: list[UUID],
    sources: list[ClaimSource],
    assessment: ClaimAssessment | None = None,
) -> ClaimVerificationResult:
    """Return a cautious verdict using only current exact quotes in selected papers.

    Candidate evidence IDs are local to this call. Refuting IDs can be supplied
    only by an upstream semantic assessor; this function never infers
    contradiction from a failed deterministic support check.
    """
    clean_claim = claim.strip()
    if not clean_claim:
        raise ValueError("claim must not be empty")

    selected = set(selected_paper_ids)
    if not selected:
        return ClaimVerificationResult(
            claim=clean_claim,
            verdict=ClaimVerdict.INSUFFICIENT,
            explanation="Select one or more papers before verifying a claim.",
            scope_note=_SCOPE_NOTE,
        )

    # Duplicate IDs make assessment references ambiguous, so ignore all entries
    # with a duplicated ID instead of guessing which quote was assessed.
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.evidence_id] = counts.get(source.evidence_id, 0) + 1

    current: dict[str, tuple[ClaimSource, CitationAnchor]] = {}
    for source in sources:
        if counts[source.evidence_id] != 1 or source.paper_id not in selected:
            continue
        anchor = resolve_exact_source_anchor(
            db,
            project_id=project_id,
            paper_id=source.paper_id,
            page_number=source.page_number,
            exact_quote=source.exact_quote,
            document_sha256=source.document_sha256,
        )
        if anchor is not None and anchor.anchor_status is AnchorStatus.VERIFIED:
            current[source.evidence_id] = (source, anchor)

    supported_ids = {
        evidence_id
        for evidence_id, (source, _anchor) in current.items()
        if check_claim_support(clean_claim, source.exact_quote)
    }
    requested_refuting = set(assessment.refuting_evidence_ids) if assessment else set()
    refuting_ids = requested_refuting & current.keys()
    distinct_refuting = refuting_ids - supported_ids

    if supported_ids and distinct_refuting:
        verdict = ClaimVerdict.MIXED
        explanation = (
            "Selected sources include textually supporting and model-assessed refuting evidence."
        )
    elif supported_ids and not refuting_ids:
        verdict = ClaimVerdict.SUPPORTED
        explanation = (
            "An exact current passage supports the claim under the conservative text checker."
        )
    elif refuting_ids and not supported_ids:
        verdict = ClaimVerdict.CONTRADICTED
        explanation = "A semantic assessor flagged a current passage as refuting the claim."
    else:
        verdict = ClaimVerdict.INSUFFICIENT
        explanation = "Selected-paper passages do not establish support or a distinct refutation."

    citation_ids = supported_ids | refuting_ids
    citations = [
        _citation(index, evidence_id, *current[evidence_id])
        for index, evidence_id in enumerate(sorted(citation_ids), start=1)
    ]
    return ClaimVerificationResult(
        claim=clean_claim,
        verdict=verdict,
        citations=citations,
        explanation=explanation,
        scope_note=_SCOPE_NOTE,
        semantic_contradiction=(
            "model_assessed" if verdict in {ClaimVerdict.CONTRADICTED, ClaimVerdict.MIXED} else None
        ),
    )


def _citation(
    index: int,
    evidence_id: str,
    source: ClaimSource,
    anchor: CitationAnchor,
) -> Citation:
    return Citation(
        citation_index=index,
        evidence_id=evidence_id,
        paper_id=source.paper_id,
        page_number=anchor.page_number,
        quote=anchor.exact_quote,
        document_sha256=anchor.document_sha256,
        parser_version=anchor.parser_version,
        anchor_status=anchor.anchor_status,
        anchors=[anchor],
    )

from typing import Any

from app.schemas.evidence import AnchorStatus, Citation, EvidenceItem

REPORT_GUIDANCE = """Write a concise research report with these sections:
## Question and scope
## Findings
## Comparison
## Limitations
## Evidence gaps
## Interpretation
## References

Use only supplied evidence for statements about the papers. Cite every factual sentence with its
evidence marker, such as [E1]. In Comparison, compare papers only when the evidence describes
compatible tasks, datasets, metrics, and conditions; otherwise state that they are not directly
comparable. Distinguish paper statements from assistant suggestions in Interpretation. Do not
claim that the selected library is comprehensive. State "Not found in retrieved evidence" when a
section lacks support. Do not invent bibliographic details."""


def report_source_manifest(
    citations: list[Citation], evidence: list[EvidenceItem]
) -> list[dict[str, Any]]:
    """Map display citation numbers to only the evidence accepted by grounded chat."""
    evidence_by_id = {item.id: item for item in evidence}
    manifest = []
    for citation in citations:
        item = evidence_by_id.get(citation.evidence_id)
        if item is None or citation.anchor_status is not AnchorStatus.VERIFIED:
            continue
        manifest.append(
            {
                "evidence_id": str(citation.citation_index),
                "source_evidence_id": citation.evidence_id,
                "citation_index": citation.citation_index,
                "paper_id": str(citation.paper_id),
                "paper_title": item.paper_title if item else None,
                "page_number": citation.page_number,
                "chunk_id": str(item.chunk_id) if item and item.chunk_id else None,
                "document_sha256": citation.document_sha256,
                "quote": citation.quote,
            }
        )
    return manifest

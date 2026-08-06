from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Paper
from app.schemas.evidence import AnchorStatus
from app.services.source_resolution import resolve_exact_source_anchor


def revalidate_source_manifest(
    db: Session, *, project_id: UUID, source_manifest: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return display-only source status without changing the stored artifact revision."""
    current: list[dict[str, Any]] = []
    for source in source_manifest:
        item = dict(source)
        try:
            paper_id = UUID(str(item.get("paper_id", "")))
            page_number = int(item.get("page_number", 0))
        except (TypeError, ValueError):
            item["availability"] = "unavailable"
            item["availability_reason"] = "source_identity_missing"
            current.append(item)
            continue

        quote = item.get("quote")
        document_sha256 = item.get("document_sha256")
        paper = db.query(Paper).filter(Paper.id == paper_id, Paper.project_id == project_id).first()
        if paper is None or paper.status != "READY":
            item["availability"] = "unavailable"
            item["availability_reason"] = "paper_unavailable"
        elif not isinstance(quote, str) or not quote.strip() or not document_sha256:
            item["availability"] = "unavailable"
            item["availability_reason"] = "source_identity_incomplete"
        else:
            anchor = resolve_exact_source_anchor(
                db,
                project_id=project_id,
                paper_id=paper_id,
                page_number=page_number,
                exact_quote=quote,
                document_sha256=str(document_sha256),
            )
            item["availability"] = (
                "current"
                if anchor and anchor.anchor_status is AnchorStatus.VERIFIED
                else "unavailable"
            )
            if anchor and anchor.anchor_status is AnchorStatus.VERIFIED:
                item["current_anchor"] = anchor.model_dump(mode="json")
            else:
                item["availability_reason"] = "source_changed_or_quote_unavailable"
        current.append(item)
    return current

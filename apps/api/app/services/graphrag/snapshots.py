"""PostgreSQL snapshot persistence for verified GraphRAG facts.

Stores typed, verified facts and verbatim provenance in the graph_fact_snapshots table
under the paper's active extraction generation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import GraphFactSnapshot
from app.schemas.graph import GraphFactCandidate
from app.services.graphrag.identity import (
    generate_entity_key,
    generate_fact_id,
    validate_and_canonicalize_external_id,
)


def _extract_external_id(entity_id: str | None, entity_type: Any) -> str | None:
    """Extract and validate external_id if entity_id represents an external identifier."""
    if not entity_id or not isinstance(entity_id, str):
        return None
    type_str = (
        entity_type.value.lower() if hasattr(entity_type, "value") else str(entity_type).lower()
    )
    # If the ID is an internal generated key (e.g. "method_...", "proj_..."), omit external_id
    if entity_id.startswith(f"{type_str}_") or entity_id.startswith("proj_"):
        return None
    if ":" in entity_id:
        try:
            return validate_and_canonicalize_external_id(entity_id)
        except ValueError:
            return None
    return None


def get_verified_fact_snapshots(
    db: Session,
    paper_id: UUID,
    generation_id: str,
) -> list[GraphFactSnapshot]:
    """Retrieve existing verified fact snapshots for a paper and generation."""
    return (
        db.query(GraphFactSnapshot)
        .filter(
            GraphFactSnapshot.paper_id == paper_id,
            GraphFactSnapshot.generation_id == generation_id,
        )
        .order_by(GraphFactSnapshot.created_at.asc())
        .all()
    )


def persist_verified_fact_snapshots(
    db: Session,
    project_id: UUID,
    paper_id: UUID,
    generation_id: str,
    event_id: UUID | None,
    verified_candidates: list[GraphFactCandidate],
    ontology_version: str = "1.0.0",
) -> list[GraphFactSnapshot]:
    """Persist verified fact candidates as immutable snapshots in PostgreSQL.

    Idempotency:
    - If snapshots already exist for (paper_id, generation_id), returns the existing
      snapshot rows without re-inserting or duplicating facts.
    - Within the batch, facts sharing identical deterministic fact IDs are deduplicated.
    """
    # 1. Check for existing snapshots (idempotent retry reuse)
    existing = get_verified_fact_snapshots(db, paper_id=paper_id, generation_id=generation_id)
    if existing:
        return existing

    if not verified_candidates:
        return []

    snapshots: list[GraphFactSnapshot] = []
    seen_fact_ids: set[str] = set()

    for candidate in verified_candidates:
        s_ext_id = _extract_external_id(candidate.subject.id, candidate.subject.type)
        o_ext_id = _extract_external_id(candidate.object.id, candidate.object.type)

        subject_key = generate_entity_key(
            project_id=project_id,
            entity_type=candidate.subject.type,
            name=candidate.subject.name,
            external_id=s_ext_id,
        )
        object_key = generate_entity_key(
            project_id=project_id,
            entity_type=candidate.object.type,
            name=candidate.object.name,
            external_id=o_ext_id,
        )

        fact_id = generate_fact_id(
            project_id=project_id,
            paper_id=paper_id,
            predicate=candidate.predicate,
            subject_key=subject_key,
            object_key=object_key,
            char_start=candidate.provenance.char_start,
            char_end=candidate.provenance.char_end,
            qualifiers=candidate.qualifiers,
            source_generation=generation_id,
        )

        if fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact_id)

        subject_type_str = (
            candidate.subject.type.value
            if hasattr(candidate.subject.type, "value")
            else str(candidate.subject.type)
        )
        predicate_str = (
            candidate.predicate.value
            if hasattr(candidate.predicate, "value")
            else str(candidate.predicate)
        )
        object_type_str = (
            candidate.object.type.value
            if hasattr(candidate.object.type, "value")
            else str(candidate.object.type)
        )
        qualifiers_dict = (
            candidate.qualifiers.model_dump(mode="json", exclude_none=True)
            if candidate.qualifiers
            else None
        )

        snapshot = GraphFactSnapshot(
            fact_id=fact_id,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            event_id=event_id,
            subject_key=subject_key,
            subject_name=candidate.subject.name,
            subject_type=subject_type_str,
            predicate=predicate_str,
            object_key=object_key,
            object_name=candidate.object.name,
            object_type=object_type_str,
            qualifiers=qualifiers_dict,
            chunk_id=candidate.provenance.chunk_id,
            page_number=candidate.provenance.page_number,
            element_id=candidate.provenance.element_id,
            char_start=candidate.provenance.char_start,
            char_end=candidate.provenance.char_end,
            exact_quote=candidate.provenance.exact_quote,
            document_sha256=candidate.provenance.document_sha256,
            validation_version=ontology_version,
        )
        snapshots.append(snapshot)

    if snapshots:
        db.add_all(snapshots)
        db.flush()

    return snapshots

"""Deterministic project-scoped identities for GraphRAG entities and facts.

Provides:
- canonicalize_name: lowercases, strips whitespace, collapses multiple whitespace runs.
- validate_and_canonicalize_external_id: validates external identifier schemes and formats.
- generate_entity_key: deterministic project-scoped entity node keys.
- canonicalize_qualifiers: sorted, deterministic representation of qualifier key-values.
- generate_fact_id: deterministic fact relation IDs preserving multi-source provenance.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.schemas.graph import EntityType, RelationshipPredicate


def canonicalize_name(name: str) -> str:
    """Canonicalize name by lowercasing, stripping, and collapsing multiple spaces."""
    if not name:
        return ""
    return " ".join(name.strip().lower().split())


def validate_and_canonicalize_external_id(external_id: str) -> str:
    """Validate external identifier format and return its canonical string representation.

    Format must be '<scheme>:<identifier>' (e.g. doi:..., arxiv:..., orcid:...).
    """
    cleaned = external_id.strip()
    if not cleaned:
        raise ValueError("external_id cannot be empty or whitespace-only")

    match = re.match(r"^([a-zA-Z0-9_\-]+):([^\s]+)$", cleaned)
    if not match:
        raise ValueError(
            f"Invalid external_id format: '{external_id}'. "
            "Must be formatted as '<scheme>:<identifier>' without internal whitespace "
            "(e.g. doi:..., arxiv:..., orcid:...)."
        )
    scheme = match.group(1).lower()
    ident = match.group(2).strip().lower()
    if not ident:
        raise ValueError(
            f"Invalid external_id format: '{external_id}'. Identifier cannot be empty."
        )
    return f"{scheme}:{ident}"


def generate_entity_key(
    project_id: UUID,
    entity_type: EntityType | str,
    name: str,
    external_id: str | None = None,
) -> str:
    """Generate a deterministic, project-scoped entity key.

    Invariants:
    - Same name in two different projects MUST produce different keys.
    - If external_id is provided, validates format and returns
      f"proj_{project_id.hex}_{entity_type.lower()}_{canonical_ext_id}".
    - Otherwise hashes f"{project_id}:{entity_type.lower()}:{canonical_name}" with SHA-256
      and returns f"{entity_type.lower()}_{hash[:16]}".
    """
    type_str = (
        entity_type.value.lower()
        if isinstance(entity_type, EntityType)
        else str(entity_type).strip().lower()
    )

    if external_id is not None and external_id.strip():
        canonical_ext_id = validate_and_canonicalize_external_id(external_id)
        proj_prefix = project_id.hex
        return f"proj_{proj_prefix}_{type_str}_{canonical_ext_id}"

    canonical_name = canonicalize_name(name)
    if not canonical_name:
        raise ValueError("Entity name cannot be empty or whitespace-only")

    payload = f"{project_id}:{type_str}:{canonical_name}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{type_str}_{h[:16]}"


def canonicalize_qualifiers(qualifiers: dict[str, Any] | BaseModel | None) -> str:
    """Canonicalize qualifier dictionary or BaseModel into a sorted, deterministic
    key-value string.
    """
    if not qualifiers:
        return ""

    if isinstance(qualifiers, BaseModel):
        data = qualifiers.model_dump(exclude_none=True)
    elif isinstance(qualifiers, dict):
        data = qualifiers
    else:
        return ""

    if not data:
        return ""

    items: list[tuple[str, str]] = []
    for k, v in data.items():
        if v is None:
            continue
        clean_k = canonicalize_name(str(k))
        if not clean_k:
            continue

        # Unwrap enum values if present
        if hasattr(v, "value"):
            v = v.value

        if isinstance(v, float):
            clean_v = f"{v:.6g}"
        elif isinstance(v, (int, bool)):
            clean_v = str(v).lower()
        elif isinstance(v, str):
            clean_v = canonicalize_name(v)
        else:
            clean_v = canonicalize_name(str(v))

        items.append((clean_k, clean_v))

    items.sort(key=lambda x: x[0])
    return ";".join(f"{k}={v}" for k, v in items)


def generate_fact_id(
    project_id: UUID,
    paper_id: UUID,
    predicate: RelationshipPredicate | str,
    subject_key: str,
    object_key: str,
    char_start: int,
    char_end: int,
    qualifiers: dict[str, Any] | BaseModel | None = None,
    source_generation: str | None = None,
) -> str:
    """Generate a deterministic, provenance-grounded fact identifier.

    Invariants:
    - Same relation asserted by two different papers MUST produce two distinct fact IDs
      (preserves both source facts).
    - Exact same input returns the exact same fact ID idempotently.
    """
    subj_str = subject_key.strip()
    obj_str = object_key.strip()
    if not subj_str:
        raise ValueError("subject_key cannot be empty or whitespace-only")
    if not obj_str:
        raise ValueError("object_key cannot be empty or whitespace-only")

    pred_str = (
        predicate.value
        if isinstance(predicate, RelationshipPredicate)
        else str(predicate).strip().upper()
    )
    canon_qual = canonicalize_qualifiers(qualifiers)
    src_gen = str(source_generation).strip() if source_generation else ""

    payload = (
        f"{project_id}:{paper_id}:{pred_str}:{subj_str}:{obj_str}:"
        f"{char_start}:{char_end}:{canon_qual}:{src_gen}"
    )
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"fact_{h[:24]}"

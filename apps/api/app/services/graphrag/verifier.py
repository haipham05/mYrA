"""Claim and source verification for GraphRAG fact candidates.

Re-resolves candidate provenance against authoritative PostgreSQL ground truth
(paper, page, chunk, element, document SHA-256) and verifies text grounding,
entity support, directional alignment, numeric consistency, and polarity matching.
"""

from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.evidence import AnchorStatus
from app.schemas.graph import (
    ClaimPolarity,
    GraphEntitySchema,
    GraphFactCandidate,
    GraphQualifierSchema,
    RelationshipPredicate,
    normalize_decimal_spaces,
)
from app.services.graphrag.provenance import resolve_graph_source_anchor

NEGATION_PATTERNS = [
    r"\bdid\s+not\b",
    r"\bdidn't\b",
    r"\bfailed\s+to\b",
    r"\bfails\s+to\b",
    r"\bfailure\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bdoes\s+not\b",
    r"\bdoesn't\b",
    r"\bwas\s+not\b",
    r"\bwasn't\b",
    r"\bcould\s+not\b",
    r"\bcouldn't\b",
    r"\bunable\s+to\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bnot\b",
]


def _is_entity_mentioned(entity: GraphEntitySchema, quote_norm: str) -> bool:
    """Check if an entity's name or any alias is mentioned in the normalized quote.

    Supports substring, token-based matching, and non-numeric descriptor matching
    for measurement/result entities (e.g. "8.5% ECE" where "ECE" is the entity
    and 8.5 is evaluated by numeric verification).
    """
    names_to_check = [entity.name]
    if entity.aliases:
        names_to_check.extend(entity.aliases)

    quote_tokens = re.findall(r"\w+", quote_norm)

    for raw_name in names_to_check:
        if not raw_name:
            continue
        cleaned = " ".join(raw_name.strip().lower().split())
        if not cleaned:
            continue

        # Strict word-boundary check for 1-2 character names to prevent false positive substrings
        if len(cleaned) <= 2:
            pattern = rf"\b{re.escape(cleaned)}\b"
            if re.search(pattern, quote_norm):
                return True
            continue

        # Direct case-insensitive substring match
        if cleaned in quote_norm:
            return True

        # Token-based match (e.g., handles punctuation variations like "GPT 4" vs "GPT-4")
        name_tokens = re.findall(r"\w+", cleaned)
        if name_tokens and all(token in quote_tokens for token in name_tokens):
            return True

        # If entity contains a numeric value (e.g. "8.5% ECE" or "2.1% ECE"), check if the
        # non-numeric semantic descriptor (e.g. "ECE") is present in the quote.
        # This allows the subsequent numeric verification check to properly flag
        # NUMERIC_VALUE_MISMATCH rather than masking it as an ungrounded entity.
        alpha_tokens = [t for t in re.findall(r"[a-zA-Z]+", cleaned) if len(t) > 1]
        if alpha_tokens and all(token in quote_tokens for token in alpha_tokens):
            return True

    return False


def _check_reversed_actor(
    predicate: RelationshipPredicate | str,
    subject: GraphEntitySchema,
    object_: GraphEntitySchema,
    quote_norm: str,
) -> bool:
    """Detect if subject and object roles are reversed in directional assertions.

    For example, if quote says 'Model Alpha extends Model Beta' but candidate
    asserts that Model Beta extends Model Alpha.
    """
    pred_str = (
        predicate.value if isinstance(predicate, RelationshipPredicate) else str(predicate).upper()
    )

    if pred_str == RelationshipPredicate.EXTENDS.value:
        extends_markers = [
            "extends",
            "extended",
            "extension of",
            "builds upon",
            "built on",
            "derived from",
        ]
        s_name = subject.name.strip().lower()
        o_name = object_.name.strip().lower()

        for marker in extends_markers:
            m_idx = quote_norm.find(marker)
            if m_idx != -1:
                s_idx = quote_norm.find(s_name)
                o_idx = quote_norm.find(o_name)
                if s_idx != -1 and o_idx != -1:
                    # Object appears before marker and subject appears after marker:
                    # e.g., "Branchformer extends Conformer" -> o_idx < m_idx < s_idx
                    if o_idx < m_idx < s_idx:
                        return True

    return False


def _check_numeric_support(
    qualifiers: GraphQualifierSchema | None,
    candidate_entities: list[GraphEntitySchema],
    quote_norm: str,
    tolerance: float = 1e-4,
) -> bool:
    """Verify that numeric claims in qualifiers or entity names are supported by the quote text."""
    quote_norm_decimals = normalize_decimal_spaces(quote_norm)

    # Extract all floats from the quote text
    found_floats: list[float] = []
    for m in re.finditer(r"[+-]?\d+(?:\.\d+)?", quote_norm_decimals):
        try:
            found_floats.append(float(m.group(0)))
        except ValueError:
            pass

    # 1. Check qualifiers numeric values
    if qualifiers is not None:
        target_val = qualifiers.result_value
        if target_val is None:
            target_val = qualifiers.numeric_value

        # Match raw_value directly if present
        if qualifiers.raw_value:
            raw_clean = normalize_decimal_spaces(qualifiers.raw_value.strip().lower())
            if raw_clean in quote_norm or raw_clean in quote_norm_decimals:
                return True

        if target_val is not None:
            # Check numeric match within floating-point tolerance
            if any(abs(f - target_val) < tolerance for f in found_floats):
                return True

            # Check string representations (e.g. "8.5" or "8" if integer)
            target_str = f"{target_val:.6g}".lower()
            if target_str in quote_norm_decimals:
                return True
            if target_val.is_integer() and str(int(target_val)) in quote_norm_decimals:
                return True

            return False

    # 2. Check numbers in entity names (e.g. Result named "8.5% ECE")
    for ent in candidate_entities:
        ent_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", ent.name)
        for num_str in ent_numbers:
            try:
                num_val = float(num_str)
                if not any(abs(f - num_val) < tolerance for f in found_floats):
                    return False
            except ValueError:
                pass

    return True


def _has_explicit_negation(text: str) -> bool:
    """Check if normalized text contains explicit negation markers."""
    t = text.lower()
    return any(re.search(pat, t) is not None for pat in NEGATION_PATTERNS)


def verify_candidate_fact(
    db: Session,
    project_id: UUID,
    candidate: GraphFactCandidate,
) -> tuple[bool, str | None]:
    """Verify an extracted candidate fact against PostgreSQL ground truth and quote text.

    Steps:
    1. Re-resolve provenance against authoritative Postgres paper, chunk, element, hash.
       If unverified -> (False, "UNRESOLVED_ANCHOR").
    2. Text Grounding & Claim Validation against candidate.provenance.exact_quote:
       - Quote normalization (lowercase, extra whitespace stripped).
       - Entity support check (both subject and object must be mentioned in quote).
       - Reversed actor check (directional relations must not be swapped).
       - Qualifier / Numeric check (claimed numbers must appear in quote).
       - Polarity / Negation check (negation markers in quote vs candidate polarity).

    Returns:
    - (True, None) if all checks pass.
    - (False, error_code) otherwise.
    """
    # 1. Re-resolve provenance anchor
    anchor, status = resolve_graph_source_anchor(db, project_id, candidate.provenance)
    if status != AnchorStatus.VERIFIED or anchor is None:
        return (False, "UNRESOLVED_ANCHOR")

    # 2. Quote normalization
    quote = candidate.provenance.exact_quote
    if not quote or not isinstance(quote, str):
        return (False, "ENTITIES_NOT_IN_QUOTE")
    quote_norm = " ".join(quote.strip().lower().split())
    if not quote_norm:
        return (False, "ENTITIES_NOT_IN_QUOTE")

    # 3. Entity support check (both subject and object must be mentioned)
    subject_ok = _is_entity_mentioned(candidate.subject, quote_norm)
    object_ok = _is_entity_mentioned(candidate.object, quote_norm)
    if not (subject_ok and object_ok):
        return (False, "ENTITIES_NOT_IN_QUOTE")

    # 4. Reversed actor check
    if _check_reversed_actor(candidate.predicate, candidate.subject, candidate.object, quote_norm):
        return (False, "REVERSED_ACTOR")

    # 5. Qualifier / Numeric check
    if not _check_numeric_support(
        candidate.qualifiers,
        [candidate.subject, candidate.object],
        quote_norm,
    ):
        return (False, "NUMERIC_VALUE_MISMATCH")

    # 6. Polarity / Negation check
    quote_for_neg = quote_norm.replace("not only", "")
    has_neg = _has_explicit_negation(quote_for_neg)
    polarity = (
        candidate.qualifiers.polarity
        if candidate.qualifiers and candidate.qualifiers.polarity
        else ClaimPolarity.POSITIVE
    )

    if has_neg and polarity == ClaimPolarity.POSITIVE:
        return (False, "POLARITY_CONTRADICTION")
    if not has_neg and polarity == ClaimPolarity.NEGATIVE:
        return (False, "POLARITY_CONTRADICTION")

    return (True, None)

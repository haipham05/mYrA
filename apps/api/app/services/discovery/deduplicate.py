"""Conservative candidate matching that never hides title-only matches."""

import re
from difflib import SequenceMatcher

from app.schemas.discovery import CatalogCandidate

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_TITLE_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _DOI_PREFIX.sub("", value.strip()).strip().rstrip(".,;)").casefold()
    return normalized or None


def _normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().casefold()
    for prefix in ("https://arxiv.org/abs/", "https://arxiv.org/pdf/", "arxiv:"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    return normalized or None


def _normalize_title(value: str) -> str:
    return _TITLE_NON_ALNUM.sub(" ", value.casefold()).strip()


def _same_work(left: CatalogCandidate, right: CatalogCandidate) -> bool:
    left_doi, right_doi = _normalize_doi(left.doi), _normalize_doi(right.doi)
    if left_doi and right_doi and left_doi == right_doi:
        return True

    left_arxiv = _normalize_arxiv_id(left.arxiv_id)
    right_arxiv = _normalize_arxiv_id(right.arxiv_id)
    has_different_dois = bool(left_doi and right_doi and left_doi != right_doi)
    # Preserve a preprint and a separately identified published version.
    return bool(left_arxiv and right_arxiv and left_arxiv == right_arxiv and not has_different_dois)


def _merge_metadata(existing: CatalogCandidate, incoming: CatalogCandidate) -> CatalogCandidate:
    updates: dict[str, object] = {}
    for field in ("doi", "arxiv_id", "abstract", "pdf_url"):
        if not getattr(existing, field) and getattr(incoming, field):
            updates[field] = getattr(incoming, field)
    for field in ("publication_year", "open_access"):
        if getattr(existing, field) is None and getattr(incoming, field) is not None:
            updates[field] = getattr(incoming, field)
    if not existing.authors and incoming.authors:
        updates["authors"] = incoming.authors
    if existing.possible_duplicate or incoming.possible_duplicate:
        updates["possible_duplicate"] = True
    return existing.model_copy(update=updates)


def deduplicate_candidates(candidates: list[CatalogCandidate]) -> list[CatalogCandidate]:
    """Coalesce exact IDs; mark similar titles without removing either candidate."""
    if len(candidates) > 50:
        raise ValueError("at most 50 catalog candidates can be deduplicated at once")

    results: list[CatalogCandidate] = []
    for candidate in candidates:
        match_index = next(
            (index for index, result in enumerate(results) if _same_work(result, candidate)),
            None,
        )
        if match_index is not None:
            results[match_index] = _merge_metadata(results[match_index], candidate)
            continue

        normalized_title = _normalize_title(candidate.title)
        similar_indexes = [
            index
            for index, result in enumerate(results)
            if normalized_title
            and SequenceMatcher(None, normalized_title, _normalize_title(result.title)).ratio()
            >= 0.92
        ]
        if similar_indexes:
            candidate = candidate.model_copy(update={"possible_duplicate": True})
            for index in similar_indexes:
                results[index] = results[index].model_copy(update={"possible_duplicate": True})
        results.append(candidate)
    return results

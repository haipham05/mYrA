"""M1 evidence resolution for GraphRAG facts and query candidates.

Re-resolves extracted graph facts against PostgreSQL ground truth
(papers, chunks, elements, pages, document SHA-256) into verified EvidenceItems
with live CitationAnchors. Prevents hallucinated or uncited assertions and
ensures bare Neo4j edges without ground truth are dropped.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import ChunkElement, GraphFactSnapshot, Paper, PaperChunk, PaperElement
from app.schemas.evidence import AnchorStatus, BoundingBox, CitationAnchor, EvidenceItem
from app.services.graphrag.provenance import resolve_graph_source_anchor

logger = logging.getLogger(__name__)


def resolve_graph_fact_to_evidence(
    db: Session,
    project_id: UUID,
    fact: str | dict[str, Any] | GraphFactSnapshot,
    evidence_id: str = "G1",
) -> tuple[EvidenceItem | None, CitationAnchor | None, AnchorStatus]:
    """Resolve a single graph fact (by ID, dict candidate, or snapshot) to verified EvidenceItem.

    Looks up GraphFactSnapshot from PostgreSQL if given fact_id (str) or candidate dict
    with fact_id or id.
    If fact is already a GraphFactSnapshot, verifies fact.project_id == project_id.
    If no snapshot found in DB and dict has provenance fields (paper_id, chunk_id, page_number,
    exact_quote, document_sha256), uses dict for resolution; otherwise if no PostgreSQL backing
    exists (e.g. bare Neo4j edge), returns (None, None, AnchorStatus.UNRESOLVED).

    Calls resolve_graph_source_anchor to verify against live PostgreSQL ground truth.
    If anchor resolution returns anything other than AnchorStatus.VERIFIED or anchor is None,
    returns (None, None, AnchorStatus.UNRESOLVED).

    Enforces:
    - Paper must exist, status == "READY", document_sha256 must match.
    - PaperChunk must exist, chunk_type == "child".
    - PaperElement on that page.
    - Parent context expansion: fetch parent chunk if available via ChunkElement
      where chunk_type == "parent" sharing element_id, or default to chunk.text.

    Returns:
    - (EvidenceItem, CitationAnchor, AnchorStatus.VERIFIED) on success
    - (None, None, AnchorStatus.UNRESOLVED) on failure
    """
    snapshot: GraphFactSnapshot | None = None
    prov_dict: dict[str, Any] | None = None

    if isinstance(fact, GraphFactSnapshot):
        if fact.project_id != project_id:
            logger.debug(
                "GraphFactSnapshot project mismatch: %s != %s", fact.project_id, project_id
            )
            return (None, None, AnchorStatus.UNRESOLVED)
        snapshot = fact
        prov_dict = {
            "paper_id": snapshot.paper_id,
            "chunk_id": snapshot.chunk_id,
            "element_id": snapshot.element_id,
            "page_number": snapshot.page_number,
            "exact_quote": snapshot.exact_quote,
            "char_start": snapshot.char_start,
            "char_end": snapshot.char_end,
            "document_sha256": snapshot.document_sha256,
            "parser_version": getattr(snapshot, "parser_version", None)
            or getattr(snapshot, "validation_version", None)
            or "v1",
        }
    elif isinstance(fact, str):
        fact_id = fact.strip()
        if not fact_id:
            return (None, None, AnchorStatus.UNRESOLVED)
        snapshot = (
            db.query(GraphFactSnapshot)
            .filter(
                GraphFactSnapshot.fact_id == fact_id,
                GraphFactSnapshot.project_id == project_id,
            )
            .first()
        )
        if not snapshot:
            logger.debug("Fact ID %s has no PostgreSQL snapshot in project %s", fact_id, project_id)
            return (None, None, AnchorStatus.UNRESOLVED)
        prov_dict = {
            "paper_id": snapshot.paper_id,
            "chunk_id": snapshot.chunk_id,
            "element_id": snapshot.element_id,
            "page_number": snapshot.page_number,
            "exact_quote": snapshot.exact_quote,
            "char_start": snapshot.char_start,
            "char_end": snapshot.char_end,
            "document_sha256": snapshot.document_sha256,
            "parser_version": getattr(snapshot, "parser_version", None)
            or getattr(snapshot, "validation_version", None)
            or "v1",
        }
    elif isinstance(fact, dict):
        cand_proj = fact.get("project_id")
        if cand_proj is not None:
            try:
                if UUID(str(cand_proj)) != project_id:
                    logger.debug("Candidate dict project mismatch: %s != %s", cand_proj, project_id)
                    return (None, None, AnchorStatus.UNRESOLVED)
            except (ValueError, AttributeError):
                return (None, None, AnchorStatus.UNRESOLVED)

        raw_id = fact.get("fact_id") or fact.get("id")
        if raw_id is not None and isinstance(raw_id, (str, int, UUID)):
            fid_str = str(raw_id).strip()
            snapshot = (
                db.query(GraphFactSnapshot)
                .filter(
                    GraphFactSnapshot.fact_id == fid_str,
                    GraphFactSnapshot.project_id == project_id,
                )
                .first()
            )

        if snapshot is not None:
            prov_dict = {
                "paper_id": snapshot.paper_id,
                "chunk_id": snapshot.chunk_id,
                "element_id": snapshot.element_id,
                "page_number": snapshot.page_number,
                "exact_quote": snapshot.exact_quote,
                "char_start": snapshot.char_start,
                "char_end": snapshot.char_end,
                "document_sha256": snapshot.document_sha256,
                "parser_version": getattr(snapshot, "parser_version", None)
                or getattr(snapshot, "validation_version", None)
                or "v1",
            }
        else:
            raw_prov = fact.get("provenance") if isinstance(fact.get("provenance"), dict) else fact
            required_prov = [
                "paper_id",
                "chunk_id",
                "page_number",
                "exact_quote",
                "document_sha256",
            ]
            if all(raw_prov.get(k) is not None for k in required_prov):
                prov_dict = {
                    "paper_id": raw_prov.get("paper_id"),
                    "chunk_id": raw_prov.get("chunk_id"),
                    "element_id": raw_prov.get("element_id"),
                    "page_number": raw_prov.get("page_number"),
                    "exact_quote": raw_prov.get("exact_quote"),
                    "char_start": raw_prov.get("char_start"),
                    "char_end": raw_prov.get("char_end"),
                    "document_sha256": raw_prov.get("document_sha256"),
                    "parser_version": raw_prov.get("parser_version") or "v1",
                }
            else:
                logger.debug("Fact dict has neither DB snapshot nor complete provenance fields")
                return (None, None, AnchorStatus.UNRESOLVED)
    else:
        return (None, None, AnchorStatus.UNRESOLVED)

    if not prov_dict:
        return (None, None, AnchorStatus.UNRESOLVED)

    # Resolve and verify source anchor against live Postgres ground truth
    anchor, status = resolve_graph_source_anchor(db, project_id, prov_dict)
    if status != AnchorStatus.VERIFIED or anchor is None:
        return (None, None, AnchorStatus.UNRESOLVED)

    try:
        paper_id = UUID(str(prov_dict["paper_id"]))
        chunk_id = UUID(str(prov_dict["chunk_id"]))
    except (ValueError, TypeError):
        return (None, None, AnchorStatus.UNRESOLVED)

    # 1. Paper verification
    paper = db.query(Paper).filter(Paper.id == paper_id, Paper.project_id == project_id).first()
    if not paper or paper.status != "READY" or not paper.document_sha256:
        return (None, None, AnchorStatus.UNRESOLVED)

    if prov_dict.get("document_sha256"):
        if (
            str(prov_dict["document_sha256"]).strip().lower()
            != paper.document_sha256.strip().lower()
        ):
            return (None, None, AnchorStatus.UNRESOLVED)

    # 2. Chunk verification (must exist and be child chunk)
    chunk = (
        db.query(PaperChunk)
        .filter(PaperChunk.id == chunk_id, PaperChunk.paper_id == paper.id)
        .first()
    )
    if not chunk or chunk.chunk_type != "child":
        return (None, None, AnchorStatus.UNRESOLVED)

    # 3. Element verification
    element: PaperElement | None = None
    if anchor.source_element_id:
        element = (
            db.query(PaperElement)
            .filter(
                PaperElement.id == anchor.source_element_id,
                PaperElement.paper_id == paper.id,
                PaperElement.page_number == anchor.page_number,
            )
            .first()
        )
    if not element and prov_dict.get("element_id"):
        try:
            raw_elem_id = UUID(str(prov_dict["element_id"]))
            element = (
                db.query(PaperElement)
                .filter(
                    PaperElement.id == raw_elem_id,
                    PaperElement.paper_id == paper.id,
                    PaperElement.page_number == anchor.page_number,
                )
                .first()
            )
        except (ValueError, TypeError):
            pass

    if not element:
        return (None, None, AnchorStatus.UNRESOLVED)

    # 4. Parent context expansion: fetch parent chunk if available via ChunkElement
    # where chunk_type == "parent" sharing element_id, or default to chunk.text
    parent_context: str | None = None
    if element:
        parent_chunk = (
            db.query(PaperChunk)
            .join(ChunkElement, ChunkElement.chunk_id == PaperChunk.id)
            .filter(
                PaperChunk.paper_id == paper.id,
                PaperChunk.chunk_type == "parent",
                ChunkElement.element_id == element.id,
            )
            .first()
        )
        if parent_chunk and parent_chunk.text:
            parent_context = parent_chunk.text

    if not parent_context:
        child_elem_ids = [
            ce.element_id
            for ce in db.query(ChunkElement.element_id)
            .filter(ChunkElement.chunk_id == chunk.id)
            .all()
        ]
        if child_elem_ids:
            parent_chunk = (
                db.query(PaperChunk)
                .join(ChunkElement, ChunkElement.chunk_id == PaperChunk.id)
                .filter(
                    PaperChunk.paper_id == paper.id,
                    PaperChunk.chunk_type == "parent",
                    ChunkElement.element_id.in_(child_elem_ids),
                )
                .first()
            )
            if parent_chunk and parent_chunk.text:
                parent_context = parent_chunk.text

    if not parent_context:
        parent_context = chunk.text

    # Extract bounding boxes
    bboxes: list[BoundingBox] = []
    if anchor.bounding_boxes:
        bboxes = list(anchor.bounding_boxes)
    elif getattr(anchor, "bounding_box", None):
        bboxes = [anchor.bounding_box]

    # Source element IDs
    source_element_ids: list[UUID] = [element.id] if element else []
    if not source_element_ids and anchor.source_element_id:
        source_element_ids = [anchor.source_element_id]

    evidence_item = EvidenceItem(
        id=evidence_id,
        paper_id=paper.id,
        paper_title=paper.filename,
        chunk_id=chunk.id,
        quote=anchor.exact_quote,
        parent_context=parent_context,
        page_number=anchor.page_number,
        bounding_boxes=bboxes,
        source_element_ids=source_element_ids,
        document_sha256=paper.document_sha256,
        parser_version=anchor.parser_version,
        anchors=[anchor],
    )
    return (evidence_item, anchor, AnchorStatus.VERIFIED)


def resolve_graph_facts_to_evidence(
    db: Session,
    project_id: UUID,
    facts: list[str | dict[str, Any] | GraphFactSnapshot],
    prefix: str = "G",
    start_index: int = 1,
) -> list[EvidenceItem]:
    """Resolve a batch of graph facts into verified EvidenceItems.

    - Resolves batch of facts into verified EvidenceItems.
    - Deduplicates identical quotes or fact IDs within the batch.
    - Drops any unverified or stale facts.
    - Assigns sequential IDs: f"{prefix}{start_index + idx}".
    - Returns list[EvidenceItem].
    """
    verified_items: list[EvidenceItem] = []
    seen_fact_ids: set[str] = set()
    seen_quotes: set[str] = set()

    for fact in facts:
        # Determine fact_id for preliminary deduplication if known
        fid: str | None = None
        if isinstance(fact, str):
            fid = fact.strip()
        elif isinstance(fact, GraphFactSnapshot):
            fid = fact.fact_id
        elif isinstance(fact, dict):
            raw_id = fact.get("fact_id") or fact.get("id")
            if raw_id is not None and isinstance(raw_id, (str, int, UUID)):
                fid = str(raw_id).strip()

        if fid and fid in seen_fact_ids:
            continue

        assigned_id = f"{prefix}{start_index + len(verified_items)}"
        item, anchor, status = resolve_graph_fact_to_evidence(
            db=db,
            project_id=project_id,
            fact=fact,
            evidence_id=assigned_id,
        )

        if status != AnchorStatus.VERIFIED or item is None:
            continue

        # Quote deduplication
        norm_quote = item.quote.strip()
        if norm_quote in seen_quotes:
            continue

        seen_quotes.add(norm_quote)
        if fid:
            seen_fact_ids.add(fid)

        item.id = f"{prefix}{start_index + len(verified_items)}"
        verified_items.append(item)

    return verified_items


def extract_fact_ids_from_candidates(candidates: list[dict[str, Any]]) -> list[str]:
    """Extract fact IDs from relationship candidates, contradiction pairs, or corpus themes.

    Supports:
    - Relationship candidates: c["fact_id"]
    - Contradiction candidates: c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]
    - Corpus themes: c["contributing_fact_ids"] or c["fact_ids"]

    Returns deduplicated list of fact IDs preserving discovery order.
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, (str, int, UUID)):
            s = str(val).strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)

    for c in candidates:
        if not isinstance(c, dict):
            continue

        # 1. Contradiction candidates: c["fact_a"], c["fact_b"]
        if "fact_a" in c or "fact_b" in c:
            fa = c.get("fact_a")
            if isinstance(fa, dict):
                _add(fa.get("fact_id") or fa.get("id"))
            elif fa:
                _add(fa)

            fb = c.get("fact_b")
            if isinstance(fb, dict):
                _add(fb.get("fact_id") or fb.get("id"))
            elif fb:
                _add(fb)

        # 2. Corpus themes: c["contributing_fact_ids"] or c["fact_ids"]
        for key in ("contributing_fact_ids", "fact_ids"):
            if key in c and isinstance(c[key], (list, tuple, set)):
                for fid in c[key]:
                    _add(fid)

        # 3. Relationship candidates: c["fact_id"] (or c["id"] if not a theme/contradiction object)
        if "fact_id" in c:
            _add(c["fact_id"])
        elif (
            "id" in c
            and "fact_a" not in c
            and "theme_type" not in c
            and "contributing_fact_ids" not in c
            and "fact_ids" not in c
        ):
            _add(c["id"])

    return result

"""Intent router and candidate graph retrieval for GraphRAG.

Routes user queries to GraphIntent (FACTUAL, RELATIONSHIP, CONTRADICTION, CORPUS_THEMES),
retrieves bounded graph candidate structures, and re-resolves candidate fact IDs
into verified EvidenceItems against PostgreSQL ground truth.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.observability.telemetry import get_telemetry
from app.schemas.evidence import EvidenceItem
from app.services.graphrag.evidence import (
    extract_fact_ids_from_candidates,
    resolve_graph_facts_to_evidence,
)
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.query_engine import (
    build_contradiction_candidates,
    build_corpus_themes,
    build_relationship_candidates,
)

logger = logging.getLogger(__name__)


class GraphIntent(StrEnum):
    """Categorization of query intent for graph-augmented retrieval."""

    FACTUAL = "factual"
    RELATIONSHIP = "relationship"
    CONTRADICTION = "contradiction"
    CORPUS_THEMES = "corpus_themes"


# Regex patterns for deterministic query classification
CONTRADICTION_PATTERNS = [
    r"\bcontradict\w*",
    r"\bconflict\w*",
    r"\bdisagree\w*",
    r"\bopposing\b",
    r"\bdiffering\s+results?\b",
    r"\bcontrasting\s+findings?\b",
    r"\bdiscrepanc\w*",
]

CORPUS_THEMES_PATTERNS = [
    r"\brecurring\s+themes?\b",
    r"\bcommon\s+themes?\b",
    r"\btrends\s+across\b",
    r"\bacross\s+all\s+papers\b",
    r"\bacross\s+papers\b",
    r"\bcorpus\s+overview\b",
    r"\bcommon\s+methods\s+across\b",
    r"\bpatterns\s+across\b",
    r"\bsummary\s+of\s+(?:the\s+)?corpus\b",
    r"\boverview\s+of\s+(?:the\s+)?corpus\b",
]

RELATIONSHIP_PATTERNS = [
    r"how\s+is\s+.+\s+related\s+to",
    r"\brelationship\s+between\b",
    r"\bconnection\s+between\b",
    r"how\s+does\s+.+\s+compare\s+to",
    r"how\s+do\s+.+\s+and\s+.+\s+relate",
    r"\bdoes\s+.+\s+use\s+.+",
    r"\bassociation\s+between\b",
    r"\blink\s+between\b",
    r"\bhow\s+are\s+.+\s+related\b",
]


def route_query_intent(query: str) -> GraphIntent:
    """Classify user query intent into factual, relationship, contradiction, or corpus themes.

    Deterministic intent classification using keyword and regex matching:
    - CONTRADICTION: contradict, conflict, disagree, opposing, differing results,
      contrasting findings, discrepanc.
    - CORPUS_THEMES: recurring theme, common theme, trends across, across all papers,
      across papers, corpus overview, common methods across, patterns across,
      summary of the corpus.
    - RELATIONSHIP: how is .* related to, relationship between, connection between,
      how does .* compare to, how do .* and .* relate, does .* use .*, association between,
      link between.
    - Default: FACTUAL.
    """
    q = (query or "").strip()
    if not q:
        return GraphIntent.FACTUAL

    for pat in CONTRADICTION_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return GraphIntent.CONTRADICTION

    for pat in CORPUS_THEMES_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return GraphIntent.CORPUS_THEMES

    for pat in RELATIONSHIP_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return GraphIntent.RELATIONSHIP

    return GraphIntent.FACTUAL


def _extract_relationship_entities(query: str) -> tuple[str | None, str | None]:
    """Attempt to extract two entity candidate names from a relationship query."""
    patterns = [
        r"(?:relationship|connection|association|link)\s+between\s+(.+?)\s+and\s+(.+)",
        r"how\s+is\s+(.+?)\s+related\s+to\s+(.+)",
        r"how\s+does\s+(.+?)\s+compare\s+to\s+(.+)",
        r"how\s+do\s+(.+?)\s+and\s+(.+?)\s+relate",
        r"does\s+(.+?)\s+use\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            ea = re.sub(r"[^\w\s-]", "", m.group(1)).strip()
            eb = re.sub(r"[^\w\s-]", "", m.group(2)).strip()
            if ea and eb:
                return (ea, eb)
    return (None, None)


def retrieve_graph_candidates_for_query(
    db: Session,
    repo: Neo4jRepository | None,
    project_id: UUID,
    query: str,
    intent: GraphIntent,
) -> tuple[list[dict[str, Any]], str | None]:
    """Retrieve raw graph candidates based on query intent.

    - If repo is None:
      Return ([], "Graph service is not configured; using text retrieval.").
    - Check connectivity repo.verify_connectivity():
      If False: Return ([], "Graph service is currently offline; using text retrieval.").
    - Based on intent:
      - CONTRADICTION:
        Call build_contradiction_candidates(db, repo, project_id, limit=10).
      - RELATIONSHIP:
        Attempt to extract entity names from query or match against repo.search_nodes:
        If two entities found, call build_relationship_candidates(db, repo, project_id, ...).
        If no pair found, search single entity nodes and get their neighbors.
      - CORPUS_THEMES:
        Call build_corpus_themes(db, repo, project_id, min_papers=2, limit=10).
      - FACTUAL:
        Optionally search entities in query and find 1-hop facts to augment context (limit=5).
    - Return (candidates, None).
    """
    if repo is None:
        return ([], "Graph service is not configured; using text retrieval.")

    try:
        if not repo.verify_connectivity():
            return ([], "Graph service is currently offline; using text retrieval.")
    except Exception as exc:
        logger.warning("Graph service connectivity check failed: %s", exc)
        return ([], "Graph service is currently offline; using text retrieval.")

    try:
        if intent == GraphIntent.CONTRADICTION:
            candidates = build_contradiction_candidates(db, repo, project_id, limit=10)
            return (candidates, None)

        elif intent == GraphIntent.CORPUS_THEMES:
            candidates = build_corpus_themes(db, repo, project_id, min_papers=2, limit=10)
            return (candidates, None)

        elif intent == GraphIntent.RELATIONSHIP:
            # 1. Attempt regex entity extraction
            entity_a, entity_b = _extract_relationship_entities(query)
            if entity_a and entity_b:
                candidates = build_relationship_candidates(
                    db, repo, project_id, entity_a, entity_b, limit=10
                )
                if candidates:
                    return (candidates, None)

            # 2. Attempt node matching against graph
            nodes = repo.search_nodes(project_id, limit=100)
            matched_nodes = [
                n for n in nodes if n.get("name") and n["name"].strip().lower() in query.lower()
            ]
            if len(matched_nodes) >= 2:
                candidates = build_relationship_candidates(
                    db,
                    repo,
                    project_id,
                    matched_nodes[0]["name"],
                    matched_nodes[1]["name"],
                    limit=10,
                )
                if candidates:
                    return (candidates, None)

            # 3. If no pair found, search single entity nodes and get neighbors
            nodes_to_traverse = matched_nodes if matched_nodes else []
            if not nodes_to_traverse:
                tokens = [
                    w
                    for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query)
                    if w.lower()
                    not in {
                        "what",
                        "which",
                        "where",
                        "when",
                        "does",
                        "have",
                        "with",
                        "between",
                        "how",
                        "and",
                        "the",
                    }
                ]
                for tok in tokens[:3]:
                    found = repo.search_nodes(project_id, query=tok, limit=2)
                    for fn in found:
                        if not any(tn.get("key") == fn.get("key") for tn in nodes_to_traverse):
                            nodes_to_traverse.append(fn)

            neighbor_candidates: list[dict[str, Any]] = []
            seen_fact_ids: set[str] = set()
            for node in nodes_to_traverse[:3]:
                neighbors = repo.get_node_neighbors(project_id, node["key"], limit=10)
                for nb in neighbors:
                    fid = nb.get("fact_id")
                    if fid and fid not in seen_fact_ids:
                        seen_fact_ids.add(fid)
                        neighbor_candidates.append(nb)
                    if len(neighbor_candidates) >= 10:
                        break
                if len(neighbor_candidates) >= 10:
                    break
            return (neighbor_candidates[:10], None)

        elif intent == GraphIntent.FACTUAL:
            nodes = repo.search_nodes(project_id, limit=50)
            matched_nodes = [
                n for n in nodes if n.get("name") and n["name"].strip().lower() in query.lower()
            ]
            if not matched_nodes:
                tokens = [
                    w
                    for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query)
                    if w.lower()
                    not in {
                        "what",
                        "which",
                        "where",
                        "when",
                        "does",
                        "have",
                        "with",
                        "between",
                        "how",
                        "and",
                        "the",
                    }
                ]
                for tok in tokens[:2]:
                    found = repo.search_nodes(project_id, query=tok, limit=2)
                    for fn in found:
                        if not any(tn.get("key") == fn.get("key") for tn in matched_nodes):
                            matched_nodes.append(fn)

            factual_candidates: list[dict[str, Any]] = []
            seen_fact_ids = set()
            for node in matched_nodes[:2]:
                neighbors = repo.get_node_neighbors(project_id, node["key"], limit=5)
                for nb in neighbors:
                    fid = nb.get("fact_id")
                    if fid and fid not in seen_fact_ids:
                        seen_fact_ids.add(fid)
                        factual_candidates.append(nb)
                    if len(factual_candidates) >= 5:
                        break
                if len(factual_candidates) >= 5:
                    break
            return (factual_candidates[:5], None)

        return ([], None)

    except Exception as exc:
        logger.warning("Error retrieving graph candidates: %s", exc)
        return ([], "Graph service is currently offline; using text retrieval.")


def retrieve_graph_evidence(
    db: Session,
    repo: Neo4jRepository | None,
    project_id: UUID,
    query: str,
    intent: GraphIntent | None = None,
) -> tuple[list[EvidenceItem], str | None]:
    """Retrieve verified EvidenceItems from graph candidates for query.

    - Determines intent if not provided: intent = route_query_intent(query).
    - Calls retrieve_graph_candidates_for_query(db, repo, project_id, query, intent).
    - Uses extract_fact_ids_from_candidates(candidates) to get candidate fact IDs.
    - Resolves into verified EvidenceItems using resolve_graph_facts_to_evidence.
    - Drops any unverified or stale facts.
    - Returns (verified_evidence_items, outage_notice).
    """
    if intent is None:
        intent = route_query_intent(query)

    telemetry = get_telemetry()
    with telemetry.stage(
        "graph.neo4j_query",
        metadata={"intent": intent.value},
    ) as query_span:
        candidates, outage_notice = retrieve_graph_candidates_for_query(
            db=db,
            repo=repo,
            project_id=project_id,
            query=query,
            intent=intent,
        )
        if query_span is not None:
            query_span.update(
                metadata={
                    "intent": intent.value,
                    "candidate_count": len(candidates),
                    "outcome": "unavailable" if outage_notice else "success",
                }
            )

    if not candidates:
        return ([], outage_notice)

    fact_ids = extract_fact_ids_from_candidates(candidates)
    if not fact_ids:
        return ([], outage_notice)

    with telemetry.stage(
        "graph.evidence_resolution",
        metadata={"candidate_fact_count": len(fact_ids)},
    ) as resolution_span:
        verified_evidence_items = resolve_graph_facts_to_evidence(
            db=db,
            project_id=project_id,
            facts=fact_ids,
            prefix="G",
        )
        if resolution_span is not None:
            resolution_span.update(
                metadata={
                    "candidate_fact_count": len(fact_ids),
                    "verified_evidence_count": len(verified_evidence_items),
                    "rejected_or_duplicate_count": max(
                        0, len(fact_ids) - len(verified_evidence_items)
                    ),
                    "outcome": "resolved" if verified_evidence_items else "unresolved",
                }
            )

    return (verified_evidence_items, outage_notice)

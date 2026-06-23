"""Graph query engine for GraphRAG relationship traversal and corpus theme aggregation.

Implements bounded graph reads, cross-paper contradiction verification, and
verifiable corpus themes strictly scoped to project boundaries.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.graph import ClaimPolarity, RelationshipPredicate, parse_numeric_value
from app.services.graphrag.neo4j_repository import Neo4jRepository

logger = logging.getLogger(__name__)

# Bounded limit caps
MAX_RELATIONSHIP_LIMIT = 50
MAX_CONTRADICTION_LIMIT = 50
MAX_THEMES_LIMIT = 20
MAX_GROUP_SIZE = 50
MAX_FACT_SCAN_LIMIT = 500


def _parse_qualifiers(raw_qualifiers: Any) -> dict[str, Any]:
    """Safely normalize qualifiers to a dict."""
    if isinstance(raw_qualifiers, dict):
        return raw_qualifiers
    if isinstance(raw_qualifiers, str) and raw_qualifiers.strip():
        try:
            parsed = json.loads(raw_qualifiers)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _find_best_node(nodes: list[dict[str, Any]], query_name: str) -> dict[str, Any] | None:
    """Find the best matching node from search results, preferring exact name match."""
    if not nodes:
        return None
    q = query_name.strip().lower()
    for n in nodes:
        if n.get("name", "").strip().lower() == q:
            return n
    for n in nodes:
        aliases = [a.strip().lower() for a in n.get("aliases", []) if isinstance(a, str)]
        if q in aliases:
            return n
    return nodes[0]


def _format_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """Format raw fact dict into a standardized relationship candidate representation."""
    qualifiers = _parse_qualifiers(fact.get("qualifiers"))
    return {
        "fact_id": fact.get("fact_id") or fact.get("id"),
        "predicate": fact.get("predicate"),
        "subject_key": fact.get("subject_key"),
        "subject_name": fact.get("subject_name"),
        "object_key": fact.get("object_key"),
        "object_name": fact.get("object_name"),
        "qualifiers": qualifiers if qualifiers else None,
        "exact_quote": fact.get("exact_quote"),
        "page_number": fact.get("page_number"),
        "paper_id": str(fact.get("paper_id")) if fact.get("paper_id") is not None else None,
    }


def _extract_comparable_context(fact: dict[str, Any]) -> dict[str, Any]:
    """Extract only explicitly supported comparison dimensions from a fact."""
    qualifiers = _parse_qualifiers(fact.get("qualifiers"))
    exact_quote = fact.get("exact_quote") or ""

    # 1. Dataset identification
    dataset: str | None = None
    if qualifiers.get("dataset"):
        dataset = str(qualifiers["dataset"]).strip()
    elif str(fact.get("object_type", "")).lower() == "dataset":
        dataset = str(fact.get("object_name", "")).strip()
    elif str(fact.get("subject_type", "")).lower() == "dataset":
        dataset = str(fact.get("subject_name", "")).strip()

    # 2. Metric identification
    metric: str | None = None
    if qualifiers.get("metric"):
        metric = str(qualifiers["metric"]).strip()
    elif str(fact.get("object_type", "")).lower() == "metric":
        metric = str(fact.get("object_name", "")).strip()
    elif str(fact.get("subject_type", "")).lower() == "metric":
        metric = str(fact.get("subject_name", "")).strip()

    # 3. Method and task
    method = str(qualifiers.get("method") or "").strip() or None
    if method is None and str(fact.get("subject_type", "")).lower() in {"method", "model"}:
        method = str(fact.get("subject_name") or "").strip() or None
    task = str(qualifiers.get("task") or "").strip() or None
    if task is None and str(fact.get("object_type", "")).lower() == "task":
        task = str(fact.get("object_name") or "").strip() or None
    if task is None and str(fact.get("subject_type", "")).lower() == "task":
        task = str(fact.get("subject_name") or "").strip() or None

    # 4. Numeric value. Missing metadata remains unknown; do not infer it from a quote.
    val: float | None = None
    for field in ("result_value", "numeric_value", "raw_value"):
        if field in qualifiers and qualifiers[field] is not None:
            parsed, _ = parse_numeric_value(qualifiers[field])
            if parsed is not None:
                val = parsed
                break

    # 5. Polarity identification. Unknown polarity is not made positive by default.
    polarity: str | None = None
    if qualifiers.get("polarity"):
        p_str = str(qualifiers["polarity"]).strip().upper()
        if p_str in {
            ClaimPolarity.POSITIVE.value,
            ClaimPolarity.NEGATIVE.value,
            ClaimPolarity.UNCERTAIN.value,
        }:
            polarity = p_str

    if polarity is None:
        quote_lower = exact_quote.lower()
        if any(
            neg in quote_lower
            for neg in ("fails to converge", "fail", "failed", "cannot", "unable", "not achieve")
        ):
            polarity = ClaimPolarity.NEGATIVE.value
        elif any(
            pos in quote_lower
            for pos in ("achieve", "improves", "outperforms", "yields", "attains")
        ):
            polarity = ClaimPolarity.POSITIVE.value

    return {
        "method": method,
        "dataset": dataset if dataset else None,
        "metric": metric if metric else None,
        "task": task,
        "split": str(qualifiers["split"]).strip() if qualifiers.get("split") else None,
        "unit": str(qualifiers["unit"]).strip() if qualifiers.get("unit") else None,
        "comparison_condition": (
            str(qualifiers["comparison_condition"]).strip()
            if qualifiers.get("comparison_condition")
            else None
        ),
        "value": val,
        "polarity": polarity,
    }


def _normalize_condition(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).casefold().split())
    return normalized or None


def _comparable_contexts(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Require explicit, matching conditions before comparing two claims."""
    required_dimensions = (
        "method",
        "dataset",
        "metric",
        "task",
        "split",
        "unit",
        "comparison_condition",
    )
    return all(
        _normalize_condition(a.get(dimension)) is not None
        and _normalize_condition(a.get(dimension)) == _normalize_condition(b.get(dimension))
        for dimension in required_dimensions
    )


def build_relationship_candidates(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID,
    entity_a_name: str,
    entity_b_name: str,
    predicate: str | RelationshipPredicate | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Resolve entity names and find relationships between them within project_id.

    - Resolves entity_a_name and entity_b_name strictly inside project_id using
      repo.search_nodes(project_id, query=entity_name).
    - If either entity cannot be found in the project, returns [].
    - Calls repo.find_relationships_between(
      project_id, node_a["key"], node_b["key"], predicate=predicate).
    - Caps results at min(limit, 50).
    - Preserves multiple source assertions and disagreement between papers (does not collapse).
    """
    if limit <= 0:
        return []

    clean_a = entity_a_name.strip() if entity_a_name else ""
    clean_b = entity_b_name.strip() if entity_b_name else ""
    if not clean_a or not clean_b:
        return []

    safe_limit = min(limit, MAX_RELATIONSHIP_LIMIT)

    # Resolve entities inside project
    nodes_a = repo.search_nodes(project_id, query=clean_a)
    nodes_b = repo.search_nodes(project_id, query=clean_b)

    if not nodes_a or not nodes_b:
        return []

    node_a = _find_best_node(nodes_a, clean_a)
    node_b = _find_best_node(nodes_b, clean_b)

    if not node_a or not node_b:
        return []

    # Find relationships (forward and backward to be direction-agnostic)
    pred_val = predicate.value if hasattr(predicate, "value") else predicate
    rels = repo.find_relationships_between(
        project_id,
        subject_key=node_a["key"],
        object_key=node_b["key"],
        predicate=pred_val,
    )

    if node_a["key"] != node_b["key"]:
        rev_rels = repo.find_relationships_between(
            project_id,
            subject_key=node_b["key"],
            object_key=node_a["key"],
            predicate=pred_val,
        )
        seen_ids = {r.get("fact_id") or r.get("id") for r in rels}
        for r in rev_rels:
            fid = r.get("fact_id") or r.get("id")
            if fid not in seen_ids:
                rels.append(r)
                seen_ids.add(fid)

    candidates: list[dict[str, Any]] = []
    for r in rels:
        candidate = _format_fact(r)
        candidates.append(candidate)
        if len(candidates) >= safe_limit:
            break

    return candidates


def build_contradiction_candidates(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Discover candidate contradiction pairs within project_id.

    - Pairs of facts asserting relations on the same subject and object (or method and dataset)
      from different papers.
    - Comparability rule:
      Both facts MUST explicitly agree on method, dataset, metric, task, split, unit,
      and comparison condition.
      - Polarity conflict: one POSITIVE and one NEGATIVE (e.g. fails vs achieves)
        OR Numeric conflict: under matching conditions, the supported values differ.
      - Missing or different comparison conditions abstain; no defaults are invented.
    - Caps results at min(limit, 50).
    - Returns both supported source facts so absence is never treated as disagreement.
    """
    if limit <= 0:
        return []

    safe_limit = min(limit, MAX_CONTRADICTION_LIMIT)

    # Retrieve project facts
    all_facts: list[dict[str, Any]] = []
    skip = 0
    page_size = 100
    while True:
        batch = repo.get_project_facts(project_id, limit=page_size, skip=skip)
        if not batch:
            break
        all_facts.extend(batch)
        if len(batch) < page_size or len(all_facts) >= MAX_FACT_SCAN_LIMIT:
            break
        skip += page_size

    if len(all_facts) < 2:
        return []

    candidates: list[dict[str, Any]] = []

    # Pre-extract comparable contexts
    contexts = [_extract_comparable_context(f) for f in all_facts]

    for i in range(len(all_facts)):
        fact_a = all_facts[i]
        ctx_a = contexts[i]
        paper_a = str(fact_a.get("paper_id")) if fact_a.get("paper_id") is not None else None

        # Every relevant comparison condition must be explicitly supported.
        if any(
            ctx_a.get(dimension) is None
            for dimension in (
                "method",
                "dataset",
                "metric",
                "task",
                "split",
                "unit",
                "comparison_condition",
            )
        ):
            continue

        s_key_a = fact_a.get("subject_key")
        s_name_a = (fact_a.get("subject_name") or "").strip().lower()

        for j in range(i + 1, len(all_facts)):
            fact_b = all_facts[j]
            paper_b = str(fact_b.get("paper_id")) if fact_b.get("paper_id") is not None else None

            # Must be from different papers
            if not paper_a or not paper_b or paper_a == paper_b:
                continue

            ctx_b = contexts[j]

            if not _comparable_contexts(ctx_a, ctx_b):
                continue

            # Must assert relations on the same subject/method or method-dataset
            s_key_b = fact_b.get("subject_key")
            s_name_b = (fact_b.get("subject_name") or "").strip().lower()

            same_subject = (s_key_a and s_key_a == s_key_b) or (s_name_a and s_name_a == s_name_b)
            if not same_subject:
                # Also check if method is specified in qualifiers
                q_a = _parse_qualifiers(fact_a.get("qualifiers"))
                q_b = _parse_qualifiers(fact_b.get("qualifiers"))
                m_a = (q_a.get("method") or "").strip().lower()
                m_b = (q_b.get("method") or "").strip().lower()
                if not (m_a and m_a == m_b):
                    continue

            # Check for conflict: Polarity or Numeric Value
            conflict_type: str | None = None

            # 1. Polarity conflict: one POSITIVE and one NEGATIVE
            if (
                ctx_a["polarity"] == ClaimPolarity.POSITIVE.value
                and ctx_b["polarity"] == ClaimPolarity.NEGATIVE.value
            ) or (
                ctx_a["polarity"] == ClaimPolarity.NEGATIVE.value
                and ctx_b["polarity"] == ClaimPolarity.POSITIVE.value
            ):
                conflict_type = "POLARITY"
            elif ctx_a["value"] is not None and ctx_b["value"] is not None:
                # 2. Numeric conflict: same metric/dataset with differing values
                if abs(ctx_a["value"] - ctx_b["value"]) > 1e-4:
                    conflict_type = "NUMERIC_VALUE"

            if conflict_type:
                candidates.append(
                    {
                        "fact_a": _format_fact(fact_a),
                        "fact_b": _format_fact(fact_b),
                        "conflict_type": conflict_type,
                        "comparison_basis": {
                            "method": ctx_a["method"],
                            "dataset": ctx_a["dataset"],
                            "metric": ctx_a["metric"],
                            "task": ctx_a["task"],
                            "split": ctx_a["split"],
                            "unit": ctx_a["unit"],
                            "comparison_condition": ctx_a["comparison_condition"],
                        },
                    }
                )
                if len(candidates) >= safe_limit:
                    return candidates

    return candidates


def build_corpus_themes(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID,
    min_papers: int = 2,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Group recurring project entities and relations appearing across multiple papers.

    - Groups recurring project entities and relations appearing across min_papers >= 2 papers.
    - Groups by entity or relation key, collects contributing paper_ids and contributing fact_ids.
    - Bounded: caps group size and response count (min(limit, 20)).
    - Does NOT generate unsupported hallucinated summaries; returns inspectable supported themes
      with contributing fact IDs. If insufficient recurring entities exist, returns empty list.
    """
    if limit <= 0:
        return []

    safe_limit = min(limit, MAX_THEMES_LIMIT)
    effective_min_papers = max(2, min_papers)

    # Retrieve project facts
    all_facts: list[dict[str, Any]] = []
    skip = 0
    page_size = 100
    while True:
        batch = repo.get_project_facts(project_id, limit=page_size, skip=skip)
        if not batch:
            break
        all_facts.extend(batch)
        if len(batch) < page_size or len(all_facts) >= MAX_FACT_SCAN_LIMIT:
            break
        skip += page_size

    if not all_facts:
        logger.info(
            "Insufficient recurring entities for project %s (0 facts found).",
            project_id,
        )
        return []

    # Entity grouping: key -> {name, type, paper_ids, fact_ids}
    entity_groups: dict[str, dict[str, Any]] = {}
    # Relation grouping: key -> {name, predicate, subject_name, object_name, paper_ids, fact_ids}
    relation_groups: dict[str, dict[str, Any]] = {}

    for f in all_facts:
        fid = f.get("fact_id") or f.get("id")
        pid = str(f.get("paper_id")) if f.get("paper_id") is not None else None
        if not fid or not pid:
            continue

        s_key = f.get("subject_key")
        s_name = f.get("subject_name") or s_key
        s_type = f.get("subject_type")

        o_key = f.get("object_key")
        o_name = f.get("object_name") or o_key
        o_type = f.get("object_type")

        predicate = f.get("predicate")

        # Track subject entity
        if s_key:
            if s_key not in entity_groups:
                entity_groups[s_key] = {
                    "key": s_key,
                    "name": s_name,
                    "type": s_type,
                    "paper_ids": set(),
                    "fact_ids": set(),
                }
            entity_groups[s_key]["paper_ids"].add(pid)
            entity_groups[s_key]["fact_ids"].add(fid)

        # Track object entity
        if o_key:
            if o_key not in entity_groups:
                entity_groups[o_key] = {
                    "key": o_key,
                    "name": o_name,
                    "type": o_type,
                    "paper_ids": set(),
                    "fact_ids": set(),
                }
            entity_groups[o_key]["paper_ids"].add(pid)
            entity_groups[o_key]["fact_ids"].add(fid)

        # Track relation
        if s_key and o_key and predicate:
            r_key = f"{s_key}:{predicate}:{o_key}"
            r_name = f"{s_name} {predicate} {o_name}"
            if r_key not in relation_groups:
                relation_groups[r_key] = {
                    "key": r_key,
                    "name": r_name,
                    "predicate": predicate,
                    "subject_name": s_name,
                    "object_name": o_name,
                    "paper_ids": set(),
                    "fact_ids": set(),
                }
            relation_groups[r_key]["paper_ids"].add(pid)
            relation_groups[r_key]["fact_ids"].add(fid)

    themes: list[dict[str, Any]] = []

    # Filter entities by effective_min_papers
    for eg in entity_groups.values():
        if len(eg["paper_ids"]) >= effective_min_papers:
            themes.append(
                {
                    "theme_type": "ENTITY",
                    "key": eg["key"],
                    "name": eg["name"],
                    "entity_type": eg["type"],
                    "predicate": None,
                    "subject_name": None,
                    "object_name": None,
                    "paper_count": len(eg["paper_ids"]),
                    "paper_ids": sorted(eg["paper_ids"])[:MAX_GROUP_SIZE],
                    "fact_ids": sorted(eg["fact_ids"])[:MAX_GROUP_SIZE],
                    "contributing_facts_count": len(eg["fact_ids"]),
                }
            )

    # Filter relations by effective_min_papers
    for rg in relation_groups.values():
        if len(rg["paper_ids"]) >= effective_min_papers:
            themes.append(
                {
                    "theme_type": "RELATION",
                    "key": rg["key"],
                    "name": rg["name"],
                    "entity_type": None,
                    "predicate": rg["predicate"],
                    "subject_name": rg["subject_name"],
                    "object_name": rg["object_name"],
                    "paper_count": len(rg["paper_ids"]),
                    "paper_ids": sorted(rg["paper_ids"])[:MAX_GROUP_SIZE],
                    "fact_ids": sorted(rg["fact_ids"])[:MAX_GROUP_SIZE],
                    "contributing_facts_count": len(rg["fact_ids"]),
                }
            )

    if not themes:
        logger.info(
            "Insufficient recurring entities or relations in project %s meeting min_papers=%d.",
            project_id,
            effective_min_papers,
        )
        return []

    # Sort deterministically: highest paper_count, then contributing_facts_count, then name
    themes.sort(key=lambda t: (-t["paper_count"], -t["contributing_facts_count"], t["name"]))

    return themes[:safe_limit]

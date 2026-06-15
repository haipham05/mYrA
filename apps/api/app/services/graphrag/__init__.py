"""GraphRAG knowledge graph extraction, provenance resolution, and identity management."""

from __future__ import annotations

from app.services.graphrag.identity import (
    canonicalize_name,
    generate_entity_key,
    generate_fact_id,
)
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor
from app.services.graphrag.provenance import resolve_graph_source_anchor
from app.services.graphrag.reconciliation import (
    rebuild_paper_graph_from_snapshots,
    reconcile_graph_drift,
)

__all__ = [
    "GraphEventProcessor",
    "Neo4jRepository",
    "canonicalize_name",
    "generate_entity_key",
    "generate_fact_id",
    "rebuild_paper_graph_from_snapshots",
    "reconcile_graph_drift",
    "resolve_graph_source_anchor",
]

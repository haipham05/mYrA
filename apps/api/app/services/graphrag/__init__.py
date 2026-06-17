"""GraphRAG knowledge graph extraction, provenance resolution, and identity management."""

from __future__ import annotations

from app.services.graphrag.extractor import (
    ExtractionResult,
    GraphExtractionAdapter,
    GraphExtractionError,
    GraphExtractionTimeoutError,
    parse_and_validate_extraction,
)
from app.services.graphrag.identity import (
    canonicalize_name,
    generate_entity_key,
    generate_fact_id,
)
from app.services.graphrag.input_selector import (
    ExtractionEvidenceItem,
    select_extraction_inputs,
)
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor
from app.services.graphrag.provenance import resolve_graph_source_anchor
from app.services.graphrag.reconciliation import (
    rebuild_paper_graph_from_snapshots,
    reconcile_graph_drift,
)
from app.services.graphrag.snapshots import (
    get_verified_fact_snapshots,
    persist_verified_fact_snapshots,
)
from app.services.graphrag.verifier import verify_candidate_fact

__all__ = [
    "ExtractionEvidenceItem",
    "ExtractionResult",
    "GraphEventProcessor",
    "GraphExtractionAdapter",
    "GraphExtractionError",
    "GraphExtractionTimeoutError",
    "Neo4jRepository",
    "canonicalize_name",
    "generate_entity_key",
    "generate_fact_id",
    "get_verified_fact_snapshots",
    "parse_and_validate_extraction",
    "persist_verified_fact_snapshots",
    "rebuild_paper_graph_from_snapshots",
    "reconcile_graph_drift",
    "resolve_graph_source_anchor",
    "select_extraction_inputs",
    "verify_candidate_fact",
]

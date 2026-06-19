"""GraphRAG knowledge graph extraction, provenance resolution, and identity management."""

from __future__ import annotations

from app.services.graphrag.evidence import (
    extract_fact_ids_from_candidates,
    resolve_graph_fact_to_evidence,
    resolve_graph_facts_to_evidence,
)
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
from app.services.graphrag.indexing import enqueue_existing_papers_for_graph
from app.services.graphrag.input_selector import (
    ExtractionEvidenceItem,
    select_extraction_inputs,
)
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor
from app.services.graphrag.provenance import resolve_graph_source_anchor
from app.services.graphrag.query_engine import (
    build_contradiction_candidates,
    build_corpus_themes,
    build_relationship_candidates,
)
from app.services.graphrag.reconciliation import (
    rebuild_paper_graph_from_snapshots,
    reconcile_graph_drift,
    scoped_rebuild_project_graph,
)
from app.services.graphrag.router import (
    GraphIntent,
    retrieve_graph_candidates_for_query,
    retrieve_graph_evidence,
    route_query_intent,
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
    "GraphIntent",
    "Neo4jRepository",
    "build_contradiction_candidates",
    "build_corpus_themes",
    "build_relationship_candidates",
    "canonicalize_name",
    "enqueue_existing_papers_for_graph",
    "extract_fact_ids_from_candidates",
    "generate_entity_key",
    "generate_fact_id",
    "get_verified_fact_snapshots",
    "parse_and_validate_extraction",
    "persist_verified_fact_snapshots",
    "rebuild_paper_graph_from_snapshots",
    "reconcile_graph_drift",
    "resolve_graph_fact_to_evidence",
    "resolve_graph_facts_to_evidence",
    "resolve_graph_source_anchor",
    "retrieve_graph_candidates_for_query",
    "retrieve_graph_evidence",
    "route_query_intent",
    "scoped_rebuild_project_graph",
    "select_extraction_inputs",
    "verify_candidate_fact",
]

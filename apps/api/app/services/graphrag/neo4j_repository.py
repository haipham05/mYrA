"""Neo4j knowledge graph repository for GraphRAG.

Manages project-scoped entities, verifiable source facts, and traversal queries
using parameterized Cypher and explicit database sessions.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from neo4j import Driver, GraphDatabase

from app.config import Settings
from app.schemas.graph import (
    EntityType,
    GraphEntitySchema,
    GraphFactCandidate,
    RelationshipPredicate,
)
from app.services.graphrag.identity import generate_entity_key, generate_fact_id

logger = logging.getLogger(__name__)

# Maximum query limits
MAX_PAGE_LIMIT = 100
DEFAULT_PAGE_LIMIT = 50


class Neo4jRepository:
    """Repository managing GraphRAG nodes and facts in Neo4j."""

    def __init__(self, driver: Driver, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database or "neo4j"

    @property
    def driver(self) -> Driver:
        """Return the underlying Neo4j driver."""
        return self._driver

    @property
    def database(self) -> str:
        """Return the configured database name."""
        return self._database

    def _get_session(self):
        """Open a session pinned to the configured database."""
        return self._driver.session(database=self._database)

    @classmethod
    def from_settings(cls, settings: Settings) -> Neo4jRepository | None:
        """Factory creating a Neo4jRepository from application Settings.

        Returns None if GraphRAG is disabled or NEO4J_URI is not configured.
        """
        if not settings.graphrag_enabled or not settings.neo4j_uri:
            return None

        auth = None
        if settings.neo4j_user and settings.neo4j_password:
            auth = (settings.neo4j_user, settings.neo4j_password)

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=auth,
            connection_timeout=settings.neo4j_timeout_seconds,
        )
        return cls(driver=driver, database=settings.neo4j_database)

    def verify_connectivity(self) -> bool:
        """Verify connectivity to the Neo4j DBMS."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.warning("Neo4j connectivity check failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the underlying Neo4j driver connection pool."""
        self._driver.close()

    def __enter__(self) -> Neo4jRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # =========================================================================
    # Schema Initialization
    # =========================================================================

    def ensure_schema(self) -> None:
        """Initialize project-scoped constraints and indexes in Neo4j.

        Compatible with Neo4j Community Edition 5.x+ and Aura/Enterprise.
        """
        schema_statements = [
            "CREATE CONSTRAINT node_key_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.key IS UNIQUE",
            "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
            "CREATE INDEX node_project_id IF NOT EXISTS FOR (n:Node) ON (n.project_id)",
            "CREATE INDEX fact_project_id IF NOT EXISTS FOR (f:Fact) ON (f.project_id)",
            "CREATE INDEX fact_paper_id IF NOT EXISTS FOR (f:Fact) ON (f.paper_id)",
            "CREATE INDEX fact_generation_id IF NOT EXISTS FOR (f:Fact) ON (f.generation_id)",
        ]

        def _tx_work(tx) -> None:
            for stmt in schema_statements:
                tx.run(stmt)

        with self._get_session() as session:
            session.execute_write(_tx_work)

    # =========================================================================
    # Managed Write Transactions
    # =========================================================================

    def upsert_nodes(
        self,
        project_id: UUID,
        nodes: list[GraphEntitySchema | dict[str, Any]],
    ) -> int:
        """Upsert entity nodes within a project.

        Merges on n.key and sets project_id, name, type, description, aliases,
        and updated_at. Deduplicates by node key before execution.
        """
        if not nodes:
            return 0

        prepared_nodes: dict[str, dict[str, Any]] = {}
        pid_str = str(project_id)

        for item in nodes:
            if isinstance(item, GraphEntitySchema):
                key = item.id
                name = item.name
                etype = item.type.value if hasattr(item.type, "value") else str(item.type)
                desc = item.description
                aliases = list(item.aliases)
            elif isinstance(item, dict):
                name = item.get("name", "").strip()
                raw_type = item.get("type", "Concept")
                etype = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
                key = item.get("key") or item.get("id")
                if not key:
                    if not name:
                        continue
                    key = generate_entity_key(
                        project_id=project_id,
                        entity_type=etype,
                        name=name,
                        external_id=item.get("external_id"),
                    )
                desc = item.get("description")
                aliases = list(item.get("aliases", []))
            else:
                continue

            prepared_nodes[key] = {
                "key": key,
                "name": name,
                "type": etype,
                "description": desc,
                "aliases": aliases,
            }

        if not prepared_nodes:
            return 0

        cypher = """
        UNWIND $batch AS row
        MERGE (n:Node {key: row.key})
        SET n.project_id = $project_id,
            n.name = row.name,
            n.type = row.type,
            n.description = row.description,
            n.aliases = row.aliases,
            n.updated_at = $updated_at
        RETURN count(n) AS cnt
        """
        params = {
            "project_id": pid_str,
            "updated_at": datetime.now(UTC).isoformat(),
            "batch": list(prepared_nodes.values()),
        }

        def _tx_work(tx) -> int:
            result = tx.run(cypher, params)
            row = result.single()
            return row["cnt"] if row else 0

        with self._get_session() as session:
            return session.execute_write(_tx_work)

    def upsert_facts(
        self,
        project_id: UUID,
        paper_id: UUID,
        generation_id: str,
        facts: list[GraphFactCandidate | dict[str, Any]],
    ) -> int:
        """Upsert provenance-grounded fact relationships between nodes.

        Ensures subject and object nodes exist within project_id, merges f:Fact,
        and merges (s)<-[:SUBJECT]-(f) and (f)-[:OBJECT]->(o).
        Idempotent: repeated MERGE on the same fact ID updates properties without
        creating duplicate nodes or duplicate edges.
        """
        if not facts:
            return 0

        pid_str = str(project_id)
        paid_str = str(paper_id)
        prepared_facts: dict[str, dict[str, Any]] = {}

        for item in facts:
            if isinstance(item, GraphFactCandidate):
                s_key = item.subject.id
                s_name = item.subject.name
                s_type = (
                    item.subject.type.value
                    if hasattr(item.subject.type, "value")
                    else str(item.subject.type)
                )
                o_key = item.object.id
                o_name = item.object.name
                o_type = (
                    item.object.type.value
                    if hasattr(item.object.type, "value")
                    else str(item.object.type)
                )
                predicate = (
                    item.predicate.value
                    if hasattr(item.predicate, "value")
                    else str(item.predicate)
                )
                qualifiers_json = (
                    json.dumps(item.qualifiers.model_dump(exclude_none=True))
                    if item.qualifiers
                    else None
                )
                char_start = item.provenance.char_start
                char_end = item.provenance.char_end
                page_number = item.provenance.page_number
                exact_quote = item.provenance.exact_quote
                fact_id = generate_fact_id(
                    project_id=project_id,
                    paper_id=paper_id,
                    predicate=item.predicate,
                    subject_key=s_key,
                    object_key=o_key,
                    char_start=char_start,
                    char_end=char_end,
                    qualifiers=item.qualifiers,
                    source_generation=generation_id,
                )
            elif isinstance(item, dict):
                # Extract subject
                s_data = item.get("subject")
                if isinstance(s_data, dict):
                    s_name = s_data.get("name", "").strip()
                    s_type_raw = s_data.get("type", "Concept")
                    s_type = s_type_raw.value if hasattr(s_type_raw, "value") else str(s_type_raw)
                    s_key = s_data.get("key") or s_data.get("id")
                    if not s_key and s_name:
                        s_key = generate_entity_key(
                            project_id,
                            s_type,
                            s_name,
                            external_id=s_data.get("external_id"),
                        )
                else:
                    s_key = item.get("subject_key") or (str(s_data) if s_data else "")
                    s_name = item.get("subject_name", s_key)
                    s_type = item.get("subject_type", "Concept")

                # Extract object
                o_data = item.get("object")
                if isinstance(o_data, dict):
                    o_name = o_data.get("name", "").strip()
                    o_type_raw = o_data.get("type", "Concept")
                    o_type = o_type_raw.value if hasattr(o_type_raw, "value") else str(o_type_raw)
                    o_key = o_data.get("key") or o_data.get("id")
                    if not o_key and o_name:
                        o_key = generate_entity_key(
                            project_id,
                            o_type,
                            o_name,
                            external_id=o_data.get("external_id"),
                        )
                else:
                    o_key = item.get("object_key") or (str(o_data) if o_data else "")
                    o_name = item.get("object_name", o_key)
                    o_type = item.get("object_type", "Concept")

                pred_raw = item.get("predicate", "")
                predicate = (
                    pred_raw.value if hasattr(pred_raw, "value") else str(pred_raw).strip().upper()
                )

                # Qualifiers
                q_raw = item.get("qualifiers")
                if q_raw is None:
                    qualifiers_json = item.get("qualifiers_json")
                elif isinstance(q_raw, str):
                    qualifiers_json = q_raw
                elif hasattr(q_raw, "model_dump"):
                    qualifiers_json = json.dumps(q_raw.model_dump(exclude_none=True))
                elif isinstance(q_raw, dict):
                    qualifiers_json = json.dumps(q_raw)
                else:
                    qualifiers_json = None

                prov = item.get("provenance", {})
                char_start = item.get(
                    "char_start",
                    prov.get("char_start", 0) if isinstance(prov, dict) else 0,
                )
                char_end = item.get(
                    "char_end",
                    prov.get("char_end", 0) if isinstance(prov, dict) else 0,
                )
                page_number = item.get(
                    "page_number",
                    prov.get("page_number", 1) if isinstance(prov, dict) else 1,
                )
                exact_quote = item.get(
                    "exact_quote",
                    prov.get("exact_quote", "") if isinstance(prov, dict) else "",
                )

                fact_id = item.get("fact_id") or item.get("id")
                if not fact_id and s_key and o_key and predicate:
                    fact_id = generate_fact_id(
                        project_id=project_id,
                        paper_id=paper_id,
                        predicate=predicate,
                        subject_key=s_key,
                        object_key=o_key,
                        char_start=char_start,
                        char_end=char_end,
                        qualifiers=q_raw,
                        source_generation=generation_id,
                    )
            else:
                continue

            if not fact_id or not s_key or not o_key:
                continue

            prepared_facts[fact_id] = {
                "fact_id": fact_id,
                "subject_key": s_key,
                "subject_name": s_name,
                "subject_type": s_type,
                "object_key": o_key,
                "object_name": o_name,
                "object_type": o_type,
                "predicate": predicate,
                "qualifiers_json": qualifiers_json,
                "char_start": char_start,
                "char_end": char_end,
                "page_number": page_number,
                "exact_quote": exact_quote,
            }

        if not prepared_facts:
            return 0

        cypher = """
        UNWIND $batch AS row
        MERGE (s:Node {key: row.subject_key})
        ON CREATE SET s.project_id = $project_id,
                      s.name = row.subject_name,
                      s.type = row.subject_type,
                      s.updated_at = $updated_at
        MERGE (o:Node {key: row.object_key})
        ON CREATE SET o.project_id = $project_id,
                      o.name = row.object_name,
                      o.type = row.object_type,
                      o.updated_at = $updated_at
        MERGE (f:Fact {id: row.fact_id})
        SET f.project_id = $project_id,
            f.paper_id = $paper_id,
            f.generation_id = $generation_id,
            f.predicate = row.predicate,
            f.qualifiers_json = row.qualifiers_json,
            f.char_start = row.char_start,
            f.char_end = row.char_end,
            f.page_number = row.page_number,
            f.exact_quote = row.exact_quote,
            f.updated_at = $updated_at
        MERGE (s)<-[:SUBJECT]-(f)
        MERGE (f)-[:OBJECT]->(o)
        RETURN count(f) AS cnt
        """
        params = {
            "project_id": pid_str,
            "paper_id": paid_str,
            "generation_id": generation_id,
            "updated_at": datetime.now(UTC).isoformat(),
            "batch": list(prepared_facts.values()),
        }

        def _tx_work(tx) -> int:
            result = tx.run(cypher, params)
            row = result.single()
            return row["cnt"] if row else 0

        with self._get_session() as session:
            return session.execute_write(_tx_work)

    def retire_older_generations(
        self,
        project_id: UUID,
        paper_id: UUID,
        active_generation_id: str,
    ) -> int:
        """Detach delete facts for a paper where generation_id <> active_generation_id."""
        pid_str = str(project_id)
        paid_str = str(paper_id)
        cypher = """
        MATCH (f:Fact {project_id: $project_id, paper_id: $paper_id})
        WHERE f.generation_id <> $active_generation_id
        WITH count(f) AS cnt, collect(f) AS facts
        FOREACH (fact IN facts | DETACH DELETE fact)
        RETURN cnt
        """
        params = {
            "project_id": pid_str,
            "paper_id": paid_str,
            "active_generation_id": active_generation_id,
        }

        def _tx_work(tx) -> int:
            result = tx.run(cypher, params)
            row = result.single()
            return row["cnt"] if row else 0

        with self._get_session() as session:
            return session.execute_write(_tx_work)

    def delete_paper_facts(self, project_id: UUID, paper_id: UUID) -> int:
        """Detach delete all facts for a specific paper within a project."""
        pid_str = str(project_id)
        paid_str = str(paper_id)
        cypher = """
        MATCH (f:Fact {project_id: $project_id, paper_id: $paper_id})
        WITH count(f) AS cnt, collect(f) AS facts
        FOREACH (fact IN facts | DETACH DELETE fact)
        RETURN cnt
        """
        params = {
            "project_id": pid_str,
            "paper_id": paid_str,
        }

        def _tx_work(tx) -> int:
            result = tx.run(cypher, params)
            row = result.single()
            return row["cnt"] if row else 0

        with self._get_session() as session:
            return session.execute_write(_tx_work)

    def delete_project_graph(self, project_id: UUID) -> dict[str, int]:
        """Detach delete all facts and nodes belonging to a project."""
        pid_str = str(project_id)

        def _tx_work(tx) -> dict[str, int]:
            q_facts = """
            MATCH (f:Fact {project_id: $project_id})
            WITH count(f) AS cnt, collect(f) AS facts
            FOREACH (fact IN facts | DETACH DELETE fact)
            RETURN cnt
            """
            facts_res = tx.run(q_facts, {"project_id": pid_str}).single()
            facts_cnt = facts_res["cnt"] if facts_res else 0

            q_nodes = """
            MATCH (n:Node {project_id: $project_id})
            WITH count(n) AS cnt, collect(n) AS nodes
            FOREACH (node IN nodes | DETACH DELETE node)
            RETURN cnt
            """
            nodes_res = tx.run(q_nodes, {"project_id": pid_str}).single()
            nodes_cnt = nodes_res["cnt"] if nodes_res else 0

            return {"facts_deleted": facts_cnt, "nodes_deleted": nodes_cnt}

        with self._get_session() as session:
            return session.execute_write(_tx_work)

    # =========================================================================
    # Managed Read Transactions
    # =========================================================================

    def search_nodes(
        self,
        project_id: UUID,
        query: str | None = None,
        entity_type: str | EntityType | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """Search nodes within a project with strictly capped pagination.

        Filters by optional query substring (matching name or aliases) and entity_type.
        """
        safe_limit = max(1, min(limit, MAX_PAGE_LIMIT))
        safe_skip = max(0, skip)
        pid_str = str(project_id)

        clean_query = query.strip() if query and query.strip() else None
        clean_type: str | None = None
        if entity_type:
            clean_type = (
                entity_type.value if hasattr(entity_type, "value") else str(entity_type).strip()
            )

        cypher = """
        MATCH (n:Node {project_id: $project_id})
        WHERE ($entity_type IS NULL OR toLower(n.type) = toLower($entity_type))
          AND ($query IS NULL
               OR toLower(n.name) CONTAINS toLower($query)
               OR any(a IN n.aliases WHERE toLower(a) CONTAINS toLower($query)))
        RETURN n.key AS key,
               n.name AS name,
               n.type AS type,
               n.description AS description,
               n.aliases AS aliases,
               n.project_id AS project_id,
               n.updated_at AS updated_at
        ORDER BY n.name ASC, n.key ASC
        SKIP $skip
        LIMIT $limit
        """
        params = {
            "project_id": pid_str,
            "query": clean_query,
            "entity_type": clean_type,
            "skip": safe_skip,
            "limit": safe_limit,
        }

        def _tx_work(tx) -> list[dict[str, Any]]:
            result = tx.run(cypher, params)
            return [dict(record) for record in result]

        with self._get_session() as session:
            return session.execute_read(_tx_work)

    def get_node_by_key(self, project_id: UUID, key: str) -> dict[str, Any] | None:
        """Retrieve a single node by key within a project."""
        clean_key = key.strip()
        if not clean_key:
            return None

        cypher = """
        MATCH (n:Node {key: $key, project_id: $project_id})
        RETURN n.key AS key,
               n.name AS name,
               n.type AS type,
               n.description AS description,
               n.aliases AS aliases,
               n.project_id AS project_id,
               n.updated_at AS updated_at
        """
        params = {
            "key": clean_key,
            "project_id": str(project_id),
        }

        def _tx_work(tx) -> dict[str, Any] | None:
            result = tx.run(cypher, params)
            record = result.single()
            return dict(record) if record else None

        with self._get_session() as session:
            return session.execute_read(_tx_work)

    def get_node_neighbors(
        self,
        project_id: UUID,
        key: str,
        direction: str = "BOTH",
        predicate: str | RelationshipPredicate | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Traverse (s)<-[:SUBJECT]-(f)-[:OBJECT]->(o) around node key within project.

        direction: 'OUTGOING', 'INCOMING', or 'BOTH'.
        Returns list of dicts with neighbor_key, neighbor_name, neighbor_type,
        direction, predicate, fact_id, and parsed qualifiers dict.
        """
        clean_key = key.strip()
        if not clean_key:
            return []

        safe_limit = max(1, min(limit, MAX_PAGE_LIMIT))
        norm_dir = direction.strip().upper()
        if norm_dir not in {"OUTGOING", "INCOMING", "BOTH"}:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'OUTGOING', 'INCOMING', or 'BOTH'."
            )

        clean_pred: str | None = None
        if predicate:
            clean_pred = (
                predicate.value if hasattr(predicate, "value") else str(predicate).strip().upper()
            )

        pid_str = str(project_id)
        params = {
            "key": clean_key,
            "project_id": pid_str,
            "predicate": clean_pred,
            "limit": safe_limit,
        }

        if norm_dir == "OUTGOING":
            cypher = """
            MATCH (s:Node {key: $key, project_id: $project_id})<-[:SUBJECT]-
                  (f:Fact {project_id: $project_id})-[:OBJECT]->
                  (neighbor:Node {project_id: $project_id})
            WHERE ($predicate IS NULL OR toUpper(f.predicate) = toUpper($predicate))
            RETURN neighbor.key AS neighbor_key,
                   neighbor.name AS neighbor_name,
                   neighbor.type AS neighbor_type,
                   "OUTGOING" AS direction,
                   f.predicate AS predicate,
                   f.id AS fact_id,
                   f.qualifiers_json AS qualifiers_json
            ORDER BY neighbor.name ASC, neighbor.key ASC, f.id ASC
            LIMIT $limit
            """
        elif norm_dir == "INCOMING":
            cypher = """
            MATCH (neighbor:Node {project_id: $project_id})<-[:SUBJECT]-
                  (f:Fact {project_id: $project_id})-[:OBJECT]->
                  (o:Node {key: $key, project_id: $project_id})
            WHERE ($predicate IS NULL OR toUpper(f.predicate) = toUpper($predicate))
            RETURN neighbor.key AS neighbor_key,
                   neighbor.name AS neighbor_name,
                   neighbor.type AS neighbor_type,
                   "INCOMING" AS direction,
                   f.predicate AS predicate,
                   f.id AS fact_id,
                   f.qualifiers_json AS qualifiers_json
            ORDER BY neighbor.name ASC, neighbor.key ASC, f.id ASC
            LIMIT $limit
            """
        else:  # BOTH
            cypher = """
            CALL () {
                MATCH (s:Node {key: $key, project_id: $project_id})<-[:SUBJECT]-
                      (f:Fact {project_id: $project_id})-[:OBJECT]->
                      (neighbor:Node {project_id: $project_id})
                WHERE ($predicate IS NULL OR toUpper(f.predicate) = toUpper($predicate))
                RETURN neighbor.key AS neighbor_key,
                       neighbor.name AS neighbor_name,
                       neighbor.type AS neighbor_type,
                       "OUTGOING" AS direction,
                       f.predicate AS predicate,
                       f.id AS fact_id,
                       f.qualifiers_json AS qualifiers_json
                UNION ALL
                MATCH (neighbor:Node {project_id: $project_id})<-[:SUBJECT]-
                      (f:Fact {project_id: $project_id})-[:OBJECT]->
                      (o:Node {key: $key, project_id: $project_id})
                WHERE ($predicate IS NULL OR toUpper(f.predicate) = toUpper($predicate))
                RETURN neighbor.key AS neighbor_key,
                       neighbor.name AS neighbor_name,
                       neighbor.type AS neighbor_type,
                       "INCOMING" AS direction,
                       f.predicate AS predicate,
                       f.id AS fact_id,
                       f.qualifiers_json AS qualifiers_json
            }
            RETURN neighbor_key,
                   neighbor_name,
                   neighbor_type,
                   direction,
                   predicate,
                   fact_id,
                   qualifiers_json
            ORDER BY neighbor_name ASC, neighbor_key ASC, fact_id ASC
            LIMIT $limit
            """

        def _tx_work(tx) -> list[dict[str, Any]]:
            result = tx.run(cypher, params)
            records: list[dict[str, Any]] = []
            for row in result:
                raw_data = dict(row)
                qualifiers = None
                if raw_data.get("qualifiers_json"):
                    try:
                        qualifiers = json.loads(raw_data["qualifiers_json"])
                    except (json.JSONDecodeError, TypeError):
                        qualifiers = None

                records.append(
                    {
                        "neighbor_key": raw_data["neighbor_key"],
                        "neighbor_name": raw_data["neighbor_name"],
                        "neighbor_type": raw_data["neighbor_type"],
                        "direction": raw_data["direction"],
                        "predicate": raw_data["predicate"],
                        "fact_id": raw_data["fact_id"],
                        "qualifiers": qualifiers,
                    }
                )
            return records

        with self._get_session() as session:
            return session.execute_read(_tx_work)

    def get_fact_by_id(self, project_id: UUID, fact_id: str) -> dict[str, Any] | None:
        """Retrieve a single fact by ID with its connected subject and object node info."""
        clean_fid = fact_id.strip()
        if not clean_fid:
            return None

        cypher = """
        MATCH (s:Node)<-[:SUBJECT]-
              (f:Fact {id: $fact_id, project_id: $project_id})-[:OBJECT]->
              (o:Node)
        RETURN f.id AS id,
               f.project_id AS project_id,
               f.paper_id AS paper_id,
               f.generation_id AS generation_id,
               f.predicate AS predicate,
               f.qualifiers_json AS qualifiers_json,
               f.char_start AS char_start,
               f.char_end AS char_end,
               f.page_number AS page_number,
               f.exact_quote AS exact_quote,
               f.updated_at AS updated_at,
               s.key AS subject_key,
               s.name AS subject_name,
               s.type AS subject_type,
               o.key AS object_key,
               o.name AS object_name,
               o.type AS object_type
        """
        params = {
            "fact_id": clean_fid,
            "project_id": str(project_id),
        }

        def _tx_work(tx) -> dict[str, Any] | None:
            result = tx.run(cypher, params)
            record = result.single()
            if not record:
                return None
            data = dict(record)
            qualifiers = None
            if data.get("qualifiers_json"):
                try:
                    qualifiers = json.loads(data["qualifiers_json"])
                except (json.JSONDecodeError, TypeError):
                    qualifiers = None
            data["qualifiers"] = qualifiers
            return data

        with self._get_session() as session:
            return session.execute_read(_tx_work)

    def find_relationships_between(
        self,
        project_id: UUID,
        subject_key: str,
        object_key: str,
        predicate: str | RelationshipPredicate | None = None,
    ) -> list[dict[str, Any]]:
        """Find facts connecting a specific subject and object within a project."""
        clean_s = subject_key.strip()
        clean_o = object_key.strip()
        if not clean_s or not clean_o:
            return []

        clean_pred: str | None = None
        if predicate:
            clean_pred = (
                predicate.value if hasattr(predicate, "value") else str(predicate).strip().upper()
            )

        cypher = """
        MATCH (s:Node {key: $subject_key, project_id: $project_id})<-[:SUBJECT]-
              (f:Fact {project_id: $project_id})-[:OBJECT]->
              (o:Node {key: $object_key, project_id: $project_id})
        WHERE ($predicate IS NULL OR toUpper(f.predicate) = toUpper($predicate))
        RETURN f.id AS fact_id,
               f.project_id AS project_id,
               f.paper_id AS paper_id,
               f.generation_id AS generation_id,
               f.predicate AS predicate,
               f.qualifiers_json AS qualifiers_json,
               f.char_start AS char_start,
               f.char_end AS char_end,
               f.page_number AS page_number,
               f.exact_quote AS exact_quote,
               f.updated_at AS updated_at,
               s.key AS subject_key,
               s.name AS subject_name,
               s.type AS subject_type,
               o.key AS object_key,
               o.name AS object_name,
               o.type AS object_type
        ORDER BY f.id ASC
        """
        params = {
            "project_id": str(project_id),
            "subject_key": clean_s,
            "object_key": clean_o,
            "predicate": clean_pred,
        }

        def _tx_work(tx) -> list[dict[str, Any]]:
            result = tx.run(cypher, params)
            records: list[dict[str, Any]] = []
            for row in result:
                data = dict(row)
                qualifiers = None
                if data.get("qualifiers_json"):
                    try:
                        qualifiers = json.loads(data["qualifiers_json"])
                    except (json.JSONDecodeError, TypeError):
                        qualifiers = None
                data["qualifiers"] = qualifiers
                records.append(data)
            return records

        with self._get_session() as session:
            return session.execute_read(_tx_work)

    def count_project_elements(self, project_id: UUID) -> dict[str, int]:
        """Count total nodes and facts within a project."""
        pid_str = str(project_id)

        def _tx_work(tx) -> dict[str, int]:
            node_res = tx.run(
                "MATCH (n:Node {project_id: $project_id}) RETURN count(n) AS cnt",
                {"project_id": pid_str},
            ).single()
            fact_res = tx.run(
                "MATCH (f:Fact {project_id: $project_id}) RETURN count(f) AS cnt",
                {"project_id": pid_str},
            ).single()
            node_cnt = node_res["cnt"] if node_res else 0
            fact_cnt = fact_res["cnt"] if fact_res else 0
            return {
                "nodes": node_cnt,
                "facts": fact_cnt,
                "node_count": node_cnt,
                "fact_count": fact_cnt,
            }

        with self._get_session() as session:
            return session.execute_read(_tx_work)

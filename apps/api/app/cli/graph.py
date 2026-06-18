"""CLI entrypoints for GraphRAG indexing and rebuild operations."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import UUID

from app.config import Settings
from app.db.session import SessionLocal
from app.services.graphrag.indexing import enqueue_existing_papers_for_graph
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.reconciliation import scoped_rebuild_project_graph


def main_index(argv: list[str] | None = None) -> int:
    """CLI entrypoint to enqueue existing READY papers for GraphRAG processing."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.graph index",
        description="Enqueue existing READY papers for GraphRAG indexing.",
    )
    parser.add_argument("--project-id", type=str, default=None, help="Target project UUID")
    parser.add_argument("--paper-id", type=str, default=None, help="Target paper UUID")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum papers to enqueue (capped at 50, default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate without enqueuing (default behavior unless --confirm)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Explicitly confirm execution (dry-run=False)",
    )

    args = parser.parse_args(argv)
    dry_run = True if args.dry_run or not args.confirm else False

    project_id: UUID | None = None
    if args.project_id:
        try:
            project_id = UUID(args.project_id)
        except (ValueError, TypeError) as exc:
            print(json.dumps({"error": f"Invalid project_id UUID: {exc}"}, indent=2))
            return 1

    paper_ids: list[UUID] | None = None
    if args.paper_id:
        try:
            paper_ids = [UUID(args.paper_id)]
        except (ValueError, TypeError) as exc:
            print(json.dumps({"error": f"Invalid paper_id UUID: {exc}"}, indent=2))
            return 1

    with SessionLocal() as db:
        result = enqueue_existing_papers_for_graph(
            db=db,
            project_id=project_id,
            paper_ids=paper_ids,
            limit=args.limit,
            dry_run=dry_run,
        )

    print(json.dumps(result, indent=2))
    return 0


def main_rebuild(
    argv: list[str] | None = None,
    repo: Neo4jRepository | None = None,
) -> int:
    """CLI entrypoint to rebuild project graph deterministically from
    stored PostgreSQL snapshots.
    """

    parser = argparse.ArgumentParser(
        prog="python -m app.cli.graph rebuild",
        description="Rebuild project graph deterministically from stored PostgreSQL snapshots.",
    )
    parser.add_argument("--project-id", type=str, default=None, help="Target project UUID")
    parser.add_argument("--paper-id", type=str, default=None, help="Target paper UUID")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate rebuild without writing to Neo4j (default behavior unless --confirm)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Explicitly confirm execution (dry-run=False)",
    )

    args = parser.parse_args(argv)

    if not args.project_id:
        print(json.dumps({"error": "--project-id is required for graph rebuild"}, indent=2))
        return 1

    try:
        project_id = UUID(args.project_id)
    except (ValueError, TypeError) as exc:
        print(json.dumps({"error": f"Invalid project_id UUID: {exc}"}, indent=2))
        return 1

    paper_id: UUID | None = None
    if args.paper_id:
        try:
            paper_id = UUID(args.paper_id)
        except (ValueError, TypeError) as exc:
            print(json.dumps({"error": f"Invalid paper_id UUID: {exc}"}, indent=2))
            return 1

    dry_run = True if args.dry_run or not args.confirm else False

    if repo is None:
        settings = Settings.from_environment()
        repo = Neo4jRepository.from_settings(settings)
        if repo is None:
            if dry_run:
                from unittest.mock import MagicMock

                repo = MagicMock(spec=Neo4jRepository)
            else:
                print(
                    json.dumps(
                        {"error": "Neo4j is not configured or disabled in settings."},
                        indent=2,
                    )
                )
                return 1

    with SessionLocal() as db:
        try:
            result = scoped_rebuild_project_graph(
                db=db,
                repo=repo,
                project_id=project_id,
                paper_id=paper_id,
                dry_run=dry_run,
            )
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint routing to index or rebuild."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m app.cli.graph [index|rebuild] [options]")
        return 1
    subcommand = args[0]
    rest = args[1:]
    if subcommand == "index":
        return main_index(rest)
    elif subcommand == "rebuild":
        return main_rebuild(rest)
    else:
        print(f"Unknown subcommand: {subcommand}. Expected 'index' or 'rebuild'.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

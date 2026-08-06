import csv
import io
import re
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Paper, ResearchArtifactRevision
from app.services.error_sanitizer import redact_secrets

_LOCAL_PATH = re.compile(r"(?:/home/[^\s,;]+|/tmp/[^\s,;]+|[A-Za-z]:\\Users\\[^\s,;]+)")


def _safe(value: str) -> str:
    return _LOCAL_PATH.sub("[local path omitted]", redact_secrets(value))


def _csv_safe(value: str) -> str:
    """Prevent spreadsheet formula execution from source-controlled cell content."""
    first_effective = next(
        (
            char
            for char in value
            if not char.isspace() and ord(char) >= 32 and ord(char) != 127 and char != "\ufeff"
        ),
        "",
    )
    if first_effective in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _source_manifest(
    revision: ResearchArtifactRevision,
    source_manifest: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    selected = source_manifest if source_manifest is not None else revision.source_manifest or []
    return [item for item in selected if isinstance(item, dict)]


def export_markdown(
    revision: ResearchArtifactRevision,
    artifact_type: str,
    *,
    source_manifest: list[dict[str, Any]] | None = None,
) -> str:
    payload = revision.payload
    body = payload.get("report_markdown") or payload.get("markdown") or payload.get("display_text")
    if not isinstance(body, str):
        sections = payload.get("sections")
        if isinstance(sections, list):
            body = "\n\n".join(
                f"## {item['title']}\n\n{item['content']}"
                for item in sections
                if isinstance(item, dict)
                and isinstance(item.get("title"), str)
                and isinstance(item.get("content"), str)
            )
        else:
            body = ""
    body = _safe(body)
    lines = [f"# {_safe(revision.title)}", "", f"Artifact type: `{artifact_type}`", "", body]
    sources = _source_manifest(revision, source_manifest)
    if sources:
        lines.extend(["", "## References", ""])
        for source in sources:
            evidence_id = _safe(
                str(source.get("evidence_id") or source.get("source_id") or "Source")
            )
            title = _safe(source.get("paper_title") or "Paper")
            page = source.get("page_number")
            quote = source.get("quote")
            reference = f"- [{evidence_id}] {title}"
            if page is not None:
                reference += f", p. {page}"
            if source.get("availability") == "unavailable":
                reference += " — source changed or unavailable; original citation retained"
            if isinstance(quote, str) and quote:
                reference += f": “{_safe(quote)}”"
            lines.append(reference)
    return "\n".join(lines).rstrip() + "\n"


def export_comparison_csv(
    revision: ResearchArtifactRevision,
    *,
    source_manifest: list[dict[str, Any]] | None = None,
) -> str:
    matrix = revision.payload.get("matrix", {})
    cells = matrix.get("cells", []) if isinstance(matrix, Mapping) else []
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "paper_id",
            "dimension",
            "status",
            "evidence_id",
            "page",
            "document_sha256",
            "quote",
            "source_availability",
        ]
    )
    source_states = {
        (str(item.get("paper_id")), str(item.get("page_number")), str(item.get("quote"))): item.get(
            "availability", "unverified"
        )
        for item in _source_manifest(revision, source_manifest)
    }
    for cell in cells if isinstance(cells, list) else []:
        if not isinstance(cell, Mapping):
            continue
        excerpts = cell.get("excerpts", [])
        if not isinstance(excerpts, list) or not excerpts:
            writer.writerow(
                [
                    _csv_safe(_safe(str(cell.get("paper_id", "")))),
                    _csv_safe(_safe(str(cell.get("dimension", "")))),
                    _csv_safe(_safe(str(cell.get("status", "")))),
                    "",
                    "",
                    "",
                    _csv_safe(_safe(str(cell.get("message", "")))),
                    "not_applicable",
                ]
            )
            continue
        for excerpt in excerpts:
            if not isinstance(excerpt, Mapping):
                continue
            evidence = excerpt.get("evidence", {})
            citation = excerpt.get("citation", {})
            evidence = evidence if isinstance(evidence, Mapping) else {}
            citation = citation if isinstance(citation, Mapping) else {}
            paper_id = str(
                evidence.get("paper_id") or citation.get("paper_id") or cell.get("paper_id", "")
            )
            page = str(evidence.get("page_number") or citation.get("page_number", ""))
            quote = str(evidence.get("quote") or citation.get("quote", ""))
            availability = source_states.get((paper_id, page, quote), "unverified")
            writer.writerow(
                [
                    _csv_safe(_safe(paper_id)),
                    _csv_safe(_safe(str(cell.get("dimension", "")))),
                    _csv_safe(_safe(str(cell.get("status", "")))),
                    _csv_safe(_safe(str(evidence.get("id") or citation.get("evidence_id", "")))),
                    _csv_safe(page),
                    _csv_safe(
                        _safe(
                            str(
                                evidence.get("document_sha256")
                                or citation.get("document_sha256", "")
                            )
                        )
                    ),
                    _csv_safe(_safe(quote)),
                    availability,
                ]
            )
    return output.getvalue()


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}")


def export_bibtex(
    db: Session,
    revision: ResearchArtifactRevision,
    project_id: UUID,
    *,
    source_manifest: list[dict[str, Any]] | None = None,
) -> str:
    paper_ids = {
        UUID(item["paper_id"])
        for item in _source_manifest(revision, source_manifest)
        if isinstance(item.get("paper_id"), str)
    }
    papers = (
        db.query(Paper)
        .filter(Paper.project_id == project_id, Paper.id.in_(paper_ids))
        .order_by(Paper.id)
        .all()
        if paper_ids
        else []
    )
    entries = []
    for paper in papers:
        key = "myra_" + re.sub(r"[^A-Za-z0-9]", "", str(paper.id))[:12]
        fields = {
            "title": paper.title,
            "author": " and ".join(paper.authors or []) or None,
            "year": str(paper.publication_year) if paper.publication_year is not None else None,
            "doi": paper.doi,
            "eprint": paper.arxiv_id,
            "url": paper.source_url,
        }
        rendered = [
            f"  {name} = {{{_bibtex_escape(_safe(value))}}}"
            for name, value in fields.items()
            if value
        ]
        source_states = {
            item.get("availability", "unverified")
            for item in _source_manifest(revision, source_manifest)
            if item.get("paper_id") == str(paper.id)
        }
        if source_states:
            state = "unavailable" if "unavailable" in source_states else next(iter(source_states))
            rendered.append(f"  note = {{Source status: {_safe(str(state))}}}")
        entries.append("@misc{" + key + ",\n" + ",\n".join(rendered) + "\n}")
    return "\n\n".join(entries) + ("\n" if entries else "")

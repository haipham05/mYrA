"""Bounded, one-shot cleanup for expired Langfuse traces."""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = ROOT / ".local" / "observability"
STATE_NAME = "retention-state.json"
RETENTION_AGE = timedelta(hours=72)
EXPORT_GRACE = timedelta(hours=2)
HTTP_TIMEOUT_SECONDS = 10
PAGE_LIMIT = 100
DELETE_BATCH_SIZE = 500
MAX_PAGES_PER_RUN = 100
MAX_DELETE_BATCHES_PER_RUN = 20
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class RetentionError(RuntimeError):
    """Safe-to-report operational failure (never contains response bodies)."""


class Response(Protocol):
    status_code: int

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        params: dict[str, str | int],
        auth: tuple[str, str],
        timeout: float,
    ) -> Response: ...

    def delete(
        self,
        url: str,
        *,
        json_body: dict[str, list[str]],
        auth: tuple[str, str],
        timeout: float,
    ) -> Response: ...


@dataclass(frozen=True)
class _UrllibResponse:
    status_code: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


class UrllibClient:
    """Small stdlib HTTP client with no logging and bounded response size."""

    MAX_RESPONSE_BYTES = 4 * 1024 * 1024

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, file_pointer, code, message, headers, new_url):
            return None

    @staticmethod
    def _request(
        method: str,
        url: str,
        *,
        auth: tuple[str, str],
        timeout: float,
        payload: dict[str, list[str]] | None = None,
    ) -> _UrllibResponse:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode("ascii")
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            opener = urllib.request.build_opener(UrllibClient._NoRedirect)
            with opener.open(request, timeout=timeout) as response:
                data = response.read(UrllibClient.MAX_RESPONSE_BYTES + 1)
                if len(data) > UrllibClient.MAX_RESPONSE_BYTES:
                    raise RetentionError("response_too_large")
                return _UrllibResponse(response.status, data)
        except urllib.error.HTTPError as error:
            raise RetentionError(f"http_{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RetentionError("network_error") from None

    def get(
        self,
        url: str,
        *,
        params: dict[str, str | int],
        auth: tuple[str, str],
        timeout: float,
    ) -> _UrllibResponse:
        query = urllib.parse.urlencode(params)
        return self._request("GET", f"{url}?{query}", auth=auth, timeout=timeout)

    def delete(
        self,
        url: str,
        *,
        json_body: dict[str, list[str]],
        auth: tuple[str, str],
        timeout: float,
    ) -> _UrllibResponse:
        return self._request("DELETE", url, auth=auth, timeout=timeout, payload=json_body)


@dataclass(frozen=True)
class ProjectCredentials:
    label: str
    public_key: str
    secret_key: str


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RetentionError("invalid_base_url")
    return value.rstrip("/")


def _private_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RetentionError("credential_file_missing_or_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RetentionError("credential_file_invalid") from None
    if not isinstance(value, dict):
        raise RetentionError("credential_file_invalid")
    return value


def load_project(state_dir: Path) -> ProjectCredentials:
    """Read the personal Langfuse project credentials."""
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise RetentionError("credential_directory_missing_or_unsafe")
    credentials = _private_json(state_dir / "credentials.json")
    personal = credentials.get("personal_project")
    if not isinstance(personal, dict):
        raise RetentionError("personal_credentials_invalid")
    personal_public = personal.get("public_key")
    personal_secret = personal.get("secret_key")
    if not all(isinstance(item, str) and item for item in (personal_public, personal_secret)):
        raise RetentionError("personal_credentials_invalid")
    return ProjectCredentials("personal", personal_public, personal_secret)


def _read_state(path: Path) -> dict[str, Any]:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RetentionError("retention_state_unsafe")
    if not path.exists():
        return {"format_version": 1, "projects": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RetentionError("retention_state_invalid") from None
    if (
        not isinstance(state, dict)
        or state.get("format_version") != 1
        or not isinstance(state.get("projects"), dict)
    ):
        raise RetentionError("retention_state_invalid")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RetentionError("retention_state_unsafe")
    if path.parent.is_symlink():
        raise RetentionError("retention_state_unsafe")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def text_export_permitted(state: dict[str, Any], now: datetime) -> bool:
    """Return true only when the personal project has a recent verified cleanup."""
    now = _utc(now)
    projects = state.get("projects")
    if not isinstance(projects, dict):
        return False
    item = projects.get("personal")
    if not isinstance(item, dict):
        return False
    # A failed run invalidates a previous success even if it is still fresh.
    if item.get("last_failure_at") is not None:
        return False
    raw_success = item.get("last_success_at")
    if not isinstance(raw_success, str):
        return False
    try:
        success = _utc(datetime.fromisoformat(raw_success.replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return False
    return success <= now and now - success <= EXPORT_GRACE


def _fetch_page(
    client: HttpClient,
    base_url: str,
    project: ProjectCredentials,
    *,
    start: datetime,
    cutoff: datetime,
    cursor: str | None,
) -> tuple[list[str], str | None]:
    params: dict[str, str | int] = {
        "fromStartTime": _iso(start),
        "toStartTime": _iso(cutoff),
        "fields": "core",
        "limit": PAGE_LIMIT,
    }
    if cursor is not None:
        params["cursor"] = cursor
    response = client.get(
        f"{base_url}/api/public/v2/observations",
        params=params,
        auth=(project.public_key, project.secret_key),
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise RetentionError(f"observations_http_{response.status_code}")
    try:
        page = response.json()
    except (ValueError, TypeError):
        raise RetentionError("observations_invalid_json") from None
    if not isinstance(page, dict) or not isinstance(page.get("data"), list):
        raise RetentionError("observations_invalid_page")
    meta = page.get("meta")
    if not isinstance(meta, dict):
        raise RetentionError("observations_missing_pagination")
    next_cursor = meta.get("cursor")
    if next_cursor is not None and (not isinstance(next_cursor, str) or not next_cursor):
        raise RetentionError("observations_invalid_cursor")
    trace_ids: list[str] = []
    for row in page["data"]:
        if not isinstance(row, dict):
            raise RetentionError("observations_invalid_item")
        trace_id = row.get("traceId")
        raw_time = row.get("startTime")
        if not isinstance(trace_id, str) or not trace_id or not isinstance(raw_time, str):
            raise RetentionError("observation_missing_trace_or_time")
        try:
            observed_at = _utc(datetime.fromisoformat(raw_time.replace("Z", "+00:00")))
        except ValueError:
            raise RetentionError("observation_invalid_time") from None
        # Don't trust server-side time filtering when choosing destructive IDs.
        if start <= observed_at < cutoff:
            trace_ids.append(trace_id)
    return trace_ids, next_cursor


def _delete_batch(
    client: HttpClient,
    base_url: str,
    project: ProjectCredentials,
    trace_ids: list[str],
) -> None:
    response = client.delete(
        f"{base_url}/api/public/traces",
        json_body={"traceIds": trace_ids},
        auth=(project.public_key, project.secret_key),
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise RetentionError(f"trace_delete_http_{response.status_code}")


def _merge_unique(destination: list[str], incoming: list[str]) -> None:
    known = set(destination)
    for trace_id in incoming:
        if trace_id not in known:
            destination.append(trace_id)
            known.add(trace_id)


def _run_project(
    client: HttpClient,
    base_url: str,
    project: ProjectCredentials,
    project_state: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Advance a bounded delete/verify pass, persisting its cursor on failure."""
    now = _utc(now)
    active = project_state.get("cycle")
    if active is None:
        raw_previous = project_state.get("last_success_cutoff")
        if raw_previous is None:
            scan_start = EPOCH
        else:
            try:
                scan_start = _utc(datetime.fromisoformat(raw_previous.replace("Z", "+00:00")))
            except (ValueError, AttributeError):
                raise RetentionError("retention_state_invalid_cutoff") from None
        cutoff = now - RETENTION_AGE
        active = {
            "start": _iso(scan_start),
            "cutoff": _iso(cutoff),
            "phase": "delete",
            "cursor": None,
            "pending_trace_ids": [],
        }
        project_state["cycle"] = active
    try:
        start = _utc(datetime.fromisoformat(active["start"].replace("Z", "+00:00")))
        cutoff = _utc(datetime.fromisoformat(active["cutoff"].replace("Z", "+00:00")))
        if start >= cutoff or active.get("phase") not in {"delete", "verify"}:
            raise RetentionError("retention_state_invalid_cycle")
        cursor = active.get("cursor")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise RetentionError("retention_state_invalid_cursor")
        pending = active.get("pending_trace_ids")
        if not isinstance(pending, list) or any(not isinstance(item, str) for item in pending):
            raise RetentionError("retention_state_invalid_pending")
        pending = list(dict.fromkeys(pending))
        # A null verify cursor starts a fresh read. Prior-pass IDs are only hints
        # for re-deletion; they must not block a fresh empty verification result.
        if active["phase"] == "verify" and cursor is None:
            pending = []
        if len(pending) > DELETE_BATCH_SIZE * MAX_DELETE_BATCHES_PER_RUN:
            raise RetentionError("pending_id_limit_reached")
        pages = batches = 0
        seen_cursors: set[str] = set()

        while pages < MAX_PAGES_PER_RUN:
            # In verify phase, records returned are still pending deletion.
            observed, next_cursor = _fetch_page(
                client, base_url, project, start=start, cutoff=cutoff, cursor=cursor
            )
            pages += 1
            if active["phase"] == "delete":
                _merge_unique(pending, observed)
                while len(pending) >= DELETE_BATCH_SIZE:
                    if batches >= MAX_DELETE_BATCHES_PER_RUN:
                        break
                    batch, pending = (
                        pending[:DELETE_BATCH_SIZE],
                        pending[DELETE_BATCH_SIZE:],
                    )
                    _delete_batch(client, base_url, project, batch)
                    batches += 1
            else:
                _merge_unique(pending, observed)
            if next_cursor is None:
                if active["phase"] == "delete":
                    while pending and batches < MAX_DELETE_BATCHES_PER_RUN:
                        batch, pending = (
                            pending[:DELETE_BATCH_SIZE],
                            pending[DELETE_BATCH_SIZE:],
                        )
                        _delete_batch(client, base_url, project, batch)
                        batches += 1
                    if pending:
                        active.update(
                            {
                                "phase": "delete",
                                "cursor": None,
                                "pending_trace_ids": pending,
                            }
                        )
                        raise RetentionError("delete_batch_limit_reached")
                    active.update({"phase": "verify", "cursor": None, "pending_trace_ids": []})
                    cursor = None
                    continue
                if pending:
                    # Re-submit what remains, then verify from the beginning next run.
                    retry_limit = DELETE_BATCH_SIZE * max(0, MAX_DELETE_BATCHES_PER_RUN - batches)
                    retry_ids = pending[:retry_limit]
                    for index in range(0, len(retry_ids), DELETE_BATCH_SIZE):
                        _delete_batch(
                            client,
                            base_url,
                            project,
                            retry_ids[index : index + DELETE_BATCH_SIZE],
                        )
                        batches += 1
                    active.update(
                        {
                            "phase": "verify",
                            "cursor": None,
                            "pending_trace_ids": pending,
                        }
                    )
                    raise RetentionError("deletion_pending_verification")
                project_state.update(
                    {
                        "last_success_at": _iso(now),
                        "last_success_cutoff": _iso(cutoff),
                        "last_failure_at": None,
                        "last_error": None,
                        "last_pages": pages,
                        "last_delete_batches": batches,
                        "last_pending_count": 0,
                    }
                )
                project_state.pop("cycle", None)
                return project_state

            if next_cursor in seen_cursors:
                raise RetentionError("pagination_cursor_repeated")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
            active["cursor"] = cursor
            active["pending_trace_ids"] = pending
            # Keep persistence bounded; never silently drop IDs after reaching caps.
            if len(pending) > DELETE_BATCH_SIZE * MAX_DELETE_BATCHES_PER_RUN:
                raise RetentionError("pending_id_limit_reached")
            if active["phase"] == "delete" and batches >= MAX_DELETE_BATCHES_PER_RUN:
                raise RetentionError("delete_batch_limit_reached")

        active.update({"cursor": cursor, "pending_trace_ids": pending})
        raise RetentionError("page_limit_reached")
    except (KeyError, TypeError, ValueError):
        raise RetentionError("retention_state_invalid_cycle") from None


def cleanup(
    *,
    base_url: str,
    project: ProjectCredentials,
    state_dir: Path = DEFAULT_STATE_DIR,
    client: HttpClient | None = None,
    clock: Any = lambda: datetime.now(timezone.utc),
) -> tuple[dict[str, Any], bool]:
    """Run a safe cleanup pass for the personal project and persist status."""
    base_url = _safe_base_url(base_url)
    if project.label != "personal":
        raise RetentionError("personal_project_required")
    if not project.public_key or not project.secret_key:
        raise RetentionError("project_credentials_not_configured")
    client = client or UrllibClient()
    now = _utc(clock())
    state_path = state_dir / STATE_NAME
    state = _read_state(state_path)
    # Preserve the personal cursor and pending IDs, while discarding legacy
    # synthetic-project state on the next authoritative write.
    personal_state = state["projects"].get("personal", {})
    if not isinstance(personal_state, dict):
        raise RetentionError("retention_state_invalid")
    state["projects"] = {"personal": personal_state}
    entry = personal_state
    entry["last_attempt_at"] = _iso(now)
    try:
        _run_project(client, base_url, project, entry, now)
    except RetentionError as error:
        cycle = entry.get("cycle", {})
        entry.update(
            {
                "last_failure_at": _iso(now),
                "last_error": str(error),
                "last_pending_count": len(cycle.get("pending_trace_ids", []))
                if isinstance(cycle, dict)
                else 0,
            }
        )
    _write_state(state_path, state)
    return state, bool(entry.get("last_success_at") and entry.get("last_failure_at") is None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--credentials-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--base-url", default=os.environ.get("LANGFUSE_BASE_URL"))
    args = parser.parse_args(argv)
    try:
        if not args.base_url:
            raise RetentionError("base_url_not_configured")
        project = load_project(args.credentials_dir)
        state, success = cleanup(
            base_url=args.base_url,
            project=project,
            state_dir=args.state_dir,
        )
    except RetentionError as error:
        print(f"retention cleanup failed: {error}", file=sys.stderr)
        return 1
    item = state["projects"].get("personal", {})
    print(
        f"personal: success={bool(item.get('last_success_at') and item.get('last_failure_at') is None)} "
        f"pending={item.get('last_pending_count', 0)} pages={item.get('last_pages', 0)} "
        f"batches={item.get('last_delete_batches', 0)}"
    )
    print(f"text_export_permitted={text_export_permitted(state, datetime.now(timezone.utc))}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

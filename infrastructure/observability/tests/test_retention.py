from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from infrastructure.observability.retention import (
    EPOCH,
    ProjectCredentials,
    cleanup,
    text_export_permitted,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
PROJECTS = (
    ProjectCredentials("personal", "pk-personal-test", "sk-personal-test"),
    ProjectCredentials("synthetic", "pk-synthetic-test", "sk-synthetic-test"),
)


class FakeResponse:
    status_code = 200

    def __init__(self, data: list[dict[str, Any]], cursor: str | None = None) -> None:
        self.payload = {"data": data, "meta": {"cursor": cursor}}

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(
        self, observations: dict[str, list[dict[str, Any]]] | None = None
    ) -> None:
        self.observations = observations or {
            "pk-personal-test": [],
            "pk-synthetic-test": [],
        }
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.pages: dict[str, list[FakeResponse]] = {}

    def get(
        self, url: str, *, params: dict[str, Any], auth: tuple[str, str], timeout: float
    ):
        self.get_calls.append(
            {"url": url, "params": params, "auth": auth, "timeout": timeout}
        )
        if auth[0] in self.pages:
            pages = self.pages[auth[0]]
            return pages.pop(0) if pages else FakeResponse([])
        cutoff = datetime.fromisoformat(
            str(params["toStartTime"]).replace("Z", "+00:00")
        )
        start = datetime.fromisoformat(
            str(params["fromStartTime"]).replace("Z", "+00:00")
        )
        rows = [
            row
            for row in self.observations[auth[0]]
            if start
            <= datetime.fromisoformat(row["startTime"].replace("Z", "+00:00"))
            < cutoff
        ]
        return FakeResponse(rows)

    def delete(
        self,
        url: str,
        *,
        json_body: dict[str, list[str]],
        auth: tuple[str, str],
        timeout: float,
    ):
        self.delete_calls.append(
            {"url": url, "json": json_body, "auth": auth, "timeout": timeout}
        )
        deleted = set(json_body["traceIds"])
        self.observations[auth[0]] = [
            row for row in self.observations[auth[0]] if row["traceId"] not in deleted
        ]
        return FakeResponse([])


class DelayedDeleteClient(FakeClient):
    def __init__(self) -> None:
        super().__init__(
            {
                "pk-personal-test": [_row("delayed", NOW - timedelta(days=4))],
                "pk-synthetic-test": [],
            }
        )
        self.apply_deletion = False

    def delete(
        self,
        url: str,
        *,
        json_body: dict[str, list[str]],
        auth: tuple[str, str],
        timeout: float,
    ):
        self.delete_calls.append(
            {"url": url, "json": json_body, "auth": auth, "timeout": timeout}
        )
        if self.apply_deletion:
            deleted = set(json_body["traceIds"])
            self.observations[auth[0]] = [
                row
                for row in self.observations[auth[0]]
                if row["traceId"] not in deleted
            ]
        return FakeResponse([])


def _row(trace_id: str, timestamp: datetime) -> dict[str, str]:
    return {
        "traceId": trace_id,
        "startTime": timestamp.isoformat().replace("+00:00", "Z"),
    }


def _state_dir(path: Path) -> Path:
    path.mkdir()
    (path / "credentials.json").write_text(
        json.dumps(
            {
                "personal_project": {
                    "public_key": "pk-personal-test",
                    "secret_key": "sk-personal-test",
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "synthetic-project.env").write_text(
        "LANGFUSE_PUBLIC_KEY=pk-synthetic-test\nLANGFUSE_SECRET_KEY=sk-synthetic-test\n",
        encoding="utf-8",
    )
    return path


def test_cleanup_deletes_only_traces_with_start_before_fixed_72h_cutoff(
    tmp_path: Path,
) -> None:
    expired_time = NOW - timedelta(hours=73)
    exact_boundary = NOW - timedelta(hours=72)
    client = FakeClient(
        {
            "pk-personal-test": [
                _row("expired", expired_time),
                _row("exact-boundary", exact_boundary),
                _row("recent", NOW - timedelta(hours=2)),
            ],
            "pk-synthetic-test": [],
        }
    )

    state, success = cleanup(
        base_url="http://langfuse-web:3000/",
        projects=PROJECTS,
        state_dir=tmp_path,
        client=client,
        clock=lambda: NOW,
    )

    assert success
    assert client.delete_calls == [
        {
            "url": "http://langfuse-web:3000/api/public/traces",
            "json": {"traceIds": ["expired"]},
            "auth": ("pk-personal-test", "sk-personal-test"),
            "timeout": 10,
        }
    ]
    assert state["projects"]["personal"]["last_pending_count"] == 0
    assert state["projects"]["personal"]["last_success_cutoff"] == (
        NOW - timedelta(hours=72)
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert client.get_calls[0]["params"]["fromStartTime"] == EPOCH.isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def test_exports_fail_closed_until_both_projects_have_recent_verified_cleanup() -> None:
    assert not text_export_permitted({"projects": {}}, NOW)
    fresh = {
        "projects": {
            label: {"last_success_at": (NOW - timedelta(minutes=30)).isoformat()}
            for label in ("personal", "synthetic")
        }
    }
    assert text_export_permitted(fresh, NOW)
    fresh["projects"]["synthetic"]["last_success_at"] = (
        NOW - timedelta(hours=2, seconds=1)
    ).isoformat()
    assert not text_export_permitted(fresh, NOW)


def test_failed_cleanup_revokes_an_otherwise_fresh_success() -> None:
    state = {
        "projects": {
            label: {
                "last_success_at": (NOW - timedelta(minutes=10)).isoformat(),
                "last_failure_at": None,
            }
            for label in ("personal", "synthetic")
        }
    }
    assert text_export_permitted(state, NOW)

    state["projects"]["synthetic"]["last_failure_at"] = (
        NOW - timedelta(minutes=1)
    ).isoformat()
    assert not text_export_permitted(state, NOW)


def test_api_acceptance_is_not_reported_as_verified_deletion(tmp_path: Path) -> None:
    client = DelayedDeleteClient()
    first_state, first_success = cleanup(
        base_url="http://langfuse-web:3000",
        projects=PROJECTS,
        state_dir=tmp_path,
        client=client,
        clock=lambda: NOW,
    )
    assert not first_success
    assert (
        first_state["projects"]["personal"]["last_error"]
        == "deletion_pending_verification"
    )
    assert first_state["projects"]["personal"]["last_pending_count"] == 1
    assert "last_success_at" not in first_state["projects"]["personal"]

    client.apply_deletion = True
    second_state, second_success = cleanup(
        base_url="http://langfuse-web:3000",
        projects=PROJECTS,
        state_dir=tmp_path,
        client=client,
        clock=lambda: NOW + timedelta(hours=1),
    )
    # The row was visible to this verification read, so success still waits.
    assert not second_success
    assert (
        second_state["projects"]["personal"]["last_error"]
        == "deletion_pending_verification"
    )

    final_state, final_success = cleanup(
        base_url="http://langfuse-web:3000",
        projects=PROJECTS,
        state_dir=tmp_path,
        client=client,
        clock=lambda: NOW + timedelta(hours=2),
    )
    assert final_success
    assert final_state["projects"]["personal"]["last_pending_count"] == 0


def test_partial_scan_is_persisted_without_claiming_success(
    tmp_path: Path, monkeypatch
) -> None:
    from infrastructure.observability import retention

    monkeypatch.setattr(retention, "MAX_PAGES_PER_RUN", 1)
    client = FakeClient()
    client.pages["pk-personal-test"] = [
        FakeResponse([_row("one", NOW - timedelta(days=4))], "next-page")
    ]
    client.pages["pk-synthetic-test"] = [
        FakeResponse([_row("two", NOW - timedelta(days=4))], "next-page")
    ]

    state, success = cleanup(
        base_url="https://langfuse.invalid",
        projects=PROJECTS,
        state_dir=tmp_path,
        client=client,
        clock=lambda: NOW,
    )

    assert not success
    assert all(
        item["last_error"] == "page_limit_reached"
        for item in state["projects"].values()
    )
    assert all("last_success_at" not in item for item in state["projects"].values())

"""Tests for docker-compose.yml configuration and Neo4j graph service setup."""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


def find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "docker-compose.yml").is_file():
            return parent
    raise FileNotFoundError("Could not find repository root containing docker-compose.yml")


REPO_ROOT = find_repo_root()
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


def load_compose_data() -> dict[str, Any]:
    """Helper to parse docker-compose.yml safely."""
    assert COMPOSE_FILE.exists(), f"Expected docker-compose.yml at {COMPOSE_FILE}"
    with open(COMPOSE_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "docker-compose.yml root should be a mapping"
    return data


def test_compose_file_exists_and_contains_expected_services() -> None:
    data = load_compose_data()
    assert "services" in data
    services = data["services"]
    assert "api" in services
    assert "web" in services
    assert "worker" in services
    assert "neo4j" in services


def test_neo4j_service_definition() -> None:
    data = load_compose_data()
    neo4j = data["services"]["neo4j"]

    assert neo4j.get("image") == "neo4j:5.26.0-community"
    assert neo4j.get("restart") == "unless-stopped"

    # Environment
    env = neo4j.get("environment")
    assert env is not None
    if isinstance(env, list):
        assert any(item.startswith("NEO4J_AUTH=") or item == "NEO4J_AUTH" for item in env)
    elif isinstance(env, dict):
        assert "NEO4J_AUTH" in env

    # Volumes
    vols = neo4j.get("volumes", [])
    assert any("neo4j_data:/data" in v for v in vols)

    # Healthcheck
    healthcheck = neo4j.get("healthcheck")
    assert isinstance(healthcheck, dict)
    assert "test" in healthcheck
    test_cmd = (
        " ".join(healthcheck["test"])
        if isinstance(healthcheck["test"], list)
        else str(healthcheck["test"])
    )
    assert "7474" in test_cmd
    assert healthcheck.get("interval") == "10s"
    assert healthcheck.get("timeout") == "5s"
    assert healthcheck.get("retries") == 5
    assert healthcheck.get("start_period") == "20s"


def test_loopback_only_port_bindings() -> None:
    data = load_compose_data()
    services = data.get("services", {})

    for svc_name, svc_conf in services.items():
        ports = svc_conf.get("ports", [])
        for port in ports:
            port_str = str(port)
            assert port_str.startswith("127.0.0.1:"), (
                f"Service '{svc_name}' port '{port_str}' is not strictly bound to 127.0.0.1"
            )
            assert "0.0.0.0" not in port_str, (
                f"Service '{svc_name}' port '{port_str}' exposes 0.0.0.0"
            )

    assert data["services"]["api"]["ports"] == ["127.0.0.1:8000:8000"]
    assert data["services"]["web"]["ports"] == ["127.0.0.1:3000:3000"]
    assert data["services"]["neo4j"]["ports"] == [
        "127.0.0.1:7474:7474",
        "127.0.0.1:7687:7687",
    ]
    assert "ports" not in data["services"]["worker"]


def test_neo4j_data_volume_declared() -> None:
    data = load_compose_data()
    assert "volumes" in data, "Top-level volumes mapping must be present"
    volumes = data["volumes"]
    assert "neo4j_data" in volumes, "neo4j_data named volume must be declared"


def test_api_and_worker_internal_neo4j_uri_configuration() -> None:
    data = load_compose_data()
    api_svc = data["services"]["api"]
    worker_svc = data["services"]["worker"]

    def extract_env_val(svc: dict[str, Any], key: str) -> str | None:
        env = svc.get("environment", {})
        if isinstance(env, dict):
            return env.get(key)
        if isinstance(env, list):
            for item in env:
                if item.startswith(f"{key}="):
                    return item.split("=", 1)[1]
        return None

    api_neo4j_uri = extract_env_val(api_svc, "NEO4J_URI")
    worker_neo4j_uri = extract_env_val(worker_svc, "NEO4J_URI")

    assert api_neo4j_uri is not None, "api service must define NEO4J_URI"
    assert "bolt://neo4j:7687" in api_neo4j_uri
    assert worker_neo4j_uri is not None, "worker service must define NEO4J_URI"
    assert "bolt://neo4j:7687" in worker_neo4j_uri


def test_docker_compose_config_validity() -> None:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        pytest.skip("docker binary not found in PATH")

    res = subprocess.run(
        [docker_bin, "compose", "-f", str(COMPOSE_FILE), "config", "--quiet"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"docker compose config failed: {res.stderr}"


def test_existing_services_structure_preserved() -> None:
    data = load_compose_data()
    api = data["services"]["api"]
    web = data["services"]["web"]
    worker = data["services"]["worker"]

    assert api.get("image") == "myra-api:latest"
    assert api.get("build", {}).get("context") == "./apps/api"
    assert "command" in api and "uvicorn" in api["command"]
    assert "volumes" in api

    assert web.get("build", {}).get("context") == "./apps/web"
    assert web.get("depends_on") == ["api"]

    assert worker.get("image") == "myra-api:latest"
    assert worker.get("build", {}).get("context") == "./apps/api"
    assert worker.get("command") == "python -m app.worker"
    assert worker.get("depends_on") == ["api"]


def test_settings_and_system_status_with_compose_default_neo4j_uri(monkeypatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    settings = Settings.from_environment()
    assert settings.neo4j_uri == "bolt://neo4j:7687"

    with TestClient(app) as client:
        resp = client.get("/api/v1/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["neo4j_configured"] is True
    assert "bolt://neo4j:7687" not in resp.text

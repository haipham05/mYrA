"""Keep the default test suite away from developer and cloud resources."""

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pytest

# conftest is imported before test modules, which import app.db.session at collection.
_test_directory = tempfile.TemporaryDirectory(prefix="myra-pytest-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_test_directory.name) / 'test.db'}"
os.environ["GCS_BUCKET_NAME"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["MYRA_LLM_MODE"] = "test"
os.environ["MYRA_EMBEDDING_PROVIDER"] = "deterministic"
os.environ["MYRA_RERANKER_PROVIDER"] = "simple-lexical"
os.environ["MYRA_USE_DOCLING"] = "false"
os.environ["MYRA_CHECK_MIGRATIONS"] = "false"


def disposable_neo4j_test_uri() -> str | None:
    """Return an explicitly acknowledged isolated Neo4j test endpoint.

    Integration tests never infer permission from a reachable local port. The
    target must use the dedicated non-production loopback port and must not
    equal the application's configured Neo4j endpoint.
    """
    if os.environ.get("MYRA_ALLOW_DISPOSABLE_NEO4J_TESTS") != "1":
        return None
    uri = os.environ.get("MYRA_TEST_NEO4J_URI", "").strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    if (
        parsed.scheme not in {"bolt", "neo4j"}
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port != 17687
        or uri == os.environ.get("NEO4J_URI")
    ):
        raise pytest.UsageError(
            "MYRA_TEST_NEO4J_URI must target the dedicated loopback port 17687 "
            "and differ from NEO4J_URI"
        )
    return uri


@pytest.fixture(scope="session")
def disposable_neo4j_uri() -> str:
    uri = disposable_neo4j_test_uri()
    if uri is None:
        pytest.skip(
            "Neo4j integration checks require MYRA_TEST_NEO4J_URI and "
            "MYRA_ALLOW_DISPOSABLE_NEO4J_TESTS=1"
        )
    return uri


def pytest_sessionfinish() -> None:
    from app.db.session import engine

    engine.dispose()
    _test_directory.cleanup()

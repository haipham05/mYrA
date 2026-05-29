"""Keep the default test suite away from developer and cloud resources."""

import os
import tempfile
from pathlib import Path

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


def pytest_sessionfinish() -> None:
    from app.db.session import engine

    engine.dispose()
    _test_directory.cleanup()

"""Validation helpers for versioned, offline generated-PDF evaluation inputs."""

import hashlib
import json
from pathlib import Path
from typing import Any

from gold_corpus import GOLD_PAPERS, GOLD_QUESTIONS

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_and_validate_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported evaluation manifest version")

    expected_files = manifest.get("fixture_hashes", {})
    actual_files = {paper["filename"] for paper in GOLD_PAPERS}
    if set(expected_files) != actual_files:
        raise ValueError("Evaluation manifest PDF list does not match gold corpus")

    for filename, expected_hash in expected_files.items():
        path = fixtures_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation fixture PDF is missing: {filename}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Evaluation fixture hash changed: {filename}")

    for name, value in (
        ("gold_papers_sha256", GOLD_PAPERS),
        ("gold_questions_sha256", GOLD_QUESTIONS),
    ):
        if canonical_hash(value) != manifest.get(name):
            raise ValueError(f"Evaluation annotation hash changed: {name}")
    return manifest


def write_report(report: dict[str, Any], *, prefix: str) -> Path:
    """Write machine-readable output to a disposable OS temp directory."""
    import tempfile

    result_dir = Path(tempfile.mkdtemp(prefix=f"myra-{prefix}-results-"))
    output_path = result_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output_path

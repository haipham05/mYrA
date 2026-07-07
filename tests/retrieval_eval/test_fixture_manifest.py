import shutil
from pathlib import Path

import pytest
from fixture_manifest import load_and_validate_manifest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_manifest_matches_four_pdf_inputs_and_annotations() -> None:
    manifest = load_and_validate_manifest(FIXTURES_DIR)

    assert manifest["manifest_version"] == 1
    assert len(manifest["fixture_hashes"]) == 4
    assert manifest["gold_questions_sha256"]
    assert manifest["gold_papers_sha256"]
    assert manifest["annotation_method"]
    assert manifest["memory_expected_facts"]


def test_manifest_rejects_changed_pdf_hash(tmp_path: Path) -> None:
    copied_fixtures = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, copied_fixtures)
    fixture = next(copied_fixtures.glob("*.pdf"))
    fixture.write_bytes(fixture.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="fixture hash changed"):
        load_and_validate_manifest(copied_fixtures)

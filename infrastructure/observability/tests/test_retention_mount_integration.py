from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("MYRA_RUN_DOCKER_INTEGRATION") != "1",
    reason="set MYRA_RUN_DOCKER_INTEGRATION=1 to verify directory bind-mount updates",
)
def test_atomic_state_replacement_is_visible_through_read_only_directory_mount(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "retention"
    state_dir.mkdir(mode=0o700)
    state_path = state_dir / "retention-state.json"
    state_path.write_text(json.dumps({"format_version": 1, "projects": {}}))
    container_name = f"myra-retention-mount-{uuid.uuid4().hex[:10]}"
    reader = (
        "import json,time; p='/state/retention-state.json'; "
        "print(json.load(open(p))['format_version'], flush=True); time.sleep(2); "
        "print(','.join(sorted(json.load(open(p))['projects'])), flush=True)"
    )

    process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container_name,
            "-v",
            f"{state_dir}:/state:ro",
            "myra-api:latest",
            "python",
            "-c",
            reader,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    initial_state_version = process.stdout.readline().strip()
    assert initial_state_version == "1"

    replacement = state_dir / ".retention-state.next"
    replacement.write_text(
        json.dumps({"format_version": 1, "projects": {"personal": {}}})
    )
    os.replace(replacement, state_path)

    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert stdout.splitlines() == ["personal"]

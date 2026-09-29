"""Real queued-stage dispatch keeps the claimed source and publication fence."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from zanzara_archive import cli
from zanzara_archive.corpus import load_manifest
from zanzara_archive.storage import SQLiteRepository


def test_worker_process_dispatches_fenced_stage_and_blocks_invalid_job(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A claimed job reaches the real stage path with its manifest episode."""

    database = tmp_path / "state.db"
    episode = load_manifest("planning/corpus-20.json").episodes[0]
    repository = SQLiteRepository.open(database)
    repository.enqueue_job(
        job_id="real-attribution", stage="attribution", source_sha256=episode.sha256
    )
    repository.close()

    def process(arguments):
        assert arguments.episode == episode.relative_filename
        assert arguments.job.job_id == "real-attribution"
        assert arguments.repository.fetch_job("real-attribution").status == "running"
        return {"artifact_path": "complete-artifact"}

    monkeypatch.setattr(cli, "_process_stage", process)
    command = ["worker", "run", "--process", "--database", str(database), "--owner", "test"]
    with Path(f"{database}.worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="2"):
            cli.main(command)
    assert "another real worker is active" in capsys.readouterr().err

    assert cli.main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["job"]["status"] == "succeeded"
    assert result["stage_result"]["artifact_path"] == "complete-artifact"

    repository = SQLiteRepository.open(database)
    repository.enqueue_job(job_id="invalid", stage="decode", source_sha256=episode.sha256)
    repository.close()
    assert cli.main(command) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["job"]["status"] == "blocked"
    assert result["error"]["code"] == "invalid_job"

"""CPU-only crash and reconciliation checks for immutable artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from zanzara_archive.artifacts import ArtifactPublicationError, ArtifactPublisher
from zanzara_archive.storage import SQLiteRepository, StorageConflictError


def test_publish_uses_hashed_layout_and_complete_manifest(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    publisher = ArtifactPublisher(tmp_path / "artifacts", repository)
    manifest = publisher.publish(
        source_sha256="a" * 64,
        stage="decode",
        stage_key="config-hash",
        files={"nested/audio.pcm": b"synthetic"},
        pipeline_version="test",
        provenance={"transform_hash": "config-hash", "decoder": "fixture"},
    )
    artifact_dir = tmp_path / "artifacts" / ("a" * 64) / "decode" / "config-hash"
    assert artifact_dir == publisher.artifact_path(
        manifest.source_sha256, manifest.stage, manifest.stage_key
    )
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "provenance.json").read_text() == (
        '{"decoder":"fixture","transform_hash":"config-hash"}'
    )
    assert repository.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
    assert (
        publisher.publish(
            source_sha256="a" * 64,
            stage="decode",
            stage_key="config-hash",
            files={"nested/audio.pcm": b"synthetic"},
            pipeline_version="test",
        )
        == manifest
    )
    repository.close()


def test_complete_orphan_is_reconciled_and_partial_output_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    without_database = ArtifactPublisher(root)
    manifest = without_database.publish(
        source_sha256="b" * 64,
        stage="asr",
        stage_key="model-hash",
        files={"words.json": b"{}"},
    )
    partial = root / ("b" * 64) / "asr" / "partial"
    partial.mkdir(parents=True)
    (partial / "words.json").write_bytes(b"unfinished")
    repository = SQLiteRepository.open(tmp_path / "state.db")
    recovered, errors = ArtifactPublisher(root, repository).reconcile()
    assert recovered == (manifest,)
    assert errors == ()
    assert repository.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
    repository.close()


def test_failed_validation_leaves_no_published_directory(tmp_path: Path) -> None:
    publisher = ArtifactPublisher(tmp_path / "artifacts")
    with pytest.raises(ArtifactPublicationError):
        publisher.publish(
            source_sha256="c" * 64,
            stage="decode",
            stage_key="key",
            files={"../escape": b"must fail"},
        )
    assert not (tmp_path / "artifacts" / ("c" * 64) / "decode" / "key").exists()
    assert not list((tmp_path / "artifacts").glob("**/*.tmp"))


def test_corrupt_completed_manifest_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    publisher = ArtifactPublisher(root)
    publisher.publish(
        source_sha256="d" * 64,
        stage="decode",
        stage_key="key",
        files={"data": b"one"},
    )
    artifact_dir = root / ("d" * 64) / "decode" / "key"
    (artifact_dir / "data").write_bytes(b"two")
    recovered, errors = publisher.reconcile()
    assert recovered == ()
    assert len(errors) == 1


def test_stale_worker_cannot_publish_artifact_or_generation(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="publish", stage="decode", source_sha256="e" * 64)
    stale = repository.claim_job("publish", "worker-a", now="2026-01-01T00:00:00+00:00")
    current = repository.claim_job("publish", "worker-b", now="2026-01-01T00:02:01+00:00")
    publisher = ArtifactPublisher(tmp_path / "artifacts", repository)

    with pytest.raises(StorageConflictError, match="stale worker"):
        publisher.publish(
            source_sha256="e" * 64,
            stage="decode",
            stage_key="stale-key",
            files={"output": b"stale"},
            job=stale,
            now="2026-01-01T00:02:02+00:00",
        )
    assert not publisher.artifact_path("e" * 64, "decode", "stale-key").exists()
    assert repository.connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0

    with pytest.raises(StorageConflictError, match="stale worker"):
        repository.publish_generation(
            generation_id="stale-generation",
            kind="speaker",
            generation_key="stale-key",
            pointer_name="active-speaker",
            expected_points=1,
            job=stale,
            now="2026-01-01T00:02:02+00:00",
        )
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM index_generations").fetchone()[0] == 0
    )
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM committed_pointers").fetchone()[0] == 0
    )

    publisher.publish(
        source_sha256="e" * 64,
        stage="decode",
        stage_key="current-key",
        files={"output": b"current"},
        job=current,
        now="2026-01-01T00:02:02+00:00",
    )
    repository.close()

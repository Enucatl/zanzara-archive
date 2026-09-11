"""CPU-only crash and reconciliation checks for immutable artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from zanzara_archive.artifacts import ArtifactPublicationError, ArtifactPublisher
from zanzara_archive.storage import SQLiteRepository


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

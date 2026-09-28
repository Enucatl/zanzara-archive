from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.calibration import build_batch, validate_batch
from zanzara_archive.chunking import Community1NativeAdaptiveConfig
from zanzara_archive.contracts import AudioChunk
from zanzara_archive.corpus import load_manifest
from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app


def _batch() -> dict:
    manifest = load_manifest("planning/corpus-20.json")
    episode = manifest.episodes[0]
    chunk = AudioChunk.create(
        episode_id=episode.relative_filename,
        source_sha256=episode.sha256,
        start_ms=0,
        end_ms=8_000,
        segmentation_fingerprint="a" * 64,
        partition="development",
    )
    return {
        "batch_id": "calibration-fixture",
        "manifest_sha256": manifest.sha256,
        "partition": "development",
        "segmentation_version": "fixture-v1",
        "split": {
            "development": [episode.relative_filename],
            "held_out": [item.relative_filename for item in manifest.episodes[1:]],
        },
        "chunks": [chunk.to_dict()],
    }


def test_batch_validation_rejects_held_out_and_normalizes() -> None:
    manifest = load_manifest("planning/corpus-20.json")
    payload = _batch()
    result = validate_batch(payload, manifest)
    assert result["partition"] == "development"
    assert result["content_sha256"]
    payload["partition"] = "held_out"
    try:
        validate_batch(payload, manifest)
    except ValueError as exc:
        assert "development" in str(exc)
    else:
        raise AssertionError("held-out batch was accepted")


def test_build_batch_selects_deterministic_development_clips() -> None:
    manifest = load_manifest("planning/corpus-20.json")
    split = {
        "development": [manifest.episodes[0].relative_filename],
        "held_out": [episode.relative_filename for episode in manifest.episodes[1:]],
    }
    first = build_batch(manifest, split, batch_id="generated-fixture", clips_per_episode=3)
    second = build_batch(manifest, split, batch_id="generated-fixture", clips_per_episode=3)
    assert first == second
    assert len(first["chunks"]) == 3
    assert all(chunk["partition"] == "development" for chunk in first["chunks"])
    assert first["segmentation_configuration"] == {
        "version": "p1r-chunk-segmentation-v1",
        "preferred_min_s": 8.0,
        "target_s": 12.0,
        "preferred_max_s": 18.0,
        "hard_max_s": 30.0,
    }


def test_build_batch_uses_only_verified_native_development_chunks() -> None:
    manifest = load_manifest("planning/corpus-20.json")
    episode = manifest.episodes[0]
    split = {
        "development": [episode.relative_filename],
        "held_out": [item.relative_filename for item in manifest.episodes[1:]],
    }
    config = Community1NativeAdaptiveConfig()
    chunk = AudioChunk.create(
        episode_id=episode.relative_filename,
        source_sha256=episode.sha256,
        start_ms=0,
        end_ms=12_000,
        segmentation_fingerprint="a" * 64,
        partition="development",
        segmentation_version=config.version,
        segmentation_configuration_hash=config.configuration_sha256,
    )
    source = {
        "algorithm": config.version,
        "corpus_manifest_sha256": manifest.sha256,
        "segmentation_version": config.version,
        "segmentation_configuration": config.to_dict(),
        "segmentation_configuration_hash": config.configuration_sha256,
        "chunks": [chunk.to_dict()],
    }
    batch = build_batch(manifest, split, batch_id="native-fixture", chunks_manifest=source)
    assert batch["chunks"] == [chunk.to_dict()]
    assert validate_batch(batch, manifest) == batch

    with pytest.raises(ValueError, match="provenance"):
        build_batch(
            manifest,
            split,
            batch_id="native-fixture",
            chunks_manifest={**source, "corpus_manifest_sha256": "0" * 64},
        )
    with pytest.raises(ValueError, match="segmentation provenance"):
        build_batch(
            manifest,
            split,
            batch_id="native-fixture",
            chunks_manifest={
                **source,
                "chunks": [{**chunk.to_dict(), "segmentation_configuration_hash": "0" * 64}],
            },
        )
    held_out = manifest.episodes[1]
    with pytest.raises(ValueError, match="outside the development split"):
        validate_batch(
            {
                **batch,
                "chunks": [
                    AudioChunk.create(
                        episode_id=held_out.relative_filename,
                        source_sha256=held_out.sha256,
                        start_ms=0,
                        end_ms=12_000,
                        segmentation_fingerprint="a" * 64,
                        partition="development",
                        segmentation_version=config.version,
                        segmentation_configuration_hash=config.configuration_sha256,
                    ).to_dict()
                ],
            },
            manifest,
        )


def test_calibration_page_decision_resume_conflict_and_export(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    client = TestClient(
        create_app(
            database,
            artifact_root=tmp_path / "artifacts",
            manifest_path="planning/corpus-20.json",
            archive_root="/export/scratch/archive/zanzara",
        )
    )
    created = client.post("/api/v1/calibration/batches", json=_batch())
    assert created.status_code == 200
    assert client.get("/calibration/calibration-fixture").status_code == 200
    chunk_id = _batch()["chunks"][0]["chunk_id"]
    decision = {
        "chunk_id": chunk_id,
        "music_level": "none",
        "reviewer": "reviewer",
        "expected_revision": 0,
        "note": "clear",
    }
    saved = client.post("/api/v1/calibration/calibration-fixture/decisions", json=decision)
    assert saved.status_code == 200
    assert saved.json()["data"]["revision"] == 1
    conflict = client.post("/api/v1/calibration/calibration-fixture/decisions", json=decision)
    assert conflict.status_code == 409
    resumed = client.get("/api/v1/calibration/calibration-fixture").json()["data"]
    assert resumed["decisions"][chunk_id]["music_level"] == "none"
    exported = client.get("/api/v1/calibration/calibration-fixture/export")
    assert exported.status_code == 200
    assert exported.json()["data"]["reviewed_count"] == 1

    repository = SQLiteRepository.open(database)
    try:
        assert repository.fetch_calibration_batch("calibration-fixture") is not None
    finally:
        repository.close()

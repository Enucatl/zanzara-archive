"""CPU-only tests for frozen manifest and read-only source validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.corpus import (
    CorpusManifest,
    CorpusValidationError,
    _hash_stable,
    load_manifest,
    resolve_source,
    verify_corpus,
)

MANIFEST = Path(__file__).parents[1] / "planning" / "corpus-20.json"


def _manifest_bytes() -> bytes:
    return MANIFEST.read_bytes()


def test_planning_manifest_is_frozen_and_complete() -> None:
    manifest = load_manifest(MANIFEST)
    assert len(manifest.episodes) == 20
    assert manifest.golden_episode == "260910-lazanzara.opus"
    assert min(item.episode_date.isoformat() for item in manifest.episodes) == "2026-07-01"
    assert max(item.episode_date.isoformat() for item in manifest.episodes) == "2026-09-10"
    assert manifest.sha256


def test_manifest_rejects_count_change() -> None:
    value = json.loads(_manifest_bytes())
    value["episodes"] = value["episodes"][:-1]
    with pytest.raises(CorpusValidationError, match="exactly 20"):
        CorpusManifest.from_bytes(json.dumps(value).encode())


def test_resolve_source_rejects_traversal_and_escaping_symlink(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "inside.opus").write_bytes(b"audio")
    outside = tmp_path / "outside.opus"
    outside.write_bytes(b"private")
    (root / "escape.opus").symlink_to(outside)

    assert resolve_source(root, "inside.opus") == (root / "inside.opus").resolve()
    with pytest.raises(CorpusValidationError):
        resolve_source(root, "../outside.opus")
    with pytest.raises(CorpusValidationError):
        resolve_source(root, str(root / "inside.opus"))
    with pytest.raises(CorpusValidationError, match="escapes archive root"):
        resolve_source(root, "escape.opus")


def test_verification_records_each_failure_without_writing_archive(tmp_path: Path) -> None:
    value = json.loads(_manifest_bytes())
    value["episodes"] = value["episodes"][:]
    manifest = CorpusManifest.from_bytes(json.dumps(value).encode())
    root = tmp_path / "archive"
    root.mkdir()
    for episode in manifest.episodes:
        (root / episode.relative_filename).write_bytes(b"synthetic")

    report = verify_corpus(manifest, root, ffprobe_binary="definitely-missing-ffprobe")
    assert not report["valid"]
    assert len(report["files"]) == 20
    assert all(item["status"] == "error" for item in report["files"])
    assert not list(root.glob("**/*.tmp"))


def test_hash_rejects_source_changed_during_read(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "episode.opus"
    source.write_bytes(b"synthetic audio")
    initial = source.stat()
    before = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
    changed = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns + 1)
    signatures = iter((before, changed))
    monkeypatch.setattr("zanzara_archive.corpus._stat_signature", lambda _: next(signatures))

    with pytest.raises(CorpusValidationError, match="changed while reading"):
        _hash_stable(source)

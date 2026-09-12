"""CPU checks for annotation revisions, immutable exports, and reference validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.annotations import (
    AnnotationValidationError,
    annotation_export_path,
    save_annotation_revision,
    validate_annotation_payload,
)
from zanzara_archive.corpus import load_manifest
from zanzara_archive.evaluation import validate_reference
from zanzara_archive.storage import SQLiteRepository, StorageConflictError


def _episode(
    repository: SQLiteRepository, *, episode_id: str, source: str, duration_ms: int
) -> None:
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        (f"manifest-{episode_id}", "b" * 64, "fixture", episode_id, "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, '2026-09-10', ?, 100, ?, 'opus', 1, 48000)""",
        (episode_id, f"manifest-{episode_id}", episode_id, source, duration_ms),
    )
    repository.connection.commit()


def _payload(*, episode_id: str = "golden.opus", source: str = "a" * 64, count: int = 2) -> dict:
    words = [
        {
            "word_id": f"word-{index}",
            "text": f"parola-{index}",
            "start_ms": index * 100,
            "end_ms": index * 100 + 80,
            "speaker_id": "SPEAKER_A",
            "overlap": index == 1,
            "unintelligible": False,
        }
        for index in range(count)
    ]
    return {
        "episode_id": episode_id,
        "source_sha256": source,
        "duration_ms": max(1_000, count * 100 + 100),
        "words": words,
        "standard_turns": [
            {
                "turn_id": "turn-a",
                "speaker_id": "SPEAKER_A",
                "start_ms": 0,
                "end_ms": max(1_000, count * 100 + 100),
            }
        ],
        "exclusive_turns": [
            {
                "turn_id": "turn-a",
                "speaker_id": "SPEAKER_A",
                "start_ms": 0,
                "end_ms": max(1_000, count * 100 + 100),
            }
        ],
        "overlap_intervals": [
            {
                "overlap_id": "overlap-a",
                "speaker_ids": ["SPEAKER_A", "SPEAKER_B"],
                "start_ms": 100,
                "end_ms": 180,
            }
        ]
        if count > 1
        else [],
        "unintelligible_spans": [],
        "provenance": {"configuration": {"fixture": True}, "models": {"asr": "synthetic"}},
    }


def test_machine_output_cannot_be_marked_reviewed() -> None:
    payload = _payload()
    payload.update(
        {
            "status": "reviewed",
            "actor_type": "machine",
            "reviewer": "machine",
            "reviewed_word_ids": ["word-0", "word-1"],
        }
    )
    with pytest.raises(AnnotationValidationError, match="machine output"):
        validate_annotation_payload(payload)


def test_revision_source_must_match_registered_episode(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    _episode(repository, episode_id="golden.opus", source="a" * 64, duration_ms=1_000)
    payload = _payload(source="b" * 64)
    with pytest.raises(AnnotationValidationError, match="source_sha256"):
        save_annotation_revision(
            repository,
            tmp_path / "artifacts",
            episode_id="golden.opus",
            payload=payload,
            expected_revision=0,
            actor_type="machine",
        )
    assert repository.fetch_annotation_revision("golden.opus") is None
    repository.close()


def test_revisions_are_append_only_and_conflicts_preserve_current_state(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    artifact_root = tmp_path / "artifacts"
    repository = SQLiteRepository.open(database)
    _episode(repository, episode_id="golden.opus", source="a" * 64, duration_ms=1_000)
    draft = save_annotation_revision(
        repository,
        artifact_root,
        episode_id="golden.opus",
        payload=_payload(),
        expected_revision=0,
        actor_type="machine",
    )
    assert draft.payload["status"] == "draft"
    assert (annotation_export_path(artifact_root, draft.payload) / "reference.json").is_file()

    reviewed_payload = dict(draft.payload)
    reviewed_payload.update(
        {
            "status": "reviewed",
            "reviewed_word_ids": ["word-0", "word-1"],
            "manual_timing_word_ids": ["word-0"],
        }
    )
    reviewed = save_annotation_revision(
        repository,
        artifact_root,
        episode_id="golden.opus",
        payload=reviewed_payload,
        expected_revision=1,
        actor_type="human",
        reviewer="reviewer@example.test",
    )
    assert reviewed.payload["status"] == "reviewed"
    assert [item["revision"] for item in repository.list_annotation_revisions("golden.opus")] == [
        2,
        1,
    ]
    with pytest.raises(StorageConflictError, match="revision conflict"):
        save_annotation_revision(
            repository,
            artifact_root,
            episode_id="golden.opus",
            payload=reviewed_payload,
            expected_revision=1,
            actor_type="human",
            reviewer="another-reviewer",
        )
    assert repository.fetch_annotation_revision("golden.opus")["revision"] == 2
    repository.close()


def test_reference_validator_checks_golden_provenance_and_two_hundred_timings(
    tmp_path: Path,
) -> None:
    corpus_path = Path("planning/corpus-20.json")
    corpus = load_manifest(corpus_path)
    golden = next(
        item for item in corpus.episodes if item.relative_filename == corpus.golden_episode
    )
    repository = SQLiteRepository.open(tmp_path / "state.db")
    _episode(
        repository,
        episode_id=golden.relative_filename,
        source=golden.sha256,
        duration_ms=golden.duration_ms,
    )
    payload = _payload(episode_id=golden.relative_filename, source=golden.sha256, count=200)
    payload["duration_ms"] = golden.duration_ms
    payload["standard_turns"][0]["end_ms"] = golden.duration_ms
    payload["exclusive_turns"][0]["end_ms"] = golden.duration_ms
    boundaries = [(index * golden.duration_ms) // 5 for index in range(6)]
    for index, word in enumerate(payload["words"]):
        start_ms = boundaries[index % 5] + (index // 5) * 100
        word.update(start_ms=start_ms, end_ms=start_ms + 80)
    payload.update(
        {
            "status": "reviewed",
            "reviewed_word_ids": [f"word-{index}" for index in range(200)],
            "manual_timing_word_ids": [f"word-{index}" for index in range(200)],
        }
    )
    saved = save_annotation_revision(
        repository,
        tmp_path / "artifacts",
        episode_id=golden.relative_filename,
        payload=payload,
        expected_revision=0,
        actor_type="human",
        reviewer="human-reviewer",
    )
    reference = annotation_export_path(tmp_path / "artifacts", saved.payload) / "reference.json"
    report = validate_reference(corpus_path, reference)
    assert report["valid"] is True
    assert report["human_judgment_validated"] is False
    repository.close()


def test_reference_validator_rejects_incomplete_review(tmp_path: Path) -> None:
    payload = _payload()
    payload.update(
        {
            "status": "reviewed",
            "reviewer": "human-reviewer",
            "reviewed_word_ids": ["word-0"],
            "manual_timing_word_ids": ["word-0"],
        }
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_reference("planning/corpus-20.json", path)
    assert report["valid"] is False
    assert any("every transcript word" in error for error in report["errors"])

"""Deterministic CPU fixtures for P1R-08 speaker metadata."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zanzara_archive.contracts import (
    AudioChunk,
    ContractValidationError,
    SpeakerConditionCorrection,
    Turn,
)
from zanzara_archive.speaker_conditions import (
    SPEAKER_CONDITION_VERSION,
    SpeakerConditionThresholds,
    derive_chunk_speaker_metadata,
    derive_speaker_conditions,
)
from zanzara_archive.storage import SQLiteRepository

SOURCE_SHA256 = "a" * 64
SEGMENTATION_SHA256 = "b" * 64


def _chunk(start_ms: int = 0, end_ms: int = 10_000) -> AudioChunk:
    return AudioChunk.create(
        episode_id="episode-speaker-conditions",
        source_sha256=SOURCE_SHA256,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=10_000,
        segmentation_fingerprint=SEGMENTATION_SHA256,
    )


def test_missing_diarization_is_explicitly_unknown() -> None:
    chunk = derive_chunk_speaker_metadata(_chunk())
    metadata = chunk.condition.speaker_metadata

    assert metadata is not None
    assert metadata.status == "unknown"
    assert metadata.speaker_condition == "uncertain"
    assert metadata.speaker_ids == ()
    assert metadata.overlap_fraction is None
    assert chunk.condition.speaker_count == 0
    assert not chunk.condition.has_overlap


def test_zero_one_and_two_speaker_conditions_use_union_speech() -> None:
    zero = derive_chunk_speaker_metadata(_chunk(), standard_turns=(), exclusive_turns=())
    one = derive_chunk_speaker_metadata(
        _chunk(),
        standard_turns=(Turn("A", 1_000, 9_000),),
        exclusive_turns=(Turn("A", 1_000, 9_000),),
    )
    two = derive_chunk_speaker_metadata(
        _chunk(),
        standard_turns=(Turn("A", 0, 3_000), Turn("B", 3_000, 6_000)),
        exclusive_turns=(Turn("A", 0, 3_000), Turn("B", 3_000, 6_000)),
    )

    assert zero.condition.speaker_metadata.speaker_condition == "uncertain"
    assert zero.condition.speaker_metadata.status == "derived"
    assert zero.condition.speaker_metadata.overlap_fraction is None
    assert one.condition.speaker_metadata.speaker_condition == "single_speaker"
    assert one.condition.speaker_metadata.speech_ms == 8_000
    assert one.condition.speaker_metadata.max_simultaneous_speakers == 1
    assert one.condition.speaker_metadata.overlap_fraction == 0.0
    assert two.condition.speaker_metadata.speaker_condition == "multi_speaker_no_overlap"
    assert two.condition.speaker_metadata.speech_ms == 6_000
    assert two.condition.speaker_metadata.overlap_ms == 0
    assert two.condition.speaker_metadata.speaker_ids == ("A", "B")


def test_partial_heavy_and_three_speaker_edge_intersections_are_reproducible() -> None:
    partial = derive_chunk_speaker_metadata(
        _chunk(3_000, 8_000),
        standard_turns=(Turn("A", 0, 8_000), Turn("B", 4_000, 5_000)),
        exclusive_turns=(Turn("A", 0, 4_000), Turn("B", 4_000, 5_000), Turn("A", 5_000, 8_000)),
    )
    heavy = derive_chunk_speaker_metadata(
        _chunk(),
        standard_turns=(
            Turn("A", 0, 8_000),
            Turn("B", 2_000, 6_000),
            Turn("C", 3_000, 5_000),
        ),
        exclusive_turns=(Turn("A", 0, 2_000), Turn("B", 2_000, 6_000), Turn("A", 6_000, 8_000)),
    )
    repeated = derive_chunk_speaker_metadata(
        _chunk(3_000, 8_000),
        standard_turns=(Turn("A", 0, 8_000), Turn("B", 4_000, 5_000)),
        exclusive_turns=(Turn("A", 0, 4_000), Turn("B", 4_000, 5_000), Turn("A", 5_000, 8_000)),
    )

    partial_metadata = partial.condition.speaker_metadata
    heavy_metadata = heavy.condition.speaker_metadata
    assert partial_metadata.speaker_condition == "partial_overlap"
    assert partial_metadata.speech_ms == 5_000
    assert partial_metadata.overlap_ms == 1_000
    assert partial_metadata.overlap_fraction == pytest.approx(0.2)
    assert partial_metadata.has_overlap
    assert [
        (item.start_ms, item.end_ms, item.speaker_ids) for item in partial.condition.overlaps
    ] == [(4_000, 5_000, ("A", "B"))]
    assert heavy_metadata.speaker_condition == "heavy_overlap"
    assert heavy_metadata.speaker_count == 3
    assert heavy_metadata.max_simultaneous_speakers == 3
    assert heavy_metadata.overlap_ms == 4_000
    assert heavy_metadata.overlap_fraction == pytest.approx(0.5)
    assert partial == repeated
    assert partial.condition.exclusive_speaker_streams[0].speaker_id == "A"


def test_threshold_version_and_review_correction_round_trip(tmp_path: Path) -> None:
    seeded = derive_chunk_speaker_metadata(
        _chunk(),
        standard_turns=(Turn("A", 0, 5_000), Turn("B", 5_000, 10_000)),
        exclusive_turns=(Turn("A", 0, 5_000), Turn("B", 5_000, 10_000)),
        thresholds=SpeakerConditionThresholds(),
    )
    machine = seeded.condition.speaker_metadata
    reviewed = replace(
        machine,
        origin="human_review",
        speaker_ids=("A",),
        speaker_count=1,
        max_simultaneous_speakers=1,
        speech_ms=10_000,
        overlap_ms=0,
        overlap_fraction=0.0,
        has_overlap=False,
        speaker_condition="single_speaker",
    )
    correction = SpeakerConditionCorrection(
        metadata=reviewed,
        reviewer="human-1",
        reviewed_at="2026-09-13T00:00:00Z",
        reason="listening correction",
        seed_version=machine.version,
    )
    corrected = replace(
        seeded,
        condition=replace(seeded.condition, speaker_correction=correction),
    )

    assert machine.origin == "machine_seed"
    assert machine.version == SPEAKER_CONDITION_VERSION
    assert corrected.condition.speaker_metadata == machine
    assert corrected.condition.speaker_correction == correction
    assert AudioChunk.from_dict(corrected.to_dict()) == corrected

    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-speakers", "c" * 64, "fixture", "episode.opus", "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            corrected.episode_id,
            "manifest-speakers",
            "episode.opus",
            "2026-09-13",
            SOURCE_SHA256,
            100,
            10_000,
            "opus",
            1,
            48_000,
        ),
    )
    repository.connection.commit()
    repository.record_audio_chunk(corrected)
    assert repository.fetch_audio_chunk(corrected.chunk_id) == corrected
    repository.close()


def test_episode_diarization_is_applied_to_all_frozen_chunks() -> None:
    chunks = (_chunk(0, 5_000), _chunk(5_000, 10_000))
    derived = derive_speaker_conditions(
        chunks,
        standard_turns=(Turn("A", 0, 10_000),),
        exclusive_turns=(Turn("A", 0, 10_000),),
        diarization_fingerprint="d" * 64,
    )

    assert [chunk.condition.speech_ms for chunk in derived] == [5_000, 5_000]
    assert all(
        chunk.condition.speaker_metadata.diarization_fingerprint == "d" * 64 for chunk in derived
    )


def test_invalid_episode_provenance_and_incomplete_views_fail() -> None:
    with pytest.raises(ContractValidationError, match="supplied together"):
        derive_chunk_speaker_metadata(_chunk(), standard_turns=(Turn("A", 0, 1_000),))
    with pytest.raises(ContractValidationError, match="exceeds diarization duration"):
        derive_chunk_speaker_metadata(
            _chunk(),
            standard_turns=(Turn("A", 0, 11_000),),
            exclusive_turns=(Turn("A", 0, 10_000),),
        )

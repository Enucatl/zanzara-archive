"""Synthetic coverage for the Community-1 adaptive boundary policy."""

from __future__ import annotations

import pytest

from zanzara_archive.chunking import (
    Community1AdaptiveConfig,
    segment_chunks_with_metadata,
)
from zanzara_archive.contracts import ContractValidationError, Turn

SOURCE_SHA256 = "a" * 64


def _segment(name: str, duration_ms: int, turns: tuple[Turn, ...]):
    return segment_chunks_with_metadata(
        name,
        SOURCE_SHA256,
        duration_ms,
        algorithm="community1-adaptive-v1",
        standard_turns=turns,
        diarization_artifact_id="diarization-synthetic-v1",
    )


def test_case_1_ideal_strong_pause_uses_gap_midpoint() -> None:
    result = _segment(
        "case-1",
        40_000,
        (Turn("A", 0, 11_700), Turn("A", 12_500, 40_000)),
    )
    chunk = result.chunks[0]
    assert (chunk.start_ms, chunk.end_ms) == (0, 12_100)
    assert chunk.boundary_end_reason == "strong_gap"
    assert chunk.boundary_gap_duration_ms == 800


def test_case_2_strong_gap_with_speaker_change_beats_closer_gap() -> None:
    result = _segment(
        "case-2",
        50_000,
        (
            Turn("A", 0, 11_500),
            Turn("A", 12_300, 12_600),
            Turn("B", 13_400, 50_000),
        ),
    )
    chunk = result.chunks[0]
    assert chunk.end_ms == 13_000
    assert chunk.boundary_end_reason == "strong_gap_and_speaker_change"
    assert chunk.boundary_speaker_change


def test_case_3_clean_speaker_change_is_a_candidate() -> None:
    result = _segment(
        "case-3",
        40_000,
        (Turn("A", 0, 12_400), Turn("B", 12_400, 40_000)),
    )
    assert result.chunks[0].end_ms == 12_400
    assert result.chunks[0].boundary_end_reason == "speaker_change"


def test_case_4_micro_gap_is_ignored() -> None:
    result = _segment(
        "case-4",
        40_000,
        (Turn("A", 0, 11_910), Turn("A", 12_090, 40_000)),
    )
    assert result.chunks[0].end_ms == 30_000
    assert result.chunks[0].boundary_end_reason == "hard_maximum"


def test_case_5_short_gap_is_accepted() -> None:
    result = _segment(
        "case-5",
        40_000,
        (Turn("A", 0, 11_800), Turn("A", 12_200, 40_000)),
    )
    assert result.chunks[0].end_ms == 12_000
    assert result.chunks[0].boundary_end_reason == "short_gap"


def test_case_6_overlap_conflicted_change_is_avoided() -> None:
    result = _segment(
        "case-6",
        50_000,
        (
            Turn("A", 0, 12_000),
            Turn("B", 11_000, 13_000),
            Turn("B", 15_000, 50_000),
        ),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.boundary_end_reason) == (14_000, "strong_gap")
    assert not chunk.boundary_overlap_conflict


def test_case_7_uninterrupted_monologue_uses_exact_hard_maximum() -> None:
    result = _segment("case-7", 70_000, (Turn("A", 0, 70_000),))
    assert result.chunks[0].end_ms == 30_000
    assert result.chunks[0].boundary_end_reason == "hard_maximum"
    assert all(chunk.end_ms - chunk.start_ms <= 30_000 for chunk in result.chunks)


def test_case_8_extension_uses_first_available_natural_pause() -> None:
    result = _segment(
        "case-8",
        50_000,
        (Turn("A", 0, 20_000), Turn("A", 20_600, 50_000)),
    )
    assert result.chunks[0].end_ms == 20_300
    assert result.chunks[0].boundary_end_reason == "strong_gap"


def test_case_9_final_remainder_merges_or_emits_short_terminal_chunk() -> None:
    merged = _segment(
        "case-9-merge",
        33_000,
        (
            Turn("A", 0, 11_000),
            Turn("A", 13_000, 29_000),
            Turn("A", 31_000, 33_000),
        ),
    )
    assert [(chunk.start_ms, chunk.end_ms) for chunk in merged.chunks] == [
        (0, 12_000),
        (12_000, 33_000),
    ]
    assert merged.chunks[-1].boundary_end_reason == "episode_end"

    short = _segment(
        "case-9-short",
        44_000,
        (
            Turn("A", 0, 11_000),
            Turn("A", 13_000, 40_000),
            Turn("A", 42_000, 44_000),
        ),
    )
    assert [(chunk.start_ms, chunk.end_ms) for chunk in short.chunks][-1] == (41_000, 44_000)
    assert short.chunks[-1].boundary_end_reason == "episode_end"
    assert short.chunks[-1].end_ms - short.chunks[-1].start_ms < 4_000


def test_case_10_same_class_tie_uses_earlier_timestamp() -> None:
    result = _segment(
        "case-10",
        50_000,
        (
            Turn("A", 0, 10_800),
            Turn("A", 11_200, 12_800),
            Turn("A", 13_200, 50_000),
        ),
    )
    assert result.chunks[0].end_ms == 11_000
    assert result.chunks[0].boundary_end_reason == "short_gap"


def test_standard_diarization_provenance_and_configuration_are_persisted() -> None:
    result = _segment("provenance", 40_000, (Turn("A", 0, 40_000),))
    chunk = result.chunks[0]
    assert chunk.diarization_artifact_id == "diarization-synthetic-v1"
    assert chunk.segmentation_version == "community1-adaptive-v1"
    assert chunk.segmentation_configuration_hash == Community1AdaptiveConfig().configuration_sha256
    assert result.configuration["strong_gap_ms"] == 600
    assert result.configuration["overlap_margin_ms"] == 250
    assert result.boundary_diagnostics[0]["reason"] == "hard_maximum"


def test_missing_or_non_monotonic_standard_diarization_blocks_generation() -> None:
    with pytest.raises(ContractValidationError, match="missing standard_turns"):
        segment_chunks_with_metadata(
            "missing",
            SOURCE_SHA256,
            40_000,
            algorithm="community1-adaptive-v1",
            diarization={},
        )
    with pytest.raises(ContractValidationError, match="requires Community-1"):
        segment_chunks_with_metadata(
            "missing-id-only",
            SOURCE_SHA256,
            40_000,
            diarization_artifact_id="diarization-synthetic-v1",
        )
    with pytest.raises(ContractValidationError, match="monotonically"):
        _segment(
            "non-monotonic",
            40_000,
            (Turn("A", 10_000, 20_000), Turn("B", 1_000, 9_000)),
        )


def test_community_generation_is_byte_identical_on_rerun() -> None:
    turns = (Turn("A", 0, 11_700), Turn("B", 12_500, 40_000))
    first = _segment("rerun", 40_000, turns)
    second = _segment("rerun", 40_000, turns)
    assert first == second
    assert first.chunks[0].to_dict() == second.chunks[0].to_dict()

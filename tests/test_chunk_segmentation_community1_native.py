"""Deterministic synthetic acceptance cases for native Community-1 chunking."""

from __future__ import annotations

import hashlib
import json

from zanzara_archive.chunking import (
    Community1NativeAdaptiveConfig,
    segment_native_activity_with_metadata,
)
from zanzara_archive.contracts import NativeActivityArtifact, NativeActivityInterval, Turn

SOURCE_SHA256 = "a" * 64


def _activity(
    duration_ms: int,
    zero_intervals: tuple[tuple[int, int], ...] = (),
    overlap=(),
) -> NativeActivityArtifact:
    boundaries = {0, duration_ms}
    for start_ms, end_ms in (*zero_intervals, *overlap):
        boundaries.update((start_ms, end_ms))
    ordered = sorted(boundaries)
    intervals = []
    for start_ms, end_ms in zip(ordered, ordered[1:], strict=False):
        count = (
            0 if (start_ms, end_ms) in zero_intervals else 2 if (start_ms, end_ms) in overlap else 1
        )
        intervals.append(NativeActivityInterval(start_ms, end_ms, count))
    return NativeActivityArtifact(
        artifact_id="native-test-v1",
        source_sha256=SOURCE_SHA256,
        duration_ms=duration_ms,
        intervals=tuple(intervals),
        frame_count=max(1, duration_ms // 100),
        frame_start_ms=0,
        frame_step_ms=100,
        frame_duration_ms=100,
        capture_version="community1-speaker-count-snapshot-v1",
    )


def _segment(
    duration_ms: int,
    *,
    zero_intervals: tuple[tuple[int, int], ...] = (),
    overlap: tuple[tuple[int, int], ...] = (),
    exclusive: tuple[Turn, ...] | None = None,
    standard: tuple[Turn, ...] | None = None,
):
    activity = _activity(duration_ms, zero_intervals, overlap)
    exclusive = exclusive or (Turn("A", 0, duration_ms),)
    return segment_native_activity_with_metadata(
        "episode-native",
        SOURCE_SHA256,
        duration_ms,
        native_activity=activity,
        standard_turns=standard or (Turn("A", 0, duration_ms),),
        exclusive_turns=exclusive,
        community1_artifact_id="community1-test-v1",
    )


def test_case_1_continuous_speech_uses_hard_maximum_without_target_candidate() -> None:
    result = _segment(40_000)
    assert result.chunks[0].end_ms == 30_000
    assert result.chunks[0].boundary_end_reason == "hard_maximum"
    assert result.chunks[0].selected_candidate_type == "hard_maximum"
    assert result.chunks[0].selection_phase == "hard_maximum"
    assert result.boundary_diagnostics[0]["ideal_target_timestamp_ms"] == 12_000
    assert result.boundary_diagnostics[0]["reason"] == "hard_maximum"
    assert result.boundary_diagnostics[0]["exclusive_transition_count_18_30"] == 0
    assert result.boundary_diagnostics[0]["native_pause_count_18_30"] == 0
    assert result.boundary_diagnostics[0]["candidate_timestamps_types"] == []
    assert result.chunks[0].end_ms <= 30_000


def test_case_2_transition_at_11_4_wins() -> None:
    result = _segment(40_000, exclusive=(Turn("A", 0, 11_400), Turn("B", 11_400, 40_000)))
    assert result.chunks[0].end_ms == 11_400
    assert result.chunks[0].selected_candidate_type == "speaker_change"


def test_case_3_late_transition_is_selected_as_the_only_structural_candidate() -> None:
    result = _segment(40_000, exclusive=(Turn("A", 0, 15_500), Turn("B", 15_500, 40_000)))
    assert result.chunks[0].end_ms == 15_500
    assert result.chunks[0].selected_candidate_type == "speaker_change"


def test_case_3_transition_after_exclusive_silence_is_still_structural() -> None:
    result = _segment(
        40_000,
        exclusive=(Turn("A", 0, 10_000), Turn("B", 11_400, 40_000)),
    )
    assert result.chunks[0].end_ms == 11_400
    assert result.chunks[0].selected_candidate_type == "speaker_change"


def test_case_4_strong_native_pause_wins() -> None:
    result = _segment(40_000, zero_intervals=((11_600, 12_000),))
    assert result.chunks[0].end_ms == 11_800
    assert result.chunks[0].boundary_end_reason == "strong_pause"
    assert result.chunks[0].pause_duration_ms == 400


def test_case_5_short_native_pause_is_a_candidate() -> None:
    result = _segment(40_000, zero_intervals=((12_100, 12_300),))
    assert result.chunks[0].end_ms == 12_200
    assert result.chunks[0].boundary_end_reason == "short_pause"


def test_case_6_micro_gap_is_ignored() -> None:
    result = _segment(40_000, zero_intervals=((11_960, 12_040),))
    assert result.chunks[0].end_ms == 30_000
    assert result.chunks[0].boundary_end_reason == "hard_maximum"


def test_case_7_pause_and_transition_use_combined_adjustment() -> None:
    result = _segment(
        40_000,
        zero_intervals=((11_600, 12_000),),
        exclusive=(Turn("A", 0, 11_900), Turn("B", 11_900, 40_000)),
    )
    assert result.chunks[0].end_ms == 11_800
    assert result.chunks[0].selected_candidate_type == "strong_pause_and_speaker_change"


def test_case_8_clean_transition_is_selected_without_an_exact_target_candidate() -> None:
    result = _segment(
        40_000,
        overlap=((11_800, 13_000),),
        exclusive=(Turn("A", 0, 11_500), Turn("B", 11_500, 40_000)),
    )
    assert result.chunks[0].end_ms == 11_500


def test_case_9_late_pause_is_selected_when_it_is_the_only_structural_candidate() -> None:
    result = _segment(40_000, zero_intervals=((15_800, 16_200),), overlap=((11_900, 12_100),))
    assert result.chunks[0].end_ms == 16_000
    assert result.chunks[0].selected_candidate_type == "strong_pause"


def test_case_10_music_metadata_does_not_change_native_activity() -> None:
    result = _segment(40_000)
    assert result.chunks[0].selected_candidate_type == "hard_maximum"
    assert result.chunks[0].selection_phase == "hard_maximum"


def test_case_11_final_short_remainder_merges() -> None:
    result = _segment(15_000)
    assert [(chunk.start_ms, chunk.end_ms) for chunk in result.chunks] == [(0, 15_000)]
    assert result.chunks[0].boundary_end_reason == "episode_end"


def test_case_12_identical_inputs_have_identical_manifest_hash() -> None:
    first = _segment(40_000)
    second = _segment(40_000)
    first_bytes = json.dumps(
        [chunk.to_dict() for chunk in first.chunks], sort_keys=True, separators=(",", ":")
    ).encode()
    second_bytes = json.dumps(
        [chunk.to_dict() for chunk in second.chunks], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()
    assert first == second
    assert all(
        chunk.end_ms - chunk.start_ms <= Community1NativeAdaptiveConfig().hard_max_ms
        for chunk in first.chunks
    )


def test_relaxation_clean_candidate_at_18_2_seconds_wins() -> None:
    result = _segment(
        40_000,
        exclusive=(Turn("A", 0, 18_200), Turn("B", 18_200, 40_000)),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (18_200, "relaxed_clean")


def test_relaxation_clean_skips_overlap_conflicted_18_2_for_19_seconds() -> None:
    result = _segment(
        40_000,
        overlap=((18_100, 18_400),),
        exclusive=(
            Turn("A", 0, 18_200),
            Turn("B", 18_200, 19_000),
            Turn("C", 19_000, 40_000),
        ),
        standard=(Turn("A", 0, 40_000), Turn("B", 18_100, 18_400)),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (19_000, "relaxed_clean")
    assert chunk.overlap_adjustment == 0.0


def test_relaxation_clean_uses_native_overlap_penalty() -> None:
    result = _segment(
        40_000,
        overlap=((18_100, 18_400),),
        exclusive=(Turn("A", 0, 18_200), Turn("B", 18_200, 40_000)),
        standard=(Turn("A", 0, 40_000),),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (18_200, "relaxed_any")
    assert chunk.overlap_adjustment == 1.5


def test_relaxation_clean_uses_23_seconds_over_earlier_overlap() -> None:
    result = _segment(
        40_000,
        overlap=((20_900, 21_100),),
        exclusive=(
            Turn("A", 0, 21_000),
            Turn("B", 21_000, 23_000),
            Turn("C", 23_000, 40_000),
        ),
        standard=(Turn("A", 0, 40_000), Turn("B", 20_900, 21_100)),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (23_000, "relaxed_clean")


def test_relaxation_any_accepts_overlap_conflicted_24_4_seconds() -> None:
    result = _segment(
        40_000,
        overlap=((24_300, 24_500),),
        exclusive=(Turn("A", 0, 24_400), Turn("B", 24_400, 40_000)),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (24_400, "relaxed_any")
    assert chunk.overlap_adjustment == 1.5


def test_relaxation_any_selects_natural_27_second_candidate() -> None:
    result = _segment(
        40_000,
        exclusive=(Turn("A", 0, 27_000), Turn("B", 27_000, 40_000)),
    )
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (27_000, "relaxed_any")


def test_relaxation_hard_maximum_is_exact_30_seconds_without_natural_candidate() -> None:
    result = _segment(40_000)
    chunk = result.chunks[0]
    assert (chunk.end_ms, chunk.selection_phase) == (30_000, "hard_maximum")
    assert chunk.selected_candidate_type == "hard_maximum"


def test_relaxed_clean_does_not_invent_native_overlap_from_standard_turns() -> None:
    result = _segment(
        40_000,
        exclusive=(
            Turn("A", 0, 18_100),
            Turn("A", 18_100, 18_200),
            Turn("B", 18_200, 40_000),
        ),
        standard=(Turn("A", 0, 40_000), Turn("B", 18_100, 18_200)),
    )
    assert (result.chunks[0].end_ms, result.chunks[0].selection_phase) == (18_200, "relaxed_clean")


def test_relaxed_any_includes_exact_24_and_30_and_earlier_timestamp_wins() -> None:
    result = _segment(
        40_000,
        exclusive=(
            Turn("A", 0, 24_000),
            Turn("B", 24_000, 30_000),
            Turn("C", 30_000, 40_000),
        ),
    )
    assert (result.chunks[0].end_ms, result.chunks[0].selection_phase) == (24_000, "relaxed_clean")
    result = _segment(
        40_000,
        exclusive=(Turn("A", 0, 30_000), Turn("B", 30_000, 40_000)),
    )
    assert (result.chunks[0].end_ms, result.chunks[0].selection_phase) == (30_000, "relaxed_any")
    assert result.chunks[0].boundary_end_reason == "speaker_change"
    # Same-time candidates retain type priority, but never outrank an earlier time.
    result = _segment(
        40_000,
        zero_intervals=((26_800, 27_200),),
        exclusive=(Turn("A", 0, 25_000), Turn("B", 25_000, 27_000), Turn("C", 27_000, 40_000)),
    )
    assert result.chunks[0].end_ms == 25_000

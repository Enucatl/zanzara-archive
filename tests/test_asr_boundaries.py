"""CPU fixtures for deterministic ASR window ownership."""

from zanzara_archive.asr import (
    assign_window_words,
    build_asr_windows,
    merge_window_results,
    window_audio_artifact,
)
from zanzara_archive.contracts import AudioArtifact, ModelFingerprint, TimedWord, TranscriptResult

MODEL = ModelFingerprint(
    name="parakeet",
    repository="nvidia/parakeet-tdt-0.6b-v3",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    runtime={"torch": "test"},
)
AUDIO = AudioArtifact(
    artifact_id="episode",
    source_sha256="c" * 64,
    format="opus",
    duration_ms=250,
    sample_rate_hz=48_000,
    channels=1,
)


def result(audio: AudioArtifact, words: tuple[TimedWord, ...]) -> TranscriptResult:
    return TranscriptResult(
        artifact_id=audio.artifact_id,
        source_sha256=audio.source_sha256,
        model=MODEL,
        text=" ".join(word.text for word in words),
        words=words,
        duration_ms=audio.duration_ms,
    )


def test_windows_clip_context_at_episode_boundaries() -> None:
    windows = build_asr_windows(250, window_ms=100, context_ms=10)
    assert [window.to_dict() for window in windows] == [
        {
            "index": 0,
            "ownership_start_ms": 0,
            "ownership_end_ms": 100,
            "decode_start_ms": 0,
            "decode_end_ms": 110,
        },
        {
            "index": 1,
            "ownership_start_ms": 100,
            "ownership_end_ms": 200,
            "decode_start_ms": 90,
            "decode_end_ms": 210,
        },
        {
            "index": 2,
            "ownership_start_ms": 200,
            "ownership_end_ms": 250,
            "decode_start_ms": 190,
            "decode_end_ms": 250,
        },
    ]


def test_midpoint_boundary_belongs_to_later_window() -> None:
    windows = build_asr_windows(250, window_ms=100, context_ms=10)
    first_audio = window_audio_artifact(AUDIO, windows[0])
    second_audio = window_audio_artifact(AUDIO, windows[1])
    boundary_from_first = TimedWord("first-boundary", "boundary", 95, 105)
    boundary_from_second = TimedWord("second-boundary", "boundary", 5, 15)
    assert assign_window_words(result(first_audio, (boundary_from_first,)), windows[0]) == ()
    assigned = assign_window_words(result(second_audio, (boundary_from_second,)), windows[1])
    assert [(word.text, word.start_ms, word.end_ms) for word in assigned] == [("boundary", 95, 105)]


def test_merge_keeps_adjacent_repeated_words_and_does_not_drop_omitted_fixture_words() -> None:
    windows = build_asr_windows(250, window_ms=100, context_ms=10)
    window_results = (
        result(
            window_audio_artifact(AUDIO, windows[0]),
            (TimedWord("first", "again", 45, 55),),
        ),
        result(
            window_audio_artifact(AUDIO, windows[1]),
            (
                TimedWord("boundary", "kept", 5, 15),
                TimedWord("omitted-in-first", "again", 45, 55),
            ),
        ),
        result(window_audio_artifact(AUDIO, windows[2]), ()),
    )
    merged = merge_window_results(AUDIO, windows, window_results, request_id="asr-test")
    assert [word.text for word in merged.words] == ["again", "kept", "again"]
    assert [word.start_ms for word in merged.words] == [45, 95, 135]

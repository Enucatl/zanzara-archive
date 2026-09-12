"""Synthetic CPU fixtures for deterministic word attribution."""

from __future__ import annotations

import pytest

from zanzara_archive.contracts import (
    ContractValidationError,
    DiarizationResult,
    ModelFingerprint,
    Overlap,
    TimedWord,
    TranscriptResult,
    Turn,
)
from zanzara_archive.transcripts import attribute_words, build_attributed_transcript

SOURCE = "a" * 64
ASR_MODEL = ModelFingerprint(
    name="asr",
    repository="test/asr",
    revision="asr-revision",
    checkpoint_sha256=("b" * 64,),
)
DIA_MODEL = ModelFingerprint(
    name="diarization",
    repository="test/diarization",
    revision="diarization-revision",
    checkpoint_sha256=("c" * 64,),
)


def test_greatest_intersection_midpoint_then_stable_id_and_unassigned_words() -> None:
    words = (
        TimedWord("greatest", "greatest", 0, 200),
        TimedWord("midpoint", "midpoint", 150, 350),
        TimedWord("stable", "stable", 100, 200),
        TimedWord("none", "none", 400, 450),
        TimedWord("long", "long", 500, 700),
    )
    exclusive = (
        Turn("B", 0, 100),
        Turn("A", 100, 250),
        Turn("C", 250, 400),
        Turn("D", 50, 150),
        Turn("A", 500, 600),
        Turn("B", 600, 700),
    )
    standard = (
        Turn("HOST", 0, 200),
        Turn("HOST", 500, 650),
        Turn("GUEST", 600, 650),
    )

    attributed = attribute_words(words, exclusive, standard)

    assert [word.speaker_id for word in attributed] == ["A", "C", "A", None, "B"]
    assert [word.overlap for word in attributed] == [False, False, False, False, True]
    assert [(word.word_id, word.start_ms, word.end_ms) for word in attributed] == [
        (word.word_id, word.start_ms, word.end_ms) for word in words
    ]


def test_equal_intersection_without_midpoint_uses_stable_speaker_id() -> None:
    word = TimedWord("tie", "tie", 100, 200)
    turns = (Turn("Z", 90, 140), Turn("A", 160, 210))

    assert attribute_words((word,), turns, ())[0].speaker_id == "A"


def test_supplied_overlap_must_match_standard_turns() -> None:
    with pytest.raises(ContractValidationError, match="overlap intervals"):
        attribute_words(
            (TimedWord("word", "ciao", 100, 200),),
            (Turn("A", 0, 300),),
            (Turn("A", 0, 300), Turn("B", 150, 250)),
            overlaps=(Overlap(("A", "B"), 100, 150),),
        )


def test_build_payload_rejects_cross_source_and_keeps_both_diarization_views() -> None:
    transcript = TranscriptResult(
        artifact_id="asr-artifact",
        source_sha256=SOURCE,
        model=ASR_MODEL,
        text="ciao",
        words=(TimedWord("word", "ciao", 10, 20),),
        duration_ms=1_000,
    )
    diarization = DiarizationResult(
        artifact_id="diarization-artifact",
        source_sha256=SOURCE,
        duration_ms=1_000,
        model=DIA_MODEL,
        standard_turns=(Turn("A", 0, 100),),
        exclusive_turns=(Turn("A", 0, 100),),
        overlaps=(),
    )

    result = build_attributed_transcript(transcript, diarization, episode_id="episode.opus")
    payload = result.to_dict()

    assert payload["episode_id"] == "episode.opus"
    assert payload["words"][0]["speaker_id"] == "A"
    assert payload["diarization"]["standard_turns"][0]["speaker_id"] == "A"
    assert payload["diarization"]["exclusive_turns"][0]["speaker_id"] == "A"
    assert payload["words"][0]["start_ms"] == 10

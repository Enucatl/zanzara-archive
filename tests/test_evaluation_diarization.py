"""Known-answer CPU fixtures for overlap-inclusive DER, JER and SA-WER."""

from __future__ import annotations

from zanzara_archive.evaluation import evaluate_documents


def _payload(
    *,
    words: list[dict] | None = None,
    turns: list[dict] | None = None,
    overlaps: list[dict] | None = None,
    duration_ms: int = 2_000,
) -> dict:
    turns = turns or []
    words = words or []
    return {
        "schema_version": 1,
        "episode_id": "fixture.opus",
        "source_sha256": "b" * 64,
        "duration_ms": duration_ms,
        "status": "draft",
        "actor_type": "machine",
        "reviewer": "machine",
        "words": words,
        "standard_turns": turns,
        "exclusive_turns": turns,
        "overlap_intervals": overlaps or [],
        "unintelligible_spans": [],
        "reviewed_word_ids": [],
        "manual_timing_word_ids": [word["word_id"] for word in words],
        "provenance": {"configuration": {"fixture": True}},
    }


def _turn(speaker_id: str, start_ms: int, end_ms: int) -> dict:
    return {
        "turn_id": f"turn-{speaker_id}-{start_ms}",
        "speaker_id": speaker_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def _word(word_id: str, text: str, speaker_id: str, start_ms: int, end_ms: int) -> dict:
    return {
        "word_id": word_id,
        "text": text,
        "speaker_id": speaker_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def test_primary_der_counts_overlap_missed_speaker_time() -> None:
    reference_turns = [_turn("A", 0, 1_000), _turn("B", 0, 1_000)]
    hypothesis_turns = [_turn("A", 0, 1_000)]
    reference = _payload(
        turns=reference_turns,
        overlaps=[{"speaker_ids": ["A", "B"], "start_ms": 0, "end_ms": 1_000}],
    )
    hypothesis = _payload(turns=hypothesis_turns)
    metrics = evaluate_documents(reference, hypothesis)["slices"]["all"]
    assert metrics["primary_der"]["components_seconds"] == {
        "missed_speech": 1.0,
        "false_alarm": 0.0,
        "speaker_confusion": 0.0,
    }
    assert metrics["primary_der"]["value"] == 0.5
    assert metrics["primary_der"]["overlap_included"] is True
    assert metrics["jer"]["value"] == 0.5


def test_speaker_confusion_and_count_error_are_explicit() -> None:
    reference_turns = [_turn("A", 0, 1_000), _turn("B", 1_000, 2_000)]
    hypothesis_turns = [_turn("X", 0, 2_000)]
    reference = _payload(turns=reference_turns, duration_ms=2_000)
    hypothesis = _payload(turns=hypothesis_turns, duration_ms=2_000)
    metrics = evaluate_documents(reference, hypothesis)["slices"]["all"]
    assert metrics["primary_der"]["components_seconds"]["speaker_confusion"] == 1.0
    assert metrics["primary_der"]["value"] == 0.5
    assert metrics["count_error"]["signed"] == -1
    assert metrics["count_error"]["absolute"] == 1


def test_sa_wer_counts_collapsed_speaker_and_unassigned_words() -> None:
    words = [
        _word("ref-a", "ciao", "A", 100, 200),
        _word("ref-b", "mario", "B", 1_100, 1_200),
    ]
    hypothesis_words = [
        _word("hyp-a", "ciao", "X", 100, 200),
        _word("hyp-b", "mario", "X", 1_100, 1_200),
    ]
    reference = _payload(
        words=words,
        turns=[_turn("A", 0, 1_000), _turn("B", 1_000, 2_000)],
    )
    hypothesis = _payload(words=hypothesis_words, turns=[_turn("X", 0, 2_000)])
    metrics = evaluate_documents(reference, hypothesis)["slices"]["all"]
    assert metrics["sa_wer"]["value"] == 1.0
    assert metrics["sa_wer"]["denominator"] == 2
    assert metrics["unassigned_rate"]["value"] == 0.0
    hypothesis_words[-1]["speaker_id"] = None
    hypothesis_unassigned = _payload(words=hypothesis_words, turns=[_turn("X", 0, 2_000)])
    metrics = evaluate_documents(reference, hypothesis_unassigned)["slices"]["all"]
    assert metrics["unassigned_rate"]["value"] == 0.5

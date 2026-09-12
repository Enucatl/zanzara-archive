"""Known-answer CPU fixtures for ASR metrics and report evidence."""

from __future__ import annotations

import json
from pathlib import Path

from zanzara_archive.evaluation import evaluate_documents, normalize_italian


def _payload(
    texts: list[str],
    *,
    source: str = "a" * 64,
    duration_ms: int = 5_000,
    mask_index: int | None = None,
    speaker: str | None = "SPEAKER_A",
) -> dict:
    words = [
        {
            "word_id": f"word-{index}",
            "text": text,
            "start_ms": 500 + index * 500,
            "end_ms": 800 + index * 500,
            "speaker_id": speaker,
            "unintelligible": index == mask_index,
        }
        for index, text in enumerate(texts)
    ]
    turns = (
        [{"turn_id": "turn-a", "speaker_id": speaker, "start_ms": 0, "end_ms": duration_ms}]
        if speaker
        else []
    )
    return {
        "schema_version": 1,
        "episode_id": "fixture.opus",
        "source_sha256": source,
        "duration_ms": duration_ms,
        "status": "draft",
        "actor_type": "machine",
        "reviewer": "machine",
        "words": words,
        "standard_turns": turns,
        "exclusive_turns": turns,
        "overlap_intervals": [],
        "unintelligible_spans": (
            [{"start_ms": 1_000, "end_ms": 1_300}] if mask_index is not None else []
        ),
        "reviewed_word_ids": [],
        "manual_timing_word_ids": [word["word_id"] for word in words],
        "provenance": {"configuration": {"fixture": True}},
    }


def _all(report: dict) -> dict:
    return report["slices"]["all"]


def test_known_answer_edit_counts_cover_substitution_deletion_insertion() -> None:
    fixture = json.loads(Path("tests/fixtures/evaluation_known_answers.json").read_text())
    for case in fixture["cases"][:3]:
        reference = _payload(case["reference"])
        hypothesis = _payload(case["hypothesis"])
        metrics = _all(evaluate_documents(reference, hypothesis))
        counts = metrics["raw_wer"]
        assert {key: counts[key] for key in ("substitutions", "deletions", "insertions")} == case[
            "expected"
        ]


def test_known_answer_raw_normalized_wer_and_cer_use_it_v1() -> None:
    fixture = json.loads(Path("tests/fixtures/evaluation_known_answers.json").read_text())
    case = fixture["cases"][3]
    metrics = _all(evaluate_documents(_payload(case["reference"]), _payload(case["hypothesis"])))
    assert normalize_italian("Città,") == "città"
    assert metrics["raw_wer"]["value"] == case["expected"]["raw_wer"]
    assert metrics["normalized_wer"]["value"] == case["expected"]["normalized_wer"]
    assert metrics["normalized_cer"]["value"] == case["expected"]["normalized_cer"]


def test_unintelligible_mask_is_scored_locally_without_mutating_ms_inputs() -> None:
    fixture = json.loads(Path("tests/fixtures/evaluation_known_answers.json").read_text())
    case = fixture["cases"][4]
    reference = _payload(case["reference"], mask_index=case["mask_index"])
    hypothesis = _payload(case["hypothesis"], mask_index=case["mask_index"])
    original_intervals = [(word["start_ms"], word["end_ms"]) for word in reference["words"]]
    metrics = _all(evaluate_documents(reference, hypothesis))
    assert metrics["raw_wer"]["value"] == 0.0
    assert metrics["normalized_wer"]["value"] == 0.0
    assert [(word["start_ms"], word["end_ms"]) for word in reference["words"]] == original_intervals
    assert metrics["slice"]["mask_count"] > 0


def test_zero_reference_denominator_is_insufficient_evidence() -> None:
    reference = _payload([])
    hypothesis = _payload(["uno"])
    metrics = _all(evaluate_documents(reference, hypothesis))
    assert metrics["raw_wer"]["status"] == "insufficient_evidence"
    assert metrics["raw_wer"]["value"] is None
    assert metrics["normalized_wer"]["status"] == "insufficient_evidence"

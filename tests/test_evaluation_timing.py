"""Known-answer CPU fixtures for millisecond-preserving timing metrics."""

from __future__ import annotations

import json

import pytest

from zanzara_archive.cli import main
from zanzara_archive.evaluation import (
    EvaluationValidationError,
    evaluate_documents,
    write_evaluation_artifacts,
)


def _payload(words: list[dict], *, span: bool = False) -> dict:
    turns = [{"turn_id": "turn-a", "speaker_id": "A", "start_ms": 0, "end_ms": 1_000}]
    return {
        "schema_version": 1,
        "episode_id": "timing-fixture.opus",
        "source_sha256": "c" * 64,
        "duration_ms": 1_000,
        "status": "draft",
        "actor_type": "machine",
        "reviewer": "machine",
        "words": words,
        "standard_turns": turns,
        "exclusive_turns": turns,
        "overlap_intervals": [],
        "unintelligible_spans": [{"start_ms": 400, "end_ms": 500}] if span else [],
        "reviewed_word_ids": [],
        "manual_timing_word_ids": [word["word_id"] for word in words],
        "provenance": {"configuration": {"fixture": True}},
    }


def _word(word_id: str, text: str, start_ms: int, end_ms: int) -> dict:
    return {
        "word_id": word_id,
        "text": text,
        "speaker_id": "A",
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def test_timing_quantiles_and_coverage_keep_deleted_reference_words() -> None:
    reference = _payload([_word("r1", "uno", 100, 200), _word("r2", "due", 600, 700)])
    hypothesis = _payload([_word("h1", "uno", 110, 220)])
    timing = evaluate_documents(reference, hypothesis)["slices"]["all"]["timing"]
    assert timing["value"]["start_median_s"] == 0.01
    assert timing["value"]["end_median_s"] == 0.02
    assert timing["value"]["combined_median_s"] == 0.015
    assert timing["matched_reference_coverage"] == 0.5
    assert timing["denominator"] == 2


def test_mask_and_split_ranges_are_millisecond_inputs_and_not_persisted_as_seconds(
    tmp_path,
) -> None:
    reference = _payload([_word("r1", "uno", 100, 200), _word("r2", "mumble", 400, 500)], span=True)
    hypothesis = _payload([_word("h1", "uno", 100, 200), _word("h2", "wrong", 400, 500)], span=True)
    report = evaluate_documents(reference, hypothesis, reviewed_commit="fixture-commit")
    assert report["canonical_time_unit"] == "milliseconds"
    assert report["metric_time_unit"] == "seconds"
    assert report["slices"]["all"]["slice"]["ranges_ms"] == [(0, 1_000)]
    assert report["private_errors"]["all"]["masks_ms"] == [(400, 500)]
    assert report["slices"]["all"]["timing"]["value"]["combined_median_s"] == 0.0
    artifacts = write_evaluation_artifacts(report, tmp_path / "run")
    assert set(artifacts["artifact_sha256"]) == {
        "metrics.json",
        "coverage.json",
        "errors.json",
        "results.json",
        "report.html",
        "run.json",
    }
    saved_metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved_metrics["canonical_time_unit"] == "milliseconds"
    assert "uno" not in (tmp_path / "run" / "report.html").read_text()


def test_score_cli_reproduces_private_run_and_truthfully_blocks_synthetic_reference(
    tmp_path,
) -> None:
    reference = _payload([_word("r1", "uno", 100, 200)])
    hypothesis = _payload([_word("h1", "uno", 100, 200)])
    reference_path = tmp_path / "reference.json"
    hypothesis_path = tmp_path / "hypothesis.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    hypothesis_path.write_text(json.dumps(hypothesis), encoding="utf-8")
    output_dir = tmp_path / "private-run"
    exit_code = main(
        [
            "evaluation",
            "score",
            "--reference",
            str(reference_path),
            "--hypothesis",
            str(hypothesis_path),
            "--output",
            str(output_dir),
            "--reviewed-commit",
            "fixture-commit",
        ]
    )
    assert exit_code == 1
    run = json.loads((output_dir / "run.json").read_text())
    assert run["verdict"] == "insufficient_evidence"
    assert run["commands_require_no_model_or_paid_call"] is True


def test_nonzero_time_origin_is_rejected_before_normalization() -> None:
    reference = _payload([_word("r1", "uno", 100, 200)])
    hypothesis = _payload([_word("h1", "uno", 100, 200)])
    reference["time_origin_ms"] = 1
    with pytest.raises(EvaluationValidationError, match="time_origin_ms"):
        evaluate_documents(reference, hypothesis)

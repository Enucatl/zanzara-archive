"""Known-answer, word-blind fixtures for P1R diarization scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.evaluation import (
    DIARIZATION_EVALUATOR_VERSION,
    DiarizationScoringError,
    score_diarization,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "p1r_diarization_known_answers.json"


def _fixtures() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda item: item["id"])
def test_analytic_diarization_values_match_fixture(fixture: dict) -> None:
    report = score_diarization(fixture["reference"], fixture["hypothesis"])
    metrics = report["slices"]["all"]
    expected = fixture["expected"]

    assert report["status"] == "scored"
    assert report["evaluator_version"] == DIARIZATION_EVALUATOR_VERSION
    assert metrics["primary_der"]["value"] == expected.get("der", metrics["primary_der"]["value"])
    assert metrics["jer"]["value"] == expected.get("jer", metrics["jer"]["value"])
    if "speaker_count_signed" in expected:
        assert metrics["speaker_count_error"]["signed"] == expected["speaker_count_signed"]
    overlap = metrics["overlap"]
    for name in ("precision", "recall", "f1"):
        assert overlap[name] == expected.get(f"overlap_{name}", overlap[name])


def test_condition_slices_are_aggregated_without_lexical_inputs() -> None:
    reference = {
        "episode_id": "fixture-conditions.opus",
        "duration_ms": 2000,
        "turns": [
            {"speaker_id": "A", "start_ms": 0, "end_ms": 1000},
            {"speaker_id": "B", "start_ms": 1000, "end_ms": 2000},
        ],
        "overlap_intervals": [],
        "words": "this malformed lexical field is ignored",
    }
    hypothesis = {
        "episode_id": "fixture-conditions.opus",
        "duration_ms": 2000,
        "turns": [{"speaker_id": "X", "start_ms": 0, "end_ms": 2000}],
        "overlap_intervals": [],
        "words": [{"not": "a timed word"}],
    }
    report = score_diarization(
        reference,
        hypothesis,
        condition_slices=[
            {"slice_id": "first", "condition": "single_speaker", "start_ms": 0, "end_ms": 1000},
            {
                "slice_id": "second",
                "condition": "multi_speaker_no_overlap",
                "start_ms": 1000,
                "end_ms": 2000,
            },
        ],
    )

    assert report["slices"]["all"]["primary_der"]["value"] == 0.5
    assert report["slices"]["first"]["primary_der"]["value"] == 0.0
    assert report["slices"]["second"]["primary_der"]["value"] == 0.0
    assert report["configuration"]["word_inputs"] == "ignored; no lexical fields are read"


def test_missing_reference_intervals_are_unscorable_not_inferred_from_words() -> None:
    report = score_diarization(
        {
            "episode_id": "fixture-missing.opus",
            "duration_ms": 1000,
            "words": [{"speaker_id": "A", "start_ms": 0, "end_ms": 1000}],
        },
        {
            "episode_id": "fixture-missing.opus",
            "duration_ms": 1000,
            "turns": [{"speaker_id": "X", "start_ms": 0, "end_ms": 1000}],
        },
    )

    assert report["status"] == "unscorable"
    assert report["slices"]["all"]["primary_der"]["status"] == "unscorable"


def test_interval_geometry_is_validated() -> None:
    with pytest.raises(DiarizationScoringError, match="bounded integer"):
        score_diarization(
            {
                "duration_ms": 1000,
                "turns": [{"speaker_id": "A", "start_ms": 0, "end_ms": 1001}],
            },
            {"duration_ms": 1000, "turns": []},
        )

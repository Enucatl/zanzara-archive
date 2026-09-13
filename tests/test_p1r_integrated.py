"""Known-answer fixtures for integrated speaker-attributed P1R scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.contracts import Overlap, Turn
from zanzara_archive.evaluation import (
    CPWER_VERSION,
    IntegratedScoringError,
    IntegratedScoringInput,
    SpeakerTextStream,
    score_cpwer,
    score_integrated,
    score_integrated_chunks,
    write_integrated_score_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "p1r_integrated_known_answers.json"


def _fixtures() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda item: item["id"])
def test_cpwer_known_answers_cover_attribution_cases(fixture: dict) -> None:
    metric = score_cpwer(fixture["reference_streams"], fixture["hypothesis_streams"])
    expected = fixture["expected"]

    assert metric["algorithm"] == CPWER_VERSION
    for field in ("value", "errors", "substitutions", "deletions", "insertions"):
        assert metric[field] == expected[field]
    if "mapping" in expected:
        assert metric["mapping"] == expected["mapping"]


def test_cpwer_preserves_overlap_channels_and_interval_provenance() -> None:
    reference_streams = (SpeakerTextStream("A", "uno due"), SpeakerTextStream("B", "tre quattro"))
    hypothesis_streams = (SpeakerTextStream("X", "uno due"), SpeakerTextStream("Y", "tre"))
    item = IntegratedScoringInput(
        reference_streams=reference_streams,
        hypothesis_streams=hypothesis_streams,
        chunk_id="chunk-1",
        source_sha256="a" * 64,
        model_fingerprint_sha256="b" * 64,
        duration_ms=2_000,
        scored_ranges_ms=((0, 2_000),),
        reference_turns=(Turn("A", 0, 2_000), Turn("B", 0, 2_000)),
        hypothesis_turns=(Turn("X", 0, 2_000), Turn("Y", 0, 2_000)),
        reference_overlaps=(Overlap(("A", "B"), 0, 2_000),),
        hypothesis_overlaps=(Overlap(("X", "Y"), 0, 1_000),),
        slice_ids=("overlap",),
    )
    report = score_integrated(item, reviewed_commit="fixture-commit")

    assert report["report_type"] == "p1r-integrated-speaker-attributed"
    assert report["metrics"]["cpwer"]["reference_denominator"] == 4
    assert report["configuration"]["overlap_treatment"].startswith("retain every explicit")
    assert report["configuration"]["interval_constraints"]["scored_ranges_ms"] == [[0, 2_000]]
    assert report["metrics"]["tcpwer"]["status"] == "deferred"
    assert report["private_details"]["mapping"]["assignments"]


def test_missing_reference_streams_are_unscorable_not_speaker_independent_wer() -> None:
    metric = score_cpwer([], [SpeakerTextStream("X", "uno")])

    assert metric["status"] == "unscorable"
    assert metric["value"] is None
    assert "no speaker-independent WER fallback" in metric["reason"]


def test_non_human_reference_is_insufficient_evidence() -> None:
    metric = score_cpwer(
        [{"speaker_id": "A", "text": "uno"}],
        [{"speaker_id": "X", "text": "uno"}],
        reference_status="draft",
    )

    assert metric["status"] == "insufficient_evidence"
    assert metric["value"] is None


def test_interval_and_stream_relationships_are_validated() -> None:
    with pytest.raises(IntegratedScoringError, match="absent from reference text streams"):
        IntegratedScoringInput(
            reference_streams=(SpeakerTextStream("A", "uno"),),
            hypothesis_streams=(),
            reference_turns=(Turn("B", 0, 1_000),),
            duration_ms=1_000,
        )
    with pytest.raises(IntegratedScoringError, match="sorted and non-overlapping"):
        IntegratedScoringInput(
            reference_streams=(SpeakerTextStream("A", "uno"),),
            hypothesis_streams=(),
            scored_ranges_ms=((500, 1_000), (0, 500)),
        )


def test_aggregate_and_report_writer_keep_integrated_output_separate(tmp_path: Path) -> None:
    item = IntegratedScoringInput(
        reference_streams=(SpeakerTextStream("A", "uno"),),
        hypothesis_streams=(SpeakerTextStream("X", "uno"),),
        chunk_id="chunk-1",
        model_fingerprint_sha256="b" * 64,
        duration_ms=1_000,
    )
    report = score_integrated_chunks((item,), synthetic_provenance=("synthetic fixture",))
    artifacts = write_integrated_score_report(report, tmp_path / "integrated")

    metrics_text = (tmp_path / "integrated" / "metrics.json").read_text()
    details = json.loads((tmp_path / "integrated" / "details.json").read_text())
    assert report["report_type"] == "p1r-integrated-speaker-attributed"
    assert "cpwer" in metrics_text
    assert "private_details" not in metrics_text
    assert details["chunks"][0]["mapping"]["assignments"]
    assert artifacts["artifact_sha256"]["run.json"]
    with pytest.raises(IntegratedScoringError, match="overwrite"):
        write_integrated_score_report(report, tmp_path / "integrated")

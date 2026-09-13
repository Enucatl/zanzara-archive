"""Known-answer fixtures for the independent P1R chunk ASR scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.contracts import (
    AudioChunk,
    ChunkBenchmarkManifest,
    ChunkCondition,
    ModelFingerprint,
    Overlap,
    SpeakerStream,
    TranscriptionHypothesis,
    TranscriptReference,
    Turn,
)
from zanzara_archive.p1r_asr import (
    ASRScoringValidationError,
    ChunkScoringInput,
    ScoringTextStream,
    score_chunks,
    score_manifest,
    score_text_pair,
    write_asr_score_report,
)

SOURCE_SHA256 = "a" * 64
MODEL_SHA256 = "b" * 64
SEGMENTATION_SHA256 = "c" * 64


def test_known_answer_substitution_deletion_and_insertion_counts() -> None:
    fixture = json.loads(Path("tests/fixtures/p1r_asr_known_answers.json").read_text())
    for case in fixture["cases"]:
        metrics = score_text_pair(case["reference"], case["hypothesis"])["normalized_wer"]
        assert {key: metrics[key] for key in ("substitutions", "deletions", "insertions")} == case[
            "expected"
        ]


def test_normalized_italian_wer_and_cer_ignore_case_and_punctuation() -> None:
    metrics = score_text_pair("CITTÀ, però", "città però")

    assert metrics["raw_wer"]["value"] == 0.5
    assert metrics["normalized_wer"]["value"] == 0.0
    assert metrics["normalized_cer"]["value"] == 0.0


def test_orc_equivalent_ignores_speaker_permutation_and_retains_overlap_streams() -> None:
    reference_streams = (
        ScoringTextStream("REFERENCE_A", "uno due"),
        ScoringTextStream("REFERENCE_B", "tre quattro"),
    )
    hypothesis_streams = (
        ScoringTextStream("HYPOTHESIS_B", "tre quattro"),
        ScoringTextStream("HYPOTHESIS_A", "uno due"),
    )

    metric = score_text_pair(
        "uno due tre quattro",
        "tre quattro uno due",
        reference_streams=reference_streams,
        hypothesis_streams=hypothesis_streams,
    )["orc_wer"]

    assert metric["status"] == "scored"
    assert metric["value"] == 0.0
    assert metric["speaker_labels_used"] is False
    assert metric["reference_stream_count"] == 2

    text_only_hypothesis = score_text_pair(
        "uno due tre quattro",
        "tre quattro uno due",
        reference_streams=reference_streams,
    )["orc_wer"]
    assert text_only_hypothesis["value"] == 0.0


def test_unintelligible_mask_requires_explicit_pair_and_is_removed_before_scoring() -> None:
    metrics = score_text_pair(
        "uno rumore tre",
        "uno parola tre",
        reference_masked_token_indices=(1,),
        hypothesis_masked_token_indices=(1,),
    )
    assert metrics["normalized_wer"]["value"] == 0.0
    assert metrics["normalized_cer"]["value"] == 0.0

    unscorable = score_text_pair(
        "uno rumore tre",
        "uno parola tre",
        reference_masked_token_indices=(1,),
    )
    assert unscorable["normalized_wer"]["status"] == "unscorable"
    assert "one side" in unscorable["normalized_wer"]["reason"]


def _model() -> ModelFingerprint:
    return ModelFingerprint(
        name="synthetic-p1r-asr",
        repository="local/p1r-asr",
        revision="fixture-v1",
        checkpoint_sha256=(MODEL_SHA256,),
        dimensions=None,
        preprocessing={"sample_rate_hz": 16000},
        precision="float32",
        runtime={"python": "3.14"},
        terms_evidence="synthetic fixture",
    )


def _input() -> ChunkScoringInput:
    condition = ChunkCondition(
        speaker_streams=(
            SpeakerStream("A", (Turn("A", 100, 1_500),)),
            SpeakerStream("B", (Turn("B", 1_000, 2_000),)),
        ),
        overlaps=(Overlap(("A", "B"), 1_000, 1_500),),
        acoustic_labels=("music",),
        music=True,
        degraded=True,
    )
    chunk = AudioChunk.create(
        episode_id="fixture-episode",
        source_sha256=SOURCE_SHA256,
        start_ms=0,
        end_ms=3_000,
        duration_ms=10_000,
        segmentation_fingerprint=SEGMENTATION_SHA256,
        condition=condition,
        partition="held_out",
    )
    reference = TranscriptReference(
        reference_id="reference-1",
        chunk_id=chunk.chunk_id,
        source_sha256=SOURCE_SHA256,
        text="uno due tre",
        review_status="human_truth",
        reviewer="reviewer-1",
    )
    hypothesis = TranscriptionHypothesis(
        chunk_id=chunk.chunk_id,
        model_fingerprint=_model(),
        text="uno due tre",
        source_sha256=SOURCE_SHA256,
    )
    return ChunkScoringInput(
        chunk=chunk,
        reference=reference,
        hypothesis=hypothesis,
        reference_streams=(ScoringTextStream("A", "uno"), ScoringTextStream("B", "due tre")),
        hypothesis_streams=(ScoringTextStream("model", "due tre uno"),),
    )


def test_aggregate_reports_denominators_and_condition_slices() -> None:
    report = score_chunks((_input(),), reviewed_commit="fixture-commit")
    model = report["models"][_model().fingerprint_sha256]

    assert model["slices"]["all"]["chunk_count"] == 1
    assert model["slices"]["all"]["duration_ms"] == 3_000
    assert model["slices"]["all"]["reference_word_denominator"] == 3
    assert model["slices"]["held_out"]["metrics"]["normalized_wer"]["value"] == 0.0
    assert model["slices"]["overlap_music"]["chunk_count"] == 1
    assert model["slices"]["degraded"]["chunk_count"] == 1
    assert report["provenance"]["reviewed_commit"] == "fixture-commit"


def test_manifest_scoring_preserves_manifest_and_model_provenance() -> None:
    item = _input()
    manifest = ChunkBenchmarkManifest(
        manifest_id="manifest-1",
        chunks=(item.chunk,),
        references=(item.reference,),
        hypotheses=(item.hypothesis,),
        partitions={"held_out": (item.chunk.chunk_id,)},
    )

    report = score_manifest(
        manifest,
        reference_streams={item.chunk.chunk_id: item.reference_streams},
        hypothesis_streams={item.hypothesis.hypothesis_id: item.hypothesis_streams},
    )

    assert report["provenance"]["manifest_sha256"] == manifest.content_sha256
    assert report["models"][_model().fingerprint_sha256]["hypothesis_count"] == 1


def test_report_writer_keeps_details_private_and_aggregate_sanitized(tmp_path: Path) -> None:
    report = score_chunks((_input(),), synthetic_provenance=("synthetic fixture",))
    artifacts = write_asr_score_report(report, tmp_path / "p1r-asr")

    metrics = json.loads((tmp_path / "p1r-asr" / "metrics.json").read_text())
    details = json.loads((tmp_path / "p1r-asr" / "details.json").read_text())
    assert "private_details" not in metrics
    assert len(details["chunks"]) == 1
    assert "uno" not in (tmp_path / "p1r-asr" / "metrics.json").read_text()
    assert artifacts["artifact_sha256"]["metrics.json"]

    with pytest.raises(ASRScoringValidationError, match="overwrite"):
        write_asr_score_report(report, tmp_path / "p1r-asr")

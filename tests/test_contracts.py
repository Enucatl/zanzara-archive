"""CPU-only contract validation and JSON round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.contracts import (
    API_V1_ROUTES,
    ApiEnvelope,
    ApiError,
    AudioArtifact,
    CandidateScore,
    ContractValidationError,
    DiarizationResult,
    EmbeddingBatch,
    ModelFingerprint,
    Overlap,
    TimedWord,
    TranscriptResult,
    Turn,
    json_round_trip,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _model() -> ModelFingerprint:
    return ModelFingerprint.from_dict(_fixture()["model"])


def test_records_round_trip_with_offsets_scope_and_provenance() -> None:
    fixture = _fixture()
    audio = AudioArtifact.from_dict(fixture["audio"])
    word = TimedWord.from_dict(fixture["word"])
    result = TranscriptResult(
        artifact_id=audio.artifact_id,
        source_sha256=audio.source_sha256,
        model=_model(),
        text=word.text,
        words=(word,),
        timestamp_granularities=("word",),
        request_id="req-fixture-1",
    )

    restored = TranscriptResult.from_dict(json_round_trip(result))
    assert restored == result
    assert restored.words[0].start_ms == 1000
    assert restored.words[0].speaker_id == "SPEAKER_00"
    assert restored.model.fingerprint_sha256 == result.model.fingerprint_sha256


def test_text_only_result_is_explicitly_not_production_usable() -> None:
    result = TranscriptResult(
        artifact_id="text-result",
        source_sha256="a" * 64,
        model=_model(),
        text="testo senza tempi",
        status="text_only",
    )
    assert not result.production_usable
    with pytest.raises(ContractValidationError, match="genuine word timestamps"):
        result.require_production()

    with pytest.raises(ContractValidationError, match="only timed"):
        TranscriptResult(
            artifact_id="bad-text-result",
            source_sha256="a" * 64,
            model=_model(),
            text="testo",
            words=(TimedWord("word-1", "testo", 0, 10),),
            status="text_only",
        )


def test_interval_vector_and_order_validation() -> None:
    with pytest.raises(ContractValidationError, match="half-open"):
        TimedWord("word-1", "ciao", 100, 100)
    with pytest.raises(ContractValidationError, match="within the audio duration"):
        DiarizationResult(
            artifact_id="diarization-1",
            source_sha256="a" * 64,
            duration_ms=100,
            model=_model(),
            standard_turns=(Turn("SPEAKER_00", 0, 101),),
            exclusive_turns=(),
            overlaps=(),
        )
    with pytest.raises(ContractValidationError, match="count mismatch"):
        EmbeddingBatch(_model(), ("one",), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    with pytest.raises(ContractValidationError, match="dimension"):
        EmbeddingBatch(_model(), ("one",), ((1.0, 0.0),))
    with pytest.raises(ContractValidationError, match="finite"):
        EmbeddingBatch(_model(), ("one",), ((float("nan"), 0.0, 1.0),))
    with pytest.raises(ContractValidationError, match="non-zero"):
        EmbeddingBatch(_model(), ("one",), ((0.0, 0.0, 0.0),))


def test_overlap_keeps_all_active_speakers_and_api_error_request_id() -> None:
    overlap = Overlap.from_dict(_fixture()["overlap"])
    assert overlap.speaker_ids == ("SPEAKER_00", "SPEAKER_01")
    assert overlap.to_dict()["speaker_ids"] == ["SPEAKER_00", "SPEAKER_01"]
    error = ApiError("unsupported_capability", "word timestamps unavailable", False, "req-1")
    envelope = ApiEnvelope("req-1", "error", error=error)
    assert ApiEnvelope.from_dict(envelope.to_dict()) == envelope
    assert "transcript" in API_V1_ROUTES
    with pytest.raises(ContractValidationError, match="match"):
        ApiEnvelope("req-2", "error", error=error)


def test_candidate_score_labels_uncalibrated_fusion() -> None:
    candidate = CandidateScore(
        candidate_id="candidate-1",
        episode_speaker_id="episode-speaker-1",
        rank=1,
        score=0.2,
        model_scores={"resnet293": 0.4, "eres2net": 0.3, "wavlm": 0.2},
    )
    assert not candidate.is_probability
    assert candidate.calibration_status == "uncalibrated_rank_fusion"

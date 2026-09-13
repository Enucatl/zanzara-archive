"""CPU-safe contract, adapter, persistence and calibration checks for P1R-08A."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.acoustic_conditions import (
    ACOUSTIC_CONDITION_VERSION,
    build_ast_windows,
    derive_acoustic_condition,
    development_confusion,
    measure_signal,
)
from zanzara_archive.contracts import (
    AcousticConditionCorrection,
    AudioChunk,
    AudioSetScore,
    AudioSetWindowScores,
    ChunkBenchmarkManifest,
    ChunkCondition,
    ContractValidationError,
    ModelFingerprint,
    SignalMeasurements,
    json_round_trip,
)
from zanzara_archive.inference import AudioSetConditionAdapter, parse_audioset_response
from zanzara_archive.storage import SCHEMA_VERSION, SQLiteRepository

SOURCE_SHA256 = "a" * 64
SEGMENTATION_SHA256 = "b" * 64


def _model() -> ModelFingerprint:
    return ModelFingerprint(
        name="audioset_ast",
        repository="MIT/ast-finetuned-audioset-10-10-0.4593",
        revision="f826b80d28226b62986cc218e5cec390b1096902",
        checkpoint_sha256=("c" * 64,),
        dimensions=527,
        preprocessing={"sample_rate_hz": 16_000},
        precision="float16 CUDA inference",
        runtime={"python": "3.14.7"},
        terms_evidence="BSD-3-Clause",
    )


def _chunk(condition: ChunkCondition | None = None) -> AudioChunk:
    return AudioChunk.create(
        episode_id="episode-p1r-08a",
        source_sha256=SOURCE_SHA256,
        start_ms=0,
        end_ms=12_000,
        segmentation_fingerprint=SEGMENTATION_SHA256,
        duration_ms=20_000,
        condition=condition,
        partition="development",
    )


def _scores(start_ms: int, end_ms: int, music: float = 0.8) -> AudioSetWindowScores:
    return AudioSetWindowScores(
        start_ms,
        end_ms,
        (
            AudioSetScore(0, "Speech", 0.6),
            AudioSetScore(16, "Laughter", 0.4),
            AudioSetScore(137, "Music", music),
        ),
    )


def test_ast_windows_and_signal_measurements_are_deterministic() -> None:
    assert build_ast_windows(0, 25_000) == (
        build_ast_windows(0, 10_000)[0],
        build_ast_windows(10_000, 20_000)[0],
        build_ast_windows(20_000, 25_000)[0],
    )
    signal = measure_signal((0.0, 0.5, 1.0, 0.0))
    assert signal.clipping_fraction == pytest.approx(0.25)
    assert signal.peak_dbfs == pytest.approx(0.0)
    assert signal.silence_fraction == pytest.approx(0.5)


def test_ast_mapping_retains_raw_windows_and_does_not_claim_speaker_fields() -> None:
    speaker_condition = ChunkCondition(acoustic_labels=("legacy",), music=False)
    chunk = _chunk(speaker_condition)
    result = derive_acoustic_condition(
        chunk,
        window_scores=(_scores(0, 10_000), _scores(10_000, 12_000)),
        signal_measurements=SignalMeasurements(0.0, -1.0, -12.0, 0.1),
        model_fingerprint="d" * 64,
        label_map_sha256="e" * 64,
        preprocessing_sha256="f" * 64,
    )
    assert result.condition is not None
    metadata = result.condition.acoustic_metadata
    assert metadata is not None
    assert metadata.music_level == "dominant"
    assert metadata.has_speech is True
    assert metadata.non_speech_activity == ("Laughter",)
    assert len(metadata.raw_scores) == 3
    assert len(metadata.window_scores) == 2
    assert result.condition.speaker_streams == ()
    assert result.condition.acoustic_labels == ("Laughter",)
    assert result.condition.music is True
    assert AudioChunk.from_dict(json_round_trip(result)) == result


def test_ast_unknown_state_and_development_confusion_are_explicit() -> None:
    unknown = derive_acoustic_condition(_chunk())
    assert unknown.condition is not None
    assert unknown.condition.acoustic_metadata is not None
    assert unknown.condition.acoustic_metadata.status == "unknown"
    assert unknown.condition.acoustic_metadata.music_level == "uncertain"

    report = development_confusion(
        (
            {"music_level": "none", "music_score": 0.05, "speech_score": 0.6},
            {"music_level": "dominant", "music_score": 0.9, "speech_score": 0.1},
        )
    )
    assert report["partition"] == "development"
    assert report["count"] == 2
    with pytest.raises(ContractValidationError, match="not development"):
        development_confusion(
            ({"partition": "held_out", "music_level": "none", "music_score": 0, "speech_score": 0},)
        )


def test_audioset_adapter_validates_request_and_response() -> None:
    chunk = _chunk()
    seen: dict[str, object] = {}

    def post(url: str, body: bytes, timeout: float) -> bytes:
        seen["url"] = url
        seen["timeout"] = timeout
        seen["payload"] = json.loads(body)
        return json.dumps(
            {
                "model": _model().repository,
                "model_revision": _model().revision,
                "chunk_id": chunk.chunk_id,
                "windows": [_scores(0, 10_000).to_dict(), _scores(10_000, 12_000).to_dict()],
                "signal_measurements": SignalMeasurements(0.0, -1.0, -12.0, 0.1).to_dict(),
                "label_map_sha256": "e" * 64,
                "preprocessing_sha256": "f" * 64,
            }
        ).encode()

    adapter = AudioSetConditionAdapter("http://audioset", _model(), http_post=post)
    parsed = adapter.classify_bytes(chunk, b"audio", request_id="request-1")
    assert parsed.result.music_level == "dominant"
    assert seen["url"] == "http://audioset/v1/audio/classify"
    assert seen["payload"]["chunk_id"] == chunk.chunk_id  # type: ignore[index]

    bad = dict(parsed.raw_response)
    bad["model_revision"] = "wrong"
    with pytest.raises(Exception, match="revision"):
        parse_audioset_response(bad, chunk=chunk, model=_model(), request_id="request-1")


def test_acoustic_correction_is_append_only_and_machine_seed_survives(tmp_path: Path) -> None:
    seed = derive_acoustic_condition(
        _chunk(),
        window_scores=(_scores(0, 10_000), _scores(10_000, 12_000)),
        signal_measurements=SignalMeasurements(0.0, -1.0, -12.0, 0.1),
        model_fingerprint="d" * 64,
    )
    manifest = ChunkBenchmarkManifest(
        manifest_id="manifest-p1r-08a",
        schema_version=1,
        chunks=(seed,),
        references=(),
        hypotheses=(),
        partitions={"development": (seed.chunk_id,)},
    )
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-episode", "f" * 64, "fixture", "episode.opus", "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            seed.episode_id,
            "manifest-episode",
            "episode.opus",
            "2026-09-13",
            SOURCE_SHA256,
            100,
            20_000,
            "opus",
            1,
            48_000,
        ),
    )
    repository.connection.commit()
    repository.record_chunk_benchmark_manifest(manifest)
    correction = AcousticConditionCorrection(
        music_level="background",
        audio_quality="clean",
        reviewer="reviewer-1",
        reviewed_at="2026-09-13T00:00:00+00:00",
        reason="fixture review",
        seed_version=ACOUSTIC_CONDITION_VERSION,
    )
    correction_id = repository.record_acoustic_condition_correction(seed.chunk_id, correction)
    assert (
        repository.record_acoustic_condition_correction(seed.chunk_id, correction) == correction_id
    )
    restored = repository.fetch_audio_chunk(seed.chunk_id)
    assert restored is not None and restored.condition is not None
    assert restored.condition.acoustic_metadata == seed.condition.acoustic_metadata
    assert restored.condition.acoustic_correction == correction
    assert repository.list_acoustic_condition_corrections(seed.chunk_id) == (correction,)
    assert repository.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    repository.close()

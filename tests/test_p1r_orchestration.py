"""CPU fixtures for P1R multi-model dispatch and reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.contracts import (
    AdapterFailure,
    ApiError,
    AudioChunk,
    CapabilityDeclaration,
    ChunkBenchmarkManifest,
    ModelFingerprint,
    TranscriptionHypothesis,
)
from zanzara_archive.inference import ParsedHypothesis
from zanzara_archive.p1r_orchestration import (
    AdapterRegistration,
    ChunkInferenceOrchestrator,
    CommonChunkAudio,
)
from zanzara_archive.storage import SQLiteRepository

SOURCE_SHA256 = "a" * 64
SEGMENTATION_SHA256 = "b" * 64


def _model(name: str, index: int) -> ModelFingerprint:
    return ModelFingerprint(
        name=name,
        repository=f"fixture/{name}",
        revision=f"revision-{name}",
        checkpoint_sha256=(f"{index:x}" * 64,),
        preprocessing={"sample_rate_hz": 16_000, "channels": 1},
        precision="float32",
        runtime={"python": "3.14"},
        terms_evidence="synthetic fixture",
    )


def _chunk() -> AudioChunk:
    return AudioChunk.create(
        episode_id="episode-p1r-orchestration",
        source_sha256=SOURCE_SHA256,
        start_ms=0,
        end_ms=3_000,
        segmentation_fingerprint=SEGMENTATION_SHA256,
        duration_ms=10_000,
        partition="development",
    )


def _repository(tmp_path: Path, chunk: AudioChunk) -> SQLiteRepository:
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
            chunk.episode_id,
            "manifest-episode",
            "episode.opus",
            "2026-09-13",
            SOURCE_SHA256,
            100,
            10_000,
            "wav",
            1,
            16_000,
        ),
    )
    repository.connection.commit()
    return repository


class FixtureAdapter:
    def __init__(self, name: str, index: int, *, failure: ApiError | None = None):
        self._model = _model(name, index)
        self.failure = failure
        self.calls: list[tuple[str, bytes]] = []

    @property
    def capabilities(self) -> CapabilityDeclaration:
        return CapabilityDeclaration(model=self._model)

    def transcribe_bytes(
        self, chunk: AudioChunk, audio_bytes: bytes, *, request_id: str
    ) -> ParsedHypothesis:
        self.calls.append((request_id, audio_bytes))
        if self.failure is not None:
            raise AdapterFailure(self.failure)
        result = TranscriptionHypothesis(
            chunk_id=chunk.chunk_id,
            model_fingerprint=self._model,
            text=f"testo {self._model.name}",
        )
        return ParsedHypothesis(
            result=result,
            raw_response={"text": result.text, "request_id": request_id},
        )


def _manifest(chunk: AudioChunk) -> ChunkBenchmarkManifest:
    return ChunkBenchmarkManifest(
        manifest_id="manifest-p1r-orchestration",
        chunks=(chunk,),
        partitions={"development": (chunk.chunk_id,)},
    )


def test_success_persists_three_independent_results_and_common_audio(
    tmp_path: Path,
) -> None:
    chunk = _chunk()
    repository = _repository(tmp_path, chunk)
    adapters = tuple(
        FixtureAdapter(name, index)
        for index, name in enumerate(("parakeet", "whisper", "voxtral"), 1)
    )
    orchestrator = ChunkInferenceOrchestrator(
        repository,
        ArtifactPublisher(tmp_path / "artifacts", repository),
        [AdapterRegistration(adapter._model.name, adapter) for adapter in adapters],
    )
    loader_calls = 0

    def load_audio(loaded_chunk: AudioChunk) -> bytes:
        nonlocal loader_calls
        loader_calls += 1
        assert loaded_chunk == chunk
        return b"common-preprocessed-wav"

    report = orchestrator.run(_manifest(chunk), load_audio)

    assert report.complete
    assert report.total_dispatches == 3
    assert report.hypothesis_count == 3
    assert report.failure_count == 0
    assert {key: value["succeeded"] for key, value in report.counts_by_model.items()} == {
        "parakeet": 1,
        "whisper": 1,
        "voxtral": 1,
    }
    assert loader_calls == 1
    assert {audio for adapter in adapters for _, audio in adapter.calls} == {
        b"common-preprocessed-wav"
    }
    rows = repository.list_chunk_inference_dispatches("manifest-p1r-orchestration")
    assert all(row["status"] == "succeeded" for row in rows)
    assert len({row["input_audio_sha256"] for row in rows}) == 1
    assert all(row["raw_artifact_id"] for row in rows)
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM transcription_hypotheses").fetchone()[0]
        == 3
    )
    assert (
        repository.connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE stage = 'chunk-inference'"
        ).fetchone()[0]
        == 3
    )

    second = orchestrator.run(
        _manifest(chunk), lambda _: pytest.fail("resumed work reloaded audio")
    )
    assert second.complete
    assert all(len(adapter.calls) == 1 for adapter in adapters)
    repository.close()


def test_one_model_failure_is_visible_without_discarding_successful_peers(
    tmp_path: Path,
) -> None:
    chunk = _chunk()
    repository = _repository(tmp_path, chunk)
    failed = FixtureAdapter(
        "whisper",
        2,
        failure=ApiError("model_unavailable", "Whisper fixture unavailable", False, "unused"),
    )
    adapters = (
        FixtureAdapter("parakeet", 1),
        failed,
        FixtureAdapter("voxtral", 3),
    )
    orchestrator = ChunkInferenceOrchestrator(
        repository,
        ArtifactPublisher(tmp_path / "artifacts", repository),
        [AdapterRegistration(adapter._model.name, adapter) for adapter in adapters],
    )

    report = orchestrator.run(_manifest(chunk), lambda _: b"common-preprocessed-wav")

    assert not report.complete
    assert report.hypothesis_count == 2
    assert report.failure_count == 1
    assert report.failures[0].adapter_id == "whisper"
    assert report.failures[0].error.code == "model_unavailable"
    assert report.counts_by_model["whisper"]["failed"] == 1
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM transcription_hypotheses").fetchone()[0]
        == 2
    )
    assert (
        repository.fetch_job(
            next(
                row["job_id"]
                for row in repository.list_chunk_inference_dispatches("manifest-p1r-orchestration")
                if row["adapter_id"] == "whisper"
            )
        ).status
        == "blocked"
    )
    repository.close()


def test_transient_failure_resumes_same_adapter_after_backoff(tmp_path: Path) -> None:
    chunk = _chunk()
    repository = _repository(tmp_path, chunk)
    adapter = FixtureAdapter("parakeet", 1)
    transient = ApiError("timeout", "temporary timeout", True, "unused")
    adapter.failure = transient
    orchestrator = ChunkInferenceOrchestrator(
        repository,
        ArtifactPublisher(tmp_path / "artifacts", repository),
        [AdapterRegistration("parakeet", adapter)],
    )
    manifest = _manifest(chunk)

    first = orchestrator.run(
        manifest, lambda _: b"common-preprocessed-wav", now="2026-09-13T00:00:00+00:00"
    )
    assert not first.complete
    assert first.failures[0].error.code == "timeout"
    adapter.failure = None
    second = orchestrator.run(
        manifest,
        lambda _: b"common-preprocessed-wav",
        now="2026-09-13T00:00:06+00:00",
    )
    assert second.complete
    dispatch = repository.list_chunk_inference_dispatches(manifest.manifest_id)[0]
    assert dispatch["attempts"] == 2
    assert len(adapter.calls) == 2
    repository.close()


def test_mismatched_common_audio_fails_each_model_without_adapter_calls(tmp_path: Path) -> None:
    chunk = _chunk()
    repository = _repository(tmp_path, chunk)
    adapters = tuple(
        FixtureAdapter(name, index) for index, name in enumerate(("one", "two", "three", "four"), 1)
    )
    orchestrator = ChunkInferenceOrchestrator(
        repository,
        ArtifactPublisher(tmp_path / "artifacts", repository),
        [AdapterRegistration(adapter._model.name, adapter) for adapter in adapters],
    )

    def mismatched_audio(loaded_chunk: AudioChunk) -> CommonChunkAudio:
        return CommonChunkAudio(
            source_sha256="c" * 64,
            start_ms=loaded_chunk.start_ms,
            end_ms=loaded_chunk.end_ms,
            audio_bytes=b"wrong-source",
        )

    report = orchestrator.run(_manifest(chunk), mismatched_audio)

    assert report.failure_count == 4
    assert all(failure.error.code == "invalid_audio" for failure in report.failures)
    assert all(not adapter.calls for adapter in adapters)
    assert len(repository.list_chunk_inference_dispatches("manifest-p1r-orchestration")) == 4
    repository.close()

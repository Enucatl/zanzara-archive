"""Synthetic contract checks for the replaceable P1R Qwen helper."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.contracts import AudioChunk, ModelFingerprint, TranscriptionHypothesis
from zanzara_archive.qwen_assistance import (
    AnnotationAssistanceRequest,
    AnnotationAssistanceService,
    AnnotationAssistanceValidationError,
    QwenAnnotationAdapter,
    build_qwen_prompt,
)
from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app

SOURCE_SHA256 = "a" * 64


def _model() -> ModelFingerprint:
    return ModelFingerprint(
        name="fixture-qwen",
        repository="Qwen/Qwen3-8B",
        revision="fixture-revision",
        checkpoint_sha256=("b" * 64,),
        runtime={"python": "3.14"},
        terms_evidence="synthetic fixture",
    )


def _chunk() -> AudioChunk:
    return AudioChunk.create(
        episode_id="episode-qwen",
        source_sha256=SOURCE_SHA256,
        start_ms=1_000,
        end_ms=4_000,
        segmentation_fingerprint="c" * 64,
        duration_ms=10_000,
        partition="development",
    )


def _repository(
    tmp_path: Path,
) -> tuple[SQLiteRepository, AudioChunk, tuple[TranscriptionHypothesis, ...]]:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-qwen", "d" * 64, "fixture", "episode-qwen", "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, '2026-09-13', ?, 100, 10000, 'wav', 1, 16000)""",
        ("episode-qwen", "manifest-qwen", "episode-qwen", SOURCE_SHA256),
    )
    chunk = _chunk()
    repository.record_audio_chunk(chunk)
    hypotheses = tuple(
        TranscriptionHypothesis(
            chunk_id=chunk.chunk_id,
            model_fingerprint=ModelFingerprint(
                name=name,
                repository=f"fixture/{name}",
                revision=f"revision-{name}",
                checkpoint_sha256=(f"{index:x}" * 64,),
                terms_evidence="synthetic fixture",
            ),
            text=text,
            source_sha256=SOURCE_SHA256,
        )
        for index, (name, text) in enumerate(
            (("parakeet", "ciao mondo"), ("whisper", "ciao mondo oggi")), 1
        )
    )
    for hypothesis in hypotheses:
        repository.record_transcription_hypothesis(hypothesis)
    return repository, chunk, hypotheses


class FakeHelper:
    def __init__(self, text: str = "ciao mondo") -> None:
        self._model = _model()
        self.text = text
        self.calls = 0

    @property
    def model(self) -> ModelFingerprint:
        return self._model

    def complete(self, request, prompt: str, *, request_id: str) -> str:
        assert request.chunk_id in prompt
        self.calls += 1
        return self.text


def test_success_persists_provenance_and_never_creates_reference(tmp_path: Path) -> None:
    repository, chunk, hypotheses = _repository(tmp_path)
    helper = FakeHelper()
    request = AnnotationAssistanceRequest.from_payload(
        chunk,
        hypotheses,
        (
            {
                "rank": 1,
                "chunk_id": "previous-1",
                "reference_text": "preceding truth",
                "draft_text": "preceding draft",
            },
        ),
    )

    first = AnnotationAssistanceService(helper).generate(repository, request)
    second = AnnotationAssistanceService(helper).generate(repository, request)

    assert first.status == "succeeded"
    assert first.draft_text == "ciao mondo"
    assert first.draft_id == second.draft_id
    assert helper.calls == 1
    assert first.prompt_version == "p1r-09-qwen-prompt-v1"
    assert first.prompt_sha256
    assert first.candidate_order
    assert first.input_payload["chunk_id"] == chunk.chunk_id
    assert first.provenance["pipeline_version"] == "p1r-09-v1"
    assert repository.fetch_annotation_assistance_draft(first.draft_id) is not None
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM transcript_references").fetchone()[0]
        == 0
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        repository.connection.execute(
            "UPDATE annotation_assistance_drafts SET draft_text = 'mutated' WHERE draft_id = ?",
            (first.draft_id,),
        )
    repository.close()


def test_unavailable_helper_is_a_visible_draft_state(tmp_path: Path) -> None:
    repository, chunk, hypotheses = _repository(tmp_path)
    request = AnnotationAssistanceRequest.from_payload(chunk, hypotheses)

    draft = AnnotationAssistanceService().generate(repository, request)

    assert draft.status == "unavailable"
    assert draft.draft_text is None
    assert draft.error is not None
    assert draft.error["code"] == "helper_unavailable"
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM transcript_references").fetchone()[0]
        == 0
    )
    repository.close()


def test_prompt_binds_output_to_current_chunk_and_limits_context() -> None:
    chunk = _chunk()
    hypotheses = (
        TranscriptionHypothesis(
            chunk_id=chunk.chunk_id,
            model_fingerprint=_model(),
            text="current hypothesis",
            source_sha256=SOURCE_SHA256,
        ),
    )
    request = AnnotationAssistanceRequest.from_payload(
        chunk,
        hypotheses,
        (
            {"rank": 1, "chunk_id": "prior-1", "reference_text": "old", "draft_text": "old"},
            {"rank": 2, "chunk_id": "prior-2", "reference_text": "older", "draft_text": "older"},
        ),
    )
    prompt = build_qwen_prompt(request)

    assert "CURRENT_CHUNK_ID: " + chunk.chunk_id in prompt
    assert '"current_chunk_text"' in prompt
    assert "Do not improve grammar" in prompt
    assert "prior-1" in prompt and "prior-2" in prompt
    with pytest.raises(AnnotationAssistanceValidationError, match="at most two"):
        AnnotationAssistanceRequest.from_payload(
            chunk,
            hypotheses,
            tuple(
                {
                    "rank": rank,
                    "chunk_id": f"prior-{rank}",
                    "reference_text": "",
                    "draft_text": "",
                }
                for rank in (1, 2, 3)
            ),
        )


def test_qwen_adapter_rejects_output_containing_prior_or_metadata() -> None:
    model = _model()

    def response(_url: str, _body: bytes, _timeout: float) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"current_chunk_text": "current", "prior_chunk_text": "leak"}
                            )
                        }
                    }
                ]
            }
        ).encode()

    adapter = QwenAnnotationAdapter("http://127.0.0.1:18089", model, http_post=response)
    chunk = _chunk()
    request = AnnotationAssistanceRequest.from_payload(
        chunk,
        (
            TranscriptionHypothesis(
                chunk_id=chunk.chunk_id,
                model_fingerprint=model,
                text="current",
                source_sha256=SOURCE_SHA256,
            ),
        ),
    )

    with pytest.raises(AnnotationAssistanceValidationError, match="only current_chunk_text"):
        adapter.complete(request, build_qwen_prompt(request), request_id="request-qwen")


def test_qwen_adapter_accepts_only_the_current_chunk_text() -> None:
    model = _model()
    captured: dict[str, object] = {}

    def response(url: str, body: bytes, _timeout: float) -> bytes:
        captured["url"] = url
        captured["body"] = json.loads(body)
        return json.dumps(
            {"choices": [{"message": {"content": json.dumps({"current_chunk_text": "current"})}}]}
        ).encode()

    adapter = QwenAnnotationAdapter("http://127.0.0.1:18089", model, http_post=response)
    chunk = _chunk()
    request = AnnotationAssistanceRequest.from_payload(
        chunk,
        (
            TranscriptionHypothesis(
                chunk_id=chunk.chunk_id,
                model_fingerprint=model,
                text="current",
                source_sha256=SOURCE_SHA256,
            ),
        ),
    )

    assert (
        adapter.complete(request, build_qwen_prompt(request), request_id="request-qwen")
        == "current"
    )
    assert captured["url"] == "http://127.0.0.1:18089/v1/chat/completions"
    assert captured["body"]["model"] == "Qwen/Qwen3-8B"
    with pytest.raises(ValueError, match="loopback"):
        QwenAnnotationAdapter("https://example.com", model)


def test_assistance_api_handles_success_and_unavailable_without_reference_mutation(
    tmp_path: Path,
) -> None:
    repository, chunk, hypotheses = _repository(tmp_path)
    repository.close()
    body = {
        "hypothesis_ids": [item.hypothesis_id for item in hypotheses],
        "preceding_chunks": [
            {
                "rank": 1,
                "chunk_id": "previous-1",
                "reference_text": "old truth",
                "draft_text": "old draft",
            }
        ],
    }

    client = TestClient(
        create_app(
            tmp_path / "state.db",
            artifact_root=tmp_path / "artifacts",
            annotation_assistant=AnnotationAssistanceService(FakeHelper("current draft")),
        )
    )
    success = client.post(f"/api/v1/chunks/{chunk.chunk_id}/annotation-assistance", json=body)
    assert success.status_code == 200
    assert success.json()["data"]["draft"]["status"] == "succeeded"
    assert success.json()["data"]["reference_unchanged"] is True

    unavailable = TestClient(
        create_app(tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    ).post(f"/api/v1/chunks/{chunk.chunk_id}/annotation-assistance", json=body)
    assert unavailable.status_code == 200
    assert unavailable.json()["data"]["draft"]["status"] == "unavailable"
    assert unavailable.json()["data"]["draft"]["error"]["code"] == "helper_unavailable"

    repository = SQLiteRepository.open(tmp_path / "state.db")
    assert (
        repository.connection.execute("SELECT COUNT(*) FROM transcript_references").fetchone()[0]
        == 0
    )
    assert (
        repository.connection.execute(
            "SELECT COUNT(*) FROM annotation_assistance_drafts"
        ).fetchone()[0]
        == 2
    )
    repository.close()

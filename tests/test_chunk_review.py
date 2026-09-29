"""Chunk review keeps candidate output separate and refuses stale writes."""

from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zanzara_archive.chunk_review import chunk_review_router
from zanzara_archive.contracts import (
    AudioChunk,
    ReferenceRevision,
    SpeakerStream,
    TranscriptionHypothesis,
    Turn,
)
from zanzara_archive.storage import SQLiteRepository


def test_chunk_review_history_validation_and_seed_optional(tmp_path: Path) -> None:
    """Review without optional seeds preserves identity and append-only history."""
    database = tmp_path / "state.db"
    repository = SQLiteRepository.open(database)
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest", "f" * 64, "fixture", "episode.opus", "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (episode_id, manifest_id, relative_filename,
        episode_date, source_sha256, size_bytes, duration_ms, codec, channels,
        sample_rate_hz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "episode",
            "manifest",
            "episode.opus",
            "2026-09-13",
            "a" * 64,
            100,
            10000,
            "opus",
            1,
            48000,
        ),
    )
    repository.connection.commit()
    chunk = AudioChunk.create(
        episode_id="episode",
        source_sha256="a" * 64,
        start_ms=1000,
        end_ms=3000,
        segmentation_fingerprint="b" * 64,
        duration_ms=10000,
    )
    repository.record_audio_chunk(chunk)
    candidate = TranscriptionHypothesis(chunk.chunk_id, "c" * 64, "machine draft")
    repository.record_transcription_hypothesis(candidate)
    repository.close()
    app = FastAPI()
    app.include_router(chunk_review_router(database))
    client = TestClient(app)
    url = "/api/v1/chunk-review/" + chunk.chunk_id
    before = client.get(url).json()["data"]
    assert before["current_reference"] is None
    assert before["candidates"][0]["text"] == "machine draft"
    body = {
        "expected_revision": 0,
        "text": "human draft",
        "reviewer": "operator",
        "review_status": "draft",
        "speaker_streams": [{"speaker_id": "A", "text": "ciao"}],
        "condition_corrections": {"music_level": "none", "speaker_condition": "single_speaker"},
    }
    saved = client.post(url, json=body)
    assert saved.status_code == 200, saved.text
    data = saved.json()["data"]
    assert data["current_reference"]["speaker_streams"][0] == {
        "speaker_id": "A",
        "turns": [],
        "text": "ciao",
    }
    assert data["revision"] == 1
    assert client.post(url, json=body).status_code == 409
    body.update(expected_revision=1, review_status="human_truth")
    assert client.post(url, json=body).status_code == 200
    data = client.get(url).json()["data"]
    assert [row["review_status"] for row in data["history"]] == ["draft", "human_truth"]
    assert data["candidates"] == before["candidates"]
    assert data["chunk"] == before["chunk"]
    assert (
        client.get("/api/v1/chunk-review").json()["data"]["chunks"][0]["review_status"]
        == "human_truth"
    )
    body["expected_revision"] = 2
    for patch in (
        {"reviewer": "machine"},
        {"speaker_streams": [{"speaker_id": []}]},
        {"speaker_streams": [{"speaker_id": "A", "turns": "bad"}]},
        {"speaker_streams": [{"speaker_id": "A", "turns": None}]},
        {"text": 12},
        {"review_status": "approved"},
        {"expected_revision": True},
        {"condition_corrections": {"music_level": "bad"}},
        {
            "speaker_streams": [
                {"speaker_id": "A", "turns": [{"speaker_id": "A", "start_ms": 0, "end_ms": 2000}]}
            ]
        },
    ):
        assert client.post(url, json={**body, **patch}).status_code == 422
    assert client.get(url).json()["data"]["revision"] == 2

    # Imported timestamps do not override the actual append order.
    repository = SQLiteRepository.open(database)
    prior = ReferenceRevision.from_dict(data["current_reference"])
    assert prior.speaker_streams[0].text == "ciao"
    latest = replace(
        prior,
        revision_id="imported-revision",
        revision=3,
        prior_revision_id=prior.revision_id,
        prior_revision=None,
        status=None,
        created_at="2000-01-01",
        text="later edit",
        speaker_streams=(SpeakerStream("A", (Turn("A", 1100, 1900),), "ciao"),),
        review_status="draft",
    )
    repository.append_reference_revision(latest)
    repository.close()
    assert client.get(url).json()["data"]["current_reference"]["text"] == "later edit"
    assert (
        client.get("/api/v1/chunk-review").json()["data"]["chunks"][0]["review_status"] == "draft"
    )

    body["expected_revision"] = 3
    updated = client.post(url, json=body)
    assert updated.status_code == 200, updated.text
    assert updated.json()["data"]["current_reference"]["speaker_streams"][0]["turns"] == [
        {"speaker_id": "A", "start_ms": 1100, "end_ms": 1900, "confidence": None}
    ]

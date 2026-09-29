"""Checks for filtered lexical search through storage, API and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.cli import main
from zanzara_archive.storage import SearchValidationError, SQLiteRepository
from zanzara_archive.web import create_app


def test_search_filters_before_pagination_and_validates_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep phrases, source offsets, ordering and filters consistent across callers."""

    database = tmp_path / "state.db"
    repository = SQLiteRepository.open(database)
    for episode_id, day, source in (
        ("a.opus", "2026-01-01", "a" * 64),
        ("b.opus", "2026-02-01", "b" * 64),
    ):
        repository.connection.execute(
            """INSERT INTO episodes
            (episode_id, relative_filename, episode_date, source_sha256, size_bytes,
             duration_ms, codec, channels, sample_rate_hz)
            VALUES (?, ?, ?, ?, 1, 10000, 'opus', 1, 48000)""",
            (episode_id, episode_id, day, source),
        )
        repository.connection.execute(
            """INSERT INTO artifacts
            (artifact_id, source_sha256, stage, stage_key, artifact_path, manifest_sha256,
             upstream_artifact_hashes_json, preprocessing_json, pipeline_version,
             artifact_schema_version, file_checksums_json, created_at)
            VALUES (?, ?, 'attribution', 'test', 'test', ?, '{}', '{}', 'test', 1, '{}', ?)""",
            (f"attr-{episode_id[0]}", source, source, day),
        )
    for chunk_id, episode_id, start_ms, content in (
        ("c2", "a.opus", 1000, "Mario Rossi parla"),
        ("c1", "b.opus", 2000, "Mario Rossi parla"),
        ("c3", "a.opus", 3000, "Mario Bianchi parla"),
    ):
        repository.connection.execute(
            """INSERT INTO text_chunks
            (chunk_id, episode_id, attribution_artifact_id, start_ms, end_ms,
             text, word_ids_json, overlap)
            VALUES (?, ?, ?, ?, ?, ?, '["w1","w2"]', 0)""",
            (chunk_id, episode_id, f"attr-{episode_id[0]}", start_ms, start_ms + 500, content),
        )
        repository.connection.execute(
            "INSERT INTO text_chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk_id, content)
        )
    repository.connection.commit()

    assert [row["chunk_id"] for row in repository.search_text('"Mario Rossi"')] == ["c1", "c2"]
    assert [row["chunk_id"] for row in repository.search_text("Mario-Rossi")] == ["c1", "c2"]
    assert (
        repository.search_text('"Mario Rossi"', episode_id="a.opus", limit=1)[0]["start_ms"] == 1000
    )
    assert [row["chunk_id"] for row in repository.search_text("Mario", date_from="2026-02-01")] == [
        "c1"
    ]
    assert repository.search_text("inesistente") == []
    assert repository.search_text('"Mario Rossi"', offset=1)[0]["chunk_id"] == "c2"
    for invalid in ("Mario*", '"Mario', '""'):
        with pytest.raises(SearchValidationError):
            repository.search_text(invalid)
    with pytest.raises(SearchValidationError):
        repository.search_text("Mario", date_from="2026-02-30")
    repository.close()

    client = TestClient(create_app(database))
    response = client.get("/api/v1/search", params={"q": '"Mario Rossi"', "episode_id": "a.opus"})
    assert response.status_code == 200
    result = response.json()["data"]["results"][0]
    assert (result["episode_id"], result["start_ms"], result["end_ms"], result["word_ids"]) == (
        "a.opus",
        1000,
        1500,
        ["w1", "w2"],
    )
    assert client.get("/api/v1/search", params={"q": "Mario*"}).status_code == 422
    assert (
        client.get("/api/v1/search", params={"q": "Mario", "global_speaker_id": "g1"}).json()[
            "error"
        ]["code"]
        == "invalid_search"
    )

    assert main(["search", '"Mario Rossi"', "--database", str(database), "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["chunk_id"] == "c1"

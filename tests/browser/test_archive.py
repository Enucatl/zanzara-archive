"""Archive navigation and unavailable-content regression checks."""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app


def test_archive_navigation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Browse real stored rows, empty state and recoverable storage failure."""
    database = tmp_path / "state.db"
    client = TestClient(create_app(database))
    assert "No episodes registered yet." in client.get("/").text
    assert "No anonymous speakers available yet." in client.get("/speakers").text
    repository = SQLiteRepository.open(database)
    repository.connection.executescript("""
        INSERT INTO episodes (episode_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz)
        VALUES ('example.opus', 'example.opus', '2026-01-01', 'source', 1, 10000, 'opus', 1, 48000);
        INSERT INTO artifacts (artifact_id, source_sha256, stage, stage_key, artifact_path,
            manifest_sha256, upstream_artifact_hashes_json, preprocessing_json,
            pipeline_version, artifact_schema_version, file_checksums_json, created_at)
        VALUES ('attr', 'source', 'attribution', 'key', 'private', 'hash', '[]', '{}',
            'test', 1, '{}', '2026-01-01');
        INSERT INTO text_chunks (chunk_id, episode_id, attribution_artifact_id,
            start_ms, end_ms, text, word_ids_json, speaker_id)
        VALUES ('chunk', 'example.opus', 'attr', 0, 1000, 'hello', '[]', '<script>');
        INSERT INTO jobs (job_id, stage, status, created_at, updated_at)
        VALUES ('job', 'asr', 'failed', '2026-01-01', '2026-01-01');
    """)
    repository.close()
    for route in ("/", "/transcripts", "/speakers"):
        page = client.get(route)
        assert page.status_code == 200
        assert f'href="{route}" aria-current="page"' in page.text
        assert "asr · failed: 1" in page.text
        assert "/annotations/example.opus" in page.text
    assert "1 indexed transcript chunks." in client.get("/transcripts").text
    assert "&lt;script&gt;" in client.get("/speakers").text
    assert 'href="/transcripts"' in client.get("/annotations/example.opus").text
    assert 'href="/speakers"' in client.get("/p1r-03d").text

    def unavailable(*args: object, **kwargs: object) -> None:
        """Simulate an unavailable database during a later request."""
        raise sqlite3.OperationalError("private storage detail")

    monkeypatch.setattr(SQLiteRepository, "open", unavailable)
    page = client.get("/transcripts")
    assert page.status_code == 503
    assert 'role="alert"' in page.text
    assert "Retry" in page.text
    assert "private storage detail" not in page.text

"""Archive navigation and unavailable-content regression checks."""

import sqlite3
from html import unescape
from pathlib import Path
from re import findall
from urllib.parse import parse_qs, urlsplit

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
    for route in ("/", "/transcripts"):
        page = client.get(route)
        assert page.status_code == 200
        assert f'href="{route}" aria-current="page"' in page.text
        assert "asr · failed: 1" in page.text
        assert "/annotations/example.opus" in page.text
    assert "1 indexed transcript chunks." in client.get("/transcripts").text
    assert "No anonymous speakers available yet." in client.get("/speakers").text
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


@pytest.mark.browser
def test_search_result_opens_correct_transcript_offset(tmp_path: Path) -> None:
    """Keep filtered search, episode playback, and unavailable states connected."""
    database = tmp_path / "state.db"
    repository = SQLiteRepository.open(database)
    repository.connection.executescript("""
        INSERT INTO episodes (episode_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz) VALUES
            ('first.opus', 'first.opus', '2026-09-01', 'first', 1, 60000, 'opus', 1, 48000),
            ('second.opus', 'second.opus', '2026-09-02', 'second', 1, 60000, 'opus', 1, 48000),
            ('unindexed.opus', 'unindexed.opus', '2026-09-03', 'unindexed',
             1, 60000, 'opus', 1, 48000);
        INSERT INTO artifacts (artifact_id, source_sha256, stage, stage_key, artifact_path,
            manifest_sha256, upstream_artifact_hashes_json, preprocessing_json,
            pipeline_version, artifact_schema_version, file_checksums_json, created_at)
        VALUES ('attr', 'second', 'attribution', 'key', 'private', 'hash', '[]', '{}',
            'test', 1, '{}', '2026-09-02');
        INSERT INTO text_chunks (chunk_id, episode_id, attribution_artifact_id,
            start_ms, end_ms, text, word_ids_json, speaker_id) VALUES
            ('first-hit', 'first.opus', 'attr', 4000, 5000, 'radar first', '[]', 'A'),
            ('second-hit', 'second.opus', 'attr', 17500, 18500, 'radar second', '[]', 'B');
        INSERT INTO text_chunks_fts (chunk_id, text) VALUES
            ('first-hit', 'radar first'), ('second-hit', 'radar second');
    """)
    repository.close()
    client = TestClient(create_app(database))
    filters = {
        "q": "radar",
        "episode_id": "second.opus",
        "date_from": "2026-09-02",
        "date_to": "2026-09-02",
    }

    for route in ("/", "/transcripts"):
        page = client.get(route, params=filters)
        assert page.status_code == 200
        assert "radar second" in page.text
        assert "radar first" not in page.text
        assert "Speaker B" in page.text
    links = [unescape(href) for href in findall(r'href="([^"]+)"', page.text)]
    matches = [
        urlsplit(href)
        for href in links
        if urlsplit(href).path == "/episodes/second.opus" and "q=" in urlsplit(href).query
    ]
    assert len(matches) == 1
    target = parse_qs(matches[0].query)
    assert float(target.pop("t")[0]) == 17.5
    assert target.pop("chunk") == ["second-hit"]
    assert target == {key: [value] for key, value in filters.items()}

    episode = client.get(matches[0].geturl())
    assert episode.status_code == 200
    assert "radar second" in episode.text
    assert "radar first" not in episode.text
    assert 'data-start-ms="17500"' in episode.text
    assert 'data-seconds="17.5"' in episode.text
    assert "Speaker B" in episode.text
    assert "Original audio unavailable" in episode.text

    empty = client.get("/transcripts", params={"q": "absent"})
    assert empty.status_code == 200
    assert "No results" in empty.text
    assert "Transcript unavailable" in client.get("/episodes/unindexed.opus").text
    missing_media = client.get("/media/second.opus")
    assert missing_media.status_code == 404
    assert missing_media.json()["error"]["code"] == "media_unavailable"

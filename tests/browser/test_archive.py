"""Archive navigation and unavailable-content regression checks."""

import os
import sqlite3
from html import unescape
from pathlib import Path
from re import findall
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.corpus import load_manifest
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


@pytest.mark.browser
def test_search_playback_keyboard_in_chromium(tmp_path: Path) -> None:
    """Follow a filtered hit to the right source offset using only the keyboard."""
    playwright = pytest.importorskip("playwright.sync_api")
    manifest_path = Path(__file__).parents[2] / "planning" / "corpus-20.json"
    first, second = load_manifest(manifest_path).episodes[:2]
    source = tmp_path / second.relative_filename
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic media")
    database = tmp_path / "state.db"
    client = TestClient(create_app(database, manifest_path=manifest_path, archive_root=tmp_path))
    repository = SQLiteRepository.open(database)
    for index, episode in enumerate((first, second)):
        repository.connection.execute(
            """INSERT INTO artifacts (artifact_id, source_sha256, stage, stage_key,
               artifact_path, manifest_sha256, upstream_artifact_hashes_json,
               preprocessing_json, pipeline_version, artifact_schema_version,
               file_checksums_json, created_at)
               VALUES (?, ?, 'attribution', 'test', 'test', ?, '[]', '{}',
               'test', 1, '{}', '2026-09-01')""",
            (f"attr-{index}", episode.sha256, episode.sha256),
        )
        repository.connection.execute(
            """INSERT INTO text_chunks (chunk_id, episode_id,
               attribution_artifact_id, start_ms, end_ms, text,
               word_ids_json, speaker_id)
               VALUES (?, ?, ?, ?, ?, ?, '[]', ?)""",
            (
                f"hit-{index}",
                episode.relative_filename,
                f"attr-{index}",
                4000 if index == 0 else 17500,
                5000 if index == 0 else 18500,
                f"radar {index}",
                "A" if index == 0 else "B",
            ),
        )
        repository.connection.execute(
            "INSERT INTO text_chunks_fts (chunk_id, text) VALUES (?, ?)",
            (f"hit-{index}", f"radar {index}"),
        )
    repository.connection.commit()
    repository.close()

    def route_request(route: playwright.Route) -> None:
        """Serve the app to Chromium without opening a network port."""
        target = urlsplit(route.request.url)
        if target.path.startswith("/media/"):
            route.fulfill(status=204)
            return
        response = client.request(
            route.request.method, target.path + ("?" + target.query if target.query else "")
        )
        route.fulfill(
            status=response.status_code,
            body=response.content,
            content_type=response.headers.get("content-type", "text/plain"),
        )

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=os.environ.get("CHROMIUM_EXECUTABLE"))
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://archive.test/**", route_request)
        page.add_init_script("""
            let clock = 0;
            Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
                get() {return clock}, set(value) {clock = value}
            });
            HTMLMediaElement.prototype.play = function() {return Promise.resolve()};
        """)
        page.goto("http://archive.test/transcripts")
        page.locator('input[name="q"]').fill("radar")
        page.locator('select[name="episode_id"]').select_option(second.relative_filename)
        page.locator('input[name="q"]').press("Enter")
        result = page.locator(".search-result a")
        playwright.expect(result).to_have_count(1)
        assert second.relative_filename in result.get_attribute("href")
        assert "t=17.500" in result.get_attribute("href")
        result.focus()
        page.keyboard.press("Enter")
        playwright.expect(page.locator("h1")).to_contain_text(second.relative_filename)
        assert page.locator(".seek").get_attribute("data-seconds") == "17.5"
        page.locator("#episode-audio").dispatch_event("loadedmetadata")
        assert page.locator("#episode-audio").evaluate("audio => audio.currentTime") == 17.5
        page.locator("#episode-audio").evaluate("audio => audio.currentTime = 0")
        page.locator(".seek").focus()
        page.keyboard.press("Enter")
        assert page.locator("#episode-audio").evaluate("audio => audio.currentTime") == 17.5
        page.get_by_text("Back to transcripts").click()
        playwright.expect(page.locator(".search-result")).to_have_count(1)
        page.locator('input[name="q"]').fill("absent")
        page.locator('input[name="q"]').press("Enter")
        playwright.expect(page.get_by_text("No results.", exact=False)).to_be_visible()
        page.locator('input[name="q"]').fill("radar*")
        page.locator('input[name="q"]').press("Enter")
        playwright.expect(page.get_by_role("alert")).to_be_visible()
        assert not errors
        browser.close()

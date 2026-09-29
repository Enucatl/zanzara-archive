"""Opt-in Chromium smoke: uv run --with playwright pytest tests/browser/test_chunk_review.py."""

import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.contracts import (
    AudioChunk,
    ChunkCondition,
    Overlap,
    SpeakerStream,
    TranscriptionHypothesis,
    Turn,
)
from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app

playwright = pytest.importorskip("playwright.sync_api")


@pytest.mark.browser
def test_chunk_review_in_chromium(tmp_path: Path) -> None:
    """Exercise real DOM edits and keyboard controls against the actual review API."""
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
    chunks = []
    hostile = '<img src=x onerror="window.injected=true">'
    for index in range(4):
        start = 1000 + index * 2000
        streams = (
            SpeakerStream("A", (Turn("A", start, start + 1000),)),
            SpeakerStream("B", (Turn("B", start + (500 if index == 2 else 1000), start + 2000),)),
        )
        condition = (
            None
            if index == 0
            else ChunkCondition(
                speaker_streams=streams,
                overlaps=(Overlap(("A", "B"), start + 500, start + 1000),) if index == 2 else (),
                rapid_turn_taking=index == 1,
            )
        )
        chunk = AudioChunk.create(
            episode_id="episode",
            source_sha256="a" * 64,
            start_ms=start,
            end_ms=start + 2000,
            segmentation_fingerprint="b" * 64,
            duration_ms=10000,
            condition=condition,
        )
        repository.record_audio_chunk(chunk)
        repository.record_transcription_hypothesis(
            TranscriptionHypothesis(chunk.chunk_id, "c" * 64, hostile if index == 0 else "ciao")
        )
        if index == 3:
            repository.record_transcription_hypothesis(
                TranscriptionHypothesis(chunk.chunk_id, "d" * 64, "buongiorno")
            )
        chunks.append(chunk)
    repository.close()
    client = TestClient(create_app(database, artifact_root=tmp_path / "artifacts"))
    urls = ["/api/v1/chunk-review/" + chunk.chunk_id for chunk in chunks]
    original = [client.get(url).json()["data"]["candidates"] for url in urls]

    def route_request(route: playwright.Route) -> None:
        """Bridge browser requests to the real local app without a server process."""
        path = urlsplit(route.request.url).path
        if path.startswith("/media/"):
            route.fulfill(status=204)
            return
        response = client.request(
            route.request.method,
            path,
            content=route.request.post_data,
            headers={"content-type": "application/json"},
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
        page.route("http://review.test/**", route_request)
        # Model the media clock deterministically; actual controls and bound logic run in Chromium.
        page.add_init_script("""
            let clock = 0, paused = true;
            Object.defineProperties(HTMLMediaElement.prototype, {
                currentTime: {get() {return clock}, set(value) {clock = value}},
                paused: {get() {return paused}}
            });
            HTMLMediaElement.prototype.play = function() {
                paused = false; this.dispatchEvent(new Event('play')); return Promise.resolve();
            };
            HTMLMediaElement.prototype.pause = function() {paused = true};
        """)
        page.goto("http://review.test/chunk-review")
        expect = playwright.expect
        expect(page.locator("#progress")).to_have_text("Chunk 1 of 4")
        expect(page.locator("#candidates pre")).to_have_text(hostile)
        assert page.locator("#candidates img, #candidates textarea").count() == 0
        assert page.evaluate("window.injected") is None
        expect(page.locator("#text")).to_have_value("")
        page.locator("#reviewer").fill("Browser operator")
        page.locator("#text").fill("human correction")
        page.keyboard.press("Alt+n")
        expect(page.locator("#message")).to_contain_text("Save your edits")
        expect(page.locator("#progress")).to_have_text("Chunk 1 of 4")
        page.keyboard.press("Control+s")
        expect(page.locator("#status")).to_have_text("draft · revision 1")
        page.keyboard.press("Alt+a")
        expect(page.locator("#status")).to_have_text("human_truth · revision 2")
        assert client.get(urls[0]).json()["data"]["current_reference"]["text"] == "human correction"

        page.keyboard.press("Alt+p")
        assert page.locator("#player").evaluate("p => p.currentTime === 1 && !p.paused")
        page.keyboard.press("Alt+ArrowLeft")
        assert page.locator("#player").evaluate("p => p.currentTime === 1")
        page.keyboard.press("Alt+ArrowRight")
        page.wait_for_function("document.querySelector('#player').paused")
        assert page.locator("#player").evaluate("p => p.currentTime === 3")
        page.keyboard.press("Alt+l")
        page.keyboard.press("Alt+p")
        page.locator("#player").evaluate("p => {p.currentTime = 3;}")
        page.wait_for_function("document.querySelector('#player').currentTime === 1")
        assert page.locator("#player").evaluate("p => !p.paused")
        page.keyboard.press("Alt+p")

        for index in (1, 2, 3):
            page.keyboard.press("Alt+n")
            expect(page.locator("#progress")).to_have_text(f"Chunk {index + 1} of 4")
            expect(page.locator("#streams textarea")).to_have_count(2)
            page.locator("#streams textarea").nth(0).fill("speaker A correction")
            page.locator("#streams textarea").nth(1).fill("speaker B correction")
            condition = "partial_overlap" if index == 2 else "multi_speaker_no_overlap"
            page.locator("#speaker_condition").select_option(condition)
            page.locator("#rapid_turn_taking").select_option("false")
            page.keyboard.press("Control+s")
            expect(page.locator("#status")).to_have_text("draft · revision 1")
            reference = client.get(urls[index]).json()["data"]["current_reference"]
            assert reference["speaker_streams"][0]["text"] == "speaker A correction"
            assert reference["speaker_streams"][0]["turns"]
            assert reference["condition_corrections"]["speaker_condition"] == condition
            assert reference["condition_corrections"]["rapid_turn_taking"] is False
            expect(page.locator("#rapid_turn_taking")).to_have_value("false")
        expect(page.locator("#candidates pre")).to_have_text(["ciao", "buongiorno"])
        page.locator("#text").fill("keep this unsaved edit")
        response = client.post(
            urls[3],
            json={
                "expected_revision": 1,
                "reviewer": "Another operator",
                "text": "concurrent edit",
                "review_status": "draft",
                "speaker_streams": [],
                "condition_corrections": {},
            },
        )
        assert response.status_code == 200
        page.keyboard.press("Control+s")
        expect(page.locator("#message")).to_contain_text("review changed")
        expect(page.locator("#text")).to_have_value("keep this unsaved edit")
        expect(page.locator("#text")).to_be_enabled()
        page.keyboard.press("Alt+b")
        expect(page.locator("#message")).to_contain_text("Save your edits")
        assert [client.get(url).json()["data"]["candidates"] for url in urls] == original
        assert not errors
        browser.close()

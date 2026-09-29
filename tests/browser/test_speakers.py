"""Voice review page and confirmed appearance regression."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.contracts import ModelFingerprint
from zanzara_archive.corpus import load_manifest
from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app


def _voice_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, list[str], list, list[str], Path]:
    """Build two synthetic voice records with one ranked candidate."""

    corpus_path = Path(__file__).parents[2] / "planning" / "corpus-20.json"
    episodes = load_manifest(corpus_path).episodes[:2]
    publisher = ArtifactPublisher(tmp_path / "artifacts")
    model = ModelFingerprint("resnet293", "test/resnet293", "rev", ("a" * 64,), 2)
    speaker_ids = []
    evidence_ids = []
    for index, episode in enumerate(episodes):
        source = episode.sha256
        publisher.publish(
            source_sha256=source,
            stage="diarization",
            stage_key=f"dia-{index}",
            files={"diarization.json": b"{}"},
            artifact_id=f"dia-{index}",
        )
        dia_path = publisher.artifact_path(source, "diarization", f"dia-{index}")
        dia_hash = hashlib.sha256((dia_path / "manifest.json").read_bytes()).hexdigest()
        exemplar = publisher.publish(
            source_sha256=source,
            stage="exemplars",
            stage_key=f"ex-{index}",
            files={"exemplars.json": b"{}"},
            upstream_artifact_hashes=(dia_hash,),
            artifact_id=f"ex-{index}",
        )
        ex_path = publisher.artifact_path(source, "exemplars", f"ex-{index}")
        ex_hash = hashlib.sha256((ex_path / "manifest.json").read_bytes()).hexdigest()
        embedding = publisher.publish(
            source_sha256=source,
            stage="speaker_embeddings_resnet293",
            stage_key=f"emb-{index}",
            artifact_id=f"emb-{index}",
            upstream_artifact_hashes=(ex_hash,),
            model_fingerprint_sha256=model.fingerprint_sha256,
            files={
                "embeddings.json": json.dumps(
                    {
                        "episode_id": episode.relative_filename,
                        "source_sha256": source,
                        "exemplars_artifact_id": exemplar.artifact_id,
                        "exemplars_manifest_sha256": ex_hash,
                        "model": model.to_dict(),
                        "speakers": {
                            "A": {
                                "status": "voice_searchable",
                                "excerpts": [
                                    {
                                        "exemplar_id": f"clip-{index}",
                                        "start_ms": 1000,
                                        "end_ms": 4000,
                                        "vector": [1.0, 0.0],
                                    }
                                ],
                            }
                        },
                    }
                ).encode()
            },
        )
        speaker_ids.append(f"{embedding.artifact_id}:A")
        evidence_ids.append(embedding.artifact_id)

    def excerpt(episode_id: str) -> dict[str, object]:
        """Return one synthetic source excerpt for the page."""
        return {"episode_id": episode_id, "start_ms": 1000, "end_ms": 4000}

    def candidates(record: dict, local_id: str, request: object) -> dict:
        """Stand in for the separately tested Qdrant candidate retriever."""
        return {
            "episode_speaker_id": speaker_ids[0],
            "generation": "test-generation",
            "score_kind": "uncalibrated_cosine_median",
            "status": "voice_searchable",
            "candidates": [
                {
                    "episode_speaker_id": speaker_ids[1],
                    "episode_id": episodes[1].relative_filename,
                    "embedding_artifact_id": evidence_ids[1],
                    "score": 0.91,
                    "centroid_score": 0.88,
                    "matches": [
                        {
                            "query_excerpt": excerpt(episodes[0].relative_filename),
                            "candidate_excerpt": excerpt(episodes[1].relative_filename),
                        }
                    ],
                    "representative_match": {
                        "query_excerpt": excerpt(episodes[0].relative_filename),
                        "candidate_excerpt": excerpt(episodes[1].relative_filename),
                    },
                }
            ],
        }

    monkeypatch.setattr("zanzara_archive.web.retrieve_candidates", candidates)
    database = tmp_path / "state.db"
    client = TestClient(
        create_app(database, artifact_root=publisher.root, manifest_path=corpus_path)
    )
    return client, speaker_ids, episodes, evidence_ids, database


def test_voice_comparison_and_reversible_appearance_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Show real artifact speakers and follow confirm, split and undo in SQLite."""
    client, speaker_ids, episodes, evidence_ids, database = _voice_client(tmp_path, monkeypatch)
    page = client.get("/speakers", params={"speaker_id": speaker_ids[0]})
    assert page.status_code == 200
    assert "ResNet293" in page.text and "uncalibrated" in page.text
    assert "Same person" in page.text and "Different person" in page.text
    assert "Transcript unavailable" in page.text
    assert "#t=1,4" in page.text

    body = {
        "left_episode_speaker_id": speaker_ids[0],
        "right_episode_speaker_id": speaker_ids[1],
        "decision": "same_person",
        "reviewer": "test reviewer",
        "evidence_artifact_ids": evidence_ids,
        "expected_revision": 0,
    }
    response = client.post("/api/v1/identity-decisions", json=body)
    assert response.status_code == 200
    decision_id = response.json()["data"]["decision_id"]
    repo = SQLiteRepository.open(database)
    global_id = repo.connection.execute(
        "SELECT global_speaker_id FROM identity_memberships WHERE episode_speaker_id = ?",
        (speaker_ids[0],),
    ).fetchone()[0]
    repo.close()
    history = client.get("/speakers").text
    assert "Confirmed appearance history" in history
    appearances = history.split("<h2>Confirmed appearance history</h2>", 1)[1]
    older, newer = sorted(episodes, key=lambda episode: episode.episode_date)
    assert appearances.index(older.relative_filename) < appearances.index(newer.relative_filename)
    assert history.count("Split this appearance") == 2
    split = client.post(
        f"/api/v1/speakers/{global_id}/split",
        json={
            "separate_episode_speaker_ids": [speaker_ids[1]],
            "retain_episode_speaker_id": speaker_ids[0],
            "reviewer": "test reviewer",
            "expected_revision": 1,
        },
    )
    assert split.status_code == 200
    assert "Split this appearance" not in client.get("/speakers").text
    body["expected_revision"] = 2
    response = client.post("/api/v1/identity-decisions", json=body)
    assert response.status_code == 200
    undo = client.post(
        f"/api/v1/identity-decisions/{response.json()['data']['decision_id']}/undo",
        json={
            "expected_revision": 3,
            "expected_decision_revision": 1,
            "retain_episode_speaker_id": speaker_ids[0],
            "reviewer": "test reviewer",
        },
    )
    assert undo.status_code == 200
    assert "Split this appearance" not in client.get("/speakers").text
    assert decision_id not in client.get("/speakers").text


@pytest.mark.browser
def test_voice_review_keyboard_in_chromium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare excerpts, confirm a match, then undo it with keyboard controls."""
    playwright = pytest.importorskip("playwright.sync_api")
    client, speaker_ids, episodes, _, _ = _voice_client(tmp_path, monkeypatch)

    def route_request(route: playwright.Route) -> None:
        """Serve the local app and JSON writes to Chromium."""
        target = urlsplit(route.request.url)
        if target.path.startswith("/media/"):
            route.fulfill(status=204)
            return
        response = client.request(
            route.request.method,
            target.path + ("?" + target.query if target.query else ""),
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
        page.route("http://archive.test/**", route_request)
        page.goto("http://archive.test/speakers")
        page.locator('select[name="speaker_id"]').select_option(speaker_ids[0])
        page.get_by_role("button", name="Find candidates").focus()
        page.keyboard.press("Enter")
        playwright.expect(page.locator(".pair")).to_have_count(1)
        assert page.locator('.pair audio[aria-label^="Selected excerpt"]').count() == 1
        assert page.locator('.pair audio[aria-label^="Candidate excerpt"]').count() == 1
        assert "#t=1,4" in page.locator(".pair audio").last.get_attribute("src")
        page.get_by_role("button", name="Same person").focus()
        page.keyboard.press("Enter")
        playwright.expect(page.get_by_role("alert")).to_contain_text("Enter a reviewer name")
        page.locator("#reviewer").fill("Browser operator")
        page.get_by_role("button", name="Same person").focus()
        page.keyboard.press("Enter")
        playwright.expect(page.get_by_role("button", name="Split this appearance")).to_have_count(2)
        appearances = page.locator(".group li")
        assert appearances.count() == 2
        older, newer = sorted(episodes, key=lambda episode: episode.episode_date)
        assert older.relative_filename in appearances.nth(0).inner_text()
        assert newer.relative_filename in appearances.nth(1).inner_text()
        page.locator("#reviewer").fill("Browser operator")
        page.get_by_role("button", name="Undo").focus()
        page.keyboard.press("Enter")
        playwright.expect(page.get_by_role("button", name="Split this appearance")).to_have_count(0)
        playwright.expect(
            page.get_by_text("No confirmed recurring appearances yet.")
        ).to_be_visible()
        assert client.get("/api/v1/identity-state").json()["data"]["revision"] == 2
        assert not errors
        browser.close()

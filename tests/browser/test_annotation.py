"""Loopback annotation route smoke checks.

These exercise the browser-facing HTML and API contract without requiring a
browser binary in CPU CI; real browser automation remains an opt-in check.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from zanzara_archive.storage import SQLiteRepository
from zanzara_archive.web import create_app


@pytest.mark.browser
def test_annotation_page_exposes_editor_and_api(tmp_path: Path) -> None:
    app = create_app(tmp_path / "state.db", artifact_root=tmp_path / "artifacts")
    client = TestClient(app)
    page = client.get("/annotations/golden.opus")
    assert page.status_code == 200
    assert "waveform" in page.text
    assert "words-pager" in page.text
    assert "PAGE_SIZE = 100" in page.text
    assert "data-end-ms" in page.text
    assert "playbackEndMs" in page.text
    assert "requestAnimationFrame" in page.text
    assert 'aria-label="Play ${escapeHtml(type)} interval"' in page.text
    assert "Mark human-reviewed" in page.text
    assert "Import draft JSON" in page.text
    assert "AST acoustic condition seed" in page.text
    assert "music_level" in page.text
    assert "audio_quality" in page.text
    response = client.get("/api/v1/annotations/golden.opus")
    assert response.status_code == 200
    assert response.json()["data"]["annotation"] is None
    assert client.get("/healthz").json() == {
        "status": "ok",
        "access": "loopback-default",
    }


@pytest.mark.browser
def test_network_access_is_reflected_by_health_endpoint(tmp_path: Path) -> None:
    client = TestClient(
        create_app(tmp_path / "state.db", artifact_root=tmp_path / "artifacts", network_access=True)
    )

    assert client.get("/healthz").json() == {
        "status": "ok",
        "access": "network-enabled",
    }


@pytest.mark.browser
def test_seed_accepts_raw_attribution_artifact(tmp_path: Path) -> None:
    source = "a" * 64
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES ("
        "'manifest-1', ?, 'fixture', 'golden.opus', '{}', 'now')",
        ("b" * 64,),
    )
    repository.connection.execute(
        "INSERT INTO episodes ("
        "episode_id, manifest_id, relative_filename, episode_date, source_sha256, "
        "size_bytes, duration_ms, codec, channels, sample_rate_hz) "
        "VALUES ('golden.opus', 'manifest-1', 'golden.opus', '2026-09-10', ?, "
        "100, 1000, 'opus', 1, 48000)",
        (source,),
    )
    repository.connection.commit()
    repository.close()

    raw_attribution = {
        "artifact_id": "attribution-1",
        "episode_id": "golden.opus",
        "source_sha256": source,
        "duration_ms": 1000,
        "words": [{"word_id": "word-1", "text": "ciao", "start_ms": 0, "end_ms": 100}],
        "diarization": {
            "standard_turns": [],
            "exclusive_turns": [],
            "overlaps": [],
        },
        "provenance": {"configuration": {"fixture": True}},
    }
    client = TestClient(create_app(tmp_path / "state.db", artifact_root=tmp_path / "artifacts"))

    response = client.post("/api/v1/annotations/golden.opus/seed", json=raw_attribution)

    assert response.status_code == 200
    assert response.json()["data"]["annotation"]["status"] == "draft"
    assert response.json()["data"]["annotation"]["actor_type"] == "machine"


@pytest.mark.browser
def test_manifest_registers_episodes_for_a_fresh_annotation_database(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            tmp_path / "state.db",
            artifact_root=tmp_path / "artifacts",
            manifest_path=Path("planning/corpus-20.json"),
        )
    )
    raw_attribution = {
        "artifact_id": "attribution-1",
        "episode_id": "260910-lazanzara.opus",
        "source_sha256": "06de18da0691a19738bcda30dace2d5be87c8be651e9bd9ed8e56b5828f64536",
        "duration_ms": 6108584,
        "words": [{"word_id": "word-1", "text": "ciao", "start_ms": 0, "end_ms": 100}],
        "diarization": {
            "standard_turns": [],
            "exclusive_turns": [],
            "overlaps": [],
        },
        "provenance": {"configuration": {"fixture": True}},
    }

    response = client.post("/api/v1/annotations/260910-lazanzara.opus/seed", json=raw_attribution)

    assert response.status_code == 200
    assert response.json()["data"]["annotation"]["status"] == "draft"


@pytest.mark.browser
def test_annotation_api_persists_private_export_and_rejects_stale_save(tmp_path: Path) -> None:
    source = "a" * 64
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES ("
        "'manifest-1', ?, 'fixture', 'golden.opus', '{}', 'now')",
        ("b" * 64,),
    )
    repository.connection.execute(
        "INSERT INTO episodes ("
        "episode_id, manifest_id, relative_filename, episode_date, source_sha256, "
        "size_bytes, duration_ms, codec, channels, sample_rate_hz) "
        "VALUES ('golden.opus', 'manifest-1', 'golden.opus', '2026-09-10', ?, "
        "100, 1000, 'opus', 1, 48000)",
        (source,),
    )
    repository.connection.commit()
    repository.close()

    payload = {
        "source_sha256": source,
        "duration_ms": 1000,
        "words": [{"word_id": "word-1", "text": "ciao", "start_ms": 0, "end_ms": 100}],
        "standard_turns": [],
        "exclusive_turns": [],
        "overlap_intervals": [],
        "unintelligible_spans": [],
        "provenance": {"configuration": {"fixture": True}},
        "actor_type": "machine",
        "reviewer": "machine",
        "expected_revision": 0,
    }
    client = TestClient(create_app(tmp_path / "state.db", artifact_root=tmp_path / "artifacts"))
    saved = client.post("/api/v1/annotations/golden.opus", json=payload)
    assert saved.status_code == 200
    current = client.get("/api/v1/annotations/golden.opus").json()["data"]
    assert current["annotation"]["words"][0]["text"] == "ciao"
    assert current["annotation"]["provenance"] == {"configuration": {"fixture": True}}
    assert [item["revision"] for item in current["history"]] == [1]
    assert '"revision":1' in client.get("/annotations/golden.opus").text
    assert client.get("/api/v1/annotations/golden.opus/exports/1/reference.json").status_code == 200
    conflict = client.post("/api/v1/annotations/golden.opus", json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "annotation_revision_conflict"

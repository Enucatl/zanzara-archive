"""Checks for speaker-aware lexical chunks and atomic FTS replacement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.storage import SQLiteRepository, StorageConflictError
from zanzara_archive.text_index import build_text_chunks

SOURCE = "a" * 64


def _payload(artifact_id: str, words: list[dict]) -> dict:
    return {
        "artifact_id": artifact_id,
        "time_origin_ms": 0,
        "duration_ms": 200_000,
        "transcript": {"status": "timed", "timestamp_granularities": ["word"]},
        "words": words,
    }


def _word(number: int, speaker: str | None, start: int, *, overlap: bool = False) -> dict:
    return {
        "word_id": f"w{number}",
        "text": f"term{number}",
        "start_ms": start,
        "end_ms": start + 100,
        "speaker_id": speaker,
        "overlap": overlap,
    }


def test_chunk_boundaries_keep_word_order_offsets_and_overlap() -> None:
    words = [_word(index, "A", index * 100) for index in range(201)]
    words += [_word(201, None, 60_000, overlap=True), _word(202, None, 121_000)]
    chunks = build_text_chunks(_payload("attribution-one", words))

    assert [len(chunk.word_ids) for chunk in chunks] == [200, 1, 1, 1]
    assert [chunk.speaker_id for chunk in chunks] == ["A", "A", None, None]
    assert chunks[2].overlap and chunks[2].start_ms == 60_000
    assert chunks[0].word_ids == tuple(f"w{index}" for index in range(200))
    assert chunks[0].text == " ".join(f"term{index}" for index in range(200))
    assert all(chunk.end_ms - chunk.start_ms <= 60_000 for chunk in chunks)
    assert build_text_chunks(_payload("attribution-one", words)) == chunks


def test_reindex_replaces_fts_and_rolls_back_failed_replacement(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        """INSERT INTO episodes
        (episode_id, relative_filename, episode_date, source_sha256, size_bytes,
         duration_ms, codec, channels, sample_rate_hz)
        VALUES ('episode.opus', 'episode.opus', '2026-01-01', ?, 1, 200000, 'opus', 1, 48000)""",
        (SOURCE,),
    )
    repository.connection.commit()
    publisher = ArtifactPublisher(tmp_path / "artifacts")

    def publish(key: str, words: list[dict]):
        payload = _payload(f"attribution-{key}", words)
        artifact = publisher.publish(
            source_sha256=SOURCE,
            stage="attribution",
            stage_key=key,
            artifact_id=payload["artifact_id"],
            files={"attributed.json": json.dumps(payload).encode()},
        )
        return (
            artifact,
            publisher.artifact_path(SOURCE, "attribution", key),
            build_text_chunks(payload),
        )

    first = publish("first", [_word(0, "A", 0), _word(1, None, 200)])
    repository.replace_text_chunks("episode.opus", *first)
    repository.replace_text_chunks("episode.opus", *first)
    assert repository.connection.execute("SELECT COUNT(*) FROM text_chunks_fts").fetchone()[0] == 2

    second = publish("second", [_word(2, "B", 500)])
    with pytest.raises(StorageConflictError):
        repository.replace_text_chunks("episode.opus", second[0], second[1], second[2] * 2)
    assert (
        repository.connection.execute(
            "SELECT COUNT(*) FROM text_chunks_fts WHERE text_chunks_fts MATCH 'term0'"
        ).fetchone()[0]
        == 1
    )

    repository.replace_text_chunks("episode.opus", *second)
    assert repository.connection.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0] == 1
    assert repository.connection.execute("SELECT COUNT(*) FROM text_chunks_fts").fetchone()[0] == 1
    assert (
        repository.connection.execute(
            "SELECT COUNT(*) FROM text_chunks_fts WHERE text_chunks_fts MATCH 'term0'"
        ).fetchone()[0]
        == 0
    )
    row = repository.connection.execute(
        "SELECT speaker_id, start_ms, end_ms, text, word_ids_json FROM text_chunks"
    ).fetchone()
    assert tuple(row) == ("B", 500, 600, "term2", '["w2"]')
    repository.close()

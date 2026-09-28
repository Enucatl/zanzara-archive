"""Deterministic speaker-aware chunks for lexical transcript search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import TimedWord


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One indexed run of attributed words from an episode."""

    chunk_id: str
    speaker_id: str | None
    start_ms: int
    end_ms: int
    text: str
    word_ids: tuple[str, ...]
    overlap: bool


def build_text_chunks(payload: dict[str, Any]) -> tuple[TextChunk, ...]:
    """Split a validated attribution payload at speaker and size boundaries."""

    if payload.get("transcript", {}).get("status") != "timed" or "word" not in payload[
        "transcript"
    ].get("timestamp_granularities", ()):
        raise ValueError("text indexing requires a timed word transcript")
    if payload.get("time_origin_ms") != 0:
        raise ValueError("text indexing requires original episode offsets")
    artifact_id = payload["artifact_id"]
    duration_ms = payload["duration_ms"]
    words = tuple(TimedWord.from_dict(value) for value in payload["words"])
    if not artifact_id or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("invalid attribution identity or duration")
    chunks: list[TextChunk] = []
    run: list[TimedWord] = []
    seen: set[str] = set()
    run_end_ms = 0
    previous_start_ms = -1

    def flush() -> None:
        if not run:
            return
        word_ids = tuple(word.word_id for word in run)
        digest = hashlib.sha256(
            json.dumps([artifact_id, word_ids[0], word_ids[-1]], separators=(",", ":")).encode()
        ).hexdigest()
        chunks.append(
            TextChunk(
                chunk_id=f"text-{digest[:32]}",
                speaker_id=run[0].speaker_id,
                start_ms=run[0].start_ms,
                end_ms=max(word.end_ms for word in run),
                text=" ".join(word.text for word in run),
                word_ids=word_ids,
                overlap=any(word.overlap for word in run),
            )
        )
        run.clear()

    for word in words:
        if word.word_id in seen or word.end_ms > duration_ms:
            raise ValueError("duplicate word ID or word exceeds episode duration")
        if word.start_ms < previous_start_ms:
            raise ValueError("attributed words must be ordered by original offset")
        if word.end_ms - word.start_ms > 60_000:
            raise ValueError("one word exceeds the 60-second chunk limit")
        seen.add(word.word_id)
        previous_start_ms = word.start_ms
        if run and (
            word.speaker_id != run[0].speaker_id
            or len(run) == 200
            or max(word.end_ms, run_end_ms) - run[0].start_ms > 60_000
        ):
            flush()
            run_end_ms = 0
        run.append(word)
        run_end_ms = max(run_end_ms, word.end_ms)
    flush()
    return tuple(chunks)

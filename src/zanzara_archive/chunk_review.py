"""Human chunk review backed by immutable candidate and reference records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from zanzara_archive.contracts import (
    ApiEnvelope,
    ChunkCondition,
    ReferenceRevision,
    SpeakerStream,
    TranscriptReference,
)
from zanzara_archive.storage import SQLiteRepository, StorageConflictError, StorageError


def _detail(repository: SQLiteRepository, chunk_id: str) -> dict[str, Any]:
    """Read seeds and immutable review history without promoting candidates."""
    chunk = repository.fetch_audio_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(404, "chunk was not found")
    history = [
        json.loads(row[0])
        for row in repository.connection.execute(
            "SELECT payload_json FROM reference_revisions WHERE chunk_id = ? ORDER BY rowid",
            (chunk_id,),
        )
    ]
    references = [
        json.loads(row[0])
        for row in repository.connection.execute(
            "SELECT payload_json FROM transcript_references WHERE chunk_id = ? ORDER BY rowid",
            (chunk_id,),
        )
    ]
    current = history[-1] if history else references[-1] if references else None
    return {
        "chunk": chunk.to_dict(),
        "candidates": [
            json.loads(row[0])
            for row in repository.connection.execute(
                "SELECT payload_json FROM transcription_hypotheses WHERE chunk_id = ? "
                "ORDER BY model_fingerprint_sha256",
                (chunk_id,),
            )
        ],
        "current_reference": current,
        "history": history,
        "references": references,
        "revision": len(history),
        "conditions": {
            "machine_seed": chunk.condition.to_dict() if chunk.condition else None,
            "human_correction": current.get("condition_corrections", {}) if current else {},
        },
    }


def _validate(body: dict[str, Any], detail: dict[str, Any]) -> tuple[SpeakerStream, ...]:
    """Validate editable fields and retain original interval evidence."""
    if type(body.get("expected_revision")) is not int or body["expected_revision"] < 0:
        raise ValueError("expected_revision must be a nonnegative integer")
    if body.get("review_status") not in ("draft", "human_truth"):
        raise ValueError("review_status must be draft or human_truth")
    reviewer = body.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.strip() == "machine":
        raise ValueError("an identified human reviewer is required")
    if not isinstance(body.get("text"), str):
        raise ValueError("text must be text")
    corrections = body.get("condition_corrections", {})
    if not isinstance(corrections, dict):
        raise ValueError("condition_corrections must be an object")
    enums = {
        "speaker_condition": (
            "single_speaker",
            "multi_speaker_no_overlap",
            "partial_overlap",
            "heavy_overlap",
            "uncertain",
        ),
        "music_level": ("none", "background", "dominant", "uncertain"),
        "audio_quality": ("clean", "degraded", "uncertain"),
    }
    for key, value in corrections.items():
        if key in enums:
            if value not in enums[key]:
                raise ValueError(f"invalid {key}")
        elif key == "notes":
            if not isinstance(value, str):
                raise ValueError("notes must be text")
        elif key == "rapid_turn_taking":
            if type(value) is not bool:
                raise ValueError("rapid_turn_taking must be boolean")
        else:
            raise ValueError(f"unknown condition correction: {key}")
    values = body.get("speaker_streams", [])
    if not isinstance(values, list):
        raise ValueError("speaker_streams must be a list")
    seed = detail["chunk"].get("condition") or {}
    previous = detail["current_reference"] or seed
    existing = {stream["speaker_id"]: stream for stream in previous.get("speaker_streams", [])}
    streams = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("speaker stream must be an object")
        merged = dict(existing.get(value.get("speaker_id"), {}))
        merged.update(value)
        streams.append(SpeakerStream.from_dict(merged))
    condition = ChunkCondition(speaker_streams=tuple(streams))
    condition.validate_bounds(detail["chunk"]["start_ms"], detail["chunk"]["end_ms"])
    return tuple(streams)


def chunk_review_router(database: Path) -> APIRouter:
    """Build the local review API using the application's database."""
    router = APIRouter()

    def response(data: dict[str, Any]) -> dict[str, Any]:
        """Use the common API envelope."""
        return ApiEnvelope(uuid4().hex, "ok", data=data).to_dict()

    @router.get("/api/v1/chunk-review")
    def queue(manifest_id: str | None = None) -> dict[str, Any]:
        """List frozen chunks and their human review progress."""
        with closing(SQLiteRepository.open(database)) as repository:
            rows = repository.connection.execute(
                """SELECT c.chunk_id, c.episode_id, c.start_ms, c.end_ms, c.partition,
                (SELECT COUNT(*) FROM reference_revisions r
                 WHERE r.chunk_id = c.chunk_id) AS revision,
                COALESCE(
                    (SELECT r.review_status FROM reference_revisions r
                     WHERE r.chunk_id = c.chunk_id ORDER BY r.rowid DESC LIMIT 1),
                    (SELECT t.review_status FROM transcript_references t
                     WHERE t.chunk_id = c.chunk_id ORDER BY t.rowid DESC LIMIT 1),
                    'unreviewed'
                ) AS review_status
                FROM audio_chunks c
                WHERE (? IS NULL OR c.manifest_id = ?)
                ORDER BY c.episode_id, c.start_ms, c.chunk_id""",
                (manifest_id, manifest_id),
            ).fetchall()
            return response({"chunks": [dict(row) for row in rows]})

    @router.get("/api/v1/chunk-review/{chunk_id:path}")
    def get_review(chunk_id: str) -> dict[str, Any]:
        """Return separate candidate drafts, human truth, and condition seeds."""
        with closing(SQLiteRepository.open(database)) as repository:
            return response(_detail(repository, chunk_id))

    @router.post("/api/v1/chunk-review/{chunk_id:path}")
    def save_review(chunk_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Append a human revision while refusing stale concurrent saves."""
        try:
            with closing(SQLiteRepository.open(database)) as repository:
                with repository.transaction() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    detail = _detail(repository, chunk_id)
                    streams = _validate(body, detail)
                    if body["expected_revision"] != detail["revision"]:
                        raise StorageConflictError("review changed; reload before saving")
                    current = detail["current_reference"]
                    reference_id = current["reference_id"] if current else "review-" + chunk_id
                    if current is None:
                        repository._record_transcript_reference(
                            connection,
                            TranscriptReference(
                                reference_id=reference_id,
                                chunk_id=chunk_id,
                                source_sha256=detail["chunk"]["source_sha256"],
                            ),
                        )
                    prior = next(
                        (
                            item
                            for item in reversed(detail["history"])
                            if item["reference_id"] == reference_id
                        ),
                        None,
                    )
                    revision = ReferenceRevision(
                        revision_id="review-revision-" + uuid4().hex,
                        reference_id=reference_id,
                        chunk_id=chunk_id,
                        revision=prior["revision"] + 1 if prior else 1,
                        prior_revision_id=prior["revision_id"] if prior else None,
                        source_sha256=detail["chunk"]["source_sha256"],
                        reviewer=body["reviewer"].strip(),
                        text=body["text"],
                        review_status=body["review_status"],
                        speaker_streams=streams,
                        created_at=datetime.now(UTC).isoformat(),
                        condition_corrections=body.get("condition_corrections", {}),
                    )
                    repository._record_reference_revision(connection, revision)
                return response(_detail(repository, chunk_id))
        except StorageConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except (StorageError, sqlite3.Error) as exc:
            raise HTTPException(503, str(exc)) from exc

    return router

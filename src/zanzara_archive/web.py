"""Local annotation application.

The CLI keeps loopback binding as its default and requires an explicit network
opt-in for LAN access. The app has no authentication, and all media/export
paths are resolved from the frozen manifest or a revision-owned artifact name.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import niquests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .annotations import (
    AnnotationValidationError,
    annotation_export_path,
    annotation_from_attribution,
    save_annotation_revision,
)
from .artifacts import ArtifactPublicationError
from .calibration import validate_batch, validate_decision
from .chunk_review import chunk_review_router
from .contracts import (
    AcousticConditionCorrection,
    ApiEnvelope,
    ApiError,
    ArtifactManifest,
    IdentityDecision,
)
from .corpus import CorpusValidationError, load_manifest, resolve_source
from .qwen_assistance import (
    AnnotationAssistanceRequest,
    AnnotationAssistanceService,
    AnnotationAssistanceValidationError,
)
from .storage import SearchValidationError, SQLiteRepository, StorageConflictError, StorageError
from .voice_index import Qdrant, load_embedding_artifacts
from .voice_search import retrieve_candidates

TEMPLATE_ROOT = Path(__file__).with_name("templates")
EXPORT_NAMES = frozenset(
    {
        "reference.json",
        "annotation.json",
        "manual_timing.json",
        "split.json",
        "transcript.txt",
        "transcript.srt",
        "transcript.vtt",
    }
)


def _request_id() -> str:
    return f"annotation-{uuid.uuid4().hex}"


def _error(request_id: str, code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiEnvelope(
            request_id,
            "error",
            error=ApiError(code, message, False, request_id),
        ).to_dict(),
    )


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _history(repository: SQLiteRepository, episode_id: str) -> list[dict[str, Any]]:
    return [
        {
            "annotation_revision_id": item.get("annotation_revision_id"),
            "revision": item.get("revision"),
            "status": item.get("status"),
            "reviewer": item.get("reviewer"),
            "created_at": item.get("created_at"),
            "export_url": (
                f"/api/v1/annotations/{quote(episode_id, safe='')}/exports/"
                f"{item.get('revision')}/reference.json"
            ),
        }
        for item in repository.list_annotation_revisions(episode_id)
    ]


def create_app(
    database: str | Path,
    *,
    artifact_root: str | Path = ".git/zanzara-artifacts",
    archive_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    p1r03d_evidence_root: str | Path | None = None,
    network_access: bool = False,
    annotation_assistant: AnnotationAssistanceService | None = None,
) -> FastAPI:
    """Create the local annotation app for one canonical SQLite state path."""

    app = FastAPI(title="Zanzara Archive — Local Annotation", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))
    database_path = Path(database).expanduser()
    app.include_router(chunk_review_router(database_path))
    artifact_path = Path(artifact_root).expanduser()
    source_root = Path(archive_root).expanduser() if archive_root is not None else None
    review_root = (
        Path(p1r03d_evidence_root).expanduser()
        if p1r03d_evidence_root is not None
        else artifact_path.parent / "zanzara-evidence" / "P1R-03D"
    )
    assistance_service = annotation_assistant or AnnotationAssistanceService()
    frozen_manifest = load_manifest(manifest_path) if manifest_path is not None else None
    source_hashes = (
        {episode.sha256 for episode in frozen_manifest.episodes}
        if frozen_manifest is not None
        else set()
    )
    voice_paths = sorted(
        path
        for path in artifact_path.glob("*/speaker_embeddings_resnet293/*/embeddings.json")
        if path.parents[2].name in source_hashes
    )
    voice_records = load_embedding_artifacts([path.parent for path in voice_paths])
    voice_speakers = {
        f"{record['embedding_artifact_id']}:{local_id}": (record, local_id)
        for record in voice_records
        for local_id in record["speakers"]
    }
    if frozen_manifest is not None or voice_records:
        current = SQLiteRepository.open(database_path)
        try:
            if frozen_manifest is not None:
                current.register_corpus_manifest(frozen_manifest)
            for record, voice_path in zip(voice_records, voice_paths, strict=True):
                source = record["source_sha256"]
                exemplar_paths = list(
                    (artifact_path / source / "exemplars").glob("*/manifest.json")
                )
                exemplar_path = next(
                    (
                        path
                        for path in exemplar_paths
                        if json.loads(path.read_text())["artifact_id"]
                        == record["exemplars_artifact_id"]
                    ),
                    None,
                )
                if exemplar_path is None:
                    raise ValueError("voice exemplar artifact is missing")
                exemplar_bytes = exemplar_path.read_bytes()
                if (
                    hashlib.sha256(exemplar_bytes).hexdigest()
                    != record["exemplars_manifest_sha256"]
                ):
                    raise ValueError("voice exemplar manifest disagrees with embedding")
                exemplar = ArtifactManifest.from_dict(json.loads(exemplar_bytes))
                if exemplar.stage != "exemplars" or exemplar.source_sha256 != source:
                    raise ValueError("voice exemplar provenance disagrees with source")
                diarization_paths = list(
                    (artifact_path / source / "diarization").glob("*/manifest.json")
                )
                diarization_path = next(
                    (
                        path
                        for path in diarization_paths
                        if hashlib.sha256(path.read_bytes()).hexdigest()
                        in exemplar.upstream_artifact_hashes
                    ),
                    None,
                )
                if diarization_path is None:
                    raise ValueError("voice diarization artifact is missing")
                diarization = ArtifactManifest.from_dict(json.loads(diarization_path.read_text()))
                if diarization.stage != "diarization" or diarization.source_sha256 != source:
                    raise ValueError("voice diarization provenance disagrees with source")
                embedding_path = voice_path.parent
                embedding = ArtifactManifest.from_dict(
                    json.loads((embedding_path / "manifest.json").read_text())
                )
                for manifest, path in (
                    (diarization, diarization_path.parent),
                    (exemplar, exemplar_path.parent),
                    (embedding, embedding_path),
                ):
                    existing = current.connection.execute(
                        "SELECT source_sha256, stage FROM artifacts WHERE artifact_id = ?",
                        (manifest.artifact_id,),
                    ).fetchone()
                    if existing is None:
                        current.record_artifact(manifest, path)
                    elif tuple(existing) != (manifest.source_sha256, manifest.stage):
                        raise StorageConflictError("voice artifact identity has changed")
                for local_id, speaker in record["speakers"].items():
                    speaker_id = f"{record['embedding_artifact_id']}:{local_id}"
                    current.register_episode_speaker(
                        speaker_id, record["episode_id"], diarization.artifact_id, local_id
                    )
                    current.connection.execute(
                        """UPDATE episode_speakers SET voice_searchable = ?
                           WHERE episode_speaker_id = ?""",
                        (speaker["status"], speaker_id),
                    )
                current.connection.commit()
        finally:
            current.close()

    @contextmanager
    def repository() -> Iterator[SQLiteRepository]:
        current = SQLiteRepository.open(database_path)
        try:
            yield current
        finally:
            current.close()

    @app.get("/", response_class=HTMLResponse)
    @app.get("/transcripts", response_class=HTMLResponse)
    def archive_page(
        request: Request,
        q: str = "",
        episode_id: str = "",
        date_from: str = "",
        date_to: str = "",
        offset: int = 0,
    ) -> HTMLResponse:
        """Show archive content and filtered transcript results."""

        episodes, speakers, jobs, results = [], [], [], []
        error = None
        search_error = None
        search_params = {
            key: value
            for key, value in (
                ("q", q),
                ("episode_id", episode_id),
                ("date_from", date_from),
                ("date_to", date_to),
                ("offset", offset if offset else ""),
            )
            if value
        }
        try:
            with repository() as current:
                episodes = current.connection.execute(
                    """SELECT e.episode_id, e.episode_date,
                       (SELECT count(*) FROM text_chunks t
                        WHERE t.episode_id = e.episode_id) AS chunk_count,
                       EXISTS(SELECT 1 FROM annotation_revisions a
                              WHERE a.episode_id = e.episode_id) AS has_annotation
                       FROM episodes e ORDER BY e.episode_date DESC, e.episode_id"""
                ).fetchall()
                speakers = current.connection.execute(
                    """SELECT DISTINCT t.speaker_id, t.episode_id
                       FROM text_chunks t JOIN episodes e USING (episode_id)
                       WHERE t.speaker_id IS NOT NULL
                       ORDER BY e.episode_date DESC, t.episode_id, t.speaker_id"""
                ).fetchall()
                jobs = current.connection.execute(
                    """SELECT stage, status, count(*) AS total FROM jobs
                       GROUP BY stage, status ORDER BY stage, status"""
                ).fetchall()
                if q.strip() and request.url.path != "/speakers":
                    try:
                        hits = current.search_text(
                            q,
                            episode_id=episode_id or None,
                            date_from=date_from or None,
                            date_to=date_to or None,
                            limit=21,
                            offset=offset,
                        )
                        results = hits[:20]
                    except SearchValidationError as exc:
                        search_error = str(exc)
                    for result in results:
                        result["episode_url"] = (
                            f"/episodes/{quote(str(result['episode_id']), safe='')}?"
                            + urlencode(
                                {
                                    **search_params,
                                    "t": f"{result['start_ms'] / 1000:.3f}",
                                    "chunk": result["chunk_id"],
                                }
                            )
                        )
        except (StorageError, sqlite3.Error):
            error = "Archive content is unavailable. Reload this page to retry."
        return templates.TemplateResponse(
            request=request,
            name="archive.html",
            context={
                "episodes": episodes,
                "speakers": speakers,
                "jobs": jobs,
                "results": results,
                "error": error,
                "search_error": search_error,
                "q": q,
                "episode_id": episode_id,
                "date_from": date_from,
                "date_to": date_to,
                "next_url": (
                    f"/transcripts?{urlencode({**search_params, 'offset': offset + 20})}"
                    if len(results) == 20 and not search_error and len(hits) > 20
                    else None
                ),
            },
            status_code=503 if error else 200,
        )

    @app.get("/speakers", response_class=HTMLResponse)
    def speakers_page(request: Request, speaker_id: str = "") -> HTMLResponse:
        """Compare indexed voices and show confirmed appearances in date order."""

        with repository() as current:
            rows = current.connection.execute(
                """SELECT s.episode_speaker_id, s.episode_id, s.local_speaker_id,
                          s.voice_searchable, e.episode_date, m.global_speaker_id,
                          g.display_name,
                          EXISTS(SELECT 1 FROM text_chunks t WHERE t.episode_id = s.episode_id)
                            AS has_transcript
                   FROM episode_speakers s JOIN episodes e USING (episode_id)
                   LEFT JOIN identity_memberships m USING (episode_speaker_id)
                   LEFT JOIN global_speakers g USING (global_speaker_id)
                   WHERE s.mapping_state = 'current'
                   ORDER BY e.episode_date, s.episode_id, s.local_speaker_id"""
            ).fetchall()
            decisions = current.connection.execute(
                """SELECT decision_id, left_episode_speaker_id, right_episode_speaker_id,
                          decision, revision FROM identity_decisions
                   WHERE state = 'active' ORDER BY created_at DESC, decision_id"""
            ).fetchall()
            revision = current.identity_revision()
        speakers = [dict(row) for row in rows]
        by_id = {item["episode_speaker_id"]: item for item in speakers}
        for item in speakers:
            record_local = voice_speakers.get(item["episode_speaker_id"])
            excerpts = (
                record_local[0]["speakers"][record_local[1]]["excerpts"] if record_local else []
            )
            item["excerpt"] = (
                {"start_ms": excerpts[0]["start_ms"], "end_ms": excerpts[0]["end_ms"]}
                if excerpts
                else None
            )
            item["media_url"] = f"/media/{quote(item['episode_id'], safe='')}" if excerpts else None
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in speakers:
            if item["global_speaker_id"]:
                groups.setdefault(item["global_speaker_id"], []).append(item)
        result = None
        error = None
        if speaker_id:
            if speaker_id not in by_id or speaker_id not in voice_speakers:
                error = "Selected voice is unavailable."
            else:
                record, local_id = voice_speakers[speaker_id]
                try:
                    result = retrieve_candidates(
                        record,
                        local_id,
                        Qdrant(os.environ.get("QDRANT_ENDPOINT", "http://127.0.0.1:6333")).request,
                    )
                except (ValueError, OSError, niquests.exceptions.RequestException):
                    error = "Voice index is unavailable. Retry when Qdrant is running."
                if result:
                    for candidate in result["candidates"]:
                        candidate["speaker"] = by_id.get(candidate["episode_speaker_id"])
                        for match in candidate["matches"]:
                            for side in ("query_excerpt", "candidate_excerpt"):
                                excerpt = match[side]
                                excerpt["media_url"] = (
                                    f"/media/{quote(excerpt['episode_id'], safe='')}"
                                    f"#t={excerpt['start_ms'] / 1000:g},"
                                    f"{excerpt['end_ms'] / 1000:g}"
                                )
                        candidate["representative_match"] = candidate["matches"][0]
        return templates.TemplateResponse(
            request=request,
            name="speakers.html",
            context={
                "speakers": speakers,
                "selected": by_id.get(speaker_id),
                "result": result,
                "error": error,
                "groups": groups,
                "decisions": decisions,
                "revision": revision,
            },
        )

    @app.get("/episodes/{episode_id:path}", response_class=HTMLResponse, response_model=None)
    def episode_page(
        request: Request,
        episode_id: str,
        t: float = 0,
        chunk: str = "",
        page: int | None = None,
    ) -> HTMLResponse | JSONResponse:
        """Show one indexed transcript beside its original audio."""

        with repository() as current:
            episode = current.connection.execute(
                "SELECT episode_date, duration_ms FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if episode is None:
                return _error(_request_id(), "unknown_episode", "episode was not found", 404)
            if not math.isfinite(t) or not 0 <= t < episode["duration_ms"] / 1000:
                return _error(
                    _request_id(), "invalid_offset", "playback offset is out of range", 422
                )
            if chunk and page is not None:
                return _error(
                    _request_id(), "invalid_page", "choose a transcript offset or page", 422
                )
            count = current.connection.execute(
                "SELECT count(*) FROM text_chunks WHERE episode_id = ?", (episode_id,)
            ).fetchone()[0]
            if chunk:
                target = current.connection.execute(
                    "SELECT start_ms FROM text_chunks WHERE episode_id = ? AND chunk_id = ?",
                    (episode_id, chunk),
                ).fetchone()
                if target is None or target["start_ms"] != round(t * 1000):
                    return _error(
                        _request_id(), "invalid_offset", "transcript offset is invalid", 422
                    )
            if page is None:
                before = current.connection.execute(
                    """SELECT count(*) FROM text_chunks WHERE episode_id = ?
                       AND (start_ms < ? OR (start_ms = ? AND chunk_id < ?))""",
                    (episode_id, round(t * 1000), round(t * 1000), chunk),
                ).fetchone()[0]
                page = before // 100
            if page < 0 or (page > 0 and page * 100 >= count):
                return _error(_request_id(), "invalid_page", "transcript page is out of range", 422)
            chunks = current.connection.execute(
                """SELECT chunk_id, start_ms, end_ms, speaker_id, text, overlap
                   FROM text_chunks WHERE episode_id = ?
                   ORDER BY start_ms, chunk_id LIMIT 100 OFFSET ?""",
                (episode_id, page * 100),
            ).fetchall()
        media_url = None
        if (
            frozen_manifest is not None
            and source_root is not None
            and any(item.relative_filename == episode_id for item in frozen_manifest.episodes)
        ):
            try:
                resolve_source(source_root, episode_id)
            except CorpusValidationError:
                pass
            else:
                media_url = f"/media/{quote(episode_id, safe='')}"
        search_params = {
            key: request.query_params[key]
            for key in ("q", "episode_id", "date_from", "date_to", "offset")
            if request.query_params.get(key)
        }
        episode_path = f"/episodes/{quote(episode_id, safe='')}"
        return templates.TemplateResponse(
            request=request,
            name="episode.html",
            context={
                "episode_id": episode_id,
                "episode_date": episode["episode_date"],
                "chunks": chunks,
                "media_url": media_url,
                "start_seconds": t,
                "page": page,
                "page_count": (count + 99) // 100,
                "previous_url": (
                    f"{episode_path}?{urlencode({**search_params, 'page': page - 1})}"
                    if page > 0
                    else None
                ),
                "next_url": (
                    f"{episode_path}?{urlencode({**search_params, 'page': page + 1})}"
                    if (page + 1) * 100 < count
                    else None
                ),
                "back_url": f"/transcripts?{urlencode(search_params)}"
                if search_params
                else "/transcripts",
            },
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "access": "network-enabled" if network_access else "loopback-default",
        }

    @app.get("/api/v1/search")
    def search(
        q: str,
        episode_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        global_speaker_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> JSONResponse:
        """Return ranked transcript chunks with original playback intervals."""

        request_id = _request_id()
        try:
            with repository() as current:
                results = current.search_text(
                    q,
                    episode_id=episode_id,
                    date_from=date_from,
                    date_to=date_to,
                    global_speaker_id=global_speaker_id,
                    limit=limit,
                    offset=offset,
                )
        except SearchValidationError as exc:
            return _error(request_id, "invalid_search", str(exc), 422)
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(
            content=ApiEnvelope(
                request_id, "ok", data={"results": results, "limit": limit, "offset": offset}
            ).to_dict()
        )

    def identity_revision(body: dict[str, Any]) -> int:
        value = body.get("expected_revision")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        return value

    @app.get("/api/v1/identity-state")
    def get_identity_state() -> JSONResponse:
        with repository() as current:
            revision = current.identity_revision()
        return JSONResponse(
            content=ApiEnvelope(_request_id(), "ok", data={"revision": revision}).to_dict()
        )

    @app.post("/api/v1/identity-decisions")
    def save_identity_decision(body: dict[str, Any]) -> JSONResponse:
        try:
            expected = identity_revision(body)
            evidence = body["evidence_artifact_ids"]
            if not isinstance(evidence, list) or not all(
                isinstance(item, str) for item in evidence
            ):
                raise ValueError("evidence_artifact_ids must be a list of artifact IDs")
            decision = IdentityDecision(
                decision_id=f"decision-{uuid.uuid4().hex}",
                left_episode_speaker_id=body["left_episode_speaker_id"],
                right_episode_speaker_id=body["right_episode_speaker_id"],
                decision=body["decision"],
                reviewer=body["reviewer"],
                evidence_artifact_ids=tuple(evidence),
                created_at=datetime.now(UTC).isoformat(),
            )
            with repository() as current:
                current.append_identity_decision(
                    decision,
                    expected_revision=expected,
                    keep_global_speaker_id=body.get("keep_global_speaker_id"),
                )
                revision = current.identity_revision()
        except StorageConflictError as exc:
            return _error(_request_id(), "identity_conflict", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(_request_id(), "invalid_identity_decision", str(exc), 422)
        return JSONResponse(
            content=ApiEnvelope(
                _request_id(),
                "ok",
                data={"decision_id": decision.decision_id, "revision": revision},
            ).to_dict()
        )

    @app.post("/api/v1/identity-decisions/{decision_id}/undo")
    def undo_identity_decision(decision_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            expected = identity_revision(body)
            decision_revision = body["expected_decision_revision"]
            if (
                isinstance(decision_revision, bool)
                or not isinstance(decision_revision, int)
                or decision_revision < 1
            ):
                raise ValueError("expected_decision_revision must be a positive integer")
            with repository() as current:
                current.undo_identity_decision(
                    decision_id,
                    expected_revision=expected,
                    expected_decision_revision=decision_revision,
                    retain_episode_speaker_id=body["retain_episode_speaker_id"],
                    reviewer=body["reviewer"],
                )
                revision = current.identity_revision()
        except StorageConflictError as exc:
            return _error(_request_id(), "identity_conflict", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(_request_id(), "invalid_identity_undo", str(exc), 422)
        return JSONResponse(
            content=ApiEnvelope(_request_id(), "ok", data={"revision": revision}).to_dict()
        )

    @app.post("/api/v1/speakers/{global_id}/split")
    def split_global_speaker(global_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            expected = identity_revision(body)
            separate = body["separate_episode_speaker_ids"]
            if not isinstance(separate, list) or not all(
                isinstance(item, str) for item in separate
            ):
                raise ValueError("separate_episode_speaker_ids must be a list of speaker IDs")
            with repository() as current:
                current.split_global_speaker(
                    global_id,
                    set(separate),
                    expected_revision=expected,
                    retain_episode_speaker_id=body["retain_episode_speaker_id"],
                    reviewer=body["reviewer"],
                )
                revision = current.identity_revision()
        except StorageConflictError as exc:
            return _error(_request_id(), "identity_conflict", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(_request_id(), "invalid_identity_split", str(exc), 422)
        return JSONResponse(
            content=ApiEnvelope(_request_id(), "ok", data={"revision": revision}).to_dict()
        )

    @app.patch("/api/v1/speakers/{global_id}/name")
    def name_global_speaker(global_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            expected = identity_revision(body)
            name = body["display_name"]
            if name is not None and not isinstance(name, str):
                raise ValueError("display_name must be text or null")
            with repository() as current:
                current.name_global_speaker(global_id, name, expected_revision=expected)
                revision = current.identity_revision()
        except StorageConflictError as exc:
            return _error(_request_id(), "identity_conflict", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(_request_id(), "invalid_speaker_name", str(exc), 422)
        return JSONResponse(
            content=ApiEnvelope(_request_id(), "ok", data={"revision": revision}).to_dict()
        )

    def p1r03d_queue(filename: str) -> dict[str, Any]:
        path = review_root / filename
        if not path.is_file():
            return {"rows": [], "content_sha256": None, "available": False}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise ValueError(f"invalid P1R-03D review queue: {path}")
        return payload

    def p1r03d_data() -> dict[str, Any]:
        with repository() as current:
            reviews = current.list_p1r03d_reviews()
        return {
            "native": p1r03d_queue("native-vad-validation-queue-4c0f98f.json"),
            "boundary": p1r03d_queue("boundary-inspection-queue-4c0f98f.json"),
            "reviews": reviews,
        }

    @app.get("/p1r-03d", response_class=HTMLResponse, response_model=None)
    def p1r03d_page(request: Request) -> HTMLResponse | JSONResponse:
        try:
            data = p1r03d_data()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error(_request_id(), "review_queue_unavailable", str(exc), 503)
        return templates.TemplateResponse(
            request=request,
            name="p1r03d.html",
            context={"data_json": _safe_json(data)},
        )

    @app.get("/api/v1/p1r-03d")
    def get_p1r03d_reviews() -> JSONResponse:
        request_id = _request_id()
        try:
            data = p1r03d_data()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error(request_id, "review_queue_unavailable", str(exc), 503)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.post("/api/v1/p1r-03d/{review_type}/{item_id:path}")
    def save_p1r03d_review(review_type: str, item_id: str, body: dict[str, Any]) -> JSONResponse:
        request_id = _request_id()
        try:
            if review_type not in {"native", "boundary"}:
                raise ValueError("review type must be native or boundary")
            queue = p1r03d_queue(
                "native-vad-validation-queue-4c0f98f.json"
                if review_type == "native"
                else "boundary-inspection-queue-4c0f98f.json"
            )
            item_key = "region_id" if review_type == "native" else "chunk_id"
            known_ids = {str(row.get(item_key)) for row in queue["rows"]}
            if item_id not in known_ids:
                return _error(request_id, "unknown_review_item", "review item was not found", 404)
            reviewer = body.get("reviewer")
            if (
                not isinstance(reviewer, str)
                or not reviewer.strip()
                or reviewer.strip() == "machine"
            ):
                raise ValueError("reviewer must be an identified human")
            expected_revision = body.get("expected_revision", 0)
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("expected_revision must be a non-negative integer")
            note = body.get("note", "")
            if not isinstance(note, str):
                raise ValueError("note must be text")
            if review_type == "native":
                speech_present = body.get("speech_present")
                overlap_correct = body.get("overlap_correct")
                if speech_present not in {True, False, None} or overlap_correct not in {
                    True,
                    False,
                    None,
                }:
                    raise ValueError("native review fields must be true, false, or null")
                if speech_present is None:
                    decision = "uncertain"
                else:
                    decision = "speech_present" if speech_present else "speech_absent"
                payload = {
                    "speech_present": speech_present,
                    "overlap_correct": overlap_correct,
                    "note": note,
                }
            else:
                quality = body.get("quality")
                if quality not in {"good", "acceptable", "awkward", "bad"}:
                    raise ValueError("boundary quality must be good, acceptable, awkward, or bad")
                decision = quality
                payload = {"quality": quality, "note": note}
            with repository() as current:
                record = current.record_p1r03d_review(
                    review_type,
                    item_id,
                    decision=decision,
                    reviewer=reviewer.strip(),
                    payload=payload,
                    expected_revision=expected_revision,
                )
        except StorageConflictError as exc:
            return _error(request_id, "review_revision_conflict", str(exc), 409)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return _error(request_id, "invalid_review", str(exc), 422)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=record).to_dict())

    @app.post("/api/v1/calibration/batches")
    def prepare_calibration_batch(body: dict[str, Any]) -> JSONResponse:
        if frozen_manifest is None:
            return _error(
                _request_id(),
                "calibration_unavailable",
                "a frozen corpus manifest is required",
                503,
            )
        try:
            batch = validate_batch(body, frozen_manifest)
            with repository() as current:
                current.record_calibration_batch(batch)
        except (ValueError, TypeError, KeyError, StorageConflictError) as exc:
            return _error(_request_id(), "invalid_calibration_batch", str(exc), 422)
        return JSONResponse(
            content=ApiEnvelope(
                _request_id(),
                "ok",
                data={
                    "batch_id": batch["batch_id"],
                    "content_sha256": batch["content_sha256"],
                    "total_count": len(batch["chunks"]),
                },
            ).to_dict()
        )

    @app.get("/calibration/{batch_id}", response_class=HTMLResponse, response_model=None)
    def calibration_page(request: Request, batch_id: str) -> HTMLResponse | JSONResponse:
        if frozen_manifest is None:
            return _error(
                _request_id(),
                "calibration_unavailable",
                "a frozen corpus manifest is required",
                503,
            )
        with repository() as current:
            batch = current.fetch_calibration_batch(batch_id)
        if batch is None:
            return _error(
                _request_id(), "unknown_calibration_batch", "calibration batch was not found", 404
            )
        return templates.TemplateResponse(
            request=request, name="calibration.html", context={"batch_id": batch_id, "data": batch}
        )

    @app.get("/api/v1/calibration/{batch_id}")
    def get_calibration(batch_id: str) -> JSONResponse:
        with repository() as current:
            batch = current.fetch_calibration_batch(batch_id)
        if batch is None:
            return _error(
                _request_id(), "unknown_calibration_batch", "calibration batch was not found", 404
            )
        return JSONResponse(content=ApiEnvelope(_request_id(), "ok", data=batch).to_dict())

    @app.post("/api/v1/calibration/{batch_id}/decisions")
    def save_calibration_decision(batch_id: str, body: dict[str, Any]) -> JSONResponse:
        try:
            label, reviewer, expected = validate_decision(
                body.get("music_level"), body.get("reviewer"), body.get("expected_revision")
            )
            note = body.get("note", "")
            if not isinstance(note, str):
                raise ValueError("note must be text")
            with repository() as current:
                decision = current.record_calibration_decision(
                    batch_id,
                    str(body.get("chunk_id", "")),
                    label=label,
                    reviewer=reviewer,
                    expected_revision=expected,
                    note=note,
                )
        except StorageConflictError as exc:
            return _error(_request_id(), "calibration_revision_conflict", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(_request_id(), "invalid_calibration_decision", str(exc), 422)
        return JSONResponse(content=ApiEnvelope(_request_id(), "ok", data=decision).to_dict())

    @app.get("/api/v1/calibration/{batch_id}/export")
    def export_calibration(batch_id: str) -> JSONResponse:
        with repository() as current:
            batch = current.fetch_calibration_batch(batch_id)
        if batch is None:
            return _error(
                _request_id(), "unknown_calibration_batch", "calibration batch was not found", 404
            )
        output = artifact_path / "calibration" / batch_id / "review.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"artifact_type": "p1r-development-music-review", "calibration": batch}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        temporary = output.with_suffix(".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(output)
        return JSONResponse(
            content=ApiEnvelope(
                _request_id(),
                "ok",
                data={
                    "path": str(output),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "reviewed_count": len(batch.get("decisions", {})),
                    "total_count": len(batch["chunks"]),
                },
            ).to_dict()
        )

    @app.get("/calibration/media/{batch_id}/{chunk_id:path}", response_model=None)
    def calibration_media(batch_id: str, chunk_id: str) -> FileResponse | JSONResponse:
        if frozen_manifest is None or source_root is None:
            return _error(
                _request_id(), "media_unavailable", "no frozen archive is configured", 404
            )
        with repository() as current:
            batch = current.fetch_calibration_batch(batch_id)
        raw = next(
            (item for item in (batch or {}).get("chunks", []) if item["chunk_id"] == chunk_id), None
        )
        if raw is None:
            return _error(
                _request_id(), "unknown_calibration_chunk", "chunk is not in this batch", 404
            )
        try:
            source = resolve_source(source_root, raw["episode_id"])
        except (CorpusValidationError, KeyError) as exc:
            return _error(_request_id(), "media_unavailable", str(exc), 404)
        return FileResponse(
            source,
            media_type=mimetypes.guess_type(source.name)[0] or "audio/ogg",
            filename=source.name,
        )

    @app.get("/chunk-review", response_class=HTMLResponse)
    def chunk_review_page(request: Request) -> HTMLResponse:
        """Review frozen chunk candidates and append human reference revisions."""

        return templates.TemplateResponse(request=request, name="chunk_review.html", context={})

    @app.get("/annotations/{episode_id:path}", response_class=HTMLResponse)
    def annotation_page(request: Request, episode_id: str) -> HTMLResponse:
        with repository() as current:
            annotation = current.fetch_annotation_revision(episode_id)
            history = _history(current, episode_id)
        initial = annotation or {"episode_id": episode_id, "status": "empty"}
        audio_url = (
            f"/media/{quote(episode_id, safe='')}" if frozen_manifest and source_root else ""
        )
        response = templates.TemplateResponse(
            request=request,
            name="annotation.html",
            context={
                "episode_id": episode_id,
                "initial_json": _safe_json(initial),
                "history_json": _safe_json(history),
                "audio_url": audio_url,
                "api_url": f"/api/v1/annotations/{quote(episode_id, safe='')}",
            },
        )
        return response

    @app.get("/api/v1/annotations/{episode_id}")
    def get_annotation(episode_id: str) -> JSONResponse:
        request_id = _request_id()
        try:
            current = SQLiteRepository.open(database_path)
            try:
                annotation = current.fetch_annotation_revision(episode_id)
                data = {
                    "episode_id": episode_id,
                    "annotation": annotation,
                    "history": _history(current, episode_id),
                }
            finally:
                current.close()
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.get("/api/v1/chunks/{chunk_id:path}/condition")
    def get_chunk_condition(chunk_id: str) -> JSONResponse:
        """Expose the AST seed and latest human correction separately."""

        request_id = _request_id()
        try:
            with repository() as current:
                chunk = current.fetch_audio_chunk(chunk_id)
                if chunk is None:
                    return _error(request_id, "unknown_chunk", "chunk was not found", 404)
                metadata = chunk.condition.acoustic_metadata if chunk.condition else None
                correction = current.fetch_acoustic_condition_correction(chunk_id)
                data = {
                    "chunk_id": chunk.chunk_id,
                    "episode_id": chunk.episode_id,
                    "machine_seed": metadata.to_dict() if metadata is not None else None,
                    "human_correction": correction.to_dict() if correction is not None else None,
                }
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.post("/api/v1/chunks/{chunk_id:path}/condition")
    def save_chunk_condition(chunk_id: str, body: dict[str, Any]) -> JSONResponse:
        """Append a human music/quality correction for one frozen chunk."""

        request_id = _request_id()
        reviewer = body.get("reviewer")
        try:
            with repository() as current:
                chunk = current.fetch_audio_chunk(chunk_id)
                if chunk is None:
                    return _error(request_id, "unknown_chunk", "chunk was not found", 404)
                metadata = chunk.condition.acoustic_metadata if chunk.condition else None
                if metadata is None:
                    return _error(
                        request_id,
                        "condition_unavailable",
                        "chunk has no AST acoustic machine seed",
                        409,
                    )
                correction = AcousticConditionCorrection(
                    music_level=body.get("music_level"),
                    audio_quality=body.get("audio_quality"),
                    reviewer=reviewer,
                    reviewed_at=body.get("reviewed_at")
                    or datetime.now(UTC).isoformat(timespec="seconds"),
                    reason=body.get("reason"),
                    seed_version=body.get("seed_version", metadata.version),
                )
                correction_id = current.record_acoustic_condition_correction(
                    chunk_id, correction, correction_id=body.get("correction_id")
                )
                current_chunk = current.fetch_audio_chunk(chunk_id)
                current_metadata = (
                    current_chunk.condition.acoustic_metadata
                    if current_chunk and current_chunk.condition
                    else metadata
                )
                data = {
                    "correction_id": correction_id,
                    "machine_seed": current_metadata.to_dict(),
                    "human_correction": correction.to_dict(),
                }
        except (StorageError, ValueError, TypeError, KeyError) as exc:
            return _error(request_id, "invalid_condition_correction", str(exc), 422)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.post("/api/v1/chunks/{chunk_id:path}/annotation-assistance")
    def annotation_assistance(chunk_id: str, body: dict[str, Any]) -> JSONResponse:
        """Create a non-authoritative current-chunk Qwen review draft."""

        request_id = _request_id()
        try:
            with repository() as current:
                chunk = current.fetch_audio_chunk(chunk_id)
                if chunk is None:
                    return _error(request_id, "unknown_chunk", "chunk was not found", 404)
                raw_ids = body.get("hypothesis_ids")
                if (
                    not isinstance(raw_ids, list)
                    or not raw_ids
                    or any(not isinstance(value, str) for value in raw_ids)
                ):
                    raise AnnotationAssistanceValidationError(
                        "hypothesis_ids must be a non-empty list of IDs"
                    )
                hypotheses = []
                for hypothesis_id in raw_ids:
                    hypothesis = current.fetch_transcription_hypothesis(hypothesis_id)
                    if hypothesis is None:
                        return _error(
                            request_id,
                            "unknown_hypothesis",
                            f"hypothesis was not found: {hypothesis_id}",
                            404,
                        )
                    hypotheses.append(hypothesis)
                preceding = body.get("preceding_chunks", ())
                if not isinstance(preceding, list):
                    raise AnnotationAssistanceValidationError("preceding_chunks must be a list")
                request = AnnotationAssistanceRequest.from_payload(chunk, hypotheses, preceding)
                draft = assistance_service.generate(current, request)
        except StorageConflictError as exc:
            return _error(request_id, "annotation_assistance_conflict", str(exc), 409)
        except AnnotationAssistanceValidationError as exc:
            return _error(request_id, "invalid_annotation_assistance", str(exc), 422)
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(
            content=ApiEnvelope(
                request_id,
                "ok",
                data={"draft": draft.to_dict(), "reference_unchanged": True},
            ).to_dict()
        )

    @app.post("/api/v1/annotations/{episode_id}")
    def save_annotation(episode_id: str, body: dict[str, Any]) -> JSONResponse:
        request_id = _request_id()
        expected_revision = body.get("expected_revision", 0)
        actor_type = body.get("actor_type", "human")
        reviewer = body.get("reviewer")
        payload = dict(body)
        for key in ("expected_revision", "actor_type", "reviewer"):
            payload.pop(key, None)
        try:
            current = SQLiteRepository.open(database_path)
            try:
                saved = save_annotation_revision(
                    current,
                    artifact_path,
                    episode_id=episode_id,
                    payload=payload,
                    expected_revision=expected_revision,
                    actor_type=actor_type,
                    reviewer=reviewer,
                )
                data = {
                    "annotation": saved.payload,
                    "export": saved.manifest.to_dict(),
                    "history": _history(current, episode_id),
                }
            finally:
                current.close()
        except StorageConflictError as exc:
            latest = None
            current = SQLiteRepository.open(database_path)
            try:
                latest = current.fetch_annotation_revision(episode_id)
            finally:
                current.close()
            return (
                _error(
                    request_id,
                    "annotation_revision_conflict",
                    f"{exc}; current annotation was preserved",
                    409,
                )
                if latest is None
                else JSONResponse(
                    status_code=409,
                    content=ApiEnvelope(
                        request_id,
                        "error",
                        data={"current": latest, "expected_revision": expected_revision},
                        error=ApiError(
                            "annotation_revision_conflict",
                            "current annotation was preserved; reload before saving",
                            False,
                            request_id,
                        ),
                    ).to_dict(),
                )
            )
        except ArtifactPublicationError as exc:
            return _error(request_id, "artifact_publication_failed", str(exc), 503)
        except (AnnotationValidationError, ValueError, KeyError) as exc:
            return _error(request_id, "invalid_annotation", str(exc), 422)
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.post("/api/v1/annotations/{episode_id:path}/seed")
    def seed_annotation(episode_id: str, body: dict[str, Any]) -> JSONResponse:
        request_id = _request_id()
        source = body.get("attribution", body.get("draft"))
        # Accept the raw P1-04 artifact too, matching the documented curl workflow.
        if source is None and ("diarization" in body or "data" in body):
            source = body
        expected_revision = body.get("expected_revision", 0)
        if not isinstance(source, Mapping):
            return _error(request_id, "invalid_annotation", "attribution draft is required", 422)
        try:
            payload = annotation_from_attribution(source)
            current = SQLiteRepository.open(database_path)
            try:
                saved = save_annotation_revision(
                    current,
                    artifact_path,
                    episode_id=episode_id,
                    payload=payload,
                    expected_revision=expected_revision,
                    actor_type="machine",
                    reviewer="machine",
                )
                data = {"annotation": saved.payload, "export": saved.manifest.to_dict()}
            finally:
                current.close()
        except StorageConflictError as exc:
            return _error(request_id, "annotation_revision_conflict", str(exc), 409)
        except ArtifactPublicationError as exc:
            return _error(request_id, "artifact_publication_failed", str(exc), 503)
        except (AnnotationValidationError, ValueError, KeyError) as exc:
            return _error(request_id, "invalid_annotation", str(exc), 422)
        except StorageError as exc:
            return _error(request_id, "storage_unavailable", str(exc), 503)
        return JSONResponse(content=ApiEnvelope(request_id, "ok", data=data).to_dict())

    @app.get("/api/v1/annotations/{episode_id}/exports/{revision}/{filename}", response_model=None)
    def annotation_export(
        episode_id: str, revision: int, filename: str
    ) -> FileResponse | JSONResponse:
        request_id = _request_id()
        if filename not in EXPORT_NAMES or revision <= 0:
            return _error(request_id, "unknown_export", "export is not registered", 404)
        current = SQLiteRepository.open(database_path)
        try:
            annotation = current.fetch_annotation_revision(episode_id, revision)
        finally:
            current.close()
        if annotation is None:
            return _error(
                request_id, "unknown_annotation", "annotation revision was not found", 404
            )
        directory = annotation_export_path(artifact_path, annotation)
        file_path = directory / filename
        if (
            not file_path.is_file()
            or file_path.is_symlink()
            or directory.resolve() not in file_path.resolve().parents
        ):
            return _error(request_id, "missing_export", "immutable export is not available", 404)
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return FileResponse(file_path, media_type=media_type, filename=filename)

    @app.get("/media/{episode_id:path}", response_model=None)
    def media(episode_id: str) -> FileResponse | JSONResponse:
        request_id = _request_id()
        if frozen_manifest is None or source_root is None:
            return _error(request_id, "media_unavailable", "no frozen archive is configured", 404)
        if not any(item.relative_filename == episode_id for item in frozen_manifest.episodes):
            return _error(
                request_id, "unknown_episode", "episode is not in the frozen manifest", 404
            )
        try:
            source = resolve_source(source_root, episode_id)
        except CorpusValidationError as exc:
            return _error(request_id, "unknown_episode", str(exc), 404)
        media_type = mimetypes.guess_type(source.name)[0] or "audio/ogg"
        return FileResponse(
            source, media_type=media_type, filename=source.name, content_disposition_type="inline"
        )

    return app


__all__ = ["create_app"]

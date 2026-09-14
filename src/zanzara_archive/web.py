"""Local annotation application.

The CLI keeps loopback binding as its default and requires an explicit network
opt-in for LAN access. The app has no authentication, and all media/export
paths are resolved from the frozen manifest or a revision-owned artifact name.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
from .contracts import AcousticConditionCorrection, ApiEnvelope, ApiError
from .corpus import CorpusValidationError, load_manifest, resolve_source
from .qwen_assistance import (
    AnnotationAssistanceRequest,
    AnnotationAssistanceService,
    AnnotationAssistanceValidationError,
)
from .storage import SQLiteRepository, StorageConflictError, StorageError

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
    artifact_path = Path(artifact_root).expanduser()
    source_root = Path(archive_root).expanduser() if archive_root is not None else None
    review_root = (
        Path(p1r03d_evidence_root).expanduser()
        if p1r03d_evidence_root is not None
        else artifact_path.parent / "zanzara-evidence" / "P1R-03D"
    )
    assistance_service = annotation_assistant or AnnotationAssistanceService()
    frozen_manifest = load_manifest(manifest_path) if manifest_path is not None else None
    if frozen_manifest is not None:
        current = SQLiteRepository.open(database_path)
        try:
            current.register_corpus_manifest(frozen_manifest)
        finally:
            current.close()

    @contextmanager
    def repository() -> Iterator[SQLiteRepository]:
        current = SQLiteRepository.open(database_path)
        try:
            yield current
        finally:
            current.close()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "access": "network-enabled" if network_access else "loopback-default",
        }

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
        return FileResponse(source, media_type=media_type, filename=source.name)

    return app


__all__ = ["create_app"]

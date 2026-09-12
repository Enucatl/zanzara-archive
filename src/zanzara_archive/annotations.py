"""Human annotation state, validation, and private reference exports.

Annotation payloads are deliberately JSON-native so the loopback editor and
the evaluation validator share one immutable representation.  Machine output
is always a draft; a reviewed payload must be saved by an identified human.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactManifest, ArtifactPublisher
from .contracts import TimedWord
from .storage import SQLiteRepository
from .transcripts import render_srt, render_txt, render_vtt

ANNOTATION_SCHEMA_VERSION = 1
ANNOTATION_STAGE = "annotation"
ANNOTATION_PIPELINE_VERSION = "p1-05"
MACHINE_REVIEWER = "machine"
ANNOTATION_STATUSES = frozenset({"draft", "reviewed"})


class AnnotationValidationError(ValueError):
    """Raised when an annotation cannot be used as canonical reference input."""


@dataclass(frozen=True, slots=True)
class AnnotationSave:
    """The persisted revision and its immutable private export manifest."""

    payload: dict[str, Any]
    manifest: ArtifactManifest


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: object) -> str:
    """Hash a JSON value without wall-clock or formatting variance."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationValidationError(f"{field_name} must be non-empty text")
    return value


def _sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AnnotationValidationError(f"{field_name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AnnotationValidationError(f"{field_name} must be a lowercase SHA-256") from exc
    if value != value.lower():
        raise AnnotationValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _integer(value: object, field_name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise AnnotationValidationError(f"{field_name} must be a {qualifier} integer")
    return value


def _interval(value: Mapping[str, Any], field_name: str, duration_ms: int) -> dict[str, Any]:
    start_ms = _integer(value.get("start_ms"), f"{field_name}.start_ms")
    end_ms = _integer(value.get("end_ms"), f"{field_name}.end_ms", positive=True)
    if start_ms >= end_ms or end_ms > duration_ms:
        raise AnnotationValidationError(f"{field_name} must be within the audio duration")
    return {"start_ms": start_ms, "end_ms": end_ms}


def _word(value: object, index: int, duration_ms: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"words[{index}] must be an object")
    word_id = _text(value.get("word_id"), f"words[{index}].word_id")
    text = value.get("text")
    if not isinstance(text, str):
        raise AnnotationValidationError(f"words[{index}].text must be text")
    interval = _interval(value, f"words[{index}]", duration_ms)
    speaker_id = value.get("speaker_id")
    if speaker_id is not None:
        speaker_id = _text(speaker_id, f"words[{index}].speaker_id")
    overlap = value.get("overlap", False)
    unintelligible = value.get("unintelligible", False)
    if not isinstance(overlap, bool) or not isinstance(unintelligible, bool):
        raise AnnotationValidationError(f"words[{index}] flags must be boolean")
    confidence = value.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
    ):
        raise AnnotationValidationError(f"words[{index}].confidence must be numeric or null")
    if confidence is not None and (
        not math.isfinite(float(confidence)) or not 0 <= confidence <= 1
    ):
        raise AnnotationValidationError(f"words[{index}].confidence must be between 0 and 1")
    return {
        "word_id": word_id,
        "text": text,
        **interval,
        "confidence": confidence,
        "speaker_id": speaker_id,
        "overlap": overlap,
        "unintelligible": unintelligible,
    }


def _turn(value: object, index: int, duration_ms: int, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"{label}[{index}] must be an object")
    speaker_id = _text(value.get("speaker_id"), f"{label}[{index}].speaker_id")
    turn_id = value.get("turn_id", f"{label}-{index}")
    turn_id = _text(turn_id, f"{label}[{index}].turn_id")
    return {
        "turn_id": turn_id,
        "speaker_id": speaker_id,
        **_interval(value, f"{label}[{index}]", duration_ms),
    }


def _overlap(value: object, index: int, duration_ms: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationValidationError(f"overlap_intervals[{index}] must be an object")
    speaker_ids = value.get("speaker_ids")
    if not isinstance(speaker_ids, Sequence) or isinstance(speaker_ids, (str, bytes)):
        raise AnnotationValidationError(f"overlap_intervals[{index}].speaker_ids must be a list")
    speakers = tuple(_text(item, f"overlap_intervals[{index}].speaker_ids") for item in speaker_ids)
    if len(speakers) < 2 or len(set(speakers)) != len(speakers):
        raise AnnotationValidationError(f"overlap_intervals[{index}] requires unique speakers")
    overlap = _interval(value, f"overlap_intervals[{index}]", duration_ms)
    return {
        "overlap_id": _text(value.get("overlap_id", f"overlap-{index}"), "overlap_id"),
        "speaker_ids": list(speakers),
        **overlap,
    }


def split_manifest(duration_ms: int, source_sha256: str) -> dict[str, Any]:
    """Build E1's five contiguous, deterministic time blocks."""

    boundaries = [(index * duration_ms) // 5 for index in range(6)]
    return {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "duration_ms": duration_ms,
        "boundaries_ms": boundaries,
        "blocks": [
            {
                "index": index,
                "start_ms": boundaries[index],
                "end_ms": boundaries[index + 1],
                "partition": "development" if index < 4 else "held_out",
            }
            for index in range(5)
        ],
    }


def _ids(value: object, field_name: str, allowed: set[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AnnotationValidationError(f"{field_name} must be a list")
    result = [_text(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise AnnotationValidationError(f"{field_name} must not contain duplicates")
    if not set(result) <= allowed:
        raise AnnotationValidationError(f"{field_name} contains an unknown word ID")
    return result


def validate_annotation_payload(
    payload: Mapping[str, Any],
    *,
    episode_id: str | None = None,
    actor_type: str | None = None,
    reviewer: str | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize an editable draft or reviewed reference."""

    if not isinstance(payload, Mapping):
        raise AnnotationValidationError("annotation payload must be an object")
    value = dict(payload)
    if value.get("schema_version", ANNOTATION_SCHEMA_VERSION) != ANNOTATION_SCHEMA_VERSION:
        raise AnnotationValidationError("annotation schema_version must be 1")
    resolved_episode = _text(episode_id or value.get("episode_id"), "episode_id")
    if value.get("episode_id", resolved_episode) != resolved_episode:
        raise AnnotationValidationError("annotation episode_id does not match the route")
    source_sha256 = _sha256(value.get("source_sha256"), "source_sha256")
    duration_ms = _integer(value.get("duration_ms"), "duration_ms", positive=True)
    status = value.get("status", "draft")
    if status == "accepted":
        status = "reviewed"
    if status not in ANNOTATION_STATUSES:
        raise AnnotationValidationError("status must be draft or reviewed")
    resolved_actor = actor_type or value.get("actor_type", "human")
    if resolved_actor not in {"machine", "human"}:
        raise AnnotationValidationError("actor_type must be machine or human")
    resolved_reviewer = reviewer or value.get("reviewer")
    if resolved_actor == "machine":
        resolved_reviewer = MACHINE_REVIEWER
    else:
        resolved_reviewer = _text(resolved_reviewer, "reviewer")
    if status == "reviewed" and resolved_actor != "human":
        raise AnnotationValidationError("machine output cannot mark an annotation reviewed")

    raw_words = value.get("words")
    if not isinstance(raw_words, Sequence) or isinstance(raw_words, (str, bytes)):
        raise AnnotationValidationError("words must be a list")
    words = [_word(item, index, duration_ms) for index, item in enumerate(raw_words)]
    word_ids = [item["word_id"] for item in words]
    if len(set(word_ids)) != len(word_ids):
        raise AnnotationValidationError("word IDs must be unique")

    def turns(label: str) -> list[dict[str, Any]]:
        raw = value.get(label, [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise AnnotationValidationError(f"{label} must be a list")
        return [_turn(item, index, duration_ms, label) for index, item in enumerate(raw)]

    standard_turns = turns("standard_turns")
    exclusive_turns = turns("exclusive_turns")
    raw_overlaps = value.get("overlap_intervals", value.get("overlaps", []))
    if not isinstance(raw_overlaps, Sequence) or isinstance(raw_overlaps, (str, bytes)):
        raise AnnotationValidationError("overlap_intervals must be a list")
    overlaps = [_overlap(item, index, duration_ms) for index, item in enumerate(raw_overlaps)]
    raw_unintelligible = value.get("unintelligible_spans", [])
    if not isinstance(raw_unintelligible, Sequence) or isinstance(raw_unintelligible, (str, bytes)):
        raise AnnotationValidationError("unintelligible_spans must be a list")
    unintelligible_spans = [
        _interval(item, f"unintelligible_spans[{index}]", duration_ms)
        for index, item in enumerate(raw_unintelligible)
        if isinstance(item, Mapping)
    ]
    if len(unintelligible_spans) != len(raw_unintelligible):
        raise AnnotationValidationError("unintelligible spans must be objects")

    allowed_ids = set(word_ids)
    reviewed_word_ids = _ids(value.get("reviewed_word_ids"), "reviewed_word_ids", allowed_ids)
    manual_timing_word_ids = _ids(
        value.get("manual_timing_word_ids"), "manual_timing_word_ids", allowed_ids
    )
    if status == "reviewed" and set(reviewed_word_ids) != allowed_ids:
        raise AnnotationValidationError("reviewed references must cover every transcript word")
    provenance = value.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise AnnotationValidationError("provenance must be an object")
    split = value.get("split")
    expected_split = split_manifest(duration_ms, source_sha256)
    if split is not None and split != expected_split:
        raise AnnotationValidationError("split manifest does not match the source duration")
    normalized: dict[str, Any] = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "episode_id": resolved_episode,
        "source_sha256": source_sha256,
        "source_artifact_id": value.get("source_artifact_id"),
        "duration_ms": duration_ms,
        "time_origin_ms": 0,
        "status": status,
        "actor_type": resolved_actor,
        "reviewer": resolved_reviewer,
        "reviewed_at": value.get("reviewed_at") if status == "reviewed" else None,
        "words": words,
        "standard_turns": standard_turns,
        "exclusive_turns": exclusive_turns,
        "overlap_intervals": overlaps,
        "unintelligible_spans": unintelligible_spans,
        "reviewed_word_ids": reviewed_word_ids,
        "manual_timing_word_ids": manual_timing_word_ids,
        "provenance": dict(provenance),
        "split": expected_split,
        "split_sha256": canonical_hash(expected_split),
    }
    if normalized["source_artifact_id"] is not None:
        normalized["source_artifact_id"] = _text(
            normalized["source_artifact_id"], "source_artifact_id"
        )
    configuration = normalized["provenance"].get("configuration", {})
    normalized["configuration_sha256"] = canonical_hash(configuration)
    return normalized


def annotation_from_attribution(
    value: Mapping[str, Any], *, actor_type: str = "machine"
) -> dict[str, Any]:
    """Convert a P1-04 attributed payload into a machine-owned draft."""

    data = value.get("data", value)
    if not isinstance(data, Mapping):
        raise AnnotationValidationError("attribution payload must contain an object")
    diarization = data.get("diarization")
    if not isinstance(diarization, Mapping):
        raise AnnotationValidationError("attribution payload is missing diarization")
    return validate_annotation_payload(
        {
            "episode_id": data.get("episode_id"),
            "source_sha256": data.get("source_sha256"),
            "source_artifact_id": data.get("artifact_id"),
            "duration_ms": data.get("duration_ms"),
            "words": data.get("words", []),
            "standard_turns": diarization.get("standard_turns", []),
            "exclusive_turns": diarization.get("exclusive_turns", []),
            "overlap_intervals": diarization.get("overlaps", []),
            "provenance": data.get("provenance", {}),
            "status": "draft",
            "actor_type": actor_type,
        },
        actor_type=actor_type,
        reviewer=MACHINE_REVIEWER,
    )


def _timed_words(payload: Mapping[str, Any]) -> tuple[TimedWord, ...]:
    return tuple(
        TimedWord(
            word_id=word["word_id"],
            text=word["text"] or "[UNINTELLIGIBLE]",
            start_ms=word["start_ms"],
            end_ms=word["end_ms"],
            confidence=word.get("confidence"),
            speaker_id=word.get("speaker_id"),
            overlap=word.get("overlap", False),
        )
        for word in payload["words"]
    )


def annotation_export_files(payload: Mapping[str, Any]) -> dict[str, bytes]:
    """Return the complete private reference/export set for one revision."""

    words = _timed_words(payload)
    reference = _canonical_json(payload).encode("utf-8")
    manual_timing = {
        "schema_version": 1,
        "annotation_revision_id": payload.get("annotation_revision_id"),
        "source_sha256": payload["source_sha256"],
        "word_ids": payload["manual_timing_word_ids"],
        "words": [
            {"word_id": word["word_id"], "start_ms": word["start_ms"], "end_ms": word["end_ms"]}
            for word in payload["words"]
            if word["word_id"] in set(payload["manual_timing_word_ids"])
        ],
    }
    split = payload["split"]
    return {
        "reference.json": reference,
        "annotation.json": reference,
        "manual_timing.json": _canonical_json(manual_timing).encode("utf-8"),
        "split.json": _canonical_json(split).encode("utf-8"),
        "transcript.txt": render_txt(words).encode("utf-8"),
        "transcript.srt": render_srt(words).encode("utf-8"),
        "transcript.vtt": render_vtt(words).encode("utf-8"),
    }


def annotation_export_path(artifact_root: str | Path, payload: Mapping[str, Any]) -> Path:
    """Resolve a revision's deterministic private artifact directory."""

    digest = canonical_hash(payload)
    revision = int(payload["revision"])
    return ArtifactPublisher(artifact_root).artifact_path(
        payload["source_sha256"], ANNOTATION_STAGE, f"revision-{revision}-{digest[:24]}"
    )


def publish_annotation_export(
    artifact_root: str | Path, payload: Mapping[str, Any]
) -> ArtifactManifest:
    """Atomically publish one private draft/reviewed revision."""

    normalized = validate_annotation_payload(payload, episode_id=payload.get("episode_id"))
    normalized.update(
        {
            key: payload[key]
            for key in ("annotation_revision_id", "revision", "created_at", "based_on_revision")
            if key in payload
        }
    )
    digest = canonical_hash(normalized)
    stage_key = f"revision-{normalized['revision']}-{digest[:24]}"
    provenance = dict(normalized["provenance"])
    provenance.update(
        {
            "annotation_revision_id": normalized.get("annotation_revision_id"),
            "revision": normalized["revision"],
            "status": normalized["status"],
            "reviewer": normalized["reviewer"],
            "source_sha256": normalized["source_sha256"],
            "split_sha256": normalized["split_sha256"],
            "configuration_sha256": normalized["configuration_sha256"],
        }
    )
    model_hash = provenance.get("model_fingerprint_sha256")
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        model_hash = None
    return ArtifactPublisher(artifact_root).publish(
        source_sha256=normalized["source_sha256"],
        stage=ANNOTATION_STAGE,
        stage_key=stage_key,
        files=annotation_export_files(normalized),
        upstream_artifact_hashes=(),
        pipeline_version=ANNOTATION_PIPELINE_VERSION,
        artifact_id=f"annotation-{digest[:24]}",
        model_fingerprint_sha256=model_hash,
        provenance=provenance,
    )


def save_annotation_revision(
    repository: SQLiteRepository,
    artifact_root: str | Path,
    *,
    episode_id: str,
    payload: Mapping[str, Any],
    expected_revision: int,
    actor_type: str,
    reviewer: str | None = None,
) -> AnnotationSave:
    """Validate, append, and export one revision with optimistic concurrency."""

    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise AnnotationValidationError("expected_revision must be a non-negative integer")
    normalized = validate_annotation_payload(
        payload, episode_id=episode_id, actor_type=actor_type, reviewer=reviewer
    )
    episode = repository.connection.execute(
        "SELECT source_sha256, duration_ms FROM episodes WHERE episode_id = ?",
        (episode_id,),
    ).fetchone()
    if episode is None:
        raise AnnotationValidationError(f"episode is not registered: {episode_id}")
    if normalized["source_sha256"] != episode["source_sha256"]:
        raise AnnotationValidationError("annotation source_sha256 does not match the episode")
    if normalized["duration_ms"] != episode["duration_ms"]:
        raise AnnotationValidationError("annotation duration_ms does not match the episode")
    db_source_artifact_id = normalized.get("source_artifact_id")
    if db_source_artifact_id and not repository.artifact_exists(db_source_artifact_id):
        db_source_artifact_id = None
    persisted = repository.append_annotation_revision(
        episode_id=episode_id,
        source_artifact_id=db_source_artifact_id,
        source_sha256=normalized["source_sha256"],
        reviewer=normalized["reviewer"],
        status="accepted" if normalized["status"] == "reviewed" else "draft",
        payload=normalized,
        expected_revision=expected_revision,
    )
    manifest = publish_annotation_export(artifact_root, persisted)
    return AnnotationSave(payload=persisted, manifest=manifest)


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "AnnotationSave",
    "AnnotationValidationError",
    "annotation_export_files",
    "annotation_export_path",
    "annotation_from_attribution",
    "canonical_hash",
    "publish_annotation_export",
    "save_annotation_revision",
    "split_manifest",
    "validate_annotation_payload",
]

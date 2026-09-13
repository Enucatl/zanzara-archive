"""Development-only music calibration batch validation and persistence helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .contracts import AudioChunk, ContractValidationError
from .corpus import CorpusManifest

MUSIC_LEVELS = ("none", "background", "dominant", "uncertain")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_batch(payload: Mapping[str, Any], corpus: CorpusManifest) -> dict[str, Any]:
    """Validate a private development-only batch and return normalized JSON."""

    required = {"batch_id", "manifest_sha256", "partition", "segmentation_version", "chunks"}
    missing = required - payload.keys()
    if missing:
        raise ContractValidationError(
            f"calibration batch missing keys: {', '.join(sorted(missing))}"
        )
    if payload["manifest_sha256"] != corpus.sha256:
        raise ContractValidationError("calibration batch manifest hash does not match corpus")
    if payload["partition"] != "development":
        raise ContractValidationError("calibration batch must use the development partition")
    if (
        not isinstance(payload["segmentation_version"], str)
        or not payload["segmentation_version"].strip()
    ):
        raise ContractValidationError("calibration batch segmentation_version must be non-empty")
    raw_chunks = payload["chunks"]
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ContractValidationError("calibration batch requires at least one chunk")
    episodes = {episode.relative_filename: episode for episode in corpus.episodes}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_chunks):
        if not isinstance(raw, Mapping):
            raise ContractValidationError(f"calibration chunks[{index}] must be an object")
        chunk = AudioChunk.from_dict(raw)
        if chunk.chunk_id in seen:
            raise ContractValidationError(f"calibration chunk {chunk.chunk_id} is duplicated")
        seen.add(chunk.chunk_id)
        episode = episodes.get(chunk.episode_id)
        if episode is None:
            raise ContractValidationError(f"calibration chunk {chunk.chunk_id} has unknown episode")
        if chunk.source_sha256 != episode.sha256:
            raise ContractValidationError(
                f"calibration chunk {chunk.chunk_id} source hash mismatch"
            )
        if chunk.end_ms > episode.duration_ms:
            raise ContractValidationError(
                f"calibration chunk {chunk.chunk_id} exceeds episode duration"
            )
        if chunk.partition != "development":
            raise ContractValidationError(f"calibration chunk {chunk.chunk_id} is not development")
        normalized.append(chunk.to_dict())
    result = {
        "batch_id": payload["batch_id"],
        "manifest_sha256": corpus.sha256,
        "partition": "development",
        "segmentation_version": payload["segmentation_version"],
        "chunks": normalized,
    }
    result["content_sha256"] = content_sha256(result)
    return result


def validate_decision(
    label: object, reviewer: object, expected_revision: object
) -> tuple[str, str, int]:
    if label not in MUSIC_LEVELS:
        raise ContractValidationError(
            "music_level must be none, background, dominant, or uncertain"
        )
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ContractValidationError("reviewer must be identified human text")
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise ContractValidationError("expected_revision must be a non-negative integer")
    return str(label), reviewer.strip(), expected_revision

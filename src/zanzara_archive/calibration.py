"""Development-only music calibration batch validation and persistence helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .chunking import ChunkSegmentationConfig, segment_chunks_with_metadata
from .contracts import AudioChunk, ContractValidationError
from .corpus import CorpusManifest

MUSIC_LEVELS = ("none", "background", "dominant", "uncertain")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_batch(
    corpus: CorpusManifest,
    split: Mapping[str, Any],
    *,
    batch_id: str,
    clips_per_episode: int = 8,
) -> dict[str, Any]:
    """Build a deterministic, model-independent development review batch."""

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ContractValidationError("calibration batch_id must be non-empty")
    if (
        isinstance(clips_per_episode, bool)
        or not isinstance(clips_per_episode, int)
        or clips_per_episode <= 0
    ):
        raise ContractValidationError("clips_per_episode must be a positive integer")
    development = split.get("development")
    held_out = split.get("held_out")
    if not isinstance(development, list) or not isinstance(held_out, list):
        raise ContractValidationError("split must contain development and held_out episode lists")
    expected = {episode.relative_filename for episode in corpus.episodes}
    development_set, held_out_set = set(development), set(held_out)
    if development_set | held_out_set != expected or development_set & held_out_set:
        raise ContractValidationError("split must assign every corpus episode exactly once")
    config = ChunkSegmentationConfig()
    chunks: list[dict[str, Any]] = []
    by_name = {episode.relative_filename: episode for episode in corpus.episodes}
    for episode_name in development:
        episode = by_name[episode_name]
        result = segment_chunks_with_metadata(
            episode.relative_filename,
            episode.sha256,
            episode.duration_ms,
            config=config,
            partition="development",
        )
        selected = result.chunks
        if len(selected) > clips_per_episode:
            positions = (
                [
                    round(i * (len(selected) - 1) / (clips_per_episode - 1))
                    for i in range(clips_per_episode)
                ]
                if clips_per_episode > 1
                else [len(selected) // 2]
            )
            selected = tuple(selected[position] for position in positions)
        chunks.extend(chunk.to_dict() for chunk in selected)
    normalized_split: dict[str, Any] = {"development": development, "held_out": held_out}
    for key in ("method", "seed"):
        if key in split:
            normalized_split[key] = split[key]
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "manifest_sha256": corpus.sha256,
        "partition": "development",
        "segmentation_version": config.version,
        "split": normalized_split,
        "chunks": chunks,
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def validate_batch(payload: Mapping[str, Any], corpus: CorpusManifest) -> dict[str, Any]:
    """Validate a private development-only batch and return normalized JSON."""

    required = {
        "batch_id",
        "manifest_sha256",
        "partition",
        "segmentation_version",
        "chunks",
        "split",
    }
    missing = required - payload.keys()
    if missing:
        raise ContractValidationError(
            f"calibration batch missing keys: {', '.join(sorted(missing))}"
        )
    if payload["manifest_sha256"] != corpus.sha256:
        raise ContractValidationError("calibration batch manifest hash does not match corpus")
    if payload["partition"] != "development":
        raise ContractValidationError("calibration batch must use the development partition")
    split = payload["split"]
    if not isinstance(split, Mapping):
        raise ContractValidationError("calibration batch split must be an object")
    development = split.get("development")
    held_out = split.get("held_out")
    expected_episodes = {episode.relative_filename for episode in corpus.episodes}
    if not isinstance(development, list) or not isinstance(held_out, list):
        raise ContractValidationError("calibration batch split must contain episode lists")
    if set(development) | set(held_out) != expected_episodes or set(development) & set(held_out):
        raise ContractValidationError(
            "calibration batch split must assign every episode exactly once"
        )
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
    normalized_split: dict[str, Any] = {"development": development, "held_out": held_out}
    for key in ("method", "seed"):
        if key in split:
            normalized_split[key] = split[key]
    result = {
        "batch_id": payload["batch_id"],
        "manifest_sha256": corpus.sha256,
        "partition": "development",
        "segmentation_version": payload["segmentation_version"],
        "split": normalized_split,
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

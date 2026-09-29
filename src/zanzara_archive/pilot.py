"""Freeze the small ASR pilot from the accepted native chunk manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import AudioChunk, ChunkBenchmarkManifest
from .storage import SQLiteRepository


def _digest(value: object) -> str:
    """Hash JSON without depending on whitespace or key order."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _difficulty(chunk: AudioChunk, queue: dict[str, dict], reviews: dict[str, str]) -> str:
    """Name the strongest non-ASR evidence of a difficult interval."""

    if reviews.get(chunk.chunk_id) in ("bad", "awkward"):
        return "human_boundary_" + reviews[chunk.chunk_id]
    row = queue.get(chunk.chunk_id, {})
    if chunk.condition is not None:
        if chunk.condition.degraded:
            return "chunk_condition_degraded"
        if chunk.condition.music:
            return "chunk_condition_music"
        if chunk.condition.rapid_turn_taking:
            return "chunk_condition_rapid_turn"
        if chunk.condition.overlaps:
            return "chunk_condition_overlap"
    if "music_containing" in row.get("categories", ()):
        return "review_queue_music"
    if "overlap_containing" in row.get("categories", ()):
        return "review_queue_overlap"
    if chunk.boundary_overlap_conflict:
        return "boundary_overlap_conflict"
    if chunk.boundary_end_reason == "hard_maximum":
        return "hard_maximum"
    return "no_difficulty_tag"


def _spread(chunks: list[AudioChunk], count: int) -> list[AudioChunk]:
    """Choose evenly spaced source times without model outputs."""

    ordered = sorted(chunks, key=lambda chunk: (chunk.start_ms, chunk.chunk_id))
    if len(ordered) < count:
        raise ValueError("insufficient chunks for representative coverage")
    return [ordered[(2 * index + 1) * len(ordered) // (2 * count)] for index in range(count)]


def freeze_pilot(
    source: dict[str, Any],
    split: dict[str, Any],
    boundary: dict[str, Any] | None = None,
    reviews: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Select 40 chunks per frozen episode partition and explain each choice."""

    if source.get("algorithm") != "community1-native-adaptive-v2":
        raise ValueError("pilot requires accepted native chunking")
    if not isinstance(source.get("content_sha256"), str) or len(source["content_sha256"]) != 64:
        raise ValueError("native chunk manifest content identity is missing")
    by_episode: dict[str, list[AudioChunk]] = defaultdict(list)
    for raw in source["chunks"]:
        chunk = AudioChunk.from_dict(raw)
        by_episode[chunk.episode_id].append(chunk)
    sets = {name: tuple(sorted(split[name])) for name in ("development", "held_out")}
    if any(len(episodes) < 2 for episodes in sets.values()):
        raise ValueError("each pilot partition needs at least two episodes")
    if set(sets["development"]) & set(sets["held_out"]):
        raise ValueError("episode partitions overlap")
    if set(sets["development"] + sets["held_out"]) != set(by_episode):
        raise ValueError("frozen episode split does not cover the native corpus")
    queue = {row["chunk_id"]: row for row in (boundary or {}).get("rows", ())}
    reviews = reviews or {}
    selected: list[tuple[AudioChunk, str, str, str]] = []
    coverage: dict[str, dict[str, int]] = {}
    for partition, episodes in sets.items():
        quotas = {
            episode: 40 // len(episodes) + (index < 40 % len(episodes))
            for index, episode in enumerate(episodes)
        }
        difficult_targets = {
            episode: 10 // len(episodes) + (index < 10 % len(episodes))
            for index, episode in enumerate(episodes)
        }
        difficult_count = 0
        for episode in episodes:
            candidates = by_episode[episode]
            if len(candidates) < quotas[episode]:
                raise ValueError(f"insufficient native chunks for {episode}")
            ranked = sorted(
                candidates,
                key=lambda chunk: (
                    (
                        "human_boundary_bad",
                        "human_boundary_awkward",
                        "chunk_condition_degraded",
                        "chunk_condition_music",
                        "chunk_condition_rapid_turn",
                        "chunk_condition_overlap",
                        "review_queue_music",
                        "review_queue_overlap",
                        "boundary_overlap_conflict",
                        "hard_maximum",
                        "no_difficulty_tag",
                    ).index(_difficulty(chunk, queue, reviews)),
                    chunk.start_ms,
                    chunk.chunk_id,
                ),
            )
            difficult = ranked[: difficult_targets[episode]]
            difficult_ids = {chunk.chunk_id for chunk in difficult}
            remaining = [chunk for chunk in candidates if chunk.chunk_id not in difficult_ids]
            neutral = [
                chunk
                for chunk in remaining
                if _difficulty(chunk, queue, reviews) == "no_difficulty_tag"
            ]
            representatives = _spread(
                neutral if len(neutral) >= quotas[episode] - len(difficult) else remaining,
                quotas[episode] - len(difficult),
            )
            selected.extend(
                (
                    replace(chunk, partition=partition),
                    partition,
                    "difficult",
                    _difficulty(chunk, queue, reviews),
                )
                for chunk in difficult
            )
            selected.extend(
                (
                    replace(chunk, partition=partition),
                    partition,
                    "representative",
                    "temporal_spread",
                )
                for chunk in representatives
            )
            difficult_count += len(difficult)
        if sum(quotas.values()) != 40 or difficult_count != 10:
            raise ValueError("pilot partition quota was not met")
        rows = [entry for entry in selected if entry[1] == partition]
        candidates_in_partition = [chunk for episode in episodes for chunk in by_episode[episode]]
        coverage[partition] = {
            "episodes": len(episodes),
            "chunks": len(rows),
            "difficult": difficult_count,
            "representative": len(rows) - difficult_count,
            "music_tagged": sum(
                "music_containing" in queue.get(chunk.chunk_id, {}).get("categories", ())
                or (chunk.condition is not None and chunk.condition.music)
                for chunk, _, _, _ in rows
            ),
            "overlap_tagged": sum(
                "overlap_containing" in queue.get(chunk.chunk_id, {}).get("categories", ())
                or (chunk.condition is not None and bool(chunk.condition.overlaps))
                for chunk, _, _, _ in rows
            ),
            "degraded_tagged": sum(
                chunk.condition is not None and chunk.condition.degraded for chunk, _, _, _ in rows
            ),
            "rapid_turn_tagged": sum(
                chunk.condition is not None and chunk.condition.rapid_turn_taking
                for chunk, _, _, _ in rows
            ),
            "boundary_overlap_conflict_selected": sum(
                chunk.boundary_overlap_conflict for chunk, _, _, _ in rows
            ),
            "untagged_difficult": sum(
                category == "difficult" and reason == "no_difficulty_tag"
                for _, _, category, reason in rows
            ),
            "tagged_representative_fallback": sum(
                category == "representative"
                and _difficulty(chunk, queue, reviews) != "no_difficulty_tag"
                for chunk, _, category, _ in rows
            ),
            "source_tag_availability": {
                "music": sum(
                    "music_containing" in queue.get(chunk.chunk_id, {}).get("categories", ())
                    or (chunk.condition is not None and chunk.condition.music)
                    for chunk in candidates_in_partition
                ),
                "overlap": sum(
                    "overlap_containing" in queue.get(chunk.chunk_id, {}).get("categories", ())
                    or (chunk.condition is not None and bool(chunk.condition.overlaps))
                    for chunk in candidates_in_partition
                ),
                "boundary_overlap_conflict": sum(
                    chunk.boundary_overlap_conflict for chunk in candidates_in_partition
                ),
                "degraded": sum(
                    chunk.condition is not None and chunk.condition.degraded
                    for chunk in candidates_in_partition
                ),
                "rapid_turn": sum(
                    chunk.condition is not None and chunk.condition.rapid_turn_taking
                    for chunk in candidates_in_partition
                ),
            },
        }
    selected.sort(key=lambda entry: (entry[1], entry[0].episode_id, entry[0].start_ms))
    chunks = tuple(entry[0] for entry in selected)
    identity = _digest(
        [source["content_sha256"], sets, [(chunk.chunk_id, chunk.partition) for chunk in chunks]]
    )
    manifest = ChunkBenchmarkManifest(
        manifest_id="p1r11-" + identity[:24],
        chunks=chunks,
        source_manifest_sha256=source["corpus_manifest_sha256"],
        partitions={
            name: tuple(chunk.chunk_id for chunk in chunks if chunk.partition == name)
            for name in sets
        },
        segmentation_version=source["segmentation_version"],
        segmentation_configuration=source["segmentation_configuration"],
        segmentation_input_fingerprints=(source["segmentation_configuration_hash"],),
    )
    return {
        "schema_version": 1,
        "source": {
            "content_sha256": source["content_sha256"],
            "algorithm": source["algorithm"],
        },
        "split": split,
        "selection": {
            "policy": "p1r11-native-episode-stratified-v1",
            "coverage": coverage,
            "rows": [
                {
                    "chunk_id": chunk.chunk_id,
                    "episode_id": chunk.episode_id,
                    "partition": partition,
                    "category": category,
                    "reason": reason,
                    "evidence_source": (
                        "human_boundary_review"
                        if reason.startswith("human_boundary_")
                        else "chunk_condition"
                        if reason.startswith("chunk_condition_")
                        else "boundary_review_queue"
                        if reason.startswith("review_queue_")
                        else "native_chunk_structure"
                        if reason in ("boundary_overlap_conflict", "hard_maximum")
                        else "sampling_policy"
                    ),
                    "start_ms": chunk.start_ms,
                    "end_ms": chunk.end_ms,
                }
                for chunk, partition, category, reason in selected
            ],
        },
        "manifest": manifest.to_dict(),
    }


def publish_pilot(wrapper: dict[str, Any], database: Path, output: Path) -> None:
    """Record immutable chunks, then atomically publish the private selection."""

    manifest = ChunkBenchmarkManifest.from_dict(wrapper["manifest"])
    if output.exists() and json.loads(output.read_text()) != wrapper:
        raise ValueError("frozen pilot output already has different membership")
    with closing(SQLiteRepository.open(database)) as repository:
        repository.record_chunk_benchmark_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(wrapper, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    """Freeze the pilot using private local provenance and the canonical DB."""

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "split", "boundary", "database", "output"):
        parser.add_argument("--" + name, type=Path, required=name != "boundary")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    split = json.loads(args.split.read_text())
    boundary = json.loads(args.boundary.read_text()) if args.boundary else None
    with sqlite3.connect(f"file:{args.database}?mode=ro", uri=True) as connection:
        reviews = {
            chunk_id: decision
            for chunk_id, decision in connection.execute(
                "SELECT item_id, decision FROM p1r03d_reviews "
                "WHERE review_type = 'boundary' AND state = 'active' "
                "ORDER BY item_id, revision"
            )
        }
    wrapper = freeze_pilot(source, split, boundary, reviews)
    publish_pilot(wrapper, args.database, args.output)
    print(json.dumps({"manifest_id": wrapper["manifest"]["manifest_id"], "path": str(args.output)}))


if __name__ == "__main__":
    main()

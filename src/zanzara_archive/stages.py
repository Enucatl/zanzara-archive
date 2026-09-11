"""Deterministic stage fingerprints and the archive invalidation graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import ContractValidationError

# Direct upstream dependencies from D4.  The order is part of the public
# fingerprint contract and is intentionally stable.
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "manifest": (),
    "decode": ("manifest",),
    "asr": ("decode",),
    "diarization": ("decode",),
    "attribution": ("asr", "diarization"),
    "exemplars": ("diarization",),
    "speaker_embeddings_resnet293": ("exemplars",),
    "speaker_embeddings_eres2net": ("exemplars",),
    "speaker_embeddings_wavlm": ("exemplars",),
    "voice_indexes": (
        "speaker_embeddings_resnet293",
        "speaker_embeddings_eres2net",
        "speaker_embeddings_wavlm",
    ),
    "candidates": ("voice_indexes",),
    "identity_decisions": ("candidates",),
    "chunks": ("attribution",),
    "text_index": ("chunks",),
}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"stage configuration is not JSON serializable: {exc}"
        ) from exc


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractValidationError(f"{field} must be a lowercase SHA-256")
    return value


def stage_fingerprint(
    stage: str,
    *,
    source_sha256: str,
    upstream_artifact_hashes: Sequence[str] = (),
    model_fingerprint_sha256: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    pipeline_version: str = "dev",
) -> str:
    """Return the immutable key for one stage invocation.

    Source and upstream hash order are preserved.  Configuration is canonicalized
    recursively so equivalent mappings produce the same key, and wall-clock data
    must not be included by callers.
    """

    if not stage or not isinstance(stage, str):
        raise ContractValidationError("stage must be non-empty text")
    _sha256(source_sha256, "source_sha256")
    for index, digest in enumerate(upstream_artifact_hashes):
        _sha256(digest, f"upstream_artifact_hashes[{index}]")
    if model_fingerprint_sha256 is not None:
        _sha256(model_fingerprint_sha256, "model_fingerprint_sha256")
    if not pipeline_version or not isinstance(pipeline_version, str):
        raise ContractValidationError("pipeline_version must be non-empty text")
    payload = {
        "configuration": json.loads(_canonical_json(configuration or {})),
        "model_fingerprint_sha256": model_fingerprint_sha256,
        "pipeline_version": pipeline_version,
        "source_sha256": source_sha256,
        "stage": stage,
        "upstream_artifact_hashes": list(upstream_artifact_hashes),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


compute_stage_key = stage_fingerprint
stage_key = stage_fingerprint


def dependencies_for(stage: str) -> tuple[str, ...]:
    """Return direct D4 dependencies for *stage*."""

    try:
        return STAGE_DEPENDENCIES[stage]
    except KeyError as exc:
        raise ContractValidationError(f"unknown archive stage: {stage}") from exc


def downstream_stages(stage: str) -> tuple[str, ...]:
    """Return all transitive dependants in deterministic graph order."""

    dependencies_for(stage)
    discovered: set[str] = set()
    pending = [stage]
    while pending:
        current = pending.pop(0)
        for candidate, dependencies in STAGE_DEPENDENCIES.items():
            if current in dependencies and candidate not in discovered:
                discovered.add(candidate)
                pending.append(candidate)
    return tuple(candidate for candidate in STAGE_DEPENDENCIES if candidate in discovered)


def invalidated_stages(changed_stages: Sequence[str]) -> tuple[str, ...]:
    """Return changed stages and every stage that must be recomputed."""

    result: set[str] = set()
    for stage in changed_stages:
        dependencies_for(stage)
        result.add(stage)
        result.update(downstream_stages(stage))
    return tuple(stage for stage in STAGE_DEPENDENCIES if stage in result)


def reusable_stages(
    changed_stages: Sequence[str], available_stages: Sequence[str]
) -> tuple[str, ...]:
    """Return available stages unaffected by the supplied changes."""

    invalidated = set(invalidated_stages(changed_stages))
    return tuple(stage for stage in available_stages if stage not in invalidated)

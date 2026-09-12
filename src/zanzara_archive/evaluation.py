"""CPU-only validation of the private P1-05 reference contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .annotations import (
    AnnotationValidationError,
    canonical_hash,
    split_manifest,
    validate_annotation_payload,
)
from .corpus import CorpusValidationError, load_manifest
from .evaluation_metrics import (
    CANONICAL_TIME_UNIT,
    DEFAULT_THRESHOLDS,
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS_VERSION,
    NORMALIZATION_VERSION,
    QUANTILE_VERSION,
    EvaluationDocument,
    EvaluationValidationError,
    current_git_commit,
    evaluate_documents,
    evaluate_files,
    normalize_italian,
    parse_evaluation_document,
    write_evaluation_artifacts,
)


class ReferenceValidationError(ValueError):
    """Raised when a reference file cannot be read as JSON."""


def _reference_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        for name in ("reference.json", "annotation.json", "reviewed-reference.json"):
            selected = candidate / name
            if selected.is_file():
                return selected
        raise ReferenceValidationError(
            f"reference directory does not contain reference.json or annotation.json: {candidate}"
        )
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceValidationError(f"cannot read reference JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReferenceValidationError("reference JSON must be an object")
    return payload


def validate_reference(corpus_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    """Validate structure and timing coverage without validating human judgment."""

    errors: list[str] = []
    try:
        corpus = load_manifest(corpus_path)
    except CorpusValidationError as exc:
        return {"valid": False, "errors": [str(exc)], "human_judgment_validated": False}
    try:
        source_path = _reference_file(reference_path)
        raw = _load_json(source_path)
    except ReferenceValidationError as exc:
        return {"valid": False, "errors": [str(exc)], "human_judgment_validated": False}

    required_fields = {
        "schema_version",
        "episode_id",
        "source_sha256",
        "duration_ms",
        "status",
        "actor_type",
        "reviewer",
        "reviewed_at",
        "words",
        "split",
        "split_sha256",
        "provenance",
    }
    errors = [
        f"reference is missing required field: {field}"
        for field in sorted(required_fields - raw.keys())
    ]
    try:
        payload = validate_annotation_payload(raw, episode_id=raw.get("episode_id"))
    except AnnotationValidationError as exc:
        return {"valid": False, "errors": [str(exc)], "human_judgment_validated": False}

    golden = next(
        episode for episode in corpus.episodes if episode.relative_filename == corpus.golden_episode
    )
    if payload["episode_id"] != corpus.golden_episode:
        errors.append("reference episode_id must be the manifest golden_episode")
    if payload["source_sha256"] != golden.sha256:
        errors.append("reference source_sha256 does not match the frozen golden source")
    if payload["duration_ms"] != golden.duration_ms:
        errors.append("reference duration_ms does not match the frozen golden source")
    if payload["status"] != "reviewed" or payload["actor_type"] != "human":
        errors.append("reference must be an identified human-reviewed revision")
    if not payload["reviewer"] or payload["reviewer"] == "machine":
        errors.append("reference reviewer must be identified and non-machine")
    if not payload.get("reviewed_at"):
        errors.append("reference reviewed_at must identify when human review was recorded")

    expected_split = split_manifest(golden.duration_ms, golden.sha256)
    if payload["split"] != expected_split:
        errors.append("reference split does not contain E1's five contiguous bounds")
    if payload["split_sha256"] != canonical_hash(expected_split):
        errors.append("reference split_sha256 does not match the split manifest")

    word_ids = {word["word_id"] for word in payload["words"]}
    reviewed_ids = set(payload["reviewed_word_ids"])
    if reviewed_ids != word_ids:
        errors.append("reference does not cover every transcript word")
    timing_ids = set(payload["manual_timing_word_ids"])
    if len(timing_ids) < 200:
        errors.append(
            f"reference has only {len(timing_ids)} manually timed words; at least 200 are required"
        )
    if not timing_ids <= word_ids:
        errors.append("manual timing sample contains an unknown word ID")

    provenance = payload["provenance"]
    if not provenance.get("configuration"):
        errors.append("reference provenance must retain the source configuration")
    if not (
        provenance.get("model_fingerprint_sha256")
        or provenance.get("models")
        or provenance.get("inputs")
    ):
        errors.append("reference provenance must retain model/input provenance")

    block_counts = {str(index): 0 for index in range(5)}
    boundaries = expected_split["boundaries_ms"]
    for word in payload["words"]:
        if word["word_id"] not in timing_ids:
            continue
        midpoint = (word["start_ms"] + word["end_ms"]) // 2
        for index in range(5):
            if boundaries[index] <= midpoint < boundaries[index + 1]:
                block_counts[str(index)] += 1
                break
    if any(count == 0 for count in block_counts.values()):
        errors.append("manual timing sample must cover all five split blocks")

    return {
        "valid": not errors,
        "errors": errors,
        "reference_path": str(source_path),
        "episode_id": payload["episode_id"],
        "source_sha256": payload["source_sha256"],
        "split_sha256": payload["split_sha256"],
        "revision": payload.get("revision"),
        "reviewer": payload["reviewer"],
        "word_count": len(payload["words"]),
        "reviewed_word_count": len(reviewed_ids),
        "manual_timing_word_count": len(timing_ids),
        "manual_timing_words_by_block": block_counts,
        "human_judgment_validated": False,
        "note": (
            "This command validates structure, provenance, coverage, and timing sample size; "
            "it does not validate human judgment."
        ),
    }


__all__ = [
    "CANONICAL_TIME_UNIT",
    "DEFAULT_THRESHOLDS",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationDocument",
    "EvaluationValidationError",
    "METRIC_DEFINITIONS_VERSION",
    "NORMALIZATION_VERSION",
    "QUANTILE_VERSION",
    "ReferenceValidationError",
    "current_git_commit",
    "evaluate_documents",
    "evaluate_files",
    "normalize_italian",
    "parse_evaluation_document",
    "validate_reference",
    "write_evaluation_artifacts",
]

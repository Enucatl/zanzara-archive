"""CPU-only ASR scoring for the P1R chunk benchmark.

P1R scores immutable chunk hypotheses against the same human reference with
three lexical views: raw WER, normalized Italian WER/CER, and an
overlap-aware, speaker-independent ORC-WER equivalent.  This module deliberately
does not import a model, call a service, or use word timing to align a model.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .contracts import (
    AudioChunk,
    ChunkBenchmarkManifest,
    TranscriptionHypothesis,
    TranscriptReference,
)

ASR_SCORING_SCHEMA_VERSION = 1
ASR_METRIC_DEFINITIONS_VERSION = "p1r-asr-v1"
ITALIAN_NORMALIZATION_VERSION = "it-v1"
ORC_WER_VERSION = "orc-equivalent-interleaving-v1"
MASK_POLICY_VERSION = "paired-token-mask-v1"


class ASRScoringValidationError(ValueError):
    """Raised when a chunk scoring input cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class ScoringTextStream:
    """Text for one speaker stream used only by the overlap-aware scorer.

    Canonical P1R ``SpeakerStream`` records intentionally carry intervals, not
    text.  This separate record keeps the lexical stream assignment explicit
    and prevents a scorer from fabricating it from ASR timing.
    """

    speaker_id: str
    text: str
    masked_token_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise ASRScoringValidationError("speaker stream speaker_id must be non-empty text")
        if not isinstance(self.text, str):
            raise ASRScoringValidationError("speaker stream text must be text")
        indices = tuple(self.masked_token_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ):
            raise ASRScoringValidationError(
                "speaker stream mask indices must be non-negative integers"
            )
        if len(set(indices)) != len(indices):
            raise ASRScoringValidationError("speaker stream mask indices must be unique")
        object.__setattr__(self, "masked_token_indices", indices)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stream without exposing it in aggregate reports."""

        return {
            "speaker_id": self.speaker_id,
            "text": self.text,
            "masked_token_indices": list(self.masked_token_indices),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScoringTextStream:
        """Build a scoring stream from JSON-compatible data."""

        return cls(
            speaker_id=value.get("speaker_id", ""),
            text=value.get("text", ""),
            masked_token_indices=tuple(value.get("masked_token_indices", ())),
        )


@dataclass(frozen=True, slots=True)
class ChunkScoringInput:
    """One immutable chunk/reference/hypothesis scoring tuple."""

    chunk: AudioChunk
    reference: TranscriptReference
    hypothesis: TranscriptionHypothesis
    reference_streams: tuple[ScoringTextStream, ...] = ()
    hypothesis_streams: tuple[ScoringTextStream, ...] = ()
    reference_masked_token_indices: tuple[int, ...] = ()
    hypothesis_masked_token_indices: tuple[int, ...] = ()
    slice_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reference.chunk_id != self.chunk.chunk_id:
            raise ASRScoringValidationError("reference does not belong to chunk")
        if self.hypothesis.chunk_id != self.chunk.chunk_id:
            raise ASRScoringValidationError("hypothesis does not belong to chunk")
        if (
            self.reference.source_sha256 != self.chunk.source_sha256
            or self.hypothesis.source_sha256 not in {None, self.chunk.source_sha256}
        ):
            raise ASRScoringValidationError("scoring source provenance does not match chunk")
        _validate_mask_indices(self.reference_masked_token_indices, "reference")
        _validate_mask_indices(self.hypothesis_masked_token_indices, "hypothesis")
        if bool(self.reference_masked_token_indices) != bool(self.hypothesis_masked_token_indices):
            raise ASRScoringValidationError(
                "text-only unintelligible masks require explicit reference and hypothesis masks"
            )
        if len({stream.speaker_id for stream in self.reference_streams}) != len(
            self.reference_streams
        ):
            raise ASRScoringValidationError("reference speaker stream IDs must be unique")
        if len({stream.speaker_id for stream in self.hypothesis_streams}) != len(
            self.hypothesis_streams
        ):
            raise ASRScoringValidationError("hypothesis speaker stream IDs must be unique")
        _validate_slice_ids(self.slice_ids)


def _validate_mask_indices(indices: Sequence[int], label: str) -> None:
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ASRScoringValidationError(f"{label} mask indices must be non-negative integers")
    if len(set(indices)) != len(indices):
        raise ASRScoringValidationError(f"{label} mask indices must be unique")


def _validate_slice_ids(slice_ids: Sequence[str]) -> None:
    if any(not isinstance(slice_id, str) or not slice_id.strip() for slice_id in slice_ids):
        raise ASRScoringValidationError("slice IDs must be non-empty text")
    if len(set(slice_ids)) != len(slice_ids):
        raise ASRScoringValidationError("slice IDs must be unique")


def normalize_italian(text: str) -> str:
    """Apply the locked P1R ``it-v1`` normalization rule."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tokenize(text: str, *, normalized: bool) -> tuple[tuple[str, ...], tuple[int, ...]]:
    raw_tokens = tuple(text.split())
    if not normalized:
        return raw_tokens, tuple(range(len(raw_tokens)))
    tokens: list[str] = []
    source_indices: list[int] = []
    for index, token in enumerate(raw_tokens):
        normalized_token = normalize_italian(token)
        if normalized_token:
            tokens.extend(normalized_token.split())
            source_indices.extend([index] * len(normalized_token.split()))
    return tuple(tokens), tuple(source_indices)


def _masked_tokens(
    text: str, *, normalized: bool, masked_indices: Sequence[int] = ()
) -> tuple[str, ...]:
    tokens, source_indices = _tokenize(text, normalized=normalized)
    masked = set(masked_indices)
    raw_count = len(text.split())
    if any(index >= raw_count for index in masked):
        raise ASRScoringValidationError("unintelligible mask index exceeds token count")
    return tuple(
        token
        for token, source_index in zip(tokens, source_indices, strict=True)
        if source_index not in masked
    )


@dataclass(frozen=True, slots=True)
class _Alignment:
    substitutions: int
    deletions: int
    insertions: int
    operations: tuple[tuple[str, int | None, int | None], ...] = ()

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def _align(reference: Sequence[str], hypothesis: Sequence[str]) -> _Alignment:
    """Return a deterministic Levenshtein alignment for one chunk."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
    for column in range(1, columns):
        costs[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row][column] = min(
                costs[row - 1][column - 1] + int(reference[row - 1] != hypothesis[column - 1]),
                costs[row - 1][column] + 1,
                costs[row][column - 1] + 1,
            )
    operations: list[tuple[str, int | None, int | None]] = []
    row, column = len(reference), len(hypothesis)
    substitutions = deletions = insertions = 0
    while row or column:
        if (
            row
            and column
            and costs[row][column]
            == costs[row - 1][column - 1] + int(reference[row - 1] != hypothesis[column - 1])
        ):
            equal = reference[row - 1] == hypothesis[column - 1]
            operations.append(("equal" if equal else "substitute", row - 1, column - 1))
            substitutions += int(not equal)
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            operations.append(("delete", row - 1, None))
            deletions += 1
            row -= 1
        else:
            operations.append(("insert", None, column - 1))
            insertions += 1
            column -= 1
    operations.reverse()
    return _Alignment(substitutions, deletions, insertions, tuple(operations))


def _metric(
    alignment: _Alignment,
    reference_denominator: int,
    hypothesis_count: int,
    *,
    unit: str,
    status: str = "scored",
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "value": alignment.errors / reference_denominator if reference_denominator else None,
        "errors": alignment.errors,
        "substitutions": alignment.substitutions,
        "deletions": alignment.deletions,
        "insertions": alignment.insertions,
        "reference_denominator": reference_denominator,
        "hypothesis_count": hypothesis_count,
        "unit": unit,
    }
    if not reference_denominator:
        result["status"] = "not_applicable"
        result["value"] = None
        result["reason"] = "zero reference denominator"
    elif reason is not None:
        result["reason"] = reason
    return result


def _invalid_metric(reason: str, *, unit: str = "tokens") -> dict[str, Any]:
    return {
        "status": "unscorable",
        "value": None,
        "errors": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "reference_denominator": None,
        "hypothesis_count": None,
        "unit": unit,
        "reason": reason,
    }


def _insufficient_metric(reason: str, *, unit: str = "tokens") -> dict[str, Any]:
    result = _invalid_metric(reason, unit=unit)
    result["status"] = "insufficient_evidence"
    return result


def _text_metrics(
    reference_text: str,
    hypothesis_text: str,
    *,
    reference_masked_token_indices: Sequence[int] = (),
    hypothesis_masked_token_indices: Sequence[int] = (),
) -> dict[str, Any]:
    if bool(reference_masked_token_indices) != bool(hypothesis_masked_token_indices):
        reason = "mask alignment is absent on one side of a text-only comparison"
        invalid = _invalid_metric(reason)
        return {
            "raw_wer": invalid.copy(),
            "normalized_wer": invalid.copy(),
            "normalized_cer": invalid.copy(),
        }
    try:
        raw_reference = _masked_tokens(
            reference_text, normalized=False, masked_indices=reference_masked_token_indices
        )
        raw_hypothesis = _masked_tokens(
            hypothesis_text, normalized=False, masked_indices=hypothesis_masked_token_indices
        )
        normalized_reference = _masked_tokens(
            reference_text, normalized=True, masked_indices=reference_masked_token_indices
        )
        normalized_hypothesis = _masked_tokens(
            hypothesis_text, normalized=True, masked_indices=hypothesis_masked_token_indices
        )
    except ASRScoringValidationError as exc:
        invalid = _invalid_metric(str(exc))
        return {
            "raw_wer": invalid.copy(),
            "normalized_wer": invalid.copy(),
            "normalized_cer": invalid.copy(),
        }

    raw_alignment = _align(raw_reference, raw_hypothesis)
    normalized_alignment = _align(normalized_reference, normalized_hypothesis)
    reference_chars = tuple(" ".join(normalized_reference))
    hypothesis_chars = tuple(" ".join(normalized_hypothesis))
    character_alignment = _align(reference_chars, hypothesis_chars)
    return {
        "raw_wer": _metric(raw_alignment, len(raw_reference), len(raw_hypothesis), unit="tokens"),
        "normalized_wer": _metric(
            normalized_alignment,
            len(normalized_reference),
            len(normalized_hypothesis),
            unit="tokens",
        ),
        "normalized_cer": _metric(
            character_alignment,
            len(reference_chars),
            len(hypothesis_chars),
            unit="Unicode code points",
        ),
    }


def _stream_tokens(stream: ScoringTextStream) -> tuple[str, ...]:
    return _masked_tokens(
        stream.text,
        normalized=True,
        masked_indices=stream.masked_token_indices,
    )


def _orc_metric(
    reference_streams: Sequence[ScoringTextStream],
    hypothesis_streams: Sequence[ScoringTextStream],
    *,
    hypothesis_text: str = "",
    reference_masked_token_indices: Sequence[int] = (),
    hypothesis_masked_token_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Score stream-free overlap by optimal interleaving of reference streams.

    The metric is intentionally a locked ORC-WER equivalent rather than a
    cpWER implementation: hypothesis speaker labels are ignored, reference
    streams retain within-speaker token order, and the scorer chooses the
    lowest-error interleaving across all streams.  This supports overlapped
    references without granting a model a speaker-label or boundary advantage.
    """

    if not reference_streams:
        return _invalid_metric("valid reference speaker streams are required for ORC-WER") | {
            "status": "not_applicable",
            "reason": "reference speaker streams are absent",
        }
    reference_stream_masks = any(stream.masked_token_indices for stream in reference_streams)
    hypothesis_stream_masks = any(stream.masked_token_indices for stream in hypothesis_streams)
    if bool(reference_masked_token_indices) != bool(hypothesis_masked_token_indices) or (
        reference_stream_masks != hypothesis_stream_masks
    ):
        return _invalid_metric(
            "mask alignment is absent on one side of the overlap-aware comparison"
        )
    try:
        reference_tokens = tuple(_stream_tokens(stream) for stream in reference_streams)
        hypothesis_tokens = (
            tuple(token for stream in hypothesis_streams for token in _stream_tokens(stream))
            if hypothesis_streams
            else _masked_tokens(
                hypothesis_text,
                normalized=True,
                masked_indices=hypothesis_masked_token_indices,
            )
        )
    except ASRScoringValidationError as exc:
        return _invalid_metric(str(exc))
    reference_count = sum(len(tokens) for tokens in reference_tokens)
    if not reference_count:
        return _invalid_metric("zero reference denominator", unit="tokens") | {
            "status": "not_applicable",
            "reason": "zero reference denominator",
        }

    @cache
    def best(
        positions: tuple[int, ...], hypothesis_index: int
    ) -> tuple[int, int, int, int, tuple[str, ...]]:
        if hypothesis_index == len(hypothesis_tokens) and all(
            position == len(tokens)
            for position, tokens in zip(positions, reference_tokens, strict=True)
        ):
            return (0, 0, 0, 0, ())
        candidates: list[tuple[int, int, int, int, tuple[str, ...], tuple[int, int]]] = []
        if hypothesis_index < len(hypothesis_tokens):
            next_result = best(positions, hypothesis_index + 1)
            candidates.append(
                (
                    next_result[0] + 1,
                    next_result[1],
                    next_result[2],
                    next_result[3] + 1,
                    next_result[4] + ("insert",),
                    (2, len(reference_tokens)),
                )
            )
        for stream_index, tokens in enumerate(reference_tokens):
            position = positions[stream_index]
            if position >= len(tokens):
                continue
            next_positions = list(positions)
            next_positions[stream_index] += 1
            next_positions_tuple = tuple(next_positions)
            delete_result = best(next_positions_tuple, hypothesis_index)
            candidates.append(
                (
                    delete_result[0] + 1,
                    delete_result[1],
                    delete_result[2] + 1,
                    delete_result[3],
                    delete_result[4] + (f"delete:{stream_index}",),
                    (1, stream_index),
                )
            )
            if hypothesis_index < len(hypothesis_tokens):
                pair_result = best(next_positions_tuple, hypothesis_index + 1)
                substitution = int(tokens[position] != hypothesis_tokens[hypothesis_index])
                candidates.append(
                    (
                        pair_result[0] + substitution,
                        pair_result[1] + substitution,
                        pair_result[2],
                        pair_result[3],
                        pair_result[4] + (f"pair:{stream_index}",),
                        (0, stream_index),
                    )
                )
        selected = min(candidates, key=lambda candidate: (*candidate[:4], candidate[5]))
        return selected[:5]

    errors, substitutions, deletions, insertions, path = best(tuple(0 for _ in reference_tokens), 0)
    return {
        "status": "scored",
        "value": errors / reference_count,
        "errors": errors,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_denominator": reference_count,
        "hypothesis_count": len(hypothesis_tokens),
        "unit": "tokens",
        "algorithm": ORC_WER_VERSION,
        "speaker_labels_used": False,
        "reference_stream_count": len(reference_streams),
        "hypothesis_stream_count": len(hypothesis_streams),
        "private_interleaving_path": list(path),
    }


def score_text_pair(
    reference_text: str,
    hypothesis_text: str,
    *,
    reference_masked_token_indices: Sequence[int] = (),
    hypothesis_masked_token_indices: Sequence[int] = (),
    reference_streams: Sequence[ScoringTextStream] = (),
    hypothesis_streams: Sequence[ScoringTextStream] = (),
) -> dict[str, Any]:
    """Score one pair of text artifacts with deterministic lexical metrics."""

    metrics = _text_metrics(
        reference_text,
        hypothesis_text,
        reference_masked_token_indices=reference_masked_token_indices,
        hypothesis_masked_token_indices=hypothesis_masked_token_indices,
    )
    metrics["orc_wer"] = _orc_metric(
        reference_streams,
        hypothesis_streams,
        hypothesis_text=hypothesis_text,
        reference_masked_token_indices=reference_masked_token_indices,
        hypothesis_masked_token_indices=hypothesis_masked_token_indices,
    )
    return metrics


def _condition_slices(chunk: AudioChunk) -> set[str]:
    slices = {"all"}
    if chunk.partition is not None:
        slices.add(chunk.partition)
    condition = chunk.condition
    if condition is None:
        return slices
    if condition.speaker_count == 1:
        slices.add("single_speaker")
    elif condition.speaker_count >= 2:
        slices.add("multi_speaker")
    if condition.has_overlap:
        slices.add("overlap")
    if condition.music or "music" in condition.acoustic_labels:
        slices.add("music")
    if condition.has_overlap and (condition.music or "music" in condition.acoustic_labels):
        slices.add("overlap_music")
    if condition.degraded:
        slices.add("degraded")
    if condition.rapid_turn_taking:
        slices.add("rapid_turn_taking")
    slices.update(f"acoustic:{label}" for label in condition.acoustic_labels)
    return slices


def _chunk_report(item: ChunkScoringInput) -> dict[str, Any]:
    if item.reference.review_status == "human_truth":
        metrics = score_text_pair(
            item.reference.text,
            item.hypothesis.text,
            reference_masked_token_indices=item.reference_masked_token_indices,
            hypothesis_masked_token_indices=item.hypothesis_masked_token_indices,
            reference_streams=item.reference_streams,
            hypothesis_streams=item.hypothesis_streams,
        )
    else:
        reason = "reference is not a current human_truth revision"
        metrics = {
            "raw_wer": _insufficient_metric(reason),
            "normalized_wer": _insufficient_metric(reason),
            "normalized_cer": _insufficient_metric(reason, unit="Unicode code points"),
            "orc_wer": _insufficient_metric(reason),
        }
    try:
        reference_word_count = len(
            _masked_tokens(
                item.reference.text,
                normalized=True,
                masked_indices=item.reference_masked_token_indices,
            )
        )
    except ASRScoringValidationError:
        reference_word_count = 0
    return {
        "chunk_id": item.chunk.chunk_id,
        "episode_id": item.chunk.episode_id,
        "partition": item.chunk.partition,
        "duration_ms": item.chunk.end_ms - item.chunk.start_ms,
        "reference_word_count": reference_word_count,
        "slice_ids": sorted(_condition_slices(item.chunk) | set(item.slice_ids)),
        "model_fingerprint_sha256": item.hypothesis.model_fingerprint_hash,
        "metrics": metrics,
    }


def _aggregate_metric(
    chunk_reports: Sequence[Mapping[str, Any]], metric_name: str
) -> dict[str, Any]:
    scored = [
        report["metrics"][metric_name]
        for report in chunk_reports
        if report["metrics"][metric_name]["status"] == "scored"
    ]
    invalid = [
        report["metrics"][metric_name]
        for report in chunk_reports
        if report["metrics"][metric_name]["status"] == "unscorable"
    ]
    insufficient = [
        report["metrics"][metric_name]
        for report in chunk_reports
        if report["metrics"][metric_name]["status"] == "insufficient_evidence"
    ]
    if not scored:
        status = (
            "unscorable"
            if invalid
            else ("insufficient_evidence" if insufficient else "not_applicable")
        )
        return {
            "status": status,
            "value": None,
            "errors": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "reference_denominator": 0,
            "hypothesis_count": 0,
            "unscorable_chunk_count": len(invalid),
            "insufficient_evidence_chunk_count": len(insufficient),
        }
    fields = ("errors", "substitutions", "deletions", "insertions", "hypothesis_count")
    totals = {
        field_name: sum(int(metric[field_name]) for metric in scored) for field_name in fields
    }
    denominator = sum(int(metric["reference_denominator"]) for metric in scored)
    return {
        "status": "partial" if invalid else "scored",
        "value": totals["errors"] / denominator if denominator else None,
        "reference_denominator": denominator,
        "unscorable_chunk_count": len(invalid),
        "insufficient_evidence_chunk_count": len(insufficient),
        **totals,
    }


def _aggregate_slice(chunk_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "chunk_count": len(chunk_reports),
        "duration_ms": sum(int(report["duration_ms"]) for report in chunk_reports),
        "reference_word_denominator": sum(
            int(report["reference_word_count"]) for report in chunk_reports
        ),
        "metrics": {
            metric_name: _aggregate_metric(chunk_reports, metric_name)
            for metric_name in ("raw_wer", "normalized_wer", "normalized_cer", "orc_wer")
        },
    }


def _aggregate_model(chunk_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slice_ids = sorted({slice_id for report in chunk_reports for slice_id in report["slice_ids"]})
    return {
        "hypothesis_count": len(chunk_reports),
        "slices": {
            slice_id: _aggregate_slice(
                [report for report in chunk_reports if slice_id in report["slice_ids"]]
            )
            for slice_id in slice_ids
        },
    }


def score_chunks(
    inputs: Iterable[ChunkScoringInput],
    *,
    manifest_sha256: str | None = None,
    reviewed_commit: str | None = None,
    synthetic_provenance: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate one or more model hypotheses over deterministic benchmark slices."""

    items = tuple(inputs)
    if not items:
        raise ASRScoringValidationError("at least one chunk scoring input is required")
    chunk_ids = [item.chunk.chunk_id for item in items]
    if len(chunk_ids) != len(set(chunk_ids)) and len(
        {(item.chunk.chunk_id, item.hypothesis.model_fingerprint_hash) for item in items}
    ) != len(items):
        raise ASRScoringValidationError("duplicate chunk/model scoring input")
    reports = tuple(_chunk_report(item) for item in items)
    model_hashes = sorted({report["model_fingerprint_sha256"] for report in reports})
    configuration = {
        "metric_definitions_version": ASR_METRIC_DEFINITIONS_VERSION,
        "normalization_version": ITALIAN_NORMALIZATION_VERSION,
        "overlap_metric_version": ORC_WER_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
        "raw_wer": "whitespace tokens preserving case and punctuation",
        "normalized_wer": "NFC/casefold/punctuation-to-space/whitespace-collapse tokens",
        "normalized_cer": "Unicode code points of normalized text including interword spaces",
        "overlap_metric": (
            "minimum Levenshtein error over all interleavings of reference speaker streams; "
            "within-stream order is fixed and speaker labels are ignored"
        ),
        "mask_policy": (
            "paired explicit token indices are removed before every lexical metric; one-sided "
            "text-only masks are unscorable"
        ),
    }
    configuration_sha256 = _sha256(configuration)
    models = {
        model_hash: _aggregate_model(
            [report for report in reports if report["model_fingerprint_sha256"] == model_hash]
        )
        for model_hash in model_hashes
    }
    return {
        "schema_version": ASR_SCORING_SCHEMA_VERSION,
        "metric_definitions_version": ASR_METRIC_DEFINITIONS_VERSION,
        "normalization_version": ITALIAN_NORMALIZATION_VERSION,
        "overlap_metric_version": ORC_WER_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
        "configuration": configuration,
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "source_sha256": sorted({item.chunk.source_sha256 for item in items}),
            "model_fingerprint_sha256": model_hashes,
            "configuration_sha256": configuration_sha256,
            "reviewed_commit": reviewed_commit,
            "synthetic_provenance": list(synthetic_provenance),
        },
        "models": models,
        "limitations": [
            "This CPU harness does not establish model quality or choose a model.",
            "Human audio-verified references and real model provenance are required for release "
            "evidence.",
            "Text-only unintelligible masks require explicit paired token alignment; ambiguous "
            "masks are unscorable.",
            "ORC-WER is a locked stream-interleaving equivalent; it is not cpWER or diarization "
            "scoring.",
        ],
        "private_details": {"chunks": list(reports)},
    }


def score_manifest(
    manifest: ChunkBenchmarkManifest,
    *,
    reference_streams: Mapping[str, Sequence[ScoringTextStream]] | None = None,
    hypothesis_streams: Mapping[str, Sequence[ScoringTextStream]] | None = None,
    reference_masks: Mapping[str, Sequence[int]] | None = None,
    hypothesis_masks: Mapping[str, Sequence[int]] | None = None,
    reviewed_commit: str | None = None,
    synthetic_provenance: Sequence[str] = (),
) -> dict[str, Any]:
    """Score every manifest hypothesis against its current human reference.

    The manifest stores text-only reference/hypothesis contracts.  Optional
    stream and mask maps are separate inputs because P1R-02 intentionally did
    not make word timing or lexical speaker assignment part of canonical gold.
    """

    references: dict[str, TranscriptReference] = {}
    for reference in manifest.references:
        current = references.get(reference.chunk_id)
        if current is None or reference.revision > current.revision:
            references[reference.chunk_id] = reference
    chunks = {chunk.chunk_id: chunk for chunk in manifest.chunks}
    reference_streams = reference_streams or {}
    hypothesis_streams = hypothesis_streams or {}
    reference_masks = reference_masks or {}
    hypothesis_masks = hypothesis_masks or {}
    inputs: list[ChunkScoringInput] = []
    for hypothesis in manifest.hypotheses:
        chunk = chunks.get(hypothesis.chunk_id)
        reference = references.get(hypothesis.chunk_id)
        if chunk is None or reference is None:
            raise ASRScoringValidationError(
                f"hypothesis {hypothesis.hypothesis_id} has no matching chunk/reference"
            )
        inputs.append(
            ChunkScoringInput(
                chunk=chunk,
                reference=reference,
                hypothesis=hypothesis,
                reference_streams=tuple(reference_streams.get(chunk.chunk_id, ())),
                hypothesis_streams=tuple(hypothesis_streams.get(hypothesis.hypothesis_id, ())),
                reference_masked_token_indices=tuple(reference_masks.get(chunk.chunk_id, ())),
                hypothesis_masked_token_indices=tuple(
                    hypothesis_masks.get(hypothesis.hypothesis_id, ())
                ),
            )
        )
    return score_chunks(
        inputs,
        manifest_sha256=manifest.content_sha256,
        reviewed_commit=reviewed_commit,
        synthetic_provenance=synthetic_provenance,
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _aggregate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "private_details"}


def _render_report_html(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for model_hash, model in report["models"].items():
        for slice_id, slice_report in model["slices"].items():
            for metric_name, metric in slice_report["metrics"].items():
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(model_hash)}</td><td>{html.escape(slice_id)}</td>"
                    f"<td>{html.escape(metric_name)}</td><td>{html.escape(str(metric['status']))}</td>"
                    f"<td>{html.escape(str(metric['value']))}</td>"
                    f"<td>{slice_report['chunk_count']}</td>"
                    f"<td>{slice_report['duration_ms']}</td>"
                    f"<td>{slice_report['reference_word_denominator']}</td>"
                    "</tr>"
                )
    return (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>P1R ASR score</title>
<h1>P1R ASR score</h1>
<p>This sanitized aggregate contains no transcript text or speaker identity.</p>
<table><thead><tr><th>Model</th><th>Slice</th><th>Metric</th><th>Status</th><th>Value</th>
<th>Chunks</th><th>Duration ms</th><th>Reference words</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + """</tbody></table>
</html>
"""
    )


def write_asr_score_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Atomically publish private details and a sanitized aggregate report."""

    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise ASRScoringValidationError(f"refusing to overwrite existing ASR report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        aggregate = _aggregate_report(report)
        private = {
            "schema_version": report["schema_version"],
            "provenance": report["provenance"],
            "chunks": report.get("private_details", {}).get("chunks", []),
        }
        files = {
            "metrics.json": _json_bytes(aggregate),
            "details.json": _json_bytes(private),
            "report.html": _render_report_html(report).encode("utf-8"),
        }
        for name, content in files.items():
            path = temporary / name
            path.write_bytes(content)
            path.chmod(0o600)
        run = {
            "schema_version": report["schema_version"],
            "run_type": "p1r-chunk-asr",
            "private": True,
            "metric_definitions_version": report["metric_definitions_version"],
            "normalization_version": report["normalization_version"],
            "overlap_metric_version": report["overlap_metric_version"],
            "provenance": report["provenance"],
            "artifact_sha256": {
                name: hashlib.sha256(content).hexdigest() for name, content in files.items()
            },
        }
        run_bytes = _json_bytes(run)
        (temporary / "run.json").write_bytes(run_bytes)
        (temporary / "run.json").chmod(0o600)
        temporary.chmod(0o700)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    hashes = {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in ("metrics.json", "details.json", "report.html", "run.json")
    }
    return {"output_dir": str(destination), "artifact_sha256": hashes}


__all__ = [
    "ASR_METRIC_DEFINITIONS_VERSION",
    "ASR_SCORING_SCHEMA_VERSION",
    "ASRScoringValidationError",
    "ChunkScoringInput",
    "ITALIAN_NORMALIZATION_VERSION",
    "MASK_POLICY_VERSION",
    "ORC_WER_VERSION",
    "ScoringTextStream",
    "normalize_italian",
    "score_chunks",
    "score_manifest",
    "score_text_pair",
    "write_asr_score_report",
]

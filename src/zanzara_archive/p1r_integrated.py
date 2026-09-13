"""CPU-only integrated speaker-attributed scoring for the P1R benchmark.

This module is deliberately separate from lexical ASR and interval diarization
scoring.  It implements a locked concatenated minimum-permutation WER (cpWER)
over explicit reference and hypothesis speaker text streams.  Speaker labels
are mapped one-to-one, while overlapping reference streams remain separate
channels and therefore retain their words in the denominator.

P1R word timing is optional, so tcpWER is reported as explicitly deferred.  A
caller may still provide validated millisecond slice and diarization interval
metadata; the scorer never invents word clipping or speaker streams from those
intervals.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .contracts import Overlap, Turn
from .p1r_asr import ScoringTextStream, normalize_italian

INTEGRATED_SCORING_SCHEMA_VERSION = 1
INTEGRATED_METRIC_DEFINITIONS_VERSION = "p1r-integrated-v1"
CPWER_VERSION = "cpwer-v1"
TCPWER_STATUS = "deferred"


class IntegratedScoringError(ValueError):
    """Raised when integrated scoring input cannot be reproduced safely."""


SpeakerTextStream = ScoringTextStream


@dataclass(frozen=True, slots=True)
class IntegratedScoringInput:
    """One chunk's explicit speaker streams and interval provenance."""

    reference_streams: tuple[SpeakerTextStream, ...]
    hypothesis_streams: tuple[SpeakerTextStream, ...]
    reference_status: str = "human_truth"
    chunk_id: str | None = None
    hypothesis_id: str | None = None
    source_sha256: str | None = None
    model_fingerprint_sha256: str | None = None
    duration_ms: int | None = None
    scored_ranges_ms: tuple[tuple[int, int], ...] = ()
    reference_turns: tuple[Turn, ...] = ()
    hypothesis_turns: tuple[Turn, ...] = ()
    reference_overlaps: tuple[Overlap, ...] = ()
    hypothesis_overlaps: tuple[Overlap, ...] = ()
    slice_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reference_streams = tuple(_coerce_stream(item) for item in self.reference_streams)
        hypothesis_streams = tuple(_coerce_stream(item) for item in self.hypothesis_streams)
        reference_turns = tuple(_coerce_turn(item) for item in self.reference_turns)
        hypothesis_turns = tuple(_coerce_turn(item) for item in self.hypothesis_turns)
        reference_overlaps = tuple(_coerce_overlap(item) for item in self.reference_overlaps)
        hypothesis_overlaps = tuple(_coerce_overlap(item) for item in self.hypothesis_overlaps)
        object.__setattr__(self, "reference_streams", reference_streams)
        object.__setattr__(self, "hypothesis_streams", hypothesis_streams)
        object.__setattr__(self, "reference_turns", reference_turns)
        object.__setattr__(self, "hypothesis_turns", hypothesis_turns)
        object.__setattr__(self, "reference_overlaps", reference_overlaps)
        object.__setattr__(self, "hypothesis_overlaps", hypothesis_overlaps)
        for label, streams in (
            ("reference", reference_streams),
            ("hypothesis", hypothesis_streams),
        ):
            speaker_ids = [stream.speaker_id for stream in streams]
            if len(speaker_ids) != len(set(speaker_ids)):
                raise IntegratedScoringError(f"{label} speaker stream IDs must be unique")
        if self.reference_status not in {"draft", "human_truth", "superseded", "rejected"}:
            raise IntegratedScoringError("reference_status is invalid")
        for name, value in (
            ("chunk_id", self.chunk_id),
            ("hypothesis_id", self.hypothesis_id),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise IntegratedScoringError(f"{name} must be non-empty text or null")
        for name, value in (
            ("source_sha256", self.source_sha256),
            ("model_fingerprint_sha256", self.model_fingerprint_sha256),
        ):
            if value is not None and not _is_sha256(value):
                raise IntegratedScoringError(f"{name} must be a SHA-256 hex digest")
        if self.duration_ms is not None:
            _require_nonnegative_int(self.duration_ms, "duration_ms")
            if self.duration_ms == 0 and self.scored_ranges_ms:
                raise IntegratedScoringError("duration_ms must be positive when ranges are present")
        ranges = _validate_ranges(self.scored_ranges_ms, self.duration_ms)
        object.__setattr__(self, "scored_ranges_ms", ranges)
        for label, turns, streams in (
            ("reference", self.reference_turns, reference_streams),
            ("hypothesis", self.hypothesis_turns, hypothesis_streams),
        ):
            _validate_turns(turns, label, streams, self.duration_ms)
        for label, overlaps, streams in (
            ("reference", self.reference_overlaps, reference_streams),
            ("hypothesis", self.hypothesis_overlaps, hypothesis_streams),
        ):
            _validate_overlaps(overlaps, label, streams, self.duration_ms)
        if any(
            not isinstance(slice_id, str) or not slice_id.strip() for slice_id in self.slice_ids
        ):
            raise IntegratedScoringError("slice IDs must be non-empty text")
        if len(set(self.slice_ids)) != len(self.slice_ids):
            raise IntegratedScoringError("slice IDs must be unique")


@dataclass(frozen=True, slots=True)
class _Alignment:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        """Return the total edit count."""

        return self.substitutions + self.deletions + self.insertions


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _require_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegratedScoringError(f"{name} must be a non-negative integer")


def _coerce_stream(value: SpeakerTextStream | Mapping[str, Any]) -> SpeakerTextStream:
    if isinstance(value, SpeakerTextStream):
        return value
    if isinstance(value, Mapping):
        try:
            return SpeakerTextStream.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegratedScoringError(f"invalid speaker text stream: {exc}") from exc
    raise IntegratedScoringError("speaker streams must be ScoringTextStream objects")


def _coerce_turn(value: Turn | Mapping[str, Any]) -> Turn:
    if isinstance(value, Turn):
        return value
    if isinstance(value, Mapping):
        try:
            return Turn.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegratedScoringError(f"invalid diarization turn: {exc}") from exc
    raise IntegratedScoringError("diarization turns must be Turn objects")


def _coerce_overlap(value: Overlap | Mapping[str, Any]) -> Overlap:
    if isinstance(value, Overlap):
        return value
    if isinstance(value, Mapping):
        try:
            return Overlap.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegratedScoringError(f"invalid overlap interval: {exc}") from exc
    raise IntegratedScoringError("overlap intervals must be Overlap objects")


def _validate_ranges(
    ranges: Sequence[tuple[int, int]], duration_ms: int | None
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    previous_end = -1
    for index, raw_range in enumerate(ranges):
        if (
            not isinstance(raw_range, Sequence)
            or isinstance(raw_range, (str, bytes))
            or len(raw_range) != 2
        ):
            raise IntegratedScoringError(f"scored_ranges_ms[{index}] must be a pair")
        start_ms, end_ms = raw_range
        _require_nonnegative_int(start_ms, f"scored_ranges_ms[{index}].start_ms")
        _require_nonnegative_int(end_ms, f"scored_ranges_ms[{index}].end_ms")
        if end_ms <= start_ms:
            raise IntegratedScoringError(
                f"scored_ranges_ms[{index}] must be half-open and non-empty"
            )
        if start_ms < previous_end:
            raise IntegratedScoringError("scored_ranges_ms must be sorted and non-overlapping")
        if duration_ms is not None and end_ms > duration_ms:
            raise IntegratedScoringError("scored_ranges_ms must remain within duration_ms")
        result.append((start_ms, end_ms))
        previous_end = end_ms
    return tuple(result)


def _validate_turns(
    turns: Sequence[Turn],
    label: str,
    streams: Sequence[SpeakerTextStream],
    duration_ms: int | None,
) -> None:
    stream_ids = {stream.speaker_id for stream in streams}
    for index, turn in enumerate(turns):
        if not isinstance(turn, Turn):
            raise IntegratedScoringError(f"{label}_turns[{index}] must be a Turn")
        if turn.speaker_id not in stream_ids:
            raise IntegratedScoringError(
                f"{label}_turns[{index}] speaker is absent from {label} text streams"
            )
        if duration_ms is not None and turn.end_ms > duration_ms:
            raise IntegratedScoringError(f"{label}_turns[{index}] exceeds duration_ms")


def _validate_overlaps(
    overlaps: Sequence[Overlap],
    label: str,
    streams: Sequence[SpeakerTextStream],
    duration_ms: int | None,
) -> None:
    stream_ids = {stream.speaker_id for stream in streams}
    for index, overlap in enumerate(overlaps):
        if not isinstance(overlap, Overlap):
            raise IntegratedScoringError(f"{label}_overlaps[{index}] must be an Overlap")
        if not set(overlap.speaker_ids) <= stream_ids:
            raise IntegratedScoringError(
                f"{label}_overlaps[{index}] contains a speaker absent from {label} text streams"
            )
        if duration_ms is not None and overlap.end_ms > duration_ms:
            raise IntegratedScoringError(f"{label}_overlaps[{index}] exceeds duration_ms")


def _tokens(stream: SpeakerTextStream) -> tuple[str, ...]:
    raw_tokens = stream.text.split()
    masked = set(stream.masked_token_indices)
    if any(index >= len(raw_tokens) for index in masked):
        raise IntegratedScoringError(
            f"speaker stream {stream.speaker_id!r} mask index exceeds token count"
        )
    result: list[str] = []
    for index, raw_token in enumerate(raw_tokens):
        if index in masked:
            continue
        result.extend(token for token in normalize_italian(raw_token).split() if token)
    return tuple(result)


def _align(reference: Sequence[str], hypothesis: Sequence[str]) -> _Alignment:
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
    row, column = len(reference), len(hypothesis)
    substitutions = deletions = insertions = 0
    while row or column:
        if (
            row
            and column
            and costs[row][column]
            == costs[row - 1][column - 1] + int(reference[row - 1] != hypothesis[column - 1])
        ):
            substitutions += int(reference[row - 1] != hypothesis[column - 1])
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return _Alignment(substitutions, deletions, insertions)


def _metric(
    alignment: _Alignment,
    reference_denominator: int,
    hypothesis_count: int,
    *,
    reference_stream_count: int,
    hypothesis_stream_count: int,
    mapping: Mapping[str, str],
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "scored" if reference_denominator else "not_applicable",
        "value": alignment.errors / reference_denominator if reference_denominator else None,
        "errors": alignment.errors if reference_denominator else None,
        "substitutions": alignment.substitutions if reference_denominator else None,
        "deletions": alignment.deletions if reference_denominator else None,
        "insertions": alignment.insertions if reference_denominator else None,
        "reference_denominator": reference_denominator,
        "hypothesis_count": hypothesis_count,
        "reference_stream_count": reference_stream_count,
        "hypothesis_stream_count": hypothesis_stream_count,
        "unit": "normalized Italian tokens",
        "algorithm": CPWER_VERSION,
        "mapping": dict(mapping),
        "assignments": [dict(assignment) for assignment in assignments],
        "definition": (
            "minimum total normalized-token edit distance over a one-to-one permutation "
            "of hypothesis speaker channels to reference channels; dummy channels account "
            "for missing or extra speakers"
        ),
    }
    if not reference_denominator:
        result["reason"] = "zero reference denominator"
    return result


def _public_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    """Remove raw channel assignments from the aggregate report surface."""

    return {key: value for key, value in metric.items() if key not in {"mapping", "assignments"}}


def _unscorable(
    reason: str, *, reference_count: int = 0, hypothesis_count: int = 0
) -> dict[str, Any]:
    return {
        "status": "unscorable",
        "value": None,
        "errors": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "reference_denominator": None,
        "hypothesis_count": hypothesis_count,
        "reference_stream_count": reference_count,
        "hypothesis_stream_count": None,
        "unit": "normalized Italian tokens",
        "algorithm": CPWER_VERSION,
        "reason": reason,
    }


def _assignment(
    reference_streams: Sequence[SpeakerTextStream],
    hypothesis_streams: Sequence[SpeakerTextStream],
) -> tuple[_Alignment, dict[str, str], tuple[dict[str, Any], ...], int]:
    references = {stream.speaker_id: _tokens(stream) for stream in reference_streams}
    hypotheses = {stream.speaker_id: _tokens(stream) for stream in hypothesis_streams}
    reference_ids = tuple(sorted(references))
    hypothesis_ids = tuple(sorted(hypotheses))
    size = max(len(reference_ids), len(hypothesis_ids))
    reference_rows: tuple[str | None, ...] = reference_ids + (None,) * (size - len(reference_ids))
    hypothesis_rows: tuple[str | None, ...] = hypothesis_ids + (None,) * (
        size - len(hypothesis_ids)
    )
    costs = {
        (hypothesis_id, reference_id): _align(
            references.get(reference_id, ()), hypotheses.get(hypothesis_id, ())
        )
        for hypothesis_id in hypothesis_rows
        for reference_id in reference_rows
    }

    @cache
    def best(row: int, used: int) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
        if row == size:
            return 0, (), ()
        candidates: list[tuple[int, tuple[str, ...], tuple[int, ...]]] = []
        for column, reference_id in enumerate(reference_rows):
            if used & (1 << column):
                continue
            alignment = costs[(hypothesis_rows[row], reference_id)]
            tail_cost, tail_key, tail_columns = best(row + 1, used | (1 << column))
            label = reference_id if reference_id is not None else "~unmapped"
            candidates.append(
                (
                    alignment.errors + tail_cost,
                    (label, *tail_key),
                    (column, *tail_columns),
                )
            )
        return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))

    total_errors, _, selected_columns = best(0, 0)
    mapping: dict[str, str] = {}
    assignments: list[dict[str, Any]] = []
    total = _Alignment(0, 0, 0)
    for row, column in enumerate(selected_columns):
        hypothesis_id = hypothesis_rows[row]
        reference_id = reference_rows[column]
        alignment = costs[(hypothesis_id, reference_id)]
        total = _Alignment(
            total.substitutions + alignment.substitutions,
            total.deletions + alignment.deletions,
            total.insertions + alignment.insertions,
        )
        if hypothesis_id is not None and reference_id is not None:
            mapping[hypothesis_id] = reference_id
        assignments.append(
            {
                "hypothesis_speaker_id": hypothesis_id,
                "reference_speaker_id": reference_id,
                "substitutions": alignment.substitutions,
                "deletions": alignment.deletions,
                "insertions": alignment.insertions,
                "errors": alignment.errors,
                "reference_tokens": len(references.get(reference_id, ())),
                "hypothesis_tokens": len(hypotheses.get(hypothesis_id, ())),
            }
        )
    if total.errors != total_errors:
        raise IntegratedScoringError("speaker permutation reconstruction was not reproducible")
    return total, mapping, tuple(assignments), sum(len(tokens) for tokens in references.values())


def _cpwer_for_input(item: IntegratedScoringInput) -> tuple[dict[str, Any], dict[str, Any]]:
    if item.reference_status != "human_truth":
        metric = {
            **_unscorable(
                "reference is not a current human_truth revision",
                reference_count=len(item.reference_streams),
                hypothesis_count=len(item.hypothesis_streams),
            ),
            "status": "insufficient_evidence",
        }
        return metric, {"mapping": {}, "assignments": [], "reason": metric["reason"]}
    if not item.reference_streams:
        metric = _unscorable(
            "explicit reference speaker text streams are required; "
            "no speaker-independent WER fallback",
            reference_count=0,
            hypothesis_count=len(item.hypothesis_streams),
        )
        return metric, {"mapping": {}, "assignments": [], "reason": metric["reason"]}
    try:
        alignment, mapping, assignments, denominator = _assignment(
            item.reference_streams, item.hypothesis_streams
        )
    except IntegratedScoringError:
        raise
    metric = _metric(
        alignment,
        denominator,
        sum(len(_tokens(stream)) for stream in item.hypothesis_streams),
        reference_stream_count=len(item.reference_streams),
        hypothesis_stream_count=len(item.hypothesis_streams),
        mapping=mapping,
        assignments=assignments,
    )
    detail = {
        "chunk_id": item.chunk_id,
        "hypothesis_id": item.hypothesis_id,
        "mapping": dict(mapping),
        "assignments": [dict(assignment) for assignment in assignments],
        "reference_stream_ids": sorted(stream.speaker_id for stream in item.reference_streams),
        "hypothesis_stream_ids": sorted(stream.speaker_id for stream in item.hypothesis_streams),
    }
    return metric, detail


def score_cpwer(
    reference_streams: Sequence[SpeakerTextStream | Mapping[str, Any]],
    hypothesis_streams: Sequence[SpeakerTextStream | Mapping[str, Any]],
    *,
    reference_status: str = "human_truth",
    duration_ms: int | None = None,
    scored_ranges_ms: Sequence[tuple[int, int]] = (),
    reference_turns: Sequence[Turn] = (),
    hypothesis_turns: Sequence[Turn] = (),
    reference_overlaps: Sequence[Overlap] = (),
    hypothesis_overlaps: Sequence[Overlap] = (),
) -> dict[str, Any]:
    """Score explicit speaker text streams with deterministic cpWER mapping."""

    item = IntegratedScoringInput(
        reference_streams=tuple(reference_streams),
        hypothesis_streams=tuple(hypothesis_streams),
        reference_status=reference_status,
        duration_ms=duration_ms,
        scored_ranges_ms=tuple(scored_ranges_ms),
        reference_turns=tuple(reference_turns),
        hypothesis_turns=tuple(hypothesis_turns),
        reference_overlaps=tuple(reference_overlaps),
        hypothesis_overlaps=tuple(hypothesis_overlaps),
    )
    return _cpwer_for_input(item)[0]


def _configuration(item: IntegratedScoringInput) -> dict[str, Any]:
    return {
        "metric_definitions_version": INTEGRATED_METRIC_DEFINITIONS_VERSION,
        "cpwer_version": CPWER_VERSION,
        "normalization_version": "it-v1",
        "channel_mapping": (
            "sort speaker IDs, then choose the minimum total edit-cost one-to-one assignment; "
            "dummy channels represent missing or extra speakers"
        ),
        "reference_streams": "explicit human-reviewed speaker text channels",
        "hypothesis_streams": "explicit model/diarization-attributed speaker text channels",
        "overlap_treatment": (
            "retain every explicit speaker stream independently; overlapping reference words "
            "count once per reference channel and are never deduplicated"
        ),
        "interval_constraints": {
            "canonical_time_unit": "milliseconds",
            "bounds": "half-open [start_ms,end_ms)",
            "scored_ranges_ms": [list(value) for value in item.scored_ranges_ms],
            "reference_turn_count": len(item.reference_turns),
            "hypothesis_turn_count": len(item.hypothesis_turns),
            "reference_overlap_count": len(item.reference_overlaps),
            "hypothesis_overlap_count": len(item.hypothesis_overlaps),
            "policy": (
                "caller supplies pre-clipped text streams; intervals are validated provenance, "
                "not word clipping"
            ),
        },
        "denominator": "all normalized reference tokens across explicit reference speaker streams",
        "tcpwer": {
            "status": TCPWER_STATUS,
            "reason": (
                "P1R word timing is optional and no reproducible time-constrained "
                "speaker-attributed "
                "alignment contract or word-timing gold is locked"
            ),
        },
    }


def _sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stream_artifact_hash(streams: Sequence[SpeakerTextStream]) -> str:
    """Hash stream content for provenance without publishing the content itself."""

    return _sha256([stream.to_dict() for stream in streams])


def _tcpwer_deferred() -> dict[str, Any]:
    return {
        "status": TCPWER_STATUS,
        "value": None,
        "unit": "ratio",
        "reason": (
            "Deferred: P1R permits text-only hypotheses and does not lock word-timing gold "
            "or a time-constrained speaker-attributed alignment semantics."
        ),
    }


def score_integrated(
    item: IntegratedScoringInput,
    *,
    reviewed_commit: str | None = None,
    synthetic_provenance: Sequence[str] = (),
) -> dict[str, Any]:
    """Produce one separately identified integrated speaker-attributed report."""

    if not isinstance(item, IntegratedScoringInput):
        raise IntegratedScoringError("score_integrated requires IntegratedScoringInput")
    metric, detail = _cpwer_for_input(item)
    configuration = _configuration(item)
    return {
        "schema_version": INTEGRATED_SCORING_SCHEMA_VERSION,
        "report_type": "p1r-integrated-speaker-attributed",
        "metric_definitions_version": INTEGRATED_METRIC_DEFINITIONS_VERSION,
        "status": metric["status"],
        "configuration": configuration,
        "configuration_sha256": _sha256(configuration),
        "provenance": {
            "chunk_id": item.chunk_id,
            "hypothesis_id": item.hypothesis_id,
            "source_sha256": item.source_sha256,
            "reference_stream_artifact_sha256": _stream_artifact_hash(item.reference_streams),
            "hypothesis_stream_artifact_sha256": _stream_artifact_hash(item.hypothesis_streams),
            "model_fingerprint_sha256": item.model_fingerprint_sha256,
            "reviewed_commit": reviewed_commit,
            "synthetic_provenance": list(synthetic_provenance),
        },
        "metrics": {"cpwer": _public_metric(metric), "tcpwer": _tcpwer_deferred()},
        "limitations": [
            "cpWER is separate from lexical WER/ORC-WER and interval DER/JER.",
            "The report does not establish real archive or model quality when fixtures are "
            "synthetic.",
            "tcpWER is deferred pending locked time-constrained semantics and timed reference "
            "evidence.",
        ],
        "private_details": {"mapping": detail},
    }


def _aggregate_cpwer(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [metric for metric in metrics if metric["status"] in {"scored", "partial"}]
    if not scored:
        status = (
            "insufficient_evidence"
            if any(metric["status"] == "insufficient_evidence" for metric in metrics)
            else "unscorable"
        )
        return {
            "status": status,
            "value": None,
            "errors": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "reference_denominator": 0,
            "chunk_count": len(metrics),
        }
    fields = ("errors", "substitutions", "deletions", "insertions", "hypothesis_count")
    totals = {field: sum(int(metric[field]) for metric in scored) for field in fields}
    denominator = sum(int(metric["reference_denominator"]) for metric in scored)
    return {
        "status": "partial" if len(scored) != len(metrics) else "scored",
        "value": totals["errors"] / denominator if denominator else None,
        "reference_denominator": denominator,
        "chunk_count": len(metrics),
        "mapping_artifact": "details.json",
        **totals,
    }


def score_integrated_chunks(
    inputs: Iterable[IntegratedScoringInput],
    *,
    manifest_sha256: str | None = None,
    reviewed_commit: str | None = None,
    synthetic_provenance: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate integrated metrics without mixing them with ASR/DER reports."""

    items = tuple(inputs)
    if not items:
        raise IntegratedScoringError("at least one integrated scoring input is required")
    identities = [(item.chunk_id, item.model_fingerprint_sha256) for item in items]
    if len(identities) != len(set(identities)):
        raise IntegratedScoringError("duplicate chunk/model integrated scoring input")
    chunk_reports: list[dict[str, Any]] = []
    for item in items:
        metric, detail = _cpwer_for_input(item)
        chunk_reports.append(
            {
                "chunk_id": item.chunk_id,
                "hypothesis_id": item.hypothesis_id,
                "model_fingerprint_sha256": item.model_fingerprint_sha256 or "unknown",
                "duration_ms": item.duration_ms,
                "slice_ids": sorted({"all", *item.slice_ids}),
                "metric": metric,
                "mapping": detail,
            }
        )
    model_ids = sorted({report["model_fingerprint_sha256"] for report in chunk_reports})
    metric_statuses = {report["metric"]["status"] for report in chunk_reports}
    models: dict[str, Any] = {}
    for model_id in model_ids:
        model_reports = [
            report for report in chunk_reports if report["model_fingerprint_sha256"] == model_id
        ]
        slice_ids = sorted(
            {slice_id for report in model_reports for slice_id in report["slice_ids"]}
        )
        models[model_id] = {
            "hypothesis_count": len(model_reports),
            "slices": {
                slice_id: {
                    "chunk_count": len(
                        [report for report in model_reports if slice_id in report["slice_ids"]]
                    ),
                    "duration_ms": sum(
                        int(report["duration_ms"] or 0)
                        for report in model_reports
                        if slice_id in report["slice_ids"]
                    ),
                    "metrics": {
                        "cpwer": _aggregate_cpwer(
                            [
                                report["metric"]
                                for report in model_reports
                                if slice_id in report["slice_ids"]
                            ]
                        ),
                        "tcpwer": _tcpwer_deferred(),
                    },
                }
                for slice_id in slice_ids
            },
        }
    configuration = _configuration(items[0])
    return {
        "schema_version": INTEGRATED_SCORING_SCHEMA_VERSION,
        "report_type": "p1r-integrated-speaker-attributed",
        "metric_definitions_version": INTEGRATED_METRIC_DEFINITIONS_VERSION,
        "status": (
            "partial"
            if "scored" in metric_statuses and len(metric_statuses) > 1
            else "scored"
            if "scored" in metric_statuses
            else (
                "insufficient_evidence"
                if "insufficient_evidence" in metric_statuses
                else "unscorable"
            )
        ),
        "configuration": configuration,
        "configuration_sha256": _sha256(configuration),
        "provenance": {
            "manifest_sha256": manifest_sha256,
            "source_sha256": sorted({item.source_sha256 for item in items if item.source_sha256}),
            "reference_stream_artifact_sha256": sorted(
                {_stream_artifact_hash(item.reference_streams) for item in items}
            ),
            "hypothesis_stream_artifact_sha256": sorted(
                {_stream_artifact_hash(item.hypothesis_streams) for item in items}
            ),
            "model_fingerprint_sha256": model_ids,
            "reviewed_commit": reviewed_commit,
            "synthetic_provenance": list(synthetic_provenance),
        },
        "models": models,
        "limitations": [
            "Integrated cpWER is separate from lexical WER/ORC-WER and interval DER/JER.",
            "tcpWER remains explicitly deferred pending locked time-constrained semantics and "
            "timed reference evidence.",
        ],
        "private_details": {"chunks": chunk_reports},
    }


def _aggregate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "private_details"}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _render_report_html(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for model_id, model in report.get("models", {}).items():
        for slice_id, slice_report in model.get("slices", {}).items():
            for metric_name, metric in slice_report.get("metrics", {}).items():
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(model_id))}</td><td>{html.escape(str(slice_id))}</td>"
                    f"<td>{html.escape(str(metric_name))}</td><td>{html.escape(str(metric.get('status')))}</td>"
                    f"<td>{html.escape(str(metric.get('value')))}</td>"
                    f"<td>{html.escape(str(slice_report.get('chunk_count')))}</td>"
                    "</tr>"
                )
    if not rows and "metrics" in report:
        for metric_name, metric in report["metrics"].items():
            rows.append(
                "<tr><td>single</td><td>all</td>"
                f"<td>{html.escape(str(metric_name))}</td>"
                f"<td>{html.escape(str(metric.get('status')))}</td>"
                f"<td>{html.escape(str(metric.get('value')))}</td><td>1</td></tr>"
            )
    return (
        '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
        "<title>P1R integrated speaker-attributed score</title>"
        "<h1>P1R integrated speaker-attributed score</h1>"
        "<p>This sanitized aggregate is separate from lexical ASR and diarization reports.</p>"
        "<table><thead><tr><th>Model</th><th>Slice</th><th>Metric</th><th>Status</th>"
        "<th>Value</th><th>Chunks</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>\n"
    )


def write_integrated_score_report(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Write a new sanitized aggregate and private mapping artifact atomically."""

    if report.get("report_type") != "p1r-integrated-speaker-attributed":
        raise IntegratedScoringError("report is not an integrated speaker-attributed report")
    destination = Path(output_dir)
    if destination.exists():
        raise IntegratedScoringError(
            f"refusing to overwrite existing report directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        aggregate = _aggregate_report(report)
        details = {
            "schema_version": report["schema_version"],
            "report_type": report["report_type"],
            "provenance": report.get("provenance", {}),
            **dict(report.get("private_details", {})),
        }
        files = {
            "metrics.json": _json_bytes(aggregate),
            "details.json": _json_bytes(details),
            "report.html": _render_report_html(report).encode("utf-8"),
        }
        checksums: dict[str, str] = {}
        for name, content in files.items():
            path = temporary / name
            path.write_bytes(content)
            path.chmod(0o600)
            checksums[name] = hashlib.sha256(content).hexdigest()
        run = {
            "schema_version": INTEGRATED_SCORING_SCHEMA_VERSION,
            "report_type": report["report_type"],
            "metric_definitions_version": report["metric_definitions_version"],
            "artifact_sha256": checksums,
            "provenance": report.get("provenance", {}),
        }
        run_path = temporary / "run.json"
        run_path.write_bytes(_json_bytes(run))
        run_path.chmod(0o600)
        checksums["run.json"] = hashlib.sha256((temporary / "run.json").read_bytes()).hexdigest()
        temporary.chmod(0o700)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"output_dir": str(destination), "artifact_sha256": checksums}


__all__ = [
    "CPWER_VERSION",
    "INTEGRATED_METRIC_DEFINITIONS_VERSION",
    "INTEGRATED_SCORING_SCHEMA_VERSION",
    "IntegratedScoringError",
    "IntegratedScoringInput",
    "SpeakerTextStream",
    "TCPWER_STATUS",
    "score_cpwer",
    "score_integrated",
    "score_integrated_chunks",
    "write_integrated_score_report",
]

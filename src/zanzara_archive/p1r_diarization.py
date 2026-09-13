"""Word-blind diarization scoring for the P1R benchmark.

This module accepts only source-relative speaker and overlap intervals.  It
intentionally does not parse transcript payloads or import the lexical scorer.
That boundary prevents ASR word timestamps from becoming a hidden source of
diarization truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

DIARIZATION_EVALUATION_SCHEMA_VERSION = 1
DIARIZATION_METRIC_DEFINITIONS_VERSION = "p1r-diarization-v1"
DIARIZATION_EVALUATOR_VERSION = "native-interval-evaluator-v1"
CANONICAL_TIME_UNIT = "milliseconds"
DEFAULT_COLLAR_MS = 0


class DiarizationScoringError(ValueError):
    """Raised when interval scoring input is malformed."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _interval_artifact_hash(
    payload: Mapping[str, Any],
    turns: Sequence[Mapping[str, Any]] | None,
    overlaps: Sequence[Mapping[str, Any]] | None,
    duration_ms: int,
) -> str:
    """Hash only diarization inputs, never unrelated lexical payload fields."""

    return _sha256(
        {
            "episode_id": payload.get("episode_id"),
            "source_sha256": payload.get("source_sha256"),
            "duration_ms": duration_ms,
            "turns": list(turns or ()),
            "overlap_intervals": list(overlaps or ()) if overlaps is not None else None,
        }
    )


def _interval(value: Mapping[str, Any], field_name: str, duration_ms: int) -> tuple[int, int]:
    start_ms = value.get("start_ms")
    end_ms = value.get("end_ms")
    if (
        isinstance(start_ms, bool)
        or not isinstance(start_ms, int)
        or isinstance(end_ms, bool)
        or not isinstance(end_ms, int)
        or start_ms < 0
        or end_ms <= start_ms
        or end_ms > duration_ms
    ):
        raise DiarizationScoringError(f"{field_name} must be a bounded integer half-open interval")
    return start_ms, end_ms


def _duration_hint(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("duration_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DiarizationScoringError("duration_ms must be a positive integer")
    return value


def _raw_turns(payload: Mapping[str, Any]) -> object:
    if "standard_turns" in payload:
        return payload["standard_turns"]
    if "turns" in payload:
        return payload["turns"]
    diarization = payload.get("diarization")
    if isinstance(diarization, Mapping):
        return diarization.get("standard_turns", diarization.get("turns"))
    return None


def _raw_overlaps(payload: Mapping[str, Any]) -> tuple[bool, object]:
    for key in ("overlap_intervals", "overlaps"):
        if key in payload:
            return True, payload[key]
    diarization = payload.get("diarization")
    if isinstance(diarization, Mapping):
        for key in ("overlap_intervals", "overlaps"):
            if key in diarization:
                return True, diarization[key]
    return False, None


def _speaker_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiarizationScoringError(f"{field_name} must be non-empty text")
    return value


def _parse_turns(
    raw: object, *, duration_ms: int, field_name: str
) -> tuple[dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DiarizationScoringError(f"{field_name} must be a list")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise DiarizationScoringError(f"{field_name}[{index}] must be an object")
        start_ms, end_ms = _interval(value, f"{field_name}[{index}]", duration_ms)
        result.append(
            {
                "speaker_id": _speaker_id(
                    value.get("speaker_id"), f"{field_name}[{index}].speaker_id"
                ),
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
    return tuple(
        sorted(result, key=lambda item: (item["start_ms"], item["end_ms"], item["speaker_id"]))
    )


def _parse_overlaps(
    raw: object, *, duration_ms: int, field_name: str
) -> tuple[dict[str, Any], ...] | None:
    if raw is None:
        raise DiarizationScoringError(f"{field_name} must be a list")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DiarizationScoringError(f"{field_name} must be a list")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise DiarizationScoringError(f"{field_name}[{index}] must be an object")
        speakers = value.get("speaker_ids")
        if not isinstance(speakers, Sequence) or isinstance(speakers, (str, bytes)):
            raise DiarizationScoringError(f"{field_name}[{index}].speaker_ids must be a list")
        speaker_ids = tuple(
            _speaker_id(speaker, f"{field_name}[{index}].speaker_ids[{speaker_index}]")
            for speaker_index, speaker in enumerate(speakers)
        )
        if len(speaker_ids) < 2 or len(set(speaker_ids)) != len(speaker_ids):
            raise DiarizationScoringError(
                f"{field_name}[{index}].speaker_ids must contain at least two unique speakers"
            )
        start_ms, end_ms = _interval(value, f"{field_name}[{index}]", duration_ms)
        result.append({"speaker_ids": speaker_ids, "start_ms": start_ms, "end_ms": end_ms})
    return tuple(sorted(result, key=lambda item: (item["start_ms"], item["end_ms"])))


def _infer_duration(payload: Mapping[str, Any], raw_turns: object, raw_overlaps: object) -> int:
    duration_ms = _duration_hint(payload)
    if duration_ms is not None:
        return duration_ms
    endpoints: list[int] = []
    for raw in (raw_turns, raw_overlaps):
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                if isinstance(item, Mapping) and isinstance(item.get("end_ms"), int):
                    endpoints.append(item["end_ms"])
    if not endpoints or max(endpoints) <= 0:
        raise DiarizationScoringError("duration_ms is required when interval inputs are empty")
    return max(endpoints)


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted((start, end) for start, end in ranges if start < end)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _parse_ranges(
    value: Mapping[str, Any], *, duration_ms: int, field_name: str
) -> tuple[tuple[int, int], ...]:
    raw_ranges = value.get("ranges_ms")
    if raw_ranges is None:
        raw_ranges = [value]
    if not isinstance(raw_ranges, Sequence) or isinstance(raw_ranges, (str, bytes)):
        raise DiarizationScoringError(f"{field_name}.ranges_ms must be a list")
    ranges: list[tuple[int, int]] = []
    for index, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, Mapping):
            if isinstance(raw_range, Sequence) and not isinstance(raw_range, (str, bytes)):
                if len(raw_range) != 2:
                    raise DiarizationScoringError(
                        f"{field_name}.ranges_ms[{index}] must have two values"
                    )
                raw_range = {"start_ms": raw_range[0], "end_ms": raw_range[1]}
            else:
                raise DiarizationScoringError(
                    f"{field_name}.ranges_ms[{index}] must be an interval"
                )
        ranges.append(_interval(raw_range, f"{field_name}.ranges_ms[{index}]", duration_ms))
    if not ranges:
        raise DiarizationScoringError(f"{field_name}.ranges_ms must not be empty")
    return _merge_ranges(ranges)


def _condition_slices(
    reference: Mapping[str, Any],
    explicit: Sequence[Mapping[str, Any]] | None,
    *,
    duration_ms: int,
) -> tuple[dict[str, Any], ...]:
    raw_slices: object = explicit
    if raw_slices is None:
        raw_slices = reference.get("condition_slices", reference.get("slices", ()))
    if not isinstance(raw_slices, Sequence) or isinstance(raw_slices, (str, bytes)):
        raise DiarizationScoringError("condition_slices must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_slices):
        if not isinstance(value, Mapping):
            raise DiarizationScoringError(f"condition_slices[{index}] must be an object")
        slice_id = value.get("slice_id", value.get("id"))
        if not isinstance(slice_id, str) or not slice_id.strip() or slice_id in seen:
            raise DiarizationScoringError(f"condition_slices[{index}].slice_id must be unique text")
        seen.add(slice_id)
        result.append(
            {
                "slice_id": slice_id,
                "label": value.get("label", value.get("condition", slice_id)),
                "condition": value.get("condition"),
                "ranges_ms": _parse_ranges(
                    value, duration_ms=duration_ms, field_name=f"condition_slices[{index}]"
                ),
            }
        )
    return tuple(result)


def _collared_ranges(
    ranges: Sequence[tuple[int, int]],
    reference_turns: Sequence[Mapping[str, Any]],
    collar_ms: int,
) -> tuple[tuple[int, int], ...]:
    if collar_ms < 0:
        raise DiarizationScoringError("collar_ms must be non-negative")
    if collar_ms == 0:
        return _merge_ranges(ranges)
    excluded = []
    for turn in reference_turns:
        for boundary in (turn["start_ms"], turn["end_ms"]):
            excluded.append((max(0, boundary - collar_ms), boundary + collar_ms))
    remaining = list(_merge_ranges(ranges))
    for excluded_start, excluded_end in _merge_ranges(excluded):
        next_ranges: list[tuple[int, int]] = []
        for start, end in remaining:
            if excluded_end <= start or excluded_start >= end:
                next_ranges.append((start, end))
                continue
            if start < excluded_start:
                next_ranges.append((start, min(end, excluded_start)))
            if excluded_end < end:
                next_ranges.append((max(start, excluded_end), end))
        remaining = next_ranges
    return tuple(remaining)


Event = tuple[int, int, frozenset[str], frozenset[str]]


def _events(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
) -> tuple[Event, ...]:
    boundaries = {point for interval in ranges for point in interval}
    for turn in (*reference, *hypothesis):
        for start, end in ranges:
            if turn["start_ms"] < end and turn["end_ms"] > start:
                boundaries.update(
                    point
                    for point in (
                        max(turn["start_ms"], start),
                        min(turn["end_ms"], end),
                    )
                    if start <= point <= end
                )
    points = sorted(boundaries)
    result: list[Event] = []
    for start, end in zip(points, points[1:], strict=False):
        if not any(range_start <= start and end <= range_end for range_start, range_end in ranges):
            continue
        reference_ids = frozenset(
            turn["speaker_id"]
            for turn in reference
            if turn["start_ms"] < end and turn["end_ms"] > start
        )
        hypothesis_ids = frozenset(
            turn["speaker_id"]
            for turn in hypothesis
            if turn["start_ms"] < end and turn["end_ms"] > start
        )
        result.append((start, end, reference_ids, hypothesis_ids))
    return tuple(result)


def _speaker_mapping(events: Sequence[Event]) -> dict[str, str]:
    scores: dict[tuple[str, str], int] = defaultdict(int)
    for start, end, reference_ids, hypothesis_ids in events:
        for reference_id in reference_ids:
            for hypothesis_id in hypothesis_ids:
                scores[(hypothesis_id, reference_id)] += end - start
    hypothesis_ids = sorted({speaker for _, _, _, speakers in events for speaker in speakers})
    reference_ids = sorted({speaker for _, _, speakers, _ in events for speaker in speakers})
    if not hypothesis_ids or not reference_ids:
        return {}

    def maximum_score(rows: Sequence[str], columns: Sequence[str]) -> int:
        if not rows or not columns:
            return 0
        weights = [
            [scores.get((row, column), 0) for column in columns] + [0] * len(rows) for row in rows
        ]
        column_count = len(weights[0])
        row_potential = [0] * (len(rows) + 1)
        column_potential = [0] * (column_count + 1)
        assigned_row = [0] * (column_count + 1)
        previous_column = [0] * (column_count + 1)
        for row_index in range(1, len(rows) + 1):
            assigned_row[0] = row_index
            current_column = 0
            minimum = [math.inf] * (column_count + 1)
            visited = [False] * (column_count + 1)
            while True:
                visited[current_column] = True
                matched_row = assigned_row[current_column]
                delta = math.inf
                next_column = 0
                for column_index in range(1, column_count + 1):
                    if visited[column_index]:
                        continue
                    reduced_cost = (
                        -weights[matched_row - 1][column_index - 1]
                        - row_potential[matched_row]
                        - column_potential[column_index]
                    )
                    if reduced_cost < minimum[column_index]:
                        minimum[column_index] = reduced_cost
                        previous_column[column_index] = current_column
                    if minimum[column_index] < delta:
                        delta = minimum[column_index]
                        next_column = column_index
                for column_index in range(column_count + 1):
                    if visited[column_index]:
                        row_potential[assigned_row[column_index]] += delta
                        column_potential[column_index] -= delta
                    else:
                        minimum[column_index] -= delta
                current_column = next_column
                if assigned_row[current_column] == 0:
                    break
            while True:
                prior = previous_column[current_column]
                assigned_row[current_column] = assigned_row[prior]
                current_column = prior
                if current_column == 0:
                    break
        return sum(
            weights[assigned_row[column_index] - 1][column_index - 1]
            for column_index in range(1, len(columns) + 1)
            if assigned_row[column_index]
        )

    selected: list[str | None] = []
    used: set[str] = set()
    remaining_score = maximum_score(hypothesis_ids, reference_ids)
    for index, hypothesis_id in enumerate(hypothesis_ids):
        options: list[str | None] = [None, *(item for item in reference_ids if item not in used)]
        options.sort(key=lambda item: item or "~", reverse=True)
        for candidate in options:
            candidate_score = scores.get((hypothesis_id, candidate), 0) if candidate else 0
            next_used = used | ({candidate} if candidate else set())
            tail_score = maximum_score(
                hypothesis_ids[index + 1 :],
                [item for item in reference_ids if item not in next_used],
            )
            if candidate_score + tail_score == remaining_score:
                selected.append(candidate)
                used = next_used
                remaining_score = tail_score
                break
        else:
            raise DiarizationScoringError("speaker mapping could not be reconstructed")
    return {
        hypothesis_id: reference_id
        for hypothesis_id, reference_id in zip(hypothesis_ids, selected, strict=True)
        if reference_id is not None
    }


def _metric(
    *,
    value: float | int | None,
    status: str,
    numerator: float | int,
    denominator: float | int,
    denominator_name: str,
    unit: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "value": value,
        "status": status,
        "numerator": numerator,
        "denominator": denominator,
        "denominator_name": denominator_name,
        "unit": unit,
        **extra,
    }


def _der(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    *,
    collar_ms: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    scoring_ranges = _collared_ranges(ranges, reference, collar_ms)
    events = _events(reference, hypothesis, scoring_ranges)
    mapping = _speaker_mapping(events)
    missed_ms = false_alarm_ms = confusion_ms = reference_speaker_ms = 0
    for start, end, reference_ids, hypothesis_ids in events:
        duration_ms = end - start
        reference_count = len(reference_ids)
        hypothesis_count = len(hypothesis_ids)
        correct = sum(
            1
            for speaker_id in hypothesis_ids
            if speaker_id in mapping and mapping[speaker_id] in reference_ids
        )
        reference_speaker_ms += reference_count * duration_ms
        missed_ms += max(0, reference_count - hypothesis_count) * duration_ms
        false_alarm_ms += max(0, hypothesis_count - reference_count) * duration_ms
        confusion_ms += max(0, min(reference_count, hypothesis_count) - correct) * duration_ms
    denominator_s = reference_speaker_ms / 1000
    numerator_s = (missed_ms + false_alarm_ms + confusion_ms) / 1000
    metric = _metric(
        value=numerator_s / denominator_s if denominator_s else None,
        status="scored" if denominator_s else "insufficient_evidence",
        numerator=numerator_s,
        denominator=denominator_s,
        denominator_name="reference_speaker_seconds",
        unit="ratio",
        components_seconds={
            "missed_speech": missed_ms / 1000,
            "false_alarm": false_alarm_ms / 1000,
            "speaker_confusion": confusion_ms / 1000,
        },
        collar_ms=collar_ms,
        overlap_included=collar_ms == 0,
        mapping=dict(mapping),
        definition=(
            "per-speaker missed, false-alarm and confusion duration divided by "
            "reference speaker-time; mapping maximizes one-to-one scored overlap"
        ),
    )
    return metric, mapping


def _jer(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    events = _events(reference, hypothesis, ranges)
    reference_ids = sorted({speaker for _, _, speakers, _ in events for speaker in speakers})
    errors: list[float] = []
    for reference_id in reference_ids:
        hypothesis_id = next(
            (speaker for speaker, target in mapping.items() if target == reference_id), None
        )
        intersection_ms = union_ms = 0
        for start, end, active_reference, active_hypothesis in events:
            reference_has = reference_id in active_reference
            hypothesis_has = hypothesis_id is not None and hypothesis_id in active_hypothesis
            if reference_has or hypothesis_has:
                union_ms += end - start
            if reference_has and hypothesis_has:
                intersection_ms += end - start
        if union_ms:
            errors.append(1 - intersection_ms / union_ms)
    return _metric(
        value=sum(errors) / len(errors) if errors else None,
        status="scored" if errors else "insufficient_evidence",
        numerator=sum(errors),
        denominator=len(errors),
        denominator_name="reference_speakers_with_scored_time",
        unit="ratio",
        mapping=dict(mapping),
        definition="mean per-reference-speaker (1 - intersection/union), using the DER map",
    )


def _count_error(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    reference_ids = {
        turn["speaker_id"]
        for turn in reference
        if any(turn["start_ms"] < end and turn["end_ms"] > start for start, end in ranges)
    }
    hypothesis_ids = {
        turn["speaker_id"]
        for turn in hypothesis
        if any(turn["start_ms"] < end and turn["end_ms"] > start for start, end in ranges)
    }
    if not reference_ids:
        return _metric(
            value=None,
            status="insufficient_evidence",
            numerator=0,
            denominator=0,
            denominator_name="reference_speakers",
            unit="speakers",
            signed=None,
            absolute=None,
            reference_count=0,
            hypothesis_count=len(hypothesis_ids),
        )
    signed = len(hypothesis_ids) - len(reference_ids)
    return _metric(
        value=signed,
        status="scored",
        numerator=signed,
        denominator=len(reference_ids),
        denominator_name="reference_speakers",
        unit="speakers",
        signed=signed,
        absolute=abs(signed),
        reference_count=len(reference_ids),
        hypothesis_count=len(hypothesis_ids),
        definition="unique active hypothesis speakers minus reference speakers",
    )


def _union_intersection(
    reference: Sequence[Mapping[str, Any]], hypothesis: Sequence[Mapping[str, Any]]
) -> tuple[int, int, int]:
    reference_ranges = _merge_ranges([(item["start_ms"], item["end_ms"]) for item in reference])
    hypothesis_ranges = _merge_ranges([(item["start_ms"], item["end_ms"]) for item in hypothesis])
    boundaries = sorted(
        {point for interval in (*reference_ranges, *hypothesis_ranges) for point in interval}
    )
    reference_ms = hypothesis_ms = intersection_ms = 0
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        in_reference = any(item[0] <= start and end <= item[1] for item in reference_ranges)
        in_hypothesis = any(item[0] <= start and end <= item[1] for item in hypothesis_ranges)
        if in_reference:
            reference_ms += end - start
        if in_hypothesis:
            hypothesis_ms += end - start
        if in_reference and in_hypothesis:
            intersection_ms += end - start
    return reference_ms, hypothesis_ms, intersection_ms


def _clip_overlaps(
    intervals: Sequence[Mapping[str, Any]], ranges: Sequence[tuple[int, int]]
) -> tuple[dict[str, Any], ...]:
    clipped: list[dict[str, Any]] = []
    for item in intervals:
        for start, end in ranges:
            clipped_start = max(item["start_ms"], start)
            clipped_end = min(item["end_ms"], end)
            if clipped_start < clipped_end:
                clipped.append(
                    {
                        "speaker_ids": item["speaker_ids"],
                        "start_ms": clipped_start,
                        "end_ms": clipped_end,
                    }
                )
    return tuple(clipped)


def _overlap_metrics(
    reference: Sequence[Mapping[str, Any]] | None,
    hypothesis: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if reference is None:
        return {
            "status": "not_available",
            "reference_supported": False,
            "precision": None,
            "recall": None,
            "f1": None,
            "definition": (
                "duration-based overlap interval detection; explicit reference truth required"
            ),
        }
    reference_ms, hypothesis_ms, intersection_ms = _union_intersection(reference, hypothesis or ())
    precision = intersection_ms / hypothesis_ms if hypothesis_ms else None
    recall = intersection_ms / reference_ms if reference_ms else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "status": "scored" if reference_ms else "insufficient_evidence",
        "reference_supported": True,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_ms": intersection_ms,
        "false_positive_ms": max(0, hypothesis_ms - intersection_ms),
        "false_negative_ms": max(0, reference_ms - intersection_ms),
        "reference_overlap_ms": reference_ms,
        "hypothesis_overlap_ms": hypothesis_ms,
        "definition": "duration-based overlap interval detection on explicit half-open intervals",
    }


def _unscorable_slice(
    label: str,
    ranges: Sequence[tuple[int, int]],
    *,
    condition: str | None = None,
) -> dict[str, Any]:
    metric = {
        "value": None,
        "status": "unscorable",
        "reason": "reference and hypothesis speaker intervals are required",
    }
    slice_metadata: dict[str, Any] = {
        "label": label,
        "ranges_ms": [list(item) for item in ranges],
    }
    if condition is not None:
        slice_metadata["condition"] = condition
    return {
        "status": "unscorable",
        "slice": slice_metadata,
        "primary_der": metric,
        "jer": metric,
        "speaker_count_error": metric,
        "overlap": {
            "status": "unscorable",
            "reference_supported": False,
            "precision": None,
            "recall": None,
            "f1": None,
        },
    }


def _score_slice(
    reference_turns: Sequence[Mapping[str, Any]],
    hypothesis_turns: Sequence[Mapping[str, Any]],
    reference_overlaps: Sequence[Mapping[str, Any]] | None,
    hypothesis_overlaps: Sequence[Mapping[str, Any]] | None,
    ranges: Sequence[tuple[int, int]],
    *,
    collar_ms: int,
    label: str,
    condition: str | None = None,
) -> dict[str, Any]:
    primary_der, mapping = _der(reference_turns, hypothesis_turns, ranges, collar_ms=collar_ms)
    clipped_reference_overlaps = (
        _clip_overlaps(reference_overlaps, ranges) if reference_overlaps is not None else None
    )
    clipped_hypothesis_overlaps = _clip_overlaps(hypothesis_overlaps or (), ranges)
    slice_metadata: dict[str, Any] = {
        "label": label,
        "ranges_ms": [list(item) for item in ranges],
    }
    if condition is not None:
        slice_metadata["condition"] = condition
    return {
        "status": "scored" if primary_der["status"] == "scored" else "insufficient_evidence",
        "slice": slice_metadata,
        "primary_der": primary_der,
        "jer": _jer(reference_turns, hypothesis_turns, _merge_ranges(ranges), mapping),
        "speaker_count_error": _count_error(reference_turns, hypothesis_turns, ranges),
        "overlap": _overlap_metrics(clipped_reference_overlaps, clipped_hypothesis_overlaps),
    }


def score_diarization(
    reference_payload: Mapping[str, Any],
    hypothesis_payload: Mapping[str, Any],
    *,
    condition_slices: Sequence[Mapping[str, Any]] | None = None,
    collar_ms: int = DEFAULT_COLLAR_MS,
) -> dict[str, Any]:
    """Score diarization intervals without reading lexical transcript fields.

    ``standard_turns``/``turns`` and explicit ``overlap_intervals`` are the
    only score inputs.  A missing reference interval stream returns an
    ``unscorable`` report; it is never reconstructed from words.
    """

    if not isinstance(reference_payload, Mapping) or not isinstance(hypothesis_payload, Mapping):
        raise DiarizationScoringError("reference and hypothesis must be objects")
    for field_name in ("episode_id", "source_sha256"):
        reference_value = reference_payload.get(field_name)
        hypothesis_value = hypothesis_payload.get(field_name)
        if (
            reference_value is not None
            and hypothesis_value is not None
            and reference_value != hypothesis_value
        ):
            raise DiarizationScoringError(f"reference and hypothesis {field_name} differ")
    reference_raw_turns = _raw_turns(reference_payload)
    hypothesis_raw_turns = _raw_turns(hypothesis_payload)
    reference_has_overlaps, reference_raw_overlaps = _raw_overlaps(reference_payload)
    _, hypothesis_raw_overlaps = _raw_overlaps(hypothesis_payload)
    reference_duration = _infer_duration(
        reference_payload, reference_raw_turns, reference_raw_overlaps
    )
    hypothesis_duration = _infer_duration(
        hypothesis_payload, hypothesis_raw_turns, hypothesis_raw_overlaps
    )
    if reference_duration != hypothesis_duration:
        raise DiarizationScoringError("reference and hypothesis duration_ms differ")
    duration_ms = reference_duration
    reference_turns = _parse_turns(
        reference_raw_turns, duration_ms=duration_ms, field_name="reference.turns"
    )
    hypothesis_turns = _parse_turns(
        hypothesis_raw_turns, duration_ms=duration_ms, field_name="hypothesis.turns"
    )
    reference_overlaps = (
        _parse_overlaps(
            reference_raw_overlaps,
            duration_ms=duration_ms,
            field_name="reference.overlap_intervals",
        )
        if reference_has_overlaps
        else None
    )
    hypothesis_overlaps = (
        _parse_overlaps(
            hypothesis_raw_overlaps,
            duration_ms=duration_ms,
            field_name="hypothesis.overlap_intervals",
        )
        if hypothesis_raw_overlaps is not None
        else ()
    )
    slices = _condition_slices(reference_payload, condition_slices, duration_ms=duration_ms)
    all_ranges = ((0, duration_ms),)
    if reference_turns is None or hypothesis_turns is None:
        all_result = _unscorable_slice("all scored audio", all_ranges)
        slice_results = {
            "all": all_result,
            **{
                item["slice_id"]: _unscorable_slice(
                    item["label"], item["ranges_ms"], condition=item["condition"]
                )
                for item in slices
            },
        }
        status = "unscorable"
    else:
        all_result = _score_slice(
            reference_turns,
            hypothesis_turns,
            reference_overlaps,
            hypothesis_overlaps,
            all_ranges,
            collar_ms=collar_ms,
            label="all scored audio",
        )
        slice_results = {"all": all_result}
        for item in slices:
            slice_results[item["slice_id"]] = _score_slice(
                reference_turns,
                hypothesis_turns,
                reference_overlaps,
                hypothesis_overlaps,
                item["ranges_ms"],
                collar_ms=collar_ms,
                label=item["label"],
                condition=item["condition"],
            )
        status = all_result["status"]
    configuration = {
        "evaluator": DIARIZATION_EVALUATOR_VERSION,
        "metric_definitions": DIARIZATION_METRIC_DEFINITIONS_VERSION,
        "canonical_time_unit": CANONICAL_TIME_UNIT,
        "der": "standard overlap-inclusive turns, one-to-one map maximizing speaker-time overlap",
        "collar_ms": collar_ms,
        "jer": "mean per-reference-speaker (1 - intersection/union) using the DER map",
        "speaker_count_error": (
            "unique active hypothesis speakers minus reference speakers per slice"
        ),
        "overlap": "duration precision/recall/F1 over explicit overlap interval unions",
        "word_inputs": "ignored; no lexical fields are read",
    }
    return {
        "schema_version": DIARIZATION_EVALUATION_SCHEMA_VERSION,
        "metric_definitions_version": DIARIZATION_METRIC_DEFINITIONS_VERSION,
        "evaluator_version": DIARIZATION_EVALUATOR_VERSION,
        "status": status,
        "provenance": {
            "episode_id": reference_payload.get("episode_id"),
            "source_sha256": reference_payload.get("source_sha256"),
            "duration_ms": duration_ms,
            "reference_artifact_sha256": _interval_artifact_hash(
                reference_payload, reference_turns, reference_overlaps, duration_ms
            ),
            "hypothesis_artifact_sha256": _interval_artifact_hash(
                hypothesis_payload, hypothesis_turns, hypothesis_overlaps, duration_ms
            ),
            "reference_status": reference_payload.get("status"),
            "reference_actor_type": reference_payload.get("actor_type"),
            "reference_interval_count": len(reference_turns or ()),
            "hypothesis_interval_count": len(hypothesis_turns or ()),
            "reference_overlap_truth": reference_has_overlaps,
        },
        "configuration": configuration,
        "configuration_sha256": _sha256(configuration),
        "slices": slice_results,
        "limitations": [
            "Synthetic interval fixtures do not establish Community-1 model quality.",
            "Missing interval truth is unscorable and is never inferred from ASR words.",
            "Overlap metrics are unavailable unless explicit reference overlap intervals "
            "are supplied.",
        ],
    }


__all__ = [
    "CANONICAL_TIME_UNIT",
    "DEFAULT_COLLAR_MS",
    "DIARIZATION_EVALUATION_SCHEMA_VERSION",
    "DIARIZATION_EVALUATOR_VERSION",
    "DIARIZATION_METRIC_DEFINITIONS_VERSION",
    "DiarizationScoringError",
    "score_diarization",
]

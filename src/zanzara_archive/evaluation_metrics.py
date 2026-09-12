"""Deterministic CPU metrics and private artifacts for E2 evaluation.

The evaluator consumes the JSON shape exported by :mod:`annotations`.  It does
not call a model, a paid provider, or an evaluation service.  Canonical input
intervals remain integer milliseconds; seconds are used only in the derived
diarization and timing metric values written to a report.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .annotations import canonical_hash, validate_annotation_payload

EVALUATION_SCHEMA_VERSION = 1
METRIC_DEFINITIONS_VERSION = "e2-v1"
NORMALIZATION_VERSION = "it-v1"
QUANTILE_VERSION = "linear-interpolation-v1"
TIMING_UNIT = "seconds"
CANONICAL_TIME_UNIT = "milliseconds"
DEFAULT_THRESHOLDS = {"normalized_wer": 0.25, "primary_der": 0.25}
_SHA256_LENGTH = 64


class EvaluationValidationError(ValueError):
    """Raised when an evaluation input cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """Validated reference or hypothesis data with source-relative timings."""

    payload: dict[str, Any]
    source_sha256: str
    duration_ms: int
    words: tuple[dict[str, Any], ...]
    standard_turns: tuple[dict[str, Any], ...]
    exclusive_turns: tuple[dict[str, Any], ...]
    overlaps: tuple[dict[str, Any], ...]
    unintelligible_spans: tuple[tuple[int, int], ...]
    split: dict[str, Any]
    split_declared: bool
    split_sha256: str | None
    slices: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Alignment:
    """One deterministic Levenshtein alignment."""

    operations: tuple[tuple[str, int | None, int | None], ...]
    substitutions: int
    deletions: int
    insertions: int

    @property
    def errors(self) -> int:
        """Return the total edit count."""

        return self.substitutions + self.deletions + self.insertions


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_italian(text: str) -> str:
    """Apply the frozen E1 ``it-v1`` normalization definition."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _validate_interval(
    value: Mapping[str, Any], field_name: str, duration_ms: int
) -> tuple[int, int]:
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
        raise EvaluationValidationError(
            f"{field_name} must be a bounded integer millisecond interval"
        )
    return start_ms, end_ms


def _validate_slices(value: object, duration_ms: int) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationValidationError("evaluation slices must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_slice in enumerate(value):
        if not isinstance(raw_slice, Mapping):
            raise EvaluationValidationError(f"slices[{index}] must be an object")
        slice_id = raw_slice.get("slice_id", raw_slice.get("id"))
        if not isinstance(slice_id, str) or not slice_id.strip() or slice_id in seen:
            raise EvaluationValidationError(f"slices[{index}].slice_id must be unique text")
        start_ms, end_ms = _validate_interval(raw_slice, f"slices[{index}]", duration_ms)
        seen.add(slice_id)
        result.append(
            {
                "slice_id": slice_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "label": raw_slice.get("label", slice_id),
            }
        )
    return tuple(result)


def _canonical_word(raw_word: Mapping[str, Any], index: int, duration_ms: int) -> dict[str, Any]:
    word_id = raw_word.get("word_id", f"word-{index}")
    text = raw_word.get("text", raw_word.get("word"))
    if not isinstance(word_id, str) or not word_id.strip():
        raise EvaluationValidationError(f"words[{index}].word_id must be non-empty text")
    if not isinstance(text, str):
        raise EvaluationValidationError(f"words[{index}].text must be text")
    start_ms, end_ms = _validate_interval(raw_word, f"words[{index}]", duration_ms)
    speaker_id = raw_word.get("speaker_id")
    if speaker_id is not None and (not isinstance(speaker_id, str) or not speaker_id.strip()):
        raise EvaluationValidationError(f"words[{index}].speaker_id must be text or null")
    return {
        "word_id": word_id,
        "text": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_id": speaker_id,
        "overlap": bool(raw_word.get("overlap", False)),
        "unintelligible": bool(raw_word.get("unintelligible", False)),
    }


def _canonical_turn(raw_turn: Mapping[str, Any], index: int, duration_ms: int) -> dict[str, Any]:
    speaker_id = raw_turn.get("speaker_id")
    if not isinstance(speaker_id, str) or not speaker_id.strip():
        raise EvaluationValidationError(f"turns[{index}].speaker_id must be non-empty text")
    start_ms, end_ms = _validate_interval(raw_turn, f"turns[{index}]", duration_ms)
    return {"speaker_id": speaker_id, "start_ms": start_ms, "end_ms": end_ms}


def _canonical_overlap(
    raw_overlap: Mapping[str, Any], index: int, duration_ms: int
) -> dict[str, Any]:
    speakers = raw_overlap.get("speaker_ids")
    if not isinstance(speakers, Sequence) or isinstance(speakers, (str, bytes)):
        raise EvaluationValidationError(f"overlaps[{index}].speaker_ids must be a list")
    speaker_ids = tuple(item for item in speakers if isinstance(item, str) and item.strip())
    if len(speaker_ids) < 2 or len(set(speaker_ids)) != len(speaker_ids):
        raise EvaluationValidationError(f"overlaps[{index}] requires unique speaker IDs")
    start_ms, end_ms = _validate_interval(raw_overlap, f"overlaps[{index}]", duration_ms)
    return {"speaker_ids": speaker_ids, "start_ms": start_ms, "end_ms": end_ms}


def parse_evaluation_document(payload: Mapping[str, Any], *, role: str) -> EvaluationDocument:
    """Validate one annotation-shaped document without judging human truth."""

    if role not in {"reference", "hypothesis"}:
        raise EvaluationValidationError("document role must be reference or hypothesis")
    if not isinstance(payload, Mapping):
        raise EvaluationValidationError(f"{role} must be a JSON object")
    raw = dict(payload)
    if raw.get("time_origin_ms", 0) != 0:
        raise EvaluationValidationError(f"{role} time_origin_ms must remain zero")
    # P1-04's attributed export keeps diarization views under one nested key.
    # Flatten that existing contract without changing the input artifact.
    diarization = raw.get("diarization")
    if isinstance(diarization, Mapping) and "standard_turns" not in raw:
        raw["standard_turns"] = diarization.get("standard_turns", [])
        raw["exclusive_turns"] = diarization.get("exclusive_turns", [])
        raw["overlap_intervals"] = diarization.get(
            "overlap_intervals", diarization.get("overlaps", [])
        )
    raw.setdefault("status", "draft")
    raw.setdefault("actor_type", "machine")
    raw.setdefault("reviewer", "machine")
    raw.setdefault("reviewed_word_ids", [])
    raw.setdefault("manual_timing_word_ids", [])
    raw.setdefault("unintelligible_spans", [])
    raw.setdefault("standard_turns", [])
    raw.setdefault("exclusive_turns", [])
    raw.setdefault("overlap_intervals", [])
    actor_type = raw.get("actor_type", "machine")
    if actor_type not in {"machine", "human"}:
        raise EvaluationValidationError("actor_type must be machine or human")
    # Reuse P1-05's schema and provenance checks.  A machine/draft document is
    # valid fixture input, but can never make the resulting quality verdict pass.
    try:
        normalized = validate_annotation_payload(
            raw,
            actor_type=actor_type,
            reviewer=raw.get("reviewer", "machine") if actor_type == "machine" else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationValidationError(f"invalid {role} annotation: {exc}") from exc
    source_sha256 = normalized["source_sha256"]
    duration_ms = normalized["duration_ms"]
    words: list[dict[str, Any]] = []
    previous_start = -1
    for index, raw_word in enumerate(normalized["words"]):
        word = _canonical_word(raw_word, index, duration_ms)
        if word["start_ms"] < previous_start:
            raise EvaluationValidationError(f"{role} word timings must be monotonic")
        previous_start = word["start_ms"]
        words.append(word)
    standard_turns = tuple(
        _canonical_turn(item, index, duration_ms)
        for index, item in enumerate(normalized["standard_turns"])
    )
    exclusive_turns = tuple(
        _canonical_turn(item, index, duration_ms)
        for index, item in enumerate(normalized["exclusive_turns"])
    )
    overlaps = tuple(
        _canonical_overlap(item, index, duration_ms)
        for index, item in enumerate(normalized["overlap_intervals"])
    )
    masks = tuple(
        _validate_interval(item, f"{role}.unintelligible_spans[{index}]", duration_ms)
        for index, item in enumerate(normalized["unintelligible_spans"])
    )
    split = normalized["split"]
    split_sha256 = normalized.get("split_sha256")
    if split_sha256 != canonical_hash(split):
        raise EvaluationValidationError(f"{role} split_sha256 does not match split")
    return EvaluationDocument(
        payload=normalized,
        source_sha256=source_sha256,
        duration_ms=duration_ms,
        words=tuple(words),
        standard_turns=standard_turns,
        exclusive_turns=exclusive_turns,
        overlaps=overlaps,
        unintelligible_spans=masks,
        split=split,
        split_declared="split" in raw,
        split_sha256=split_sha256,
        slices=_validate_slices(raw.get("slices", raw.get("evaluation_slices")), duration_ms),
    )


def _token_text(word: Mapping[str, Any], *, normalized: bool) -> str:
    text = str(word["text"])
    return normalize_italian(text) if normalized else text


def _is_masked(word: Mapping[str, Any], masks: Sequence[tuple[int, int]]) -> bool:
    if word.get("unintelligible", False):
        return True
    return any(
        start_ms < word["end_ms"] and end_ms > word["start_ms"] for start_ms, end_ms in masks
    )


def _in_ranges(word: Mapping[str, Any], ranges: Sequence[tuple[int, int]]) -> bool:
    midpoint = (word["start_ms"] + word["end_ms"]) // 2
    return any(start_ms <= midpoint < end_ms for start_ms, end_ms in ranges)


def _scored_words(
    words: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    masks: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], ...]:
    return tuple(word for word in words if _in_ranges(word, ranges) and not _is_masked(word, masks))


def _edit_alignment(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Return a stable word/character Levenshtein alignment."""

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
                costs[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1]),
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
            == costs[row - 1][column - 1] + (reference[row - 1] != hypothesis[column - 1])
        ):
            equal = reference[row - 1] == hypothesis[column - 1]
            operations.append(("equal" if equal else "substitute", row - 1, column - 1))
            substitutions += not equal
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
    return Alignment(tuple(operations), substitutions, deletions, insertions)


def _metric(
    value: float | int | None,
    *,
    numerator: int | float,
    denominator: int | float,
    denominator_name: str,
    unit: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "value": value,
        "status": "scored" if denominator > 0 else "insufficient_evidence",
        "numerator": numerator,
        "denominator": denominator,
        "denominator_name": denominator_name,
        "unit": unit,
    }
    if extra:
        result.update(extra)
    return result


def _text_metrics(
    reference: Sequence[Mapping[str, Any]], hypothesis: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    raw_reference = tuple(str(word["text"]) for word in reference)
    raw_hypothesis = tuple(str(word["text"]) for word in hypothesis)
    normalized_reference = tuple(
        token for word in reference for token in _token_text(word, normalized=True).split()
    )
    normalized_hypothesis = tuple(
        token for word in hypothesis for token in _token_text(word, normalized=True).split()
    )
    raw_alignment = _edit_alignment(raw_reference, raw_hypothesis)
    normalized_alignment = _edit_alignment(normalized_reference, normalized_hypothesis)
    normalized_reference_text = " ".join(normalized_reference)
    normalized_hypothesis_text = " ".join(normalized_hypothesis)
    cer_alignment = _edit_alignment(
        tuple(normalized_reference_text), tuple(normalized_hypothesis_text)
    )
    return {
        "raw_wer": _metric(
            raw_alignment.errors / len(raw_reference) if raw_reference else None,
            numerator=raw_alignment.errors,
            denominator=len(raw_reference),
            denominator_name="reference_raw_tokens",
            unit="ratio",
            extra={
                "substitutions": raw_alignment.substitutions,
                "deletions": raw_alignment.deletions,
                "insertions": raw_alignment.insertions,
                "definition": "(S+D+I)/N over whitespace tokens with case and punctuation intact",
            },
        ),
        "normalized_wer": _metric(
            normalized_alignment.errors / len(normalized_reference)
            if normalized_reference
            else None,
            numerator=normalized_alignment.errors,
            denominator=len(normalized_reference),
            denominator_name="reference_it_v1_tokens",
            unit="ratio",
            extra={
                "substitutions": normalized_alignment.substitutions,
                "deletions": normalized_alignment.deletions,
                "insertions": normalized_alignment.insertions,
                "definition": "(S+D+I)/N after it-v1 normalization",
            },
        ),
        "normalized_cer": _metric(
            cer_alignment.errors / len(normalized_reference_text)
            if normalized_reference_text
            else None,
            numerator=cer_alignment.errors,
            denominator=len(normalized_reference_text),
            denominator_name="normalized_reference_unicode_code_points",
            unit="ratio",
            extra={
                "definition": "Unicode code-point Levenshtein over it-v1 text with single spaces",
            },
        ),
    }


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start_ms, end_ms) for start_ms, end_ms in ranges if start_ms < end_ms)
    merged: list[tuple[int, int]] = []
    for start_ms, end_ms in ordered:
        if merged and start_ms <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def _subtract_ranges(
    ranges: Sequence[tuple[int, int]], excluded: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    remaining = list(ranges)
    for excluded_start, excluded_end in _merge_ranges(excluded):
        next_remaining: list[tuple[int, int]] = []
        for start_ms, end_ms in remaining:
            if excluded_end <= start_ms or excluded_start >= end_ms:
                next_remaining.append((start_ms, end_ms))
                continue
            if start_ms < excluded_start:
                next_remaining.append((start_ms, min(end_ms, excluded_start)))
            if excluded_end < end_ms:
                next_remaining.append((max(start_ms, excluded_end), end_ms))
        remaining = next_remaining
    return remaining


def _collared_ranges(
    ranges: Sequence[tuple[int, int]], reference_turns: Sequence[Mapping[str, Any]], collar_ms: int
) -> list[tuple[int, int]]:
    if collar_ms <= 0:
        return list(ranges)
    excluded: list[tuple[int, int]] = []
    for turn in reference_turns:
        for boundary in (turn["start_ms"], turn["end_ms"]):
            excluded.append((max(0, boundary - collar_ms), boundary + collar_ms))
    return _subtract_ranges(ranges, excluded)


def _duration_events(
    reference_turns: Sequence[Mapping[str, Any]],
    hypothesis_turns: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int, frozenset[str], frozenset[str]], ...]:
    boundaries = {point for interval in ranges for point in interval}
    for turn in (*reference_turns, *hypothesis_turns):
        for start_ms, end_ms in [(turn["start_ms"], turn["end_ms"])]:
            for range_start, range_end in ranges:
                if start_ms < range_end and end_ms > range_start:
                    boundaries.update(
                        point
                        for point in (max(start_ms, range_start), min(end_ms, range_end))
                        if range_start <= point <= range_end
                    )
    points = sorted(boundaries)
    events: list[tuple[int, int, frozenset[str], frozenset[str]]] = []
    for start_ms, end_ms in zip(points, points[1:], strict=False):
        if not any(
            range_start <= start_ms and end_ms <= range_end for range_start, range_end in ranges
        ):
            continue
        reference = frozenset(
            turn["speaker_id"]
            for turn in reference_turns
            if turn["start_ms"] < end_ms and turn["end_ms"] > start_ms
        )
        hypothesis = frozenset(
            turn["speaker_id"]
            for turn in hypothesis_turns
            if turn["start_ms"] < end_ms and turn["end_ms"] > start_ms
        )
        events.append((start_ms, end_ms, reference, hypothesis))
    return tuple(events)


def _speaker_overlap_scores(
    events: Sequence[tuple[int, int, frozenset[str], frozenset[str]]],
) -> dict[tuple[str, str], int]:
    scores: dict[tuple[str, str], int] = defaultdict(int)
    for start_ms, end_ms, reference, hypothesis in events:
        for reference_speaker in reference:
            for hypothesis_speaker in hypothesis:
                scores[(hypothesis_speaker, reference_speaker)] += end_ms - start_ms
    return dict(scores)


def _speaker_mapping(
    events: Sequence[tuple[int, int, frozenset[str], frozenset[str]]],
) -> dict[str, str]:
    """Choose a deterministic one-to-one hypothesis-to-reference map."""

    scores = _speaker_overlap_scores(events)
    hypothesis_speakers = sorted({speaker for _, _, _, speakers in events for speaker in speakers})
    reference_speakers = sorted({speaker for _, _, speakers, _ in events for speaker in speakers})
    if not hypothesis_speakers or not reference_speakers:
        return {}
    reference_index = {speaker: index for index, speaker in enumerate(reference_speakers)}

    @cache
    def best(index: int, used: int) -> tuple[int, tuple[str | None, ...]]:
        if index == len(hypothesis_speakers):
            return 0, ()
        hypothesis_speaker = hypothesis_speakers[index]
        options: list[tuple[int, tuple[str | None, ...]]] = []
        tail_score, tail_map = best(index + 1, used)
        options.append((tail_score, (None, *tail_map)))
        for reference_speaker in reference_speakers:
            bit = 1 << reference_index[reference_speaker]
            if used & bit:
                continue
            tail_score, tail_map = best(index + 1, used | bit)
            options.append(
                (
                    scores.get((hypothesis_speaker, reference_speaker), 0) + tail_score,
                    (reference_speaker, *tail_map),
                )
            )
        # Maximize overlap, then prefer a lexicographically stable mapping;
        # unmapped speakers sort after real IDs.
        return max(
            options,
            key=lambda option: (option[0], tuple(item or "~" for item in option[1])),
        )

    _, selected = best(0, 0)
    return {
        hypothesis_speaker: reference_speaker
        for hypothesis_speaker, reference_speaker in zip(hypothesis_speakers, selected, strict=True)
        if reference_speaker is not None
    }


def _diarization_metric(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    *,
    collar_ms: int,
) -> dict[str, Any]:
    scoring_ranges = _collared_ranges(ranges, reference, collar_ms)
    events = _duration_events(reference, hypothesis, scoring_ranges)
    mapping = _speaker_mapping(events)
    missed_ms = false_alarm_ms = confusion_ms = reference_speaker_ms = 0
    for start_ms, end_ms, reference_speakers, hypothesis_speakers in events:
        duration_ms = end_ms - start_ms
        reference_count = len(reference_speakers)
        hypothesis_count = len(hypothesis_speakers)
        correct = sum(
            1
            for speaker in hypothesis_speakers
            if speaker in mapping and mapping[speaker] in reference_speakers
        )
        reference_speaker_ms += reference_count * duration_ms
        missed_ms += max(0, reference_count - hypothesis_count) * duration_ms
        false_alarm_ms += max(0, hypothesis_count - reference_count) * duration_ms
        confusion_ms += max(0, min(reference_count, hypothesis_count) - correct) * duration_ms
    denominator_s = reference_speaker_ms / 1000
    numerator_s = (missed_ms + false_alarm_ms + confusion_ms) / 1000
    return {
        "value": numerator_s / denominator_s if denominator_s else None,
        "status": "scored" if denominator_s > 0 else "insufficient_evidence",
        "numerator": numerator_s,
        "denominator": denominator_s,
        "denominator_name": "reference_speaker_seconds",
        "unit": "ratio",
        "components_seconds": {
            "missed_speech": missed_ms / 1000,
            "false_alarm": false_alarm_ms / 1000,
            "speaker_confusion": confusion_ms / 1000,
        },
        "collar_ms": collar_ms,
        "overlap_included": collar_ms == 0,
        "mapping": mapping,
        "definition": (
            "per-speaker missed/false-alarm/confusion duration divided by reference speaker-time; "
            "mapping maximizes scored overlap"
        ),
    }


def _jer_metric(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    events = _duration_events(reference, hypothesis, ranges)
    reference_speakers = sorted({speaker for _, _, speakers, _ in events for speaker in speakers})
    errors: list[float] = []
    for reference_speaker in reference_speakers:
        mapped_hypothesis = next(
            (speaker for speaker, target in mapping.items() if target == reference_speaker), None
        )
        intersection_ms = union_ms = 0
        for start_ms, end_ms, reference_active, hypothesis_active in events:
            reference_has = reference_speaker in reference_active
            hypothesis_has = (
                mapped_hypothesis is not None and mapped_hypothesis in hypothesis_active
            )
            if reference_has or hypothesis_has:
                union_ms += end_ms - start_ms
            if reference_has and hypothesis_has:
                intersection_ms += end_ms - start_ms
        if union_ms:
            errors.append(1 - intersection_ms / union_ms)
    return {
        "value": sum(errors) / len(errors) if errors else None,
        "status": "scored" if errors else "insufficient_evidence",
        "numerator": sum(errors),
        "denominator": len(errors),
        "denominator_name": "reference_speakers_with_scored_time",
        "unit": "ratio",
        "mapping": dict(mapping),
        "definition": "mean per-reference-speaker (1 - intersection/union), using the DER map",
    }


def _count_error_metric(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    reference_speakers = {turn["speaker_id"] for turn in reference if _turn_in_ranges(turn, ranges)}
    hypothesis_speakers = {
        turn["speaker_id"] for turn in hypothesis if _turn_in_ranges(turn, ranges)
    }
    if not reference_speakers:
        return {
            "value": None,
            "status": "insufficient_evidence",
            "numerator": 0,
            "denominator": 0,
            "denominator_name": "reference_speakers",
            "unit": "speakers",
            "signed": None,
            "absolute": None,
            "reference_count": len(reference_speakers),
            "hypothesis_count": len(hypothesis_speakers),
        }
    signed = len(hypothesis_speakers) - len(reference_speakers)
    return {
        "value": signed,
        "status": "scored",
        "numerator": signed,
        "denominator": len(reference_speakers),
        "denominator_name": "reference_speakers",
        "unit": "speakers",
        "signed": signed,
        "absolute": abs(signed),
        "reference_count": len(reference_speakers),
        "hypothesis_count": len(hypothesis_speakers),
        "definition": "hypothesis unique active speakers minus reference unique active speakers",
    }


def _turn_in_ranges(turn: Mapping[str, Any], ranges: Sequence[tuple[int, int]]) -> bool:
    return any(
        turn["start_ms"] < end_ms and turn["end_ms"] > start_ms for start_ms, end_ms in ranges
    )


def _sa_wer_metrics(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    masks: Sequence[tuple[int, int]],
    mapping: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_words = _scored_words(reference, ranges, masks)
    hypothesis_words = _scored_words(hypothesis, ranges, masks)
    reference_speakers = sorted(
        {word["speaker_id"] for word in reference_words if word.get("speaker_id") is not None}
    )
    buckets: dict[str, tuple[list[str], list[str]]] = {
        speaker: ([], []) for speaker in reference_speakers
    }
    buckets["__unassigned__"] = ([], [])
    unassigned_count = 0
    for word in reference_words:
        target = word.get("speaker_id")
        bucket = target if target in buckets else "__unassigned__"
        buckets[bucket][0].append(_token_text(word, normalized=True))
    for word in hypothesis_words:
        source = word.get("speaker_id")
        target = mapping.get(source) if source is not None else None
        bucket = target if target in buckets else "__unassigned__"
        if bucket == "__unassigned__":
            unassigned_count += 1
        buckets[bucket][1].append(_token_text(word, normalized=True))
    total_errors = 0
    reference_count = 0
    bucket_results: dict[str, Any] = {}
    for bucket, (reference_tokens, hypothesis_tokens) in buckets.items():
        reference_tokens = [token for token in reference_tokens if token]
        hypothesis_tokens = [token for token in hypothesis_tokens if token]
        alignment = _edit_alignment(reference_tokens, hypothesis_tokens)
        total_errors += alignment.errors
        reference_count += len(reference_tokens)
        bucket_results[bucket] = {
            "substitutions": alignment.substitutions,
            "deletions": alignment.deletions,
            "insertions": alignment.insertions,
            "reference_tokens": len(reference_tokens),
            "hypothesis_tokens": len(hypothesis_tokens),
        }
    sa_wer = _metric(
        total_errors / reference_count if reference_count else None,
        numerator=total_errors,
        denominator=reference_count,
        denominator_name="scored_reference_it_v1_tokens",
        unit="ratio",
        extra={
            "mapping": dict(mapping),
            "buckets": bucket_results,
            "definition": (
                "map hypothesis speakers by maximum scored overlap, align normalized words "
                "chronologically per mapped speaker, and score unmapped words as insertions"
            ),
        },
    )
    unassigned = _metric(
        unassigned_count / len(hypothesis_words) if hypothesis_words else None,
        numerator=unassigned_count,
        denominator=len(hypothesis_words),
        denominator_name="all_timed_hypothesis_words_in_scored_regions",
        unit="ratio",
        extra={
            "definition": "unmapped or null hypothesis speaker words / all timed hypothesis words"
        },
    )
    return sa_wer, unassigned


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _timing_metrics(
    reference: Sequence[Mapping[str, Any]],
    hypothesis: Sequence[Mapping[str, Any]],
    ranges: Sequence[tuple[int, int]],
    masks: Sequence[tuple[int, int]],
    manual_timing_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_words = tuple(
        word
        for word in _scored_words(reference, ranges, masks)
        if word["word_id"] in manual_timing_ids
    )
    hypothesis_words = _scored_words(hypothesis, ranges, masks)
    reference_tokens = tuple(_token_text(word, normalized=True) for word in reference_words)
    hypothesis_tokens = tuple(_token_text(word, normalized=True) for word in hypothesis_words)
    alignment = _edit_alignment(reference_tokens, hypothesis_tokens)
    start_errors_ms: list[int] = []
    end_errors_ms: list[int] = []
    combined_errors_ms: list[int] = []
    matched: list[dict[str, Any]] = []
    for operation, reference_index, hypothesis_index in alignment.operations:
        if operation != "equal" or reference_index is None or hypothesis_index is None:
            continue
        reference_word = reference_words[reference_index]
        hypothesis_word = hypothesis_words[hypothesis_index]
        start_error_ms = abs(hypothesis_word["start_ms"] - reference_word["start_ms"])
        end_error_ms = abs(hypothesis_word["end_ms"] - reference_word["end_ms"])
        start_errors_ms.append(start_error_ms)
        end_errors_ms.append(end_error_ms)
        combined_errors_ms.extend((start_error_ms, end_error_ms))
        matched.append(
            {
                "reference_word_id": reference_word["word_id"],
                "hypothesis_word_id": hypothesis_word["word_id"],
                "start_error_ms": start_error_ms,
                "end_error_ms": end_error_ms,
            }
        )
    denominator = len(reference_words)
    timing = {
        "status": "scored" if denominator else "insufficient_evidence",
        "value": {
            "start_median_s": _quantile([value / 1000 for value in start_errors_ms], 0.5),
            "start_p95_s": _quantile([value / 1000 for value in start_errors_ms], 0.95),
            "end_median_s": _quantile([value / 1000 for value in end_errors_ms], 0.5),
            "end_p95_s": _quantile([value / 1000 for value in end_errors_ms], 0.95),
            "combined_median_s": _quantile([value / 1000 for value in combined_errors_ms], 0.5),
            "combined_p95_s": _quantile([value / 1000 for value in combined_errors_ms], 0.95),
        },
        "numerator": len(matched),
        "denominator": denominator,
        "denominator_name": "eligible_manual_timing_reference_words",
        "matched_reference_words": len(matched),
        "matched_reference_coverage": len(matched) / denominator if denominator else None,
        "unit": TIMING_UNIT,
        "quantile_definition": QUANTILE_VERSION,
        "definition": (
            "absolute start/end errors for lexically equal words; deleted words stay in "
            "coverage denominator"
        ),
    }
    private = {
        "start_errors_ms": start_errors_ms,
        "end_errors_ms": end_errors_ms,
        "combined_errors_ms": combined_errors_ms,
        "matched": matched,
    }
    return timing, private


def _slice_catalog(reference: EvaluationDocument) -> tuple[dict[str, Any], ...]:
    duration_ms = reference.duration_ms
    slices: list[dict[str, Any]] = [
        {"slice_id": "all", "label": "all scored audio", "ranges_ms": [(0, duration_ms)]}
    ]
    blocks = reference.split.get("blocks", [])
    for block in blocks:
        slices.append(
            {
                "slice_id": f"block-{block['index']}",
                "label": block["partition"],
                "ranges_ms": [(block["start_ms"], block["end_ms"])],
                "partition": block["partition"],
            }
        )
    if blocks:
        for partition, selected in (
            ("development", blocks[:4]),
            ("held_out", blocks[4:]),
        ):
            slices.append(
                {
                    "slice_id": partition,
                    "label": partition,
                    "ranges_ms": [(block["start_ms"], block["end_ms"]) for block in selected],
                    "partition": partition,
                }
            )
    for custom in reference.slices:
        slices.append(
            {
                "slice_id": custom["slice_id"],
                "label": custom["label"],
                "ranges_ms": [(custom["start_ms"], custom["end_ms"])],
            }
        )
    overlap_ranges = [(item["start_ms"], item["end_ms"]) for item in reference.overlaps]
    if overlap_ranges:
        slices.append(
            {"slice_id": "overlap", "label": "reference overlap", "ranges_ms": overlap_ranges}
        )
    if reference.unintelligible_spans:
        slices.append(
            {
                "slice_id": "unintelligible",
                "label": "reference unintelligible regions",
                "ranges_ms": list(reference.unintelligible_spans),
            }
        )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in slices:
        if item["slice_id"] in seen:
            continue
        seen.add(item["slice_id"])
        unique.append(item)
    return tuple(unique)


def _model_hashes(*payloads: Mapping[str, Any]) -> list[str]:
    found: set[str] = set()
    for payload in payloads:
        candidates: list[object] = [payload.get("model_fingerprint_sha256")]
        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping):
            candidates.append(provenance.get("model_fingerprint_sha256"))
            models = provenance.get("models")
            if isinstance(models, Mapping):
                candidates.extend(models.values())
        for candidate in candidates:
            if isinstance(candidate, str) and len(candidate) == _SHA256_LENGTH:
                try:
                    int(candidate, 16)
                except ValueError:
                    continue
                found.add(candidate)
            elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                for item in candidate:
                    if isinstance(item, str) and len(item) == _SHA256_LENGTH:
                        try:
                            int(item, 16)
                        except ValueError:
                            continue
                        found.add(item)
            elif (
                isinstance(candidate, Mapping)
                and {
                    "name",
                    "repository",
                    "revision",
                    "checkpoint_sha256",
                }
                <= candidate.keys()
            ):
                found.add(_sha256(candidate))
    return sorted(found)


def _quality_verdict(
    reference: EvaluationDocument,
    slice_metrics: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
    model_hashes: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    held_out = slice_metrics.get("held_out")
    checks: dict[str, Any] = {}
    if not reference.split_declared:
        return "insufficient_evidence", {
            "reason": "reference did not declare a frozen split",
            "checks": checks,
        }
    if (
        reference.payload.get("status") != "reviewed"
        or reference.payload.get("actor_type") != "human"
    ):
        return "insufficient_evidence", {
            "reason": "reference is not an identified human-reviewed revision",
            "checks": checks,
        }
    if (
        not reference.payload.get("reviewer")
        or reference.payload.get("reviewer") == "machine"
        or not reference.payload.get("reviewed_at")
    ):
        return "insufficient_evidence", {
            "reason": "reference lacks identified reviewer and review timestamp",
            "checks": checks,
        }
    if len(reference.payload.get("manual_timing_word_ids", [])) < 200:
        return "insufficient_evidence", {
            "reason": "reference has fewer than E1's 200 manually timed words",
            "checks": checks,
        }
    if not model_hashes:
        return "insufficient_evidence", {
            "reason": "model fingerprint hashes are missing",
            "checks": checks,
        }
    if not isinstance(held_out, Mapping):
        return "insufficient_evidence", {"reason": "held-out slice is missing", "checks": checks}
    for metric_name, threshold in thresholds.items():
        metric = held_out.get(metric_name)
        passed = (
            isinstance(metric, Mapping)
            and metric.get("status") == "scored"
            and isinstance(metric.get("value"), (int, float))
            and metric["value"] <= threshold
        )
        checks[metric_name] = {
            "threshold": threshold,
            "value": metric.get("value") if isinstance(metric, Mapping) else None,
            "passed": passed,
            "status": metric.get("status") if isinstance(metric, Mapping) else "missing",
        }
    if not all(check["passed"] for check in checks.values()):
        return "blocked", {
            "reason": "a required held-out quality threshold failed",
            "checks": checks,
        }
    return "pass", {
        "reason": "required held-out thresholds passed on the supplied reviewed reference",
        "checks": checks,
    }


def evaluate_documents(
    reference_payload: Mapping[str, Any],
    hypothesis_payload: Mapping[str, Any],
    *,
    reviewed_commit: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Score one reference/hypothesis pair and return aggregate/private data."""

    reference = parse_evaluation_document(reference_payload, role="reference")
    hypothesis = parse_evaluation_document(hypothesis_payload, role="hypothesis")
    if reference.source_sha256 != hypothesis.source_sha256:
        raise EvaluationValidationError("reference and hypothesis source hashes differ")
    if reference.duration_ms != hypothesis.duration_ms:
        raise EvaluationValidationError("reference and hypothesis duration_ms differ")
    configuration = {
        "metric_definitions": METRIC_DEFINITIONS_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "quantiles": QUANTILE_VERSION,
        "canonical_time_unit": CANONICAL_TIME_UNIT,
        "metric_time_unit": TIMING_UNIT,
        "mask_policy": "union of reference/hypothesis unintelligible spans and word flags",
        "split_assignment": "reference word midpoint in integer milliseconds",
        "primary_der": "zero collar, standard overlap-inclusive turns",
        "secondary_der": "250ms reference-boundary exclusion collar",
        "sa_wer": "maximum-overlap one-to-one map, normalized chronological per-speaker alignment",
        "thresholds": dict(thresholds or DEFAULT_THRESHOLDS),
    }
    configuration_sha256 = canonical_hash(configuration)
    masks = _merge_ranges((*reference.unintelligible_spans, *hypothesis.unintelligible_spans))
    slice_metrics: dict[str, dict[str, Any]] = {}
    private_errors: dict[str, Any] = {}
    manual_timing_ids = set(reference.payload.get("manual_timing_word_ids", []))
    for item in _slice_catalog(reference):
        slice_id = item["slice_id"]
        ranges = item["ranges_ms"]
        ref_words = _scored_words(reference.words, ranges, masks)
        hyp_words = _scored_words(hypothesis.words, ranges, masks)
        text = _text_metrics(ref_words, hyp_words)
        primary = _diarization_metric(
            reference.standard_turns, hypothesis.standard_turns, ranges, collar_ms=0
        )
        secondary = _diarization_metric(
            reference.standard_turns, hypothesis.standard_turns, ranges, collar_ms=250
        )
        jer = _jer_metric(
            reference.standard_turns,
            hypothesis.standard_turns,
            ranges,
            primary["mapping"],
        )
        count_error = _count_error_metric(
            reference.standard_turns, hypothesis.standard_turns, ranges
        )
        sa_wer, unassigned = _sa_wer_metrics(
            reference.words,
            hypothesis.words,
            ranges,
            masks,
            primary["mapping"],
        )
        timing, private_timing = _timing_metrics(
            reference.words,
            hypothesis.words,
            ranges,
            masks,
            manual_timing_ids,
        )
        slice_metrics[slice_id] = {
            **text,
            "primary_der": primary,
            "secondary_der_250ms": secondary,
            "secondary_der": secondary,
            "jer": jer,
            "count_error": count_error,
            "sa_wer": sa_wer,
            "unassigned_rate": unassigned,
            "timing": timing,
            "timing_quantiles": timing,
            "slice": {
                "label": item["label"],
                "ranges_ms": ranges,
                "reference_word_count": len(ref_words),
                "hypothesis_word_count": len(hyp_words),
                "mask_count": sum(
                    1
                    for word in (*reference.words, *hypothesis.words)
                    if _is_masked(word, masks) and _in_ranges(word, ranges)
                ),
            },
        }
        private_errors[slice_id] = {
            "timing": private_timing,
            "masks_ms": list(masks),
        }
    verdict, pass_block = _quality_verdict(
        reference,
        slice_metrics,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        model_hashes=_model_hashes(reference.payload, hypothesis.payload),
    )
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "quantile_version": QUANTILE_VERSION,
        "canonical_time_unit": CANONICAL_TIME_UNIT,
        "metric_time_unit": TIMING_UNIT,
        "provenance": {
            "source_sha256": reference.source_sha256,
            "duration_ms": reference.duration_ms,
            "reference_artifact_sha256": _sha256(reference.payload),
            "hypothesis_artifact_sha256": _sha256(hypothesis.payload),
            "split_sha256": reference.split_sha256,
            "configuration_sha256": configuration_sha256,
            "model_fingerprint_sha256": _model_hashes(reference.payload, hypothesis.payload),
            "reviewed_commit": reviewed_commit,
            "reference_status": reference.payload.get("status", "unspecified"),
            "reference_actor_type": reference.payload.get("actor_type", "unspecified"),
        },
        "configuration": configuration,
        "slices": slice_metrics,
        "pass_block": pass_block,
        "verdict": verdict,
        "limitations": [
            "This harness does not establish model quality or a baseline verdict.",
            "Human reference judgment and real model provenance are required for release evidence.",
            "Metrics with zero reference denominators are insufficient evidence, never a pass.",
        ],
        "private_errors": private_errors,
    }
    return report


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError(f"cannot read evaluation JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"evaluation JSON must be an object: {path}")
    return value


def _document_file(path: str | Path, role: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        names = (
            ("reference.json", "annotation.json")
            if role == "reference"
            else ("hypothesis.json", "transcript.json")
        )
        for name in names:
            if (candidate / name).is_file():
                return candidate / name
        raise EvaluationValidationError(f"{role} directory has no supported JSON file: {candidate}")
    return candidate


def evaluate_files(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    *,
    reviewed_commit: str | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Load and score two private JSON artifacts."""

    reference_file = _document_file(reference_path, "reference")
    hypothesis_file = _document_file(hypothesis_path, "hypothesis")
    return evaluate_documents(
        _load_json(reference_file),
        _load_json(hypothesis_file),
        reviewed_commit=reviewed_commit,
        thresholds=thresholds,
    )


def _json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _write_private_file(path: Path, value: object) -> None:
    path.write_bytes(_json_bytes(value))
    path.chmod(0o600)


def _render_report_html(report: Mapping[str, Any]) -> str:
    provenance = report["provenance"]
    rows: list[str] = []
    metric_names = (
        "raw_wer",
        "normalized_wer",
        "normalized_cer",
        "primary_der",
        "secondary_der_250ms",
        "jer",
        "count_error",
        "sa_wer",
        "unassigned_rate",
        "timing",
    )
    for slice_id, metrics in report["slices"].items():
        for metric_name in metric_names:
            metric = metrics[metric_name]
            value = metric.get("value")
            if isinstance(value, Mapping):
                value = "; ".join(f"{key}={item}" for key, item in value.items())
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(slice_id))}</td>"
                f"<td>{html.escape(metric_name)}</td>"
                f"<td>{html.escape(str(value))}</td>"
                f"<td>{html.escape(str(metric.get('status')))}</td>"
                f"<td>{html.escape(str(metric.get('denominator')))}</td>"
                "</tr>"
            )

    def provenance_row(label: str, key: str) -> str:
        value = html.escape(str(provenance.get(key)))
        return f"<dt>{label}</dt><dd><code>{value}</code></dd>"

    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><meta charset="utf-8"><title>Zanzara evaluation aggregate</title>',
            "<style>"
            "body { font: 14px system-ui, sans-serif; max-width: 1100px; "
            "margin: 2rem auto; padding: 0 1rem } "
            "table { border-collapse: collapse; width: 100% } "
            "td, th { border: 1px solid #bbb; padding: .35rem; text-align: left } "
            "code { overflow-wrap: anywhere }"
            "</style>",
            "<h1>Zanzara evaluation aggregate</h1>",
            f"<p><strong>Verdict:</strong> {html.escape(str(report['verdict']))}</p>",
            "<p>This aggregate contains no transcript, identity, or audio content. "
            "It does not claim baseline quality.</p>",
            "<dl>",
            provenance_row("Source hash", "source_sha256"),
            provenance_row("Reference artifact hash", "reference_artifact_sha256"),
            provenance_row("Hypothesis artifact hash", "hypothesis_artifact_sha256"),
            provenance_row("Split hash", "split_sha256"),
            provenance_row("Configuration hash", "configuration_sha256"),
            provenance_row("Reviewed commit", "reviewed_commit"),
            "</dl>",
            "<table><thead><tr><th>Slice</th><th>Metric</th><th>Value</th>"
            f"<th>Status</th><th>Denominator</th></tr></thead><tbody>{''.join(rows)}</tbody></table>",
            "<h2>Pass/block logic</h2>"
            f"<pre>{html.escape(json.dumps(report['pass_block'], indent=2, sort_keys=True))}</pre>",
            "</html>",
        )
    )


def write_evaluation_artifacts(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Atomically write the E6 run files to a new private directory."""

    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise EvaluationValidationError(
            f"refusing to overwrite existing evaluation run: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        aggregate = {key: value for key, value in report.items() if key not in {"private_errors"}}
        coverage = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "provenance": report["provenance"],
            "slices": {
                slice_id: metrics["slice"] for slice_id, metrics in report["slices"].items()
            },
            "limitations": report["limitations"],
        }
        errors = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "provenance": report["provenance"],
            "timing_alignment": report["private_errors"],
        }
        files: dict[str, bytes] = {
            "metrics.json": _json_bytes(aggregate),
            "coverage.json": _json_bytes(coverage),
            "errors.json": _json_bytes(errors),
            "results.json": _json_bytes(report),
            "report.html": _render_report_html(report).encode("utf-8"),
        }
        for name, content in files.items():
            (temporary / name).write_bytes(content)
            (temporary / name).chmod(0o600)
        artifact_hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        }
        run = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "run_type": "asr-diarization-attribution-timing",
            "private": True,
            "verdict": report["verdict"],
            "provenance": report["provenance"],
            "commands_require_no_model_or_paid_call": True,
            "artifact_sha256": artifact_hashes,
        }
        _write_private_file(temporary / "run.json", run)
        temporary.chmod(0o700)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(destination),
        "verdict": report["verdict"],
        "artifact_sha256": {
            **run["artifact_sha256"],
            "run.json": hashlib.sha256((destination / "run.json").read_bytes()).hexdigest(),
        },
    }


def current_git_commit() -> str | None:
    """Return the checked-out commit when the CLI is run in a Git worktree."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


__all__ = [
    "CANONICAL_TIME_UNIT",
    "DEFAULT_THRESHOLDS",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationDocument",
    "EvaluationValidationError",
    "METRIC_DEFINITIONS_VERSION",
    "NORMALIZATION_VERSION",
    "QUANTILE_VERSION",
    "evaluate_documents",
    "evaluate_files",
    "current_git_commit",
    "normalize_italian",
    "parse_evaluation_document",
    "write_evaluation_artifacts",
]

"""Deterministic speaker and overlap metadata for frozen P1R chunks.

Community-1 is run once for an entire episode.  This module only intersects
that episode-level result with immutable chunk intervals; it never runs a
diarizer on an individual benchmark chunk and never uses ASR output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import (
    AudioChunk,
    ChunkCondition,
    ContractValidationError,
    DiarizationResult,
    Overlap,
    SpeakerConditionMetadata,
    SpeakerStream,
    Turn,
)

SPEAKER_CONDITION_VERSION = "p1r-speaker-condition-v1"
HEAVY_OVERLAP_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class SpeakerConditionThresholds:
    """Frozen thresholds used to label machine-seeded speaker conditions."""

    version: str = SPEAKER_CONDITION_VERSION
    heavy_overlap_fraction: float = HEAVY_OVERLAP_FRACTION

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ContractValidationError("speaker threshold version must be non-empty text")
        if isinstance(self.heavy_overlap_fraction, bool) or not isinstance(
            self.heavy_overlap_fraction, (int, float)
        ):
            raise ContractValidationError("heavy_overlap_fraction must be a number")
        if not 0.0 < float(self.heavy_overlap_fraction) <= 1.0:
            raise ContractValidationError("heavy_overlap_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy so a manifest can retain the exact thresholds."""

        return {
            "version": self.version,
            "heavy_overlap_fraction": float(self.heavy_overlap_fraction),
        }


def _hash_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _turns(values: Sequence[object], field_name: str) -> tuple[Turn, ...]:
    result: list[Turn] = []
    for index, value in enumerate(values):
        try:
            turn = value if isinstance(value, Turn) else Turn.from_dict(value)  # type: ignore[arg-type]
        except (TypeError, KeyError, ValueError) as exc:
            raise ContractValidationError(f"{field_name}[{index}] is not a valid turn") from exc
        result.append(turn)
    return tuple(result)


def _diarization_parts(
    diarization: DiarizationResult | Mapping[str, Any] | object,
) -> tuple[tuple[Turn, ...], tuple[Turn, ...], str, str, str | None, int | None]:
    standard = _value(diarization, "standard_turns")
    exclusive = _value(diarization, "exclusive_turns")
    if not isinstance(standard, Sequence) or isinstance(standard, (str, bytes)):
        raise ContractValidationError("diarization standard_turns must be a sequence")
    if not isinstance(exclusive, Sequence) or isinstance(exclusive, (str, bytes)):
        raise ContractValidationError("diarization exclusive_turns must be a sequence")
    source_sha256 = _value(diarization, "source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ContractValidationError("diarization source_sha256 is required")
    duration_ms = _value(diarization, "duration_ms")
    if duration_ms is not None and (
        isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0
    ):
        raise ContractValidationError("diarization duration_ms must be positive")
    artifact_id = _value(diarization, "artifact_id")
    if artifact_id is not None and not isinstance(artifact_id, str):
        raise ContractValidationError("diarization artifact_id must be text")
    model = _value(diarization, "model")
    fingerprint = _value(model, "fingerprint_sha256") if model is not None else None
    standard_turns = _turns(standard, "standard_turns")
    exclusive_turns = _turns(exclusive, "exclusive_turns")
    if not isinstance(fingerprint, str):
        fingerprint = _hash_json(
            {
                "source_sha256": source_sha256,
                "duration_ms": duration_ms,
                "standard_turns": [turn.to_dict() for turn in standard_turns],
                "exclusive_turns": [turn.to_dict() for turn in exclusive_turns],
            }
        )
    return (
        standard_turns,
        exclusive_turns,
        source_sha256,
        fingerprint,
        artifact_id,
        duration_ms,
    )


def _clip_turns(
    turns: Sequence[Turn],
    *,
    chunk: AudioChunk,
    duration_ms: int | None,
    field_name: str,
) -> tuple[Turn, ...]:
    if (
        duration_ms is not None
        and chunk.duration_ms is not None
        and duration_ms != chunk.duration_ms
    ):
        raise ContractValidationError("diarization duration does not match chunk source duration")
    if duration_ms is not None and chunk.end_ms > duration_ms:
        raise ContractValidationError("chunk interval exceeds diarization duration")
    clipped: list[Turn] = []
    for index, turn in enumerate(turns):
        if duration_ms is not None and turn.end_ms > duration_ms:
            raise ContractValidationError(f"{field_name}[{index}] exceeds diarization duration")
        start_ms = max(chunk.start_ms, turn.start_ms)
        end_ms = min(chunk.end_ms, turn.end_ms)
        if start_ms < end_ms:
            clipped.append(Turn(turn.speaker_id, start_ms, end_ms, turn.confidence))
    return tuple(sorted(clipped, key=lambda item: (item.speaker_id, item.start_ms, item.end_ms)))


def _streams(turns: Sequence[Turn]) -> tuple[SpeakerStream, ...]:
    grouped: dict[str, list[Turn]] = {}
    for turn in turns:
        grouped.setdefault(turn.speaker_id, []).append(turn)
    return tuple(
        SpeakerStream(speaker_id, tuple(grouped[speaker_id])) for speaker_id in sorted(grouped)
    )


def _standard_segments(
    turns: Sequence[Turn],
) -> tuple[int, int, int, tuple[Overlap, ...], tuple[str, ...]]:
    boundaries = sorted({point for turn in turns for point in (turn.start_ms, turn.end_ms)})
    speech_ms = 0
    overlap_ms = 0
    maximum = 0
    overlaps: list[Overlap] = []
    speaker_ids = tuple(sorted({turn.speaker_id for turn in turns}))
    for start_ms, end_ms in zip(boundaries, boundaries[1:], strict=False):
        active = tuple(
            sorted(
                {
                    turn.speaker_id
                    for turn in turns
                    if turn.start_ms < end_ms and turn.end_ms > start_ms
                }
            )
        )
        if not active:
            continue
        duration_ms = end_ms - start_ms
        speech_ms += duration_ms
        maximum = max(maximum, len(active))
        if len(active) < 2:
            continue
        overlap_ms += duration_ms
        interval = Overlap(active, start_ms, end_ms)
        if overlaps and overlaps[-1].end_ms == start_ms and overlaps[-1].speaker_ids == active:
            overlaps[-1] = Overlap(active, overlaps[-1].start_ms, end_ms)
        else:
            overlaps.append(interval)
    return speech_ms, overlap_ms, maximum, tuple(overlaps), speaker_ids


def _metadata(
    *,
    turns: Sequence[Turn],
    thresholds: SpeakerConditionThresholds,
    diarization_fingerprint: str | None,
    diarization_artifact_id: str | None,
    status: str = "derived",
) -> SpeakerConditionMetadata:
    speech_ms, overlap_ms, maximum, _, speaker_ids = _standard_segments(turns)
    overlap_fraction = overlap_ms / speech_ms if speech_ms else None
    if not speaker_ids or not speech_ms:
        condition = "uncertain"
    elif overlap_ms == 0 and len(speaker_ids) == 1:
        condition = "single_speaker"
    elif overlap_ms == 0:
        condition = "multi_speaker_no_overlap"
    elif overlap_fraction is not None and overlap_fraction >= thresholds.heavy_overlap_fraction:
        condition = "heavy_overlap"
    else:
        condition = "partial_overlap"
    return SpeakerConditionMetadata(
        version=thresholds.version,
        threshold_version=thresholds.version,
        speaker_ids=speaker_ids,
        speaker_count=len(speaker_ids),
        max_simultaneous_speakers=maximum,
        speech_ms=speech_ms,
        overlap_ms=overlap_ms,
        overlap_fraction=overlap_fraction,
        has_overlap=overlap_ms > 0,
        speaker_condition=condition,
        status=status,  # type: ignore[arg-type]
        origin="machine_seed",
        diarization_fingerprint=diarization_fingerprint,
        diarization_artifact_id=diarization_artifact_id,
    )


def _unknown_metadata(thresholds: SpeakerConditionThresholds) -> SpeakerConditionMetadata:
    return SpeakerConditionMetadata(
        version=thresholds.version,
        threshold_version=thresholds.version,
        speaker_ids=(),
        speaker_count=0,
        max_simultaneous_speakers=0,
        speech_ms=0,
        overlap_ms=0,
        overlap_fraction=None,
        has_overlap=False,
        speaker_condition="uncertain",
        status="unknown",
        origin="machine_seed",
    )


def derive_chunk_speaker_metadata(
    chunk: AudioChunk,
    *,
    diarization: DiarizationResult | Mapping[str, Any] | object | None = None,
    standard_turns: Sequence[object] | None = None,
    exclusive_turns: Sequence[object] | None = None,
    diarization_fingerprint: str | None = None,
    thresholds: SpeakerConditionThresholds | None = None,
) -> AudioChunk:
    """Attach deterministic speaker metadata to one frozen chunk.

    The standard tracks are the only source for speech and overlap metrics.
    Exclusive tracks are retained separately for later attribution consumers.
    When no episode-level diarization is available, the result is explicitly
    ``unknown``/``uncertain`` and contains no invented speaker labels.
    """

    policy = thresholds or SpeakerConditionThresholds()
    if diarization is not None and (standard_turns is not None or exclusive_turns is not None):
        raise ContractValidationError("provide diarization or explicit turns, not both")
    if diarization is None and standard_turns is None and exclusive_turns is None:
        metadata = _unknown_metadata(policy)
        standard = exclusive = ()
        overlaps: tuple[Overlap, ...] = ()
    else:
        if diarization is not None:
            (
                standard_raw,
                exclusive_raw,
                diarization_source_sha256,
                input_fingerprint,
                artifact_id,
                duration_ms,
            ) = _diarization_parts(diarization)
            if diarization_source_sha256 != chunk.source_sha256:
                raise ContractValidationError(
                    "diarization source hash does not match the chunk source"
                )
        else:
            if standard_turns is None or exclusive_turns is None:
                raise ContractValidationError(
                    "standard_turns and exclusive_turns must be supplied together"
                )
            standard_raw = _turns(standard_turns, "standard_turns")
            exclusive_raw = _turns(exclusive_turns, "exclusive_turns")
            input_fingerprint = _hash_json(
                {
                    "standard_turns": [turn.to_dict() for turn in standard_raw],
                    "exclusive_turns": [turn.to_dict() for turn in exclusive_raw],
                }
            )
            artifact_id = None
            duration_ms = chunk.duration_ms
        if diarization_fingerprint is not None:
            if len(diarization_fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in diarization_fingerprint
            ):
                raise ContractValidationError("diarization_fingerprint must be a SHA-256")
            input_fingerprint = diarization_fingerprint
        standard = _clip_turns(
            standard_raw,
            chunk=chunk,
            duration_ms=duration_ms,
            field_name="standard_turns",
        )
        exclusive = _clip_turns(
            exclusive_raw,
            chunk=chunk,
            duration_ms=duration_ms,
            field_name="exclusive_turns",
        )
        _, _, _, overlaps, _ = _standard_segments(standard)
        metadata = _metadata(
            turns=standard,
            thresholds=policy,
            diarization_fingerprint=input_fingerprint,
            diarization_artifact_id=artifact_id,
        )

    current = chunk.condition or ChunkCondition()
    condition = replace(
        current,
        speaker_streams=_streams(standard),
        exclusive_speaker_streams=_streams(exclusive),
        overlaps=overlaps,
        speaker_metadata=metadata,
        speaker_correction=None,
    )
    return replace(chunk, condition=condition)


def derive_speaker_conditions(
    chunks: Sequence[AudioChunk],
    *,
    diarization: DiarizationResult | Mapping[str, Any] | object | None = None,
    standard_turns: Sequence[object] | None = None,
    exclusive_turns: Sequence[object] | None = None,
    diarization_fingerprint: str | None = None,
    thresholds: SpeakerConditionThresholds | None = None,
) -> tuple[AudioChunk, ...]:
    """Attach the same episode-wide diarization result to every frozen chunk."""

    return tuple(
        derive_chunk_speaker_metadata(
            chunk,
            diarization=diarization,
            standard_turns=standard_turns,
            exclusive_turns=exclusive_turns,
            diarization_fingerprint=diarization_fingerprint,
            thresholds=thresholds,
        )
        for chunk in chunks
    )


# Descriptive aliases for callers that use metadata/attachment terminology.
attach_speaker_condition = derive_chunk_speaker_metadata
derive_speaker_condition_metadata = derive_chunk_speaker_metadata


__all__ = [
    "HEAVY_OVERLAP_FRACTION",
    "SPEAKER_CONDITION_VERSION",
    "SpeakerConditionThresholds",
    "attach_speaker_condition",
    "derive_chunk_speaker_metadata",
    "derive_speaker_condition_metadata",
    "derive_speaker_conditions",
]

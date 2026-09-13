"""Deterministic, model-independent P1R audio chunk segmentation.

The benchmark chunker operates on source-relative millisecond intervals.  It
does not inspect text or ASR output: callers provide optional VAD/acoustic
boundary evidence and, when available, episode-wide diarization turns.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .contracts import AudioChunk, ContractValidationError, Turn
from .stages import stage_fingerprint

BoundaryKind = Literal["silence", "speaker_turn", "acoustic"]
Interval = tuple[int, int]

_BOUNDARY_PRIORITIES: dict[BoundaryKind, int] = {
    "silence": 0,
    "speaker_turn": 1,
    "acoustic": 2,
}
_BOUNDARY_ALIASES: dict[str, BoundaryKind] = {
    "acoustic": "acoustic",
    "energy": "acoustic",
    "pause": "silence",
    "silence": "silence",
    "speaker_turn": "speaker_turn",
    "turn": "speaker_turn",
    "vad": "silence",
    "vad_silence": "silence",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ContractValidationError(f"{field_name} must be a positive finite number")
    return number


def _milliseconds(seconds: float, field_name: str) -> int:
    milliseconds = round(seconds * 1000)
    if milliseconds <= 0:
        raise ContractValidationError(f"{field_name} must be at least one millisecond")
    return milliseconds


@dataclass(frozen=True, slots=True)
class ChunkSegmentationConfig:
    """Versioned adaptive chunk policy, expressed in seconds."""

    version: str = "p1r-chunk-segmentation-v1"
    preferred_min_s: float = 8.0
    target_s: float = 12.0
    preferred_max_s: float = 18.0
    hard_max_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ContractValidationError("segmentation version must be non-empty text")
        values = {
            "preferred_min_s": _positive_number(self.preferred_min_s, "preferred_min_s"),
            "target_s": _positive_number(self.target_s, "target_s"),
            "preferred_max_s": _positive_number(self.preferred_max_s, "preferred_max_s"),
            "hard_max_s": _positive_number(self.hard_max_s, "hard_max_s"),
        }
        if not values["preferred_min_s"] <= values["target_s"] <= values["preferred_max_s"]:
            raise ContractValidationError(
                "segmentation durations must satisfy preferred_min_s <= target_s <= preferred_max_s"
            )
        if values["preferred_max_s"] > values["hard_max_s"]:
            raise ContractValidationError("preferred_max_s cannot exceed hard_max_s")
        for field_name, value in values.items():
            object.__setattr__(self, field_name, value)

    @property
    def preferred_min_ms(self) -> int:
        """Return the minimum preferred chunk duration in milliseconds."""

        return _milliseconds(self.preferred_min_s, "preferred_min_s")

    @property
    def target_ms(self) -> int:
        """Return the target chunk duration in milliseconds."""

        return _milliseconds(self.target_s, "target_s")

    @property
    def preferred_max_ms(self) -> int:
        """Return the maximum preferred chunk duration in milliseconds."""

        return _milliseconds(self.preferred_max_s, "preferred_max_s")

    @property
    def hard_max_ms(self) -> int:
        """Return the hard chunk duration limit in milliseconds."""

        return _milliseconds(self.hard_max_s, "hard_max_s")

    @property
    def configuration_sha256(self) -> str:
        """Return the stable hash of this policy configuration."""

        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy used to produce a segmentation manifest."""

        return {
            "version": self.version,
            "preferred_min_s": self.preferred_min_s,
            "target_s": self.target_s,
            "preferred_max_s": self.preferred_max_s,
            "hard_max_s": self.hard_max_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkSegmentationConfig:
        """Restore a policy from JSON-compatible data."""

        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class AcousticBoundary:
    """One candidate boundary with an explicit deterministic priority class."""

    time_ms: int
    kind: BoundaryKind = "acoustic"
    confidence: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.time_ms, bool) or not isinstance(self.time_ms, int) or self.time_ms < 0:
            raise ContractValidationError("boundary time_ms must be a non-negative integer")
        try:
            normalized_kind = _BOUNDARY_ALIASES[self.kind]
        except (KeyError, TypeError) as exc:
            raise ContractValidationError(
                "boundary kind must be silence, speaker_turn, or acoustic"
            ) from exc
        object.__setattr__(self, "kind", normalized_kind)
        if self.confidence is not None:
            try:
                confidence = float(self.confidence)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError(
                    "boundary confidence must be between 0 and 1"
                ) from exc
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ContractValidationError("boundary confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the boundary for fingerprints and evidence."""

        return {
            "time_ms": self.time_ms,
            "kind": self.kind,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AcousticBoundary:
        """Restore a boundary from JSON-compatible data."""

        return cls(**dict(value))


# A descriptive alias for callers that use "candidate" terminology.
BoundaryCandidate = AcousticBoundary


@dataclass(frozen=True, slots=True)
class ChunkSegmentationResult:
    """Chunks plus the metadata needed to publish their segmentation."""

    chunks: tuple[AudioChunk, ...]
    version: str
    configuration: Mapping[str, Any]
    input_fingerprints: tuple[str, ...]
    segmentation_fingerprint: str

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ContractValidationError("segmentation must produce at least one chunk")
        if not self.version:
            raise ContractValidationError("segmentation version must be non-empty")
        if not isinstance(self.configuration, Mapping):
            raise ContractValidationError("segmentation configuration must be a mapping")
        try:
            json.dumps(self.configuration, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                "segmentation configuration must be JSON serializable"
            ) from exc
        if len(self.segmentation_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.segmentation_fingerprint
        ):
            raise ContractValidationError("segmentation fingerprint must be a SHA-256")
        for index, digest in enumerate(self.input_fingerprints):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ContractValidationError(
                    f"input_fingerprints[{index}] must be a lowercase SHA-256"
                )

    def manifest_metadata(self) -> dict[str, Any]:
        """Return fields accepted by ``ChunkBenchmarkManifest``."""

        return {
            "segmentation_version": self.version,
            "segmentation_configuration": dict(self.configuration),
            "segmentation_input_fingerprints": list(self.input_fingerprints),
        }


def _normalize_interval(value: object, *, field_name: str, duration_ms: int) -> Interval:
    if isinstance(value, Mapping):
        start = value.get("start_ms")
        end = value.get("end_ms")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        start, end = value
    else:
        raise ContractValidationError(f"{field_name} must be a [start_ms, end_ms] interval")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= duration_ms
    ):
        raise ContractValidationError(
            f"{field_name} must be a non-empty interval within the source duration"
        )
    return start, end


def _normalize_regions(
    selected_regions: Sequence[object] | None, duration_ms: int
) -> tuple[Interval, ...]:
    raw_regions = selected_regions if selected_regions is not None else ((0, duration_ms),)
    regions = tuple(
        _normalize_interval(value, field_name=f"selected_regions[{index}]", duration_ms=duration_ms)
        for index, value in enumerate(raw_regions)
    )
    if not regions:
        raise ContractValidationError("selected_regions must contain at least one interval")
    ordered = tuple(sorted(regions))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current[0] < previous[1]:
            raise ContractValidationError("selected_regions cannot overlap")
    return ordered


def _normalize_boundary(
    value: object, *, default_kind: BoundaryKind, field_name: str, duration_ms: int
) -> AcousticBoundary:
    if isinstance(value, AcousticBoundary):
        boundary = value
    elif isinstance(value, Mapping):
        payload = dict(value)
        if "time_ms" not in payload:
            if "position_ms" in payload:
                payload["time_ms"] = payload.pop("position_ms")
            else:
                raise ContractValidationError(f"{field_name} must contain time_ms")
        payload.setdefault("kind", default_kind)
        boundary = AcousticBoundary.from_dict(payload)
    elif isinstance(value, bool) or not isinstance(value, int):
        raise ContractValidationError(f"{field_name} must be a boundary time or object")
    else:
        boundary = AcousticBoundary(value, default_kind)
    if boundary.time_ms > duration_ms:
        raise ContractValidationError(f"{field_name} is outside the source duration")
    return boundary


def _normalize_boundaries(
    values: Sequence[object], *, default_kind: BoundaryKind, field_name: str, duration_ms: int
) -> tuple[AcousticBoundary, ...]:
    return tuple(
        _normalize_boundary(
            value,
            default_kind=default_kind,
            field_name=f"{field_name}[{index}]",
            duration_ms=duration_ms,
        )
        for index, value in enumerate(values)
    )


def _normalize_turns(
    values: Sequence[object], *, field_name: str, duration_ms: int
) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    for index, value in enumerate(values):
        try:
            turn = value if isinstance(value, Turn) else Turn.from_dict(value)  # type: ignore[arg-type]
        except (TypeError, KeyError, ValueError) as exc:
            raise ContractValidationError(
                f"{field_name}[{index}] is not a valid speaker turn"
            ) from exc
        _normalize_interval(
            (turn.start_ms, turn.end_ms),
            field_name=f"{field_name}[{index}]",
            duration_ms=duration_ms,
        )
        turns.append(turn)
    return tuple(turns)


def _diarization_turns(value: object, *, duration_ms: int) -> tuple[Turn, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw = value.get("standard_turns", ())
    else:
        raw = getattr(value, "standard_turns", ())
    if not isinstance(raw, Sequence):
        raise ContractValidationError("diarization standard_turns must be a sequence")
    return _normalize_turns(raw, field_name="diarization.standard_turns", duration_ms=duration_ms)


def _candidate_records(
    *,
    vad_boundaries: tuple[AcousticBoundary, ...],
    acoustic_boundaries: tuple[AcousticBoundary, ...],
    speaker_turns: tuple[Turn, ...],
) -> tuple[AcousticBoundary, ...]:
    turn_boundaries = tuple(
        AcousticBoundary(time_ms=time, kind="speaker_turn")
        for turn in speaker_turns
        for time in (turn.start_ms, turn.end_ms)
    )
    return vad_boundaries + acoustic_boundaries + turn_boundaries


def _input_digest(label: str, values: object) -> str:
    return _canonical_hash({"label": label, "values": values})


def _provided_digest(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _choose_boundary(
    current_ms: int,
    region_end_ms: int,
    candidates: Sequence[AcousticBoundary],
    config: ChunkSegmentationConfig,
) -> int:
    remaining_ms = region_end_ms - current_ms
    if remaining_ms <= config.hard_max_ms:
        return region_end_ms

    preferred_min = current_ms + config.preferred_min_ms
    preferred_max = current_ms + config.preferred_max_ms
    hard_max = min(region_end_ms, current_ms + config.hard_max_ms)
    target = current_ms + config.target_ms

    def viable(candidate: AcousticBoundary, upper_bound: int) -> bool:
        return (
            preferred_min <= candidate.time_ms <= upper_bound
            and candidate.time_ms < region_end_ms
            and region_end_ms - candidate.time_ms >= config.preferred_min_ms
        )

    preferred = [candidate for candidate in candidates if viable(candidate, preferred_max)]
    available = preferred or [candidate for candidate in candidates if viable(candidate, hard_max)]
    if available:
        selected = min(
            available,
            key=lambda candidate: (
                _BOUNDARY_PRIORITIES[candidate.kind],
                abs(candidate.time_ms - target),
                -(candidate.confidence or 0.0),
                candidate.time_ms,
            ),
        )
        return selected.time_ms

    # Reserve a preferred-sized final interval when a hard cut would leave a
    # tiny tail.  This still keeps every emitted interval at or below hard_max.
    return min(current_ms + config.hard_max_ms, region_end_ms - config.preferred_min_ms)


def _validate_coverage(
    chunks: Sequence[AudioChunk],
    regions: Sequence[Interval],
    *,
    hard_max_ms: int,
) -> None:
    if not regions:
        raise ContractValidationError("selected_regions must contain at least one interval")
    selected_end = max(end for _, end in regions)
    if any(chunk.start_ms < 0 or chunk.end_ms > selected_end for chunk in chunks):
        raise ContractValidationError("chunk interval is outside selected source regions")
    cursor = 0
    consumed: set[str] = set()
    for region_start, region_end in regions:
        region_chunks = [
            chunk
            for chunk in chunks
            if region_start <= chunk.start_ms and chunk.end_ms <= region_end
        ]
        if not region_chunks:
            raise ContractValidationError("segmentation produced no chunk for a selected region")
        expected = region_start
        for chunk in region_chunks:
            consumed.add(chunk.chunk_id)
            if chunk.start_ms != expected:
                raise ContractValidationError("chunks do not completely cover selected regions")
            if chunk.end_ms <= chunk.start_ms or chunk.end_ms - chunk.start_ms > hard_max_ms:
                raise ContractValidationError("chunk duration violates the hard maximum")
            expected = chunk.end_ms
        if expected != region_end:
            raise ContractValidationError("chunks do not completely cover selected regions")
        if region_start < cursor:
            raise ContractValidationError("selected-region coverage overlaps")
        cursor = region_end
    if len(consumed) != len(chunks):
        raise ContractValidationError("chunks include intervals outside selected regions")


def _prepare_inputs(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    config: ChunkSegmentationConfig,
    vad_boundaries: Sequence[object],
    acoustic_boundaries: Sequence[object],
    speaker_turns: Sequence[object],
    diarization: object | None,
    selected_regions: Sequence[object] | None,
    vad_fingerprint: str | None,
    acoustic_fingerprint: str | None,
    diarization_fingerprint: str | None,
    input_fingerprints: Sequence[str],
) -> tuple[
    tuple[Interval, ...],
    tuple[AcousticBoundary, ...],
    tuple[AcousticBoundary, ...],
    tuple[Turn, ...],
    tuple[str, ...],
    str,
]:
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ContractValidationError("episode_id must be non-empty text")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ContractValidationError("source_sha256 must be a lowercase SHA-256")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ContractValidationError("duration_ms must be a positive integer")
    regions = _normalize_regions(selected_regions, duration_ms)
    vad = _normalize_boundaries(
        vad_boundaries,
        default_kind="silence",
        field_name="vad_boundaries",
        duration_ms=duration_ms,
    )
    acoustic = _normalize_boundaries(
        acoustic_boundaries,
        default_kind="acoustic",
        field_name="acoustic_boundaries",
        duration_ms=duration_ms,
    )
    turns = _normalize_turns(speaker_turns, field_name="speaker_turns", duration_ms=duration_ms)
    turns += _diarization_turns(diarization, duration_ms=duration_ms)

    normalized_inputs = {
        "vad": [boundary.to_dict() for boundary in vad],
        "acoustic": [boundary.to_dict() for boundary in acoustic],
        "speaker_turns": [turn.to_dict() for turn in turns],
    }
    digests = (
        _provided_digest(vad_fingerprint, field_name="vad_fingerprint")
        or _input_digest("vad", normalized_inputs["vad"]),
        _provided_digest(acoustic_fingerprint, field_name="acoustic_fingerprint")
        or _input_digest("acoustic", normalized_inputs["acoustic"]),
        _provided_digest(diarization_fingerprint, field_name="diarization_fingerprint")
        or _input_digest("diarization", normalized_inputs["speaker_turns"]),
    )
    extra_digests = tuple(
        _provided_digest(value, field_name=f"input_fingerprints[{index}]") or value
        for index, value in enumerate(input_fingerprints)
    )
    all_digests = digests + extra_digests
    fingerprint = stage_fingerprint(
        "benchmark_chunks",
        source_sha256=source_sha256,
        upstream_artifact_hashes=all_digests,
        configuration={
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
            "segmentation": config.to_dict(),
        },
        pipeline_version=config.version,
    )
    return regions, vad, acoustic, turns, all_digests, fingerprint


def segment_chunks_with_metadata(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    *,
    config: ChunkSegmentationConfig | None = None,
    vad_boundaries: Sequence[object] = (),
    acoustic_boundaries: Sequence[object] = (),
    speaker_turns: Sequence[object] = (),
    diarization_turns: Sequence[object] = (),
    diarization: object | None = None,
    selected_regions: Sequence[object] | None = None,
    partition: Literal["development", "held_out"] | None = None,
    vad_fingerprint: str | None = None,
    acoustic_fingerprint: str | None = None,
    diarization_fingerprint: str | None = None,
    input_fingerprints: Sequence[str] = (),
) -> ChunkSegmentationResult:
    """Build deterministic chunks and their publication metadata.

    Boundary times are always interpreted in the original episode timeline.
    ``diarization_turns`` is an alias for supplying Community-1 standard turns;
    it is combined with ``speaker_turns`` for compatibility with callers that
    use either name.
    """

    policy = config or ChunkSegmentationConfig()
    regions, vad, acoustic, turns, input_hashes, fingerprint = _prepare_inputs(
        episode_id=episode_id,
        source_sha256=source_sha256,
        duration_ms=duration_ms,
        config=policy,
        vad_boundaries=vad_boundaries,
        acoustic_boundaries=acoustic_boundaries,
        speaker_turns=tuple(speaker_turns) + tuple(diarization_turns),
        diarization=diarization,
        selected_regions=selected_regions,
        vad_fingerprint=vad_fingerprint,
        acoustic_fingerprint=acoustic_fingerprint,
        diarization_fingerprint=diarization_fingerprint,
        input_fingerprints=input_fingerprints,
    )
    candidates = _candidate_records(
        vad_boundaries=vad,
        acoustic_boundaries=acoustic,
        speaker_turns=turns,
    )
    chunks: list[AudioChunk] = []
    for region_start, region_end in regions:
        current = region_start
        while current < region_end:
            end = _choose_boundary(current, region_end, candidates, policy)
            chunks.append(
                AudioChunk.create(
                    episode_id=episode_id,
                    source_sha256=source_sha256,
                    start_ms=current,
                    end_ms=end,
                    duration_ms=duration_ms,
                    segmentation_fingerprint=fingerprint,
                    partition=partition,
                )
            )
            current = end
    result = ChunkSegmentationResult(
        chunks=tuple(chunks),
        version=policy.version,
        configuration={
            **policy.to_dict(),
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
        },
        input_fingerprints=input_hashes,
        segmentation_fingerprint=fingerprint,
    )
    _validate_coverage(result.chunks, regions, hard_max_ms=policy.hard_max_ms)
    return result


def segment_chunks(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    **kwargs: Any,
) -> tuple[AudioChunk, ...]:
    """Return only the immutable ``AudioChunk`` records for one episode."""

    return segment_chunks_with_metadata(episode_id, source_sha256, duration_ms, **kwargs).chunks


def validate_chunk_coverage(
    chunks: Sequence[AudioChunk],
    selected_regions: Sequence[object],
    *,
    duration_ms: int,
    hard_max_ms: int = 30_000,
) -> None:
    """Validate complete selected-region coverage and the hard duration limit."""

    regions = _normalize_regions(selected_regions, duration_ms)
    if isinstance(hard_max_ms, bool) or not isinstance(hard_max_ms, int) or hard_max_ms <= 0:
        raise ContractValidationError("hard_max_ms must be a positive integer")
    _validate_coverage(chunks, regions, hard_max_ms=hard_max_ms)


# Natural aliases for callers using the issue's terminology.
segment_audio_chunks = segment_chunks
build_audio_chunks = segment_chunks


__all__ = [
    "AcousticBoundary",
    "BoundaryCandidate",
    "ChunkSegmentationConfig",
    "ChunkSegmentationResult",
    "build_audio_chunks",
    "segment_audio_chunks",
    "segment_chunks",
    "segment_chunks_with_metadata",
    "validate_chunk_coverage",
]

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

from .contracts import (
    AudioChunk,
    ContractValidationError,
    DiarizationResult,
    NativeActivityArtifact,
    NativeActivityInterval,
    Turn,
)
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

        if value.get("algorithm") == "community1-adaptive-v1" and cls is ChunkSegmentationConfig:
            return Community1AdaptiveConfig.from_dict(value)
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Community1AdaptiveConfig(ChunkSegmentationConfig):
    """Versioned Community-1 standard-diarization boundary policy.

    The parent ``ChunkSegmentationConfig`` remains available for legacy P1R
    callers.  This policy is intentionally expressed in integer milliseconds:
    the values are part of the configuration fingerprint and never appear as
    unversioned literals in the selector.
    """

    version: str = "community1-adaptive-v1"
    strong_gap_ms: int = 600
    short_gap_ms: int = 250
    speaker_change_tolerance_ms: int = 250
    overlap_margin_ms: int = 250
    minimum_chunk_ms: int = 4_000

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in (
            "strong_gap_ms",
            "short_gap_ms",
            "speaker_change_tolerance_ms",
            "overlap_margin_ms",
            "minimum_chunk_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractValidationError(f"{field_name} must be a positive integer")
        if self.strong_gap_ms <= self.short_gap_ms:
            raise ContractValidationError("strong_gap_ms must exceed short_gap_ms")
        if self.minimum_chunk_ms > self.preferred_min_ms:
            raise ContractValidationError("minimum_chunk_ms cannot exceed preferred_min_ms")

    def to_dict(self) -> dict[str, Any]:
        """Serialize every value that influences Community-1 segmentation."""

        return {
            "version": self.version,
            "algorithm": "community1-adaptive-v1",
            "preferred_min_ms": self.preferred_min_ms,
            "target_ms": self.target_ms,
            "preferred_max_ms": self.preferred_max_ms,
            "hard_max_ms": self.hard_max_ms,
            "strong_gap_ms": self.strong_gap_ms,
            "short_gap_ms": self.short_gap_ms,
            "speaker_change_tolerance_ms": self.speaker_change_tolerance_ms,
            "overlap_margin_ms": self.overlap_margin_ms,
            "minimum_chunk_ms": self.minimum_chunk_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Community1AdaptiveConfig:
        """Restore the millisecond policy from a manifest."""

        payload = dict(value)
        for seconds, milliseconds in (
            ("preferred_min_s", "preferred_min_ms"),
            ("target_s", "target_ms"),
            ("preferred_max_s", "preferred_max_ms"),
            ("hard_max_s", "hard_max_ms"),
        ):
            if milliseconds in payload:
                payload[seconds] = payload.pop(milliseconds) / 1000
        for key in (
            "algorithm",
            "configuration_sha256",
            "duration_ms",
            "episode_id",
            "selected_regions_ms",
            "diarization_artifact_id",
            "diarization_fingerprint",
            "diarization_model_fingerprint",
        ):
            payload.pop(key, None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class Community1NativeAdaptiveConfig(ChunkSegmentationConfig):
    """Fixed native ``speaker_counting`` policy using 12 seconds as a score reference."""

    version: str = "community1-native-adaptive-v1"
    preferred_min_s: float = 8.0
    target_s: float = 12.0
    preferred_max_s: float = 16.0
    relaxation_start_s: float = 18.0
    relaxed_clean_max_s: float = 24.0
    hard_max_s: float = 30.0
    strong_pause_ms: int = 400
    short_pause_ms: int = 150
    overlap_margin_ms: int = 250
    minimum_chunk_ms: int = 4_000

    def __post_init__(self) -> None:
        super().__post_init__()
        for field_name in ("relaxation_start_s", "relaxed_clean_max_s"):
            _positive_number(getattr(self, field_name), field_name)
        if not (
            self.preferred_max_s
            <= self.relaxation_start_s
            <= self.relaxed_clean_max_s
            <= self.hard_max_s
        ):
            raise ContractValidationError(
                "relaxation thresholds must be ordered after preferred_max_s"
            )
        for field_name in (
            "strong_pause_ms",
            "short_pause_ms",
            "overlap_margin_ms",
            "minimum_chunk_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractValidationError(f"{field_name} must be a positive integer")
        if self.strong_pause_ms <= self.short_pause_ms:
            raise ContractValidationError("strong_pause_ms must exceed short_pause_ms")
        if self.minimum_chunk_ms > self.preferred_min_ms:
            raise ContractValidationError("minimum_chunk_ms cannot exceed preferred_min_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "algorithm": self.version,
            "preferred_min_ms": self.preferred_min_ms,
            "target_ms": self.target_ms,
            "preferred_max_ms": self.preferred_max_ms,
            "relaxation_start_ms": self.relaxation_start_ms,
            "relaxed_clean_max_ms": self.relaxed_clean_max_ms,
            "hard_max_ms": self.hard_max_ms,
            "strong_pause_ms": self.strong_pause_ms,
            "short_pause_ms": self.short_pause_ms,
            "overlap_margin_ms": self.overlap_margin_ms,
            "minimum_chunk_ms": self.minimum_chunk_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Community1NativeAdaptiveConfig:
        payload = dict(value)
        for seconds, milliseconds in (
            ("preferred_min_s", "preferred_min_ms"),
            ("target_s", "target_ms"),
            ("preferred_max_s", "preferred_max_ms"),
            ("relaxation_start_s", "relaxation_start_ms"),
            ("relaxed_clean_max_s", "relaxed_clean_max_ms"),
            ("hard_max_s", "hard_max_ms"),
        ):
            if milliseconds in payload:
                payload[seconds] = payload.pop(milliseconds) / 1000
        for key in (
            "algorithm",
            "configuration_sha256",
            "duration_ms",
            "episode_id",
            "selected_regions_ms",
            "community1_artifact_id",
            "native_activity_artifact_id",
            "diarization_fingerprint",
        ):
            payload.pop(key, None)
        return cls(**payload)

    @property
    def relaxation_start_ms(self) -> int:
        """Return the first duration at which relaxed selection begins."""

        return _milliseconds(self.relaxation_start_s, "relaxation_start_s")

    @property
    def relaxed_clean_max_ms(self) -> int:
        """Return the inclusive end of the clean relaxation phase."""

        return _milliseconds(self.relaxed_clean_max_s, "relaxed_clean_max_s")


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
class _CommunityBoundary:
    """One boundary derived only from Community-1 standard turns."""

    time_ms: int
    reason: str
    gap_duration_ms: int | None = None
    speaker_change: bool = False
    overlap_conflict: bool = False

    @property
    def priority(self) -> int:
        return {
            "strong_gap_and_speaker_change": 1,
            "strong_gap": 2,
            "speaker_change": 3,
            "short_gap_and_speaker_change": 4,
            "short_gap": 5,
        }[self.reason]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate for private deterministic diagnostics."""

        return {
            "time_ms": self.time_ms,
            "reason": self.reason,
            "gap_duration_ms": self.gap_duration_ms,
            "speaker_change": self.speaker_change,
            "overlap_conflict": self.overlap_conflict,
        }


@dataclass(frozen=True, slots=True)
class ChunkSegmentationResult:
    """Chunks plus the metadata needed to publish their segmentation."""

    chunks: tuple[AudioChunk, ...]
    version: str
    configuration: Mapping[str, Any]
    input_fingerprints: tuple[str, ...]
    segmentation_fingerprint: str
    boundary_diagnostics: tuple[Mapping[str, Any], ...] = ()

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


def _diarization_value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _community_turns(
    *,
    duration_ms: int,
    diarization: object | None,
    standard_turns: Sequence[object] | None,
    diarization_turns: Sequence[object],
    speaker_turns: Sequence[object],
    source_sha256: str,
) -> tuple[tuple[Turn, ...], str | None, str | None]:
    """Read and validate the Community-1 standard view and its provenance."""

    if diarization is not None:
        if isinstance(diarization, Mapping) and "standard_turns" not in diarization:
            raise ContractValidationError("diarization artifact is missing standard_turns")
        if not isinstance(diarization, Mapping) and not hasattr(diarization, "standard_turns"):
            raise ContractValidationError("diarization artifact is missing standard_turns")
        reported_source = _diarization_value(diarization, "source_sha256")
        if reported_source is not None and reported_source != source_sha256:
            raise ContractValidationError("diarization source hash does not match the episode")
        reported_duration = _diarization_value(diarization, "duration_ms")
        if reported_duration is not None and reported_duration != duration_ms:
            raise ContractValidationError("diarization duration does not match the episode")
        artifact_id = _diarization_value(diarization, "artifact_id")
        model = _diarization_value(diarization, "model")
        model_fingerprint = (
            _diarization_value(model, "fingerprint_sha256")
            if model is not None
            else _diarization_value(diarization, "model_fingerprint_sha256")
        )
        if standard_turns is not None:
            raw_turns = standard_turns
        else:
            raw_turns = _diarization_turns(diarization, duration_ms=duration_ms)
    elif standard_turns is not None:
        artifact_id = None
        model_fingerprint = None
        raw_turns = standard_turns
    elif diarization_turns:
        artifact_id = None
        model_fingerprint = None
        raw_turns = diarization_turns
    else:
        # ``speaker_turns`` is the P1R-03 name.  It remains accepted as a
        # standard-turn alias by the explicit Community-1 wrapper.
        artifact_id = None
        model_fingerprint = None
        raw_turns = speaker_turns

    turns = _normalize_turns(
        raw_turns,
        field_name="diarization.standard_turns",
        duration_ms=duration_ms,
    )
    previous: tuple[int, int] | None = None
    for index, turn in enumerate(turns):
        current = (turn.start_ms, turn.end_ms)
        if previous is not None and current < previous:
            raise ContractValidationError(
                f"diarization.standard_turns[{index}] is not monotonically ordered"
            )
        previous = current
    if artifact_id is not None:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ContractValidationError("diarization artifact_id must be non-empty text")
    if model_fingerprint is not None:
        _provided_digest(model_fingerprint, field_name="diarization model fingerprint")
    return turns, artifact_id, model_fingerprint


def _active_segments(
    turns: Sequence[Turn], duration_ms: int
) -> tuple[tuple[int, int, frozenset[str]], ...]:
    """Return maximal source intervals with a constant active speaker set."""

    points = sorted(
        {0, duration_ms, *(point for turn in turns for point in (turn.start_ms, turn.end_ms))}
    )
    segments: list[tuple[int, int, frozenset[str]]] = []
    for start_ms, end_ms in zip(points, points[1:], strict=False):
        active = frozenset(
            turn.speaker_id for turn in turns if turn.start_ms < end_ms and turn.end_ms > start_ms
        )
        if segments and segments[-1][2] == active and segments[-1][1] == start_ms:
            segments[-1] = (segments[-1][0], end_ms, active)
        else:
            segments.append((start_ms, end_ms, active))
    return tuple(segments)


def _overlap_intervals(segments: Sequence[tuple[int, int, frozenset[str]]]) -> tuple[Interval, ...]:
    return tuple((start_ms, end_ms) for start_ms, end_ms, active in segments if len(active) >= 2)


def _is_overlap_conflicted(
    time_ms: int,
    overlaps: Sequence[Interval],
    margin_ms: int,
) -> bool:
    return any(
        start_ms - margin_ms <= time_ms <= end_ms + margin_ms for start_ms, end_ms in overlaps
    )


def _edge_speaker(turns: Sequence[Turn], *, boundary_ms: int, before: bool) -> str | None:
    """Choose the deterministic last/first speaker at a speech-gap edge."""

    matching = (
        [turn for turn in turns if turn.end_ms == boundary_ms]
        if before
        else [turn for turn in turns if turn.start_ms == boundary_ms]
    )
    if not matching:
        return None
    if before:
        selected = max(matching, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_id))
    else:
        selected = min(matching, key=lambda turn: (turn.end_ms, turn.start_ms, turn.speaker_id))
    return selected.speaker_id


def _community_boundaries(
    turns: Sequence[Turn],
    *,
    duration_ms: int,
    config: Community1AdaptiveConfig,
) -> tuple[_CommunityBoundary, ...]:
    """Extract gap and clean-transition candidates from standard diarization."""

    segments = _active_segments(turns, duration_ms)
    overlaps = _overlap_intervals(segments)
    candidates: list[_CommunityBoundary] = []
    for start_ms, end_ms, active in segments:
        if active:
            continue
        gap_duration_ms = end_ms - start_ms
        if gap_duration_ms < config.short_gap_ms:
            continue
        midpoint_ms = (start_ms + end_ms) // 2
        outgoing = _edge_speaker(turns, boundary_ms=start_ms, before=True)
        incoming = _edge_speaker(turns, boundary_ms=end_ms, before=False)
        speaker_change = outgoing is not None and incoming is not None and outgoing != incoming
        if gap_duration_ms >= config.strong_gap_ms:
            reason = "strong_gap_and_speaker_change" if speaker_change else "strong_gap"
        else:
            reason = "short_gap_and_speaker_change" if speaker_change else "short_gap"
        candidates.append(
            _CommunityBoundary(
                time_ms=midpoint_ms,
                reason=reason,
                gap_duration_ms=gap_duration_ms,
                speaker_change=speaker_change,
                overlap_conflict=_is_overlap_conflicted(
                    midpoint_ms, overlaps, config.overlap_margin_ms
                ),
            )
        )

    # A clean change may have a micro-gap, or two near-contiguous turns.  Pair
    # only turns whose boundaries are within the fixed tolerance, then apply
    # the active-speaker and overlap checks to the resulting midpoint.
    for outgoing in turns:
        for incoming in turns:
            if outgoing.speaker_id == incoming.speaker_id:
                continue
            delta_ms = incoming.start_ms - outgoing.end_ms
            if abs(delta_ms) > config.speaker_change_tolerance_ms:
                continue
            midpoint_ms = (outgoing.end_ms + incoming.start_ms) // 2
            before_active = frozenset(
                turn.speaker_id for turn in turns if turn.start_ms < midpoint_ms <= turn.end_ms
            )
            after_active = frozenset(
                turn.speaker_id for turn in turns if turn.start_ms <= midpoint_ms < turn.end_ms
            )
            overlap_conflict = _is_overlap_conflicted(
                midpoint_ms, overlaps, config.overlap_margin_ms
            )
            if (
                len(before_active) != 1
                or len(after_active) != 1
                or before_active == after_active
                or overlap_conflict
            ):
                continue
            candidates.append(
                _CommunityBoundary(
                    time_ms=midpoint_ms,
                    reason="speaker_change",
                    speaker_change=True,
                    overlap_conflict=False,
                )
            )

    # Distinct extraction paths can describe the same timestamp.  Prefer the
    # stronger class and retain one deterministic record per timestamp.
    unique: dict[int, _CommunityBoundary] = {}
    for candidate in candidates:
        previous = unique.get(candidate.time_ms)
        if previous is None or (
            candidate.priority,
            candidate.time_ms,
        ) < (previous.priority, previous.time_ms):
            unique[candidate.time_ms] = candidate
    return tuple(sorted(unique.values(), key=lambda item: (item.time_ms, item.priority)))


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


def _select_community_boundary(
    current_ms: int,
    region_end_ms: int,
    candidates: Sequence[_CommunityBoundary],
    config: Community1AdaptiveConfig,
) -> tuple[int, str, _CommunityBoundary | None]:
    """Select one Community-1 boundary using the fixed priority policy."""

    remaining_ms = region_end_ms - current_ms
    if remaining_ms <= config.hard_max_ms:
        return region_end_ms, "episode_end", None

    preferred_min = current_ms + config.preferred_min_ms
    preferred_max = current_ms + config.preferred_max_ms
    hard_max = min(region_end_ms, current_ms + config.hard_max_ms)
    target = current_ms + config.target_ms

    def in_range(candidate: _CommunityBoundary, lower: int, upper: int) -> bool:
        return lower <= candidate.time_ms <= upper and candidate.time_ms < region_end_ms

    natural = [
        candidate
        for candidate in candidates
        if in_range(candidate, preferred_min, preferred_max) and not candidate.overlap_conflict
    ]
    if natural:
        selected = min(
            natural,
            key=lambda candidate: (
                candidate.priority,
                abs(candidate.time_ms - target),
                candidate.time_ms,
            ),
        )
        return selected.time_ms, selected.reason, selected

    extension = [
        candidate
        for candidate in candidates
        if in_range(candidate, preferred_max + 1, hard_max) and not candidate.overlap_conflict
    ]
    if extension:
        highest_priority = min(candidate.priority for candidate in extension)
        selected = min(
            (candidate for candidate in extension if candidate.priority == highest_priority),
            key=lambda candidate: candidate.time_ms,
        )
        return selected.time_ms, selected.reason, selected

    # An overlap-conflicted transition is allowed only when the clean search
    # would otherwise fall through to hard maximum.  Apply the same priority
    # rules in the preferred interval and earliest-in-class rule in extension.
    conflicted_preferred = [
        candidate
        for candidate in candidates
        if in_range(candidate, preferred_min, preferred_max) and candidate.overlap_conflict
    ]
    if conflicted_preferred:
        selected = min(
            conflicted_preferred,
            key=lambda candidate: (
                candidate.priority,
                abs(candidate.time_ms - target),
                candidate.time_ms,
            ),
        )
        return selected.time_ms, selected.reason, selected

    conflicted_extension = [
        candidate
        for candidate in candidates
        if in_range(candidate, preferred_max + 1, hard_max) and candidate.overlap_conflict
    ]
    if conflicted_extension:
        highest_priority = min(candidate.priority for candidate in conflicted_extension)
        selected = min(
            (
                candidate
                for candidate in conflicted_extension
                if candidate.priority == highest_priority
            ),
            key=lambda candidate: candidate.time_ms,
        )
        return selected.time_ms, selected.reason, selected
    return hard_max, "hard_maximum", None


def _community_chunk(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    start_ms: int,
    end_ms: int,
    reason: str,
    candidate: _CommunityBoundary | None,
    start_reason: str | None,
    partition: Literal["development", "held_out"] | None,
    fingerprint: str,
    config: Community1AdaptiveConfig,
    diarization_artifact_id: str | None,
) -> AudioChunk:
    return AudioChunk.create(
        episode_id=episode_id,
        source_sha256=source_sha256,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        segmentation_fingerprint=fingerprint,
        partition=partition,
        boundary_start_reason=start_reason,
        boundary_end_reason=reason,  # type: ignore[arg-type]
        boundary_gap_duration_ms=(candidate.gap_duration_ms if candidate else None),
        boundary_speaker_change=bool(candidate and candidate.speaker_change),
        boundary_overlap_conflict=bool(candidate and candidate.overlap_conflict),
        distance_from_target_ms=abs(end_ms - start_ms - config.target_ms),
        diarization_artifact_id=diarization_artifact_id,
        segmentation_version=config.version,
        segmentation_configuration_hash=config.configuration_sha256,
    )


def _segment_community1(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    config: Community1AdaptiveConfig,
    turns: tuple[Turn, ...],
    selected_regions: tuple[Interval, ...],
    partition: Literal["development", "held_out"] | None,
    diarization_artifact_id: str | None,
    fingerprint: str,
) -> tuple[tuple[AudioChunk, ...], tuple[Mapping[str, Any], ...]]:
    candidates = _community_boundaries(turns, duration_ms=duration_ms, config=config)
    chunks: list[AudioChunk] = []
    diagnostics: list[Mapping[str, Any]] = []
    for region_start, region_end in selected_regions:
        current = region_start
        region_chunks: list[AudioChunk] = []
        while current < region_end:
            remaining_ms = region_end - current
            if remaining_ms < config.preferred_min_ms and region_chunks:
                previous = region_chunks[-1]
                if region_end - previous.start_ms <= config.hard_max_ms:
                    merged = _community_chunk(
                        episode_id=episode_id,
                        source_sha256=source_sha256,
                        duration_ms=duration_ms,
                        start_ms=previous.start_ms,
                        end_ms=region_end,
                        reason="episode_end",
                        candidate=None,
                        start_reason=previous.boundary_start_reason,
                        partition=partition,
                        fingerprint=fingerprint,
                        config=config,
                        diarization_artifact_id=diarization_artifact_id,
                    )
                    region_chunks[-1] = merged
                    diagnostics[-1] = {
                        **diagnostics[-1],
                        "end_ms": region_end,
                        "reason": "episode_end",
                        "distance_from_target_ms": merged.distance_from_target_ms,
                    }
                    current = region_end
                    continue
            end_ms, reason, candidate = _select_community_boundary(
                current, region_end, candidates, config
            )
            chunk = _community_chunk(
                episode_id=episode_id,
                source_sha256=source_sha256,
                duration_ms=duration_ms,
                start_ms=current,
                end_ms=end_ms,
                reason=reason,
                candidate=candidate,
                start_reason=(
                    region_chunks[-1].boundary_end_reason if region_chunks else "episode_start"
                ),
                partition=partition,
                fingerprint=fingerprint,
                config=config,
                diarization_artifact_id=diarization_artifact_id,
            )
            region_chunks.append(chunk)
            diagnostics.append(
                {
                    "start_ms": current,
                    "end_ms": end_ms,
                    "reason": reason,
                    "gap_duration_ms": candidate.gap_duration_ms if candidate else None,
                    "speaker_change": bool(candidate and candidate.speaker_change),
                    "overlap_conflict": bool(candidate and candidate.overlap_conflict),
                    "distance_from_target_ms": chunk.distance_from_target_ms,
                }
            )
            current = end_ms
        chunks.extend(region_chunks)
    return tuple(chunks), tuple(diagnostics)


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


def _prepare_community_inputs(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    config: Community1AdaptiveConfig,
    standard_turns: Sequence[object] | None,
    speaker_turns: Sequence[object],
    diarization_turns: Sequence[object],
    diarization: object | None,
    selected_regions: Sequence[object] | None,
    diarization_fingerprint: str | None,
    input_fingerprints: Sequence[str],
    diarization_artifact_id: str | None,
) -> tuple[
    tuple[Interval, ...],
    tuple[Turn, ...],
    tuple[str, ...],
    str,
    str | None,
    str | None,
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
    turns, artifact_id, model_fingerprint = _community_turns(
        duration_ms=duration_ms,
        diarization=diarization,
        standard_turns=standard_turns,
        diarization_turns=diarization_turns,
        speaker_turns=speaker_turns,
        source_sha256=source_sha256,
    )
    if (
        diarization_artifact_id is not None
        and artifact_id is not None
        and diarization_artifact_id != artifact_id
    ):
        raise ContractValidationError(
            "diarization_artifact_id does not match the supplied artifact"
        )
    artifact_id = diarization_artifact_id or artifact_id
    if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id.strip()):
        raise ContractValidationError("diarization_artifact_id must be non-empty text")
    normalized_turns = [turn.to_dict() for turn in turns]
    diarization_values: dict[str, Any] = {
        "standard_turns": normalized_turns,
        "artifact_id": artifact_id,
        "model_fingerprint_sha256": model_fingerprint,
    }
    digest = _provided_digest(
        diarization_fingerprint, field_name="diarization_fingerprint"
    ) or _input_digest("community1-standard-diarization", diarization_values)
    extra_digests = tuple(
        _provided_digest(value, field_name=f"input_fingerprints[{index}]") or value
        for index, value in enumerate(input_fingerprints)
    )
    all_digests = (digest,) + extra_digests
    fingerprint = stage_fingerprint(
        "benchmark_chunks",
        source_sha256=source_sha256,
        upstream_artifact_hashes=all_digests,
        configuration={
            "algorithm": config.version,
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
            "segmentation": config.to_dict(),
        },
        pipeline_version=config.version,
    )
    return regions, turns, all_digests, fingerprint, artifact_id, model_fingerprint


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
    standard_turns: Sequence[object] | None = None,
    exclusive_turns: Sequence[object] | None = None,
    native_activity: NativeActivityArtifact | Mapping[str, Any] | None = None,
    diarization: object | None = None,
    selected_regions: Sequence[object] | None = None,
    partition: Literal["development", "held_out"] | None = None,
    vad_fingerprint: str | None = None,
    acoustic_fingerprint: str | None = None,
    diarization_fingerprint: str | None = None,
    input_fingerprints: Sequence[str] = (),
    diarization_artifact_id: str | None = None,
    community1_artifact_id: str | None = None,
    algorithm: str | None = None,
) -> ChunkSegmentationResult:
    """Build deterministic chunks and their publication metadata.

    Boundary times are always interpreted in the original episode timeline.
    Supplying ``standard_turns``, ``diarization_turns`` or ``diarization``
    selects ``community1-adaptive-v1`` and uses only standard Community-1
    turns. The explicit ``algorithm`` argument can require that path even when
    the input is absent, which produces a validation error instead of a
    fallback. Calls without those inputs retain the legacy P1R policy.
    """

    if algorithm not in (
        None,
        "p1r-chunk-segmentation-v1",
        "community1-adaptive-v1",
        "community1-native-adaptive-v1",
    ):
        raise ContractValidationError(f"unsupported chunk segmentation algorithm: {algorithm}")
    if algorithm == "community1-native-adaptive-v1" or native_activity is not None:
        native_config = (
            config
            if isinstance(config, Community1NativeAdaptiveConfig)
            else Community1NativeAdaptiveConfig()
        )
        return segment_native_activity_with_metadata(
            episode_id,
            source_sha256,
            duration_ms,
            native_activity=native_activity,
            diarization=diarization,
            standard_turns=standard_turns,
            exclusive_turns=exclusive_turns,
            community1_artifact_id=community1_artifact_id or diarization_artifact_id,
            config=native_config,
            selected_regions=selected_regions,
            partition=partition,
        )
    use_community = algorithm == "community1-adaptive-v1" or any(
        (
            standard_turns is not None,
            diarization is not None,
            bool(diarization_turns),
            diarization_artifact_id is not None,
        )
    )
    if use_community:
        community_config = (
            config
            if isinstance(config, Community1AdaptiveConfig)
            else Community1AdaptiveConfig(
                preferred_min_s=(config.preferred_min_s if config else 8.0),
                target_s=(config.target_s if config else 12.0),
                preferred_max_s=(config.preferred_max_s if config else 18.0),
                hard_max_s=(config.hard_max_s if config else 30.0),
            )
        )
        if (
            standard_turns is None
            and diarization is None
            and not diarization_turns
            and not speaker_turns
        ):
            raise ContractValidationError(
                "community1-adaptive-v1 requires Community-1 standard diarization"
            )
        regions, turns, input_hashes, fingerprint, artifact_id, model_fingerprint = (
            _prepare_community_inputs(
                episode_id=episode_id,
                source_sha256=source_sha256,
                duration_ms=duration_ms,
                config=community_config,
                standard_turns=standard_turns,
                speaker_turns=speaker_turns,
                diarization_turns=diarization_turns,
                diarization=diarization,
                selected_regions=selected_regions,
                diarization_fingerprint=diarization_fingerprint,
                input_fingerprints=input_fingerprints,
                diarization_artifact_id=diarization_artifact_id,
            )
        )
        chunks, diagnostics = _segment_community1(
            episode_id=episode_id,
            source_sha256=source_sha256,
            duration_ms=duration_ms,
            config=community_config,
            turns=turns,
            selected_regions=regions,
            partition=partition,
            diarization_artifact_id=artifact_id,
            fingerprint=fingerprint,
        )
        configuration = {
            **community_config.to_dict(),
            "configuration_sha256": community_config.configuration_sha256,
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
            "diarization_artifact_id": artifact_id,
            "diarization_fingerprint": input_hashes[0],
            "diarization_model_fingerprint": model_fingerprint,
        }
        result = ChunkSegmentationResult(
            chunks=chunks,
            version=community_config.version,
            configuration=configuration,
            input_fingerprints=input_hashes,
            segmentation_fingerprint=fingerprint,
            boundary_diagnostics=diagnostics,
        )
        _validate_coverage(result.chunks, regions, hard_max_ms=community_config.hard_max_ms)
        return result

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


@dataclass(frozen=True, slots=True)
class _NativeCandidate:
    time_ms: int
    candidate_type: str
    type_adjustment: float
    pause_duration_ms: int | None = None
    exclusive_speaker_before: str | None = None
    exclusive_speaker_after: str | None = None


_NATIVE_TYPE_ADJUSTMENTS = {
    "strong_pause": -2.0,
    "short_pause": -1.0,
    "speaker_change": -1.5,
    "strong_pause_and_speaker_change": -2.5,
    "short_pause_and_speaker_change": -1.75,
}
_NATIVE_TYPE_PRIORITIES = {
    "strong_pause_and_speaker_change": 0,
    "strong_pause": 1,
    "short_pause_and_speaker_change": 2,
    "speaker_change": 3,
    "short_pause": 4,
}


def _native_activity_value(
    value: object, *, source_sha256: str, duration_ms: int
) -> NativeActivityArtifact:
    if isinstance(value, NativeActivityArtifact):
        artifact = value
    elif isinstance(value, Mapping):
        try:
            artifact = NativeActivityArtifact.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError(f"invalid native activity artifact: {exc}") from exc
    else:
        raise ContractValidationError("native activity artifact is required")
    if artifact.source_sha256 != source_sha256:
        raise ContractValidationError("native activity source hash does not match the episode")
    if artifact.duration_ms != duration_ms:
        raise ContractValidationError("native activity duration does not match the episode")
    return artifact


def _exclusive_turns(
    value: object | None,
    *,
    duration_ms: int,
    explicit: Sequence[object] | None,
) -> tuple[Turn, ...]:
    if explicit is not None:
        raw = explicit
    elif isinstance(value, DiarizationResult):
        raw = value.exclusive_turns
    elif isinstance(value, Mapping):
        raw = value.get("exclusive_turns", ())
    elif value is not None:
        raw = getattr(value, "exclusive_turns", ())
    else:
        raw = ()
    if not isinstance(raw, Sequence):
        raise ContractValidationError("diarization exclusive_turns must be a sequence")
    return _normalize_turns(raw, field_name="diarization.exclusive_turns", duration_ms=duration_ms)


def _native_transitions(turns: Sequence[Turn]) -> tuple[tuple[int, str, str], ...]:
    ordered = sorted(turns, key=lambda turn: (turn.start_ms, turn.end_ms, turn.speaker_id))
    transitions: list[tuple[int, str, str]] = []
    for before, after in zip(ordered, ordered[1:], strict=False):
        if before.speaker_id != after.speaker_id:
            transitions.append((after.start_ms, before.speaker_id, after.speaker_id))
    return tuple(transitions)


def _native_overlap_adjustment(
    time_ms: int,
    activity: Sequence[NativeActivityInterval],
    *,
    margin_ms: int,
) -> float:
    instant_overlap = any(
        interval.speaker_count >= 2 and interval.start_ms <= time_ms < interval.end_ms
        for interval in activity
    )
    if instant_overlap:
        return 1.5
    nearby_overlap = any(
        interval.speaker_count >= 2
        and interval.end_ms > time_ms - margin_ms
        and interval.start_ms < time_ms + margin_ms
        for interval in activity
    )
    return 0.75 if nearby_overlap else 0.0


def _native_candidates(
    *,
    start_ms: int,
    region_end_ms: int,
    activity: NativeActivityArtifact,
    exclusive_turns: Sequence[Turn],
    config: Community1NativeAdaptiveConfig,
) -> tuple[_NativeCandidate, ...]:
    target_ms = start_ms + config.target_ms
    lower = start_ms + config.preferred_min_ms
    upper = min(start_ms + config.hard_max_ms, region_end_ms)
    if target_ms >= region_end_ms:
        return ()
    transitions = _native_transitions(exclusive_turns)
    candidates: list[_NativeCandidate] = []
    for interval in activity.intervals:
        if interval.speaker_count != 0:
            continue
        pause_duration_ms = interval.end_ms - interval.start_ms
        if pause_duration_ms < config.short_pause_ms:
            continue
        midpoint_ms = (interval.start_ms + interval.end_ms) // 2
        if not lower <= midpoint_ms <= upper:
            continue
        nearby_transition = min(
            (
                transition
                for transition in transitions
                if abs(transition[0] - midpoint_ms) <= config.overlap_margin_ms
            ),
            key=lambda transition: (abs(transition[0] - midpoint_ms), transition[0]),
            default=None,
        )
        strong = pause_duration_ms >= config.strong_pause_ms
        base_type = "strong_pause" if strong else "short_pause"
        candidate_type = f"{base_type}_and_speaker_change" if nearby_transition else base_type
        candidates.append(
            _NativeCandidate(
                midpoint_ms,
                candidate_type,
                _NATIVE_TYPE_ADJUSTMENTS[candidate_type],
                pause_duration_ms,
                nearby_transition[1] if nearby_transition else None,
                nearby_transition[2] if nearby_transition else None,
            )
        )
    pause_times = tuple(
        candidate.time_ms for candidate in candidates if "pause" in candidate.candidate_type
    )
    for time_ms, before, after in transitions:
        if not lower <= time_ms <= upper:
            continue
        if any(abs(time_ms - pause_time) <= config.overlap_margin_ms for pause_time in pause_times):
            continue
        candidates.append(
            _NativeCandidate(
                time_ms,
                "speaker_change",
                _NATIVE_TYPE_ADJUSTMENTS["speaker_change"],
                exclusive_speaker_before=before,
                exclusive_speaker_after=after,
            )
        )
    return tuple(candidates)


def _select_native_candidate(
    *,
    start_ms: int,
    region_end_ms: int,
    activity: NativeActivityArtifact,
    exclusive_turns: Sequence[Turn],
    config: Community1NativeAdaptiveConfig,
) -> tuple[int, str, str, _NativeCandidate | None, dict[str, float]]:
    remaining_ms = region_end_ms - start_ms
    if remaining_ms <= config.target_ms:
        return (
            region_end_ms,
            "episode_end",
            "preferred",
            None,
            {
                "distance_cost": abs(remaining_ms - config.target_ms) / 1000.0,
                "type_adjustment": 0.0,
                "overlap_adjustment": 0.0,
                "total_score": abs(remaining_ms - config.target_ms) / 1000.0,
            },
        )
    candidates = _native_candidates(
        start_ms=start_ms,
        region_end_ms=region_end_ms,
        activity=activity,
        exclusive_turns=exclusive_turns,
        config=config,
    )
    target_ms = start_ms + config.target_ms

    def details(candidate: _NativeCandidate) -> dict[str, float]:
        distance_cost = abs(candidate.time_ms - target_ms) / 1000.0
        overlap_adjustment = _native_overlap_adjustment(
            candidate.time_ms, activity.intervals, margin_ms=config.overlap_margin_ms
        )
        return {
            "distance_cost": distance_cost,
            "type_adjustment": candidate.type_adjustment,
            "overlap_adjustment": overlap_adjustment,
            "total_score": distance_cost + candidate.type_adjustment + overlap_adjustment,
        }

    def scored(candidate: _NativeCandidate) -> tuple[float, float, float, float, int]:
        score = details(candidate)
        return (
            score["total_score"],
            abs(candidate.time_ms - target_ms),
            candidate.type_adjustment,
            score["overlap_adjustment"],
            candidate.time_ms,
        )

    preferred_end = start_ms + config.relaxation_start_ms
    relaxed_clean_end = start_ms + config.relaxed_clean_max_ms
    preferred = tuple(candidate for candidate in candidates if candidate.time_ms <= preferred_end)
    if preferred:
        selected = min(preferred, key=scored)
        return selected.time_ms, selected.candidate_type, "preferred", selected, details(selected)

    relaxed_clean = tuple(
        candidate
        for candidate in candidates
        if candidate.time_ms <= relaxed_clean_end
        and _native_overlap_adjustment(
            candidate.time_ms, activity.intervals, margin_ms=config.overlap_margin_ms
        )
        <= 0.75
    )
    if relaxed_clean:
        selected = min(
            relaxed_clean,
            key=lambda candidate: (
                candidate.time_ms,
                _NATIVE_TYPE_PRIORITIES[candidate.candidate_type],
            ),
        )
        return (
            selected.time_ms,
            selected.candidate_type,
            "relaxed_clean",
            selected,
            details(selected),
        )

    relaxed_any = tuple(
        candidate for candidate in candidates if candidate.time_ms <= start_ms + config.hard_max_ms
    )
    if relaxed_any:
        selected = min(
            relaxed_any,
            key=lambda candidate: (
                candidate.time_ms,
                _NATIVE_TYPE_PRIORITIES[candidate.candidate_type],
            ),
        )
        return selected.time_ms, selected.candidate_type, "relaxed_any", selected, details(selected)

    end_ms = min(start_ms + config.hard_max_ms, region_end_ms)
    reason = "episode_end" if end_ms == region_end_ms else "hard_maximum"
    phase = "relaxed_any" if reason == "episode_end" else "hard_maximum"
    distance_cost = abs(end_ms - target_ms) / 1000.0
    return (
        end_ms,
        reason,
        phase,
        None,
        {
            "distance_cost": distance_cost,
            "type_adjustment": 0.0,
            "overlap_adjustment": 0.0,
            "total_score": distance_cost,
        },
    )


def _native_chunk(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    start_ms: int,
    end_ms: int,
    reason: str,
    selection_phase: str,
    candidate: _NativeCandidate | None,
    score: Mapping[str, float],
    start_reason: str | None,
    partition: Literal["development", "held_out"] | None,
    fingerprint: str,
    config: Community1NativeAdaptiveConfig,
    community1_artifact_id: str,
    native_activity_artifact_id: str,
) -> AudioChunk:
    return AudioChunk.create(
        episode_id=episode_id,
        source_sha256=source_sha256,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        segmentation_fingerprint=fingerprint,
        partition=partition,
        boundary_start_reason=start_reason,  # type: ignore[arg-type]
        boundary_end_reason=reason,  # type: ignore[arg-type]
        selection_phase=selection_phase,  # type: ignore[arg-type]
        boundary_speaker_change=bool(candidate and candidate.exclusive_speaker_before),
        boundary_overlap_conflict=bool(score.get("overlap_adjustment", 0.0)),
        distance_from_target_ms=abs(end_ms - start_ms - config.target_ms),
        diarization_artifact_id=community1_artifact_id,
        community1_artifact_id=community1_artifact_id,
        native_activity_artifact_id=native_activity_artifact_id,
        selected_candidate_type=(candidate.candidate_type if candidate else reason),
        selected_candidate_timestamp_ms=(candidate.time_ms if candidate else end_ms),
        distance_cost=score.get("distance_cost"),
        type_adjustment=score.get("type_adjustment"),
        overlap_adjustment=score.get("overlap_adjustment"),
        total_score=score.get("total_score"),
        pause_duration_ms=(candidate.pause_duration_ms if candidate else None),
        exclusive_speaker_before=(candidate.exclusive_speaker_before if candidate else None),
        exclusive_speaker_after=(candidate.exclusive_speaker_after if candidate else None),
        segmentation_version=config.version,
        segmentation_configuration_hash=config.configuration_sha256,
    )


def _segment_native_activity(
    *,
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    config: Community1NativeAdaptiveConfig,
    activity: NativeActivityArtifact,
    exclusive_turns: tuple[Turn, ...],
    selected_regions: tuple[Interval, ...],
    partition: Literal["development", "held_out"] | None,
    fingerprint: str,
    community1_artifact_id: str,
) -> tuple[tuple[AudioChunk, ...], tuple[Mapping[str, Any], ...]]:
    chunks: list[AudioChunk] = []
    diagnostics: list[Mapping[str, Any]] = []
    for region_start, region_end in selected_regions:
        current = region_start
        region_chunks: list[AudioChunk] = []
        while current < region_end:
            remaining_ms = region_end - current
            if remaining_ms < config.minimum_chunk_ms and region_chunks:
                previous = region_chunks[-1]
                if region_end - previous.start_ms <= config.hard_max_ms:
                    merged = _native_chunk(
                        episode_id=episode_id,
                        source_sha256=source_sha256,
                        duration_ms=duration_ms,
                        start_ms=previous.start_ms,
                        end_ms=region_end,
                        reason="episode_end",
                        selection_phase=previous.selection_phase or "preferred",
                        candidate=None,
                        score={
                            "distance_cost": abs(region_end - previous.start_ms - config.target_ms)
                            / 1000.0,
                            "type_adjustment": 0.0,
                            "overlap_adjustment": 0.0,
                            "total_score": abs(region_end - previous.start_ms - config.target_ms)
                            / 1000.0,
                        },
                        start_reason=previous.boundary_start_reason,
                        partition=partition,
                        fingerprint=fingerprint,
                        config=config,
                        community1_artifact_id=community1_artifact_id,
                        native_activity_artifact_id=activity.artifact_id,
                    )
                    region_chunks[-1] = merged
                    diagnostics[-1] = {
                        **diagnostics[-1],
                        "end_ms": region_end,
                        "reason": "episode_end",
                    }
                    current = region_end
                    continue
            end_ms, reason, selection_phase, candidate, score = _select_native_candidate(
                start_ms=current,
                region_end_ms=region_end,
                activity=activity,
                exclusive_turns=exclusive_turns,
                config=config,
            )
            chunk = _native_chunk(
                episode_id=episode_id,
                source_sha256=source_sha256,
                duration_ms=duration_ms,
                start_ms=current,
                end_ms=end_ms,
                reason=reason,
                selection_phase=selection_phase,
                candidate=candidate,
                score=score,
                start_reason=(
                    region_chunks[-1].boundary_end_reason if region_chunks else "episode_start"
                ),
                partition=partition,
                fingerprint=fingerprint,
                config=config,
                community1_artifact_id=community1_artifact_id,
                native_activity_artifact_id=activity.artifact_id,
            )
            region_chunks.append(chunk)
            diagnostics.append(
                {
                    "start_ms": current,
                    "end_ms": end_ms,
                    "reason": reason,
                    "selection_phase": selection_phase,
                    "ideal_target_timestamp_ms": current + config.target_ms,
                    "selected_candidate_type": chunk.selected_candidate_type,
                    "selected_candidate_timestamp_ms": chunk.selected_candidate_timestamp_ms,
                    **score,
                }
            )
            current = end_ms
        chunks.extend(region_chunks)
    return tuple(chunks), tuple(diagnostics)


def segment_native_activity_with_metadata(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    *,
    native_activity: NativeActivityArtifact | Mapping[str, Any] | None = None,
    diarization: object | None = None,
    standard_turns: Sequence[object] | None = None,
    exclusive_turns: Sequence[object] | None = None,
    community1_artifact_id: str | None = None,
    config: Community1NativeAdaptiveConfig | None = None,
    selected_regions: Sequence[object] | None = None,
    partition: Literal["development", "held_out"] | None = None,
) -> ChunkSegmentationResult:
    """Build chunks from one Community-1 run's native speaker-count artifact."""

    policy = config or Community1NativeAdaptiveConfig()
    if not isinstance(policy, Community1NativeAdaptiveConfig):
        raise ContractValidationError("native activity requires Community1NativeAdaptiveConfig")
    if isinstance(diarization, DiarizationResult):
        artifact = native_activity or diarization.native_activity
        standard = standard_turns or diarization.standard_turns
        exclusive = _exclusive_turns(diarization, duration_ms=duration_ms, explicit=exclusive_turns)
        default_artifact_id = diarization.artifact_id
    else:
        artifact = native_activity
        standard = standard_turns
        exclusive = _exclusive_turns(diarization, duration_ms=duration_ms, explicit=exclusive_turns)
        default_artifact_id = (
            _diarization_value(diarization, "artifact_id") if diarization else None
        )
    if artifact is None:
        raise ContractValidationError("community1-native-adaptive-v1 requires native activity")
    native = _native_activity_value(artifact, source_sha256=source_sha256, duration_ms=duration_ms)
    if standard is None:
        raise ContractValidationError("community1-native-adaptive-v1 requires standard diarization")
    standard_normalized = _normalize_turns(
        standard, field_name="diarization.standard_turns", duration_ms=duration_ms
    )
    if not exclusive:
        raise ContractValidationError(
            "community1-native-adaptive-v1 requires exclusive diarization"
        )
    regions = _normalize_regions(selected_regions, duration_ms)
    artifact_id = community1_artifact_id or default_artifact_id
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ContractValidationError("community1_artifact_id is required")
    input_hashes = (
        _input_digest(
            "community1-standard-diarization",
            [turn.to_dict() for turn in standard_normalized],
        ),
        _input_digest("community1-exclusive-diarization", [turn.to_dict() for turn in exclusive]),
        _input_digest("community1-native-activity", native.to_dict()),
    )
    fingerprint = stage_fingerprint(
        "benchmark_chunks",
        source_sha256=source_sha256,
        upstream_artifact_hashes=input_hashes,
        configuration={
            "algorithm": policy.version,
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
            "segmentation": policy.to_dict(),
            "community1_artifact_id": artifact_id,
            "native_activity_artifact_id": native.artifact_id,
        },
        pipeline_version=policy.version,
    )
    chunks, diagnostics = _segment_native_activity(
        episode_id=episode_id,
        source_sha256=source_sha256,
        duration_ms=duration_ms,
        config=policy,
        activity=native,
        exclusive_turns=exclusive,
        selected_regions=regions,
        partition=partition,
        fingerprint=fingerprint,
        community1_artifact_id=artifact_id,
    )
    result = ChunkSegmentationResult(
        chunks=chunks,
        version=policy.version,
        configuration={
            **policy.to_dict(),
            "configuration_sha256": policy.configuration_sha256,
            "duration_ms": duration_ms,
            "episode_id": episode_id,
            "selected_regions_ms": [list(region) for region in regions],
            "community1_artifact_id": artifact_id,
            "native_activity_artifact_id": native.artifact_id,
            "native_activity_timeline": native.to_dict()["timeline"],
        },
        input_fingerprints=input_hashes,
        segmentation_fingerprint=fingerprint,
        boundary_diagnostics=diagnostics,
    )
    _validate_coverage(result.chunks, regions, hard_max_ms=policy.hard_max_ms)
    return result


def segment_native_activity_chunks(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    **kwargs: Any,
) -> tuple[AudioChunk, ...]:
    """Return only chunks from the native Community-1 adaptive policy."""

    return segment_native_activity_with_metadata(
        episode_id, source_sha256, duration_ms, **kwargs
    ).chunks


def segment_chunks(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    **kwargs: Any,
) -> tuple[AudioChunk, ...]:
    """Return only the immutable ``AudioChunk`` records for one episode."""

    return segment_chunks_with_metadata(episode_id, source_sha256, duration_ms, **kwargs).chunks


def segment_community1_chunks(
    episode_id: str,
    source_sha256: str,
    duration_ms: int,
    *,
    standard_turns: Sequence[object] | None = None,
    diarization: object | None = None,
    diarization_artifact_id: str | None = None,
    config: Community1AdaptiveConfig | None = None,
    selected_regions: Sequence[object] | None = None,
    partition: Literal["development", "held_out"] | None = None,
    diarization_fingerprint: str | None = None,
) -> tuple[AudioChunk, ...]:
    """Build chunks from Community-1 standard turns and no other signal."""

    return segment_chunks_with_metadata(
        episode_id,
        source_sha256,
        duration_ms,
        algorithm="community1-adaptive-v1",
        config=config,
        standard_turns=standard_turns,
        diarization=diarization,
        diarization_artifact_id=diarization_artifact_id,
        selected_regions=selected_regions,
        partition=partition,
        diarization_fingerprint=diarization_fingerprint,
    ).chunks


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
    "Community1AdaptiveConfig",
    "Community1NativeAdaptiveConfig",
    "build_audio_chunks",
    "segment_community1_chunks",
    "segment_audio_chunks",
    "segment_chunks",
    "segment_chunks_with_metadata",
    "segment_native_activity_chunks",
    "segment_native_activity_with_metadata",
    "validate_chunk_coverage",
]

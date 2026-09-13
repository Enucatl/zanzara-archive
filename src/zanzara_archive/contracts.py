"""Typed boundaries shared by archive stages and internal services.

The contracts in this module deliberately contain no model or network code.  They
validate data at the adapter boundary so that malformed timing, vectors, and
provenance cannot be published as canonical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ContractStatus = Literal["timed", "text_only", "no_words", "missing_asr"]
TimestampGranularity = Literal["word", "segment"]
ReferenceReviewStatus = Literal["draft", "human_truth", "superseded", "rejected"]
ChunkPartition = Literal["development", "held_out"]
JobState = Literal["queued", "running", "retry_wait", "succeeded", "failed", "cancelled", "blocked"]
IdentityAction = Literal["same_person", "different_person", "uncertain"]
IdentityState = Literal["active", "superseded"]
CalibrationStatus = Literal["uncalibrated_rank_fusion", "calibrated_estimate", "unavailable"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")


class ContractValidationError(ValueError):
    """Raised when a value cannot cross a typed contract boundary."""


class AdapterFailure(RuntimeError):
    """A typed failure returned by an inference or service adapter."""

    def __init__(self, error: ApiError) -> None:
        super().__init__(error.message)
        self.error = error

    @property
    def code(self) -> str:
        """Return the machine-readable failure code."""
        return self.error.code

    @property
    def retryable(self) -> bool:
        """Return whether the failed request may be retried."""
        return self.error.retryable

    @property
    def request_id(self) -> str:
        """Return the request identifier associated with the failure."""
        return self.error.request_id


class UnsupportedCapabilityError(AdapterFailure):
    """The selected adapter cannot provide a requested capability."""


class InvalidAudioError(AdapterFailure):
    """The adapter rejected audio or its registered artifact."""


class ModelUnavailableError(AdapterFailure):
    """The requested model is not loaded or cannot be accessed."""


class RequestTimeoutError(AdapterFailure):
    """An adapter request exceeded its bounded timeout."""


class BudgetBlockedError(AdapterFailure):
    """A request was refused by the cost ledger before network I/O."""


class GenerationMismatchError(AdapterFailure):
    """A response references a stale or incompatible published generation."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be non-empty text")
    return value


def _one_of(value: object, choices: set[str], field_name: str) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ContractValidationError(f"{field_name} must be one of: {allowed}")


def _require_id(value: object, field_name: str) -> str:
    value = _require_text(value, field_name)
    if not _ID_RE.fullmatch(value):
        raise ContractValidationError(f"{field_name} contains unsupported characters")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractValidationError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractValidationError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value


def _interval(
    start_ms: object, end_ms: object, field_name: str, duration_ms: int | None = None
) -> tuple[int, int]:
    start = _require_nonnegative_int(start_ms, f"{field_name}.start_ms")
    end = _require_positive_int(end_ms, f"{field_name}.end_ms")
    if start >= end:
        raise ContractValidationError(f"{field_name} must be a non-empty half-open interval")
    if duration_ms is not None and end > duration_ms:
        raise ContractValidationError(f"{field_name} must be within the audio duration")
    return start, end


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{field_name} must be a finite number")
    return number


def _confidence(value: object | None, field_name: str) -> float | None:
    if value is None:
        return None
    confidence = _finite_number(value, field_name)
    if not 0.0 <= confidence <= 1.0:
        raise ContractValidationError(f"{field_name} must be between 0 and 1")
    return confidence


def _tuple_text(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    result = tuple(
        _require_text(value, f"{field_name}[{index}]") for index, value in enumerate(values)
    )
    return result


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ApiError:
    """Stable error envelope returned by internal and public API adapters."""

    code: str
    message: str
    retryable: bool
    request_id: str

    def __post_init__(self) -> None:
        _require_id(self.code, "code")
        _require_text(self.message, "message")
        _require_id(self.request_id, "request_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error envelope to JSON-compatible data."""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ApiError:
        """Build an error envelope from JSON-compatible data."""
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ApiEnvelope:
    """Versioned response envelope with a request ID on success or failure."""

    request_id: str
    status: Literal["ok", "error"]
    data: Mapping[str, Any] | None = None
    error: ApiError | None = None

    def __post_init__(self) -> None:
        _require_id(self.request_id, "request_id")
        _one_of(self.status, {"ok", "error"}, "status")
        if self.status == "ok" and self.error is not None:
            raise ContractValidationError("successful API responses cannot contain an error")
        if self.status == "error" and self.error is None:
            raise ContractValidationError("error API responses require an error envelope")
        if self.error is not None and self.error.request_id != self.request_id:
            raise ContractValidationError("error request_id must match the response request_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response envelope without dropping explicit statuses."""
        result: dict[str, Any] = {"request_id": self.request_id, "status": self.status}
        if self.data is not None:
            result["data"] = dict(self.data)
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ApiEnvelope:
        """Build a response envelope from JSON-compatible data."""
        payload = dict(value)
        error = payload.get("error")
        payload["error"] = ApiError.from_dict(error) if isinstance(error, Mapping) else None
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AudioArtifact:
    """A validated registered audio artifact with source provenance and time origin."""

    artifact_id: str
    source_sha256: str
    format: str
    duration_ms: int
    sample_rate_hz: int
    channels: int
    time_origin_ms: int = 0
    transform_hash: str | None = None
    decoder_version: str | None = None
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "artifact_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_text(self.format, "format")
        _require_positive_int(self.duration_ms, "duration_ms")
        _require_positive_int(self.sample_rate_hz, "sample_rate_hz")
        _require_positive_int(self.channels, "channels")
        _require_nonnegative_int(self.time_origin_ms, "time_origin_ms")
        if self.transform_hash is not None:
            _require_sha256(self.transform_hash, "transform_hash")
        if self.decoder_version is not None:
            _require_text(self.decoder_version, "decoder_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact while preserving media provenance."""
        return {
            "artifact_id": self.artifact_id,
            "source_sha256": self.source_sha256,
            "format": self.format,
            "duration_ms": self.duration_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "time_origin_ms": self.time_origin_ms,
            "transform_hash": self.transform_hash,
            "decoder_version": self.decoder_version,
            "preprocessing": dict(self.preprocessing),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AudioArtifact:
        """Build an audio artifact from JSON-compatible data."""
        return cls(**dict(value))

    @property
    def audio_format(self) -> str:
        """Return the source format using the descriptive compatibility alias."""
        return self.format


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """Immutable model identity and preprocessing/runtime evidence."""

    name: str
    repository: str
    revision: str
    checkpoint_sha256: tuple[str, ...]
    dimensions: int | None = None
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    precision: str | None = None
    runtime: Mapping[str, str] = field(default_factory=dict)
    terms_evidence: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.name, "name")
        _require_text(self.repository, "repository")
        _require_text(self.revision, "revision")
        if not self.checkpoint_sha256:
            raise ContractValidationError("checkpoint_sha256 must contain at least one hash")
        for index, digest in enumerate(self.checkpoint_sha256):
            _require_sha256(digest, f"checkpoint_sha256[{index}]")
        if self.dimensions is not None:
            _require_positive_int(self.dimensions, "dimensions")
        if self.precision is not None:
            _require_text(self.precision, "precision")
        if self.terms_evidence is not None:
            _require_text(self.terms_evidence, "terms_evidence")

    @property
    def fingerprint_sha256(self) -> str:
        """Return a deterministic hash of the complete model fingerprint."""
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Serialize model identity without recording wall-clock state."""
        return {
            "name": self.name,
            "repository": self.repository,
            "revision": self.revision,
            "checkpoint_sha256": list(self.checkpoint_sha256),
            "dimensions": self.dimensions,
            "preprocessing": dict(self.preprocessing),
            "precision": self.precision,
            "runtime": dict(self.runtime),
            "terms_evidence": self.terms_evidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelFingerprint:
        """Build a model fingerprint from JSON-compatible data."""
        payload = dict(value)
        payload["checkpoint_sha256"] = tuple(payload["checkpoint_sha256"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class CapabilityDeclaration:
    """Capabilities advertised by an adapter alongside its model fingerprint."""

    model: ModelFingerprint
    supports_timestamps: bool = False
    timestamp_granularities: tuple[TimestampGranularity, ...] = ()
    max_audio_ms: int | None = None
    max_payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if any(
            granularity not in {"word", "segment"} for granularity in self.timestamp_granularities
        ):
            raise ContractValidationError("timestamp granularities must be word or segment")
        if self.supports_timestamps and not self.timestamp_granularities:
            raise ContractValidationError(
                "timestamp granularities are required when timestamps are supported"
            )
        if not self.supports_timestamps and self.timestamp_granularities:
            raise ContractValidationError("timestamp granularities require timestamp support")
        if self.max_audio_ms is not None:
            _require_positive_int(self.max_audio_ms, "max_audio_ms")
        if self.max_payload_bytes is not None:
            _require_positive_int(self.max_payload_bytes, "max_payload_bytes")

    @property
    def fingerprint(self) -> ModelFingerprint:
        """Return the advertised model fingerprint."""
        return self.model

    def to_dict(self) -> dict[str, Any]:
        """Serialize capabilities and their model provenance."""
        return {
            "model": self.model.to_dict(),
            "supports_timestamps": self.supports_timestamps,
            "timestamp_granularities": list(self.timestamp_granularities),
            "max_audio_ms": self.max_audio_ms,
            "max_payload_bytes": self.max_payload_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityDeclaration:
        """Build capabilities from JSON-compatible data."""
        payload = dict(value)
        payload["model"] = ModelFingerprint.from_dict(payload["model"])
        payload["timestamp_granularities"] = tuple(payload.get("timestamp_granularities", ()))
        return cls(**payload)


AdapterCapabilities = CapabilityDeclaration


@dataclass(frozen=True, slots=True)
class TimedWord:
    """A word with real source-relative integer millisecond timing."""

    word_id: str
    text: str
    start_ms: int
    end_ms: int
    confidence: float | None = None
    speaker_id: str | None = None
    overlap: bool = False

    def __post_init__(self) -> None:
        _require_id(self.word_id, "word_id")
        _require_text(self.text, "text")
        _interval(self.start_ms, self.end_ms, "word")
        _confidence(self.confidence, "confidence")
        if self.speaker_id is not None:
            _require_id(self.speaker_id, "speaker_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize a timed word with optional actual confidence and attribution."""
        return {
            "word_id": self.word_id,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "overlap": self.overlap,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TimedWord:
        """Build a timed word from JSON-compatible data."""
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Turn:
    """A diarization turn scoped to one episode-speaker artifact."""

    speaker_id: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_id(self.speaker_id, "speaker_id")
        _interval(self.start_ms, self.end_ms, "turn")
        _confidence(self.confidence, "confidence")

    def to_dict(self) -> dict[str, Any]:
        """Serialize a diarization turn."""
        return {
            "speaker_id": self.speaker_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Turn:
        """Build a turn from JSON-compatible data."""
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class Overlap:
    """A standard-diarization interval with all simultaneously active speakers."""

    speaker_ids: tuple[str, ...]
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if len(self.speaker_ids) < 2:
            raise ContractValidationError("overlap requires at least two speakers")
        for index, speaker_id in enumerate(self.speaker_ids):
            _require_id(speaker_id, f"speaker_ids[{index}]")
        if len(set(self.speaker_ids)) != len(self.speaker_ids):
            raise ContractValidationError("overlap speaker IDs must be unique")
        _interval(self.start_ms, self.end_ms, "overlap")

    def to_dict(self) -> dict[str, Any]:
        """Serialize an overlap interval without discarding active speakers."""
        return {
            "speaker_ids": list(self.speaker_ids),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Overlap:
        """Build an overlap interval from JSON-compatible data."""
        payload = dict(value)
        payload["speaker_ids"] = tuple(payload["speaker_ids"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class SpeakerStream:
    """One speaker's interval stream inside a benchmark chunk."""

    speaker_id: str
    turns: tuple[Turn, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.speaker_id, "speaker_id")
        previous_start = -1
        for turn in self.turns:
            if turn.speaker_id != self.speaker_id:
                raise ContractValidationError("speaker stream turn IDs must match speaker_id")
            if turn.start_ms < previous_start:
                raise ContractValidationError("speaker stream turns must be monotonic")
            previous_start = turn.start_ms

    def to_dict(self) -> dict[str, Any]:
        """Serialize interval evidence without introducing word timing."""

        return {
            "speaker_id": self.speaker_id,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpeakerStream:
        """Build a speaker stream from JSON-compatible data."""

        payload = dict(value)
        payload["turns"] = tuple(Turn.from_dict(turn) for turn in payload.get("turns", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ChunkCondition:
    """Condition metadata for turn taking, overlap and acoustic slices."""

    speaker_streams: tuple[SpeakerStream, ...] = ()
    overlaps: tuple[Overlap, ...] = ()
    acoustic_labels: tuple[str, ...] = ()
    rapid_turn_taking: bool = False
    music: bool = False
    degraded: bool = False

    def __post_init__(self) -> None:
        speaker_ids = tuple(stream.speaker_id for stream in self.speaker_streams)
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ContractValidationError("chunk condition speaker IDs must be unique")
        _tuple_text(self.acoustic_labels, "acoustic_labels")
        if len(set(self.acoustic_labels)) != len(self.acoustic_labels):
            raise ContractValidationError("chunk condition acoustic labels must be unique")
        for overlap in self.overlaps:
            unknown = set(overlap.speaker_ids) - set(speaker_ids)
            if unknown:
                raise ContractValidationError(
                    "chunk condition overlap references an unknown speaker stream"
                )

    def validate_bounds(self, start_ms: int, end_ms: int) -> None:
        """Require all condition intervals to remain inside the chunk."""

        for stream in self.speaker_streams:
            for turn in stream.turns:
                _interval(turn.start_ms, turn.end_ms, "speaker stream turn")
                if turn.start_ms < start_ms or turn.end_ms > end_ms:
                    raise ContractValidationError("speaker stream turn exceeds chunk interval")
        for overlap in self.overlaps:
            if overlap.start_ms < start_ms or overlap.end_ms > end_ms:
                raise ContractValidationError("overlap interval exceeds chunk interval")

    @property
    def speaker_count(self) -> int:
        """Return the number of distinct speaker streams in the condition."""

        return len(self.speaker_streams)

    @property
    def has_overlap(self) -> bool:
        """Return whether genuine simultaneous speaker activity is recorded."""

        return bool(self.overlaps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize condition metadata, including all active overlap speakers."""

        return {
            "speaker_streams": [stream.to_dict() for stream in self.speaker_streams],
            "overlaps": [overlap.to_dict() for overlap in self.overlaps],
            "acoustic_labels": list(self.acoustic_labels),
            "rapid_turn_taking": self.rapid_turn_taking,
            "music": self.music,
            "degraded": self.degraded,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkCondition:
        """Build condition metadata from JSON-compatible data."""

        payload = dict(value)
        payload["speaker_streams"] = tuple(
            SpeakerStream.from_dict(stream) for stream in payload.get("speaker_streams", ())
        )
        payload["overlaps"] = tuple(
            Overlap.from_dict(overlap) for overlap in payload.get("overlaps", ())
        )
        payload["acoustic_labels"] = tuple(payload.get("acoustic_labels", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A deterministic, source-relative half-open benchmark interval."""

    chunk_id: str
    episode_id: str
    source_sha256: str
    start_ms: int
    end_ms: int
    segmentation_fingerprint: str
    duration_ms: int | None = None
    condition: ChunkCondition | None = None
    partition: ChunkPartition | None = None

    def __post_init__(self) -> None:
        _require_id(self.chunk_id, "chunk_id")
        _require_id(self.episode_id, "episode_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.segmentation_fingerprint, "segmentation_fingerprint")
        if self.duration_ms is not None:
            _require_positive_int(self.duration_ms, "duration_ms")
        _interval(self.start_ms, self.end_ms, "chunk", self.duration_ms)
        if self.partition is not None:
            _one_of(self.partition, {"development", "held_out"}, "partition")
        if self.condition is not None:
            self.condition.validate_bounds(self.start_ms, self.end_ms)

    @staticmethod
    def deterministic_id(
        episode_id: str,
        source_sha256: str,
        start_ms: int,
        end_ms: int,
        segmentation_fingerprint: str,
    ) -> str:
        """Return the stable ID for one source interval and segmentation config."""

        payload = {
            "episode_id": episode_id,
            "source_sha256": source_sha256,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "segmentation_fingerprint": segmentation_fingerprint,
        }
        return "chunk-" + _canonical_hash(payload)

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        source_sha256: str,
        start_ms: int,
        end_ms: int,
        segmentation_fingerprint: str,
        duration_ms: int | None = None,
        condition: ChunkCondition | None = None,
        partition: ChunkPartition | None = None,
    ) -> AudioChunk:
        """Create a chunk with its deterministic ID derived from immutable inputs."""

        return cls(
            chunk_id=cls.deterministic_id(
                episode_id, source_sha256, start_ms, end_ms, segmentation_fingerprint
            ),
            episode_id=episode_id,
            source_sha256=source_sha256,
            start_ms=start_ms,
            end_ms=end_ms,
            segmentation_fingerprint=segmentation_fingerprint,
            duration_ms=duration_ms,
            condition=condition,
            partition=partition,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk identity, interval and optional condition metadata."""

        return {
            "chunk_id": self.chunk_id,
            "episode_id": self.episode_id,
            "source_sha256": self.source_sha256,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segmentation_fingerprint": self.segmentation_fingerprint,
            "duration_ms": self.duration_ms,
            "condition": self.condition.to_dict() if self.condition is not None else None,
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AudioChunk:
        """Build a chunk from JSON-compatible data."""

        payload = dict(value)
        condition = payload.get("condition")
        payload["condition"] = (
            ChunkCondition.from_dict(condition) if isinstance(condition, Mapping) else None
        )
        return cls(**payload)


def _model_fingerprint_sha256(
    model_fingerprint: ModelFingerprint | str | None,
    explicit_sha256: str | None,
) -> str:
    """Normalize a full model record or its immutable fingerprint hash."""

    if isinstance(model_fingerprint, ModelFingerprint):
        derived = model_fingerprint.fingerprint_sha256
    elif isinstance(model_fingerprint, str):
        derived = _require_sha256(model_fingerprint, "model_fingerprint")
    elif model_fingerprint is None:
        derived = None
    else:
        raise ContractValidationError("model_fingerprint must be a ModelFingerprint or SHA-256")
    if explicit_sha256 is not None:
        explicit_sha256 = _require_sha256(explicit_sha256, "model_fingerprint_sha256")
        if derived is not None and derived != explicit_sha256:
            raise ContractValidationError("model fingerprint hash does not match model metadata")
        derived = explicit_sha256
    if derived is None:
        raise ContractValidationError("model_fingerprint is required")
    return derived


@dataclass(frozen=True, slots=True)
class TranscriptionHypothesis:
    """An immutable model candidate for one chunk; timing metadata is optional."""

    chunk_id: str
    model_fingerprint: ModelFingerprint | str | None = None
    text: str = ""
    words: tuple[TimedWord, ...] = ()
    segments: tuple[Mapping[str, Any], ...] = ()
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp_granularities: tuple[TimestampGranularity, ...] | None = None
    source_sha256: str | None = None
    artifact_id: str | None = None
    hypothesis_id: str | None = None
    model_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.chunk_id, "chunk_id")
        if not isinstance(self.text, str):
            raise ContractValidationError("hypothesis text must be text")
        object.__setattr__(
            self,
            "model_fingerprint_sha256",
            _model_fingerprint_sha256(self.model_fingerprint, self.model_fingerprint_sha256),
        )
        if self.source_sha256 is not None:
            _require_sha256(self.source_sha256, "source_sha256")
        if self.artifact_id is not None:
            _require_id(self.artifact_id, "artifact_id")
        word_ids = [word.word_id for word in self.words]
        if len(set(word_ids)) != len(word_ids):
            raise ContractValidationError("hypothesis word IDs must be unique")
        for previous, current in zip(self.words, self.words[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ContractValidationError("hypothesis words must be monotonic")
        for segment in self.segments:
            if not isinstance(segment, Mapping):
                raise ContractValidationError("hypothesis segments must be JSON objects")
            try:
                json.dumps(segment, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError(
                    "hypothesis segment is not JSON serializable"
                ) from exc
        try:
            json.dumps(self.raw_metadata, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("hypothesis metadata is not JSON serializable") from exc
        granularities = self.timestamp_granularities
        if granularities is None:
            granularities = tuple(
                granularity
                for granularity, present in (
                    ("word", bool(self.words)),
                    ("segment", bool(self.segments)),
                )
                if present
            )
            object.__setattr__(self, "timestamp_granularities", granularities)
        if len(set(granularities)) != len(granularities) or any(
            value not in {"word", "segment"} for value in granularities
        ):
            raise ContractValidationError("hypothesis timestamp granularities are invalid")
        if self.words and "word" not in granularities:
            raise ContractValidationError("word artifacts require word timestamp granularity")
        if self.segments and "segment" not in granularities:
            raise ContractValidationError("segment artifacts require segment timestamp granularity")
        if self.hypothesis_id is None:
            object.__setattr__(self, "hypothesis_id", self.deterministic_id(self))
        else:
            _require_id(self.hypothesis_id, "hypothesis_id")

    @property
    def model_fingerprint_hash(self) -> str:
        """Return the stable model hash regardless of the input representation."""

        return _model_fingerprint_sha256(self.model_fingerprint, self.model_fingerprint_sha256)

    @staticmethod
    def deterministic_id(hypothesis: TranscriptionHypothesis) -> str:
        """Return the stable identity of a candidate payload."""

        payload = {
            "chunk_id": hypothesis.chunk_id,
            "model_fingerprint_sha256": hypothesis.model_fingerprint_hash,
            "text": hypothesis.text,
            "words": [word.to_dict() for word in hypothesis.words],
            "segments": [dict(segment) for segment in hypothesis.segments],
            "raw_metadata": dict(hypothesis.raw_metadata),
        }
        return "hypothesis-" + _canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize text and optional immutable timing/model artifacts."""

        model = (
            self.model_fingerprint.to_dict()
            if isinstance(self.model_fingerprint, ModelFingerprint)
            else self.model_fingerprint
        )
        return {
            "hypothesis_id": self.hypothesis_id,
            "chunk_id": self.chunk_id,
            "model_fingerprint": model,
            "model_fingerprint_sha256": self.model_fingerprint_hash,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "segments": [dict(segment) for segment in self.segments],
            "raw_metadata": dict(self.raw_metadata),
            "timestamp_granularities": list(self.timestamp_granularities),
            "source_sha256": self.source_sha256,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranscriptionHypothesis:
        """Build a hypothesis from JSON-compatible data."""

        payload = dict(value)
        model = payload.get("model_fingerprint") or payload.get("model")
        payload["model_fingerprint"] = (
            ModelFingerprint.from_dict(model) if isinstance(model, Mapping) else model
        )
        payload["words"] = tuple(TimedWord.from_dict(word) for word in payload.get("words", ()))
        payload["segments"] = tuple(payload.get("segments", ()))
        payload["timestamp_granularities"] = tuple(payload.get("timestamp_granularities", ()))
        payload.setdefault("raw_metadata", {})
        payload.pop("model", None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TranscriptReference:
    """The current immutable view of one chunk's append-only human reference."""

    reference_id: str
    chunk_id: str
    source_sha256: str | None = None
    text: str = ""
    speaker_streams: tuple[SpeakerStream, ...] = ()
    review_status: ReferenceReviewStatus = "draft"
    revision: int = 1
    reviewer: str | None = None
    created_at: str | None = None
    source_hash: str | None = None
    status: ReferenceReviewStatus | None = None

    def __post_init__(self) -> None:
        _require_id(self.reference_id, "reference_id")
        _require_id(self.chunk_id, "chunk_id")
        source_sha256 = self.source_sha256 or self.source_hash
        if source_sha256 is None:
            raise ContractValidationError("reference source_sha256 is required")
        _require_sha256(source_sha256, "source_sha256")
        if self.source_sha256 is not None and self.source_hash is not None:
            if self.source_sha256 != self.source_hash:
                raise ContractValidationError("reference source hashes do not match")
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "source_hash", source_sha256)
        if self.status is not None:
            _one_of(self.status, {"draft", "human_truth", "superseded", "rejected"}, "status")
            if self.review_status != "draft" and self.review_status != self.status:
                raise ContractValidationError("reference review statuses do not match")
            object.__setattr__(self, "review_status", self.status)
        _one_of(
            self.review_status,
            {"draft", "human_truth", "superseded", "rejected"},
            "review_status",
        )
        object.__setattr__(self, "status", self.review_status)
        _require_positive_int(self.revision, "revision")
        if self.reviewer is not None:
            _require_text(self.reviewer, "reviewer")
        if self.review_status == "human_truth" and self.reviewer is None:
            raise ContractValidationError("human truth references require a reviewer")
        if self.created_at is not None:
            _require_text(self.created_at, "created_at")
        speaker_ids = [stream.speaker_id for stream in self.speaker_streams]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ContractValidationError("reference speaker IDs must be unique")

    @property
    def source_hash_value(self) -> str:
        """Compatibility alias for callers that call the checksum a source hash."""

        return self.source_sha256  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reference without flattening speaker overlap streams."""

        return {
            "reference_id": self.reference_id,
            "chunk_id": self.chunk_id,
            "source_sha256": self.source_sha256,
            "text": self.text,
            "speaker_streams": [stream.to_dict() for stream in self.speaker_streams],
            "review_status": self.review_status,
            "revision": self.revision,
            "reviewer": self.reviewer,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranscriptReference:
        """Build a reference from JSON-compatible data."""

        payload = dict(value)
        payload["speaker_streams"] = tuple(
            SpeakerStream.from_dict(stream) for stream in payload.get("speaker_streams", ())
        )
        if "source_sha256" not in payload and "source_hash" in payload:
            payload["source_sha256"] = payload["source_hash"]
        if "review_status" not in payload and "status" in payload:
            payload["review_status"] = payload["status"]
        payload.pop("status", None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ReferenceRevision:
    """One append-only revision retaining provenance and review history."""

    revision_id: str
    reference_id: str
    chunk_id: str
    revision: int
    reviewer: str
    source_sha256: str | None = None
    prior_revision_id: str | None = None
    review_status: ReferenceReviewStatus = "draft"
    text: str = ""
    speaker_streams: tuple[SpeakerStream, ...] = ()
    created_at: str | None = None
    source_hash: str | None = None
    prior_revision: str | None = None
    status: ReferenceReviewStatus | None = None

    def __post_init__(self) -> None:
        _require_id(self.revision_id, "revision_id")
        _require_id(self.reference_id, "reference_id")
        _require_id(self.chunk_id, "chunk_id")
        _require_positive_int(self.revision, "revision")
        _require_text(self.reviewer, "reviewer")
        source_sha256 = self.source_sha256 or self.source_hash
        if source_sha256 is None:
            raise ContractValidationError("reference revision source_sha256 is required")
        _require_sha256(source_sha256, "source_sha256")
        if self.source_sha256 is not None and self.source_hash is not None:
            if self.source_sha256 != self.source_hash:
                raise ContractValidationError("reference revision source hashes do not match")
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "source_hash", source_sha256)
        prior = self.prior_revision_id or self.prior_revision
        if self.prior_revision_id is not None and self.prior_revision is not None:
            if self.prior_revision_id != self.prior_revision:
                raise ContractValidationError("reference revision predecessors do not match")
        if prior == self.revision_id:
            raise ContractValidationError("reference revision cannot point to itself")
        object.__setattr__(self, "prior_revision_id", prior)
        object.__setattr__(self, "prior_revision", prior)
        if self.status is not None:
            _one_of(self.status, {"draft", "human_truth", "superseded", "rejected"}, "status")
            if self.review_status != "draft" and self.review_status != self.status:
                raise ContractValidationError("reference revision review statuses do not match")
            object.__setattr__(self, "review_status", self.status)
        _one_of(
            self.review_status,
            {"draft", "human_truth", "superseded", "rejected"},
            "review_status",
        )
        object.__setattr__(self, "status", self.review_status)
        if self.created_at is not None:
            _require_text(self.created_at, "created_at")

    @property
    def source_hash_value(self) -> str:
        """Compatibility alias for source_sha256."""

        return self.source_sha256  # type: ignore[return-value]

    @property
    def prior_revision_value(self) -> str | None:
        """Compatibility alias for prior_revision_id."""

        return self.prior_revision_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize one immutable review revision."""

        return {
            "revision_id": self.revision_id,
            "reference_id": self.reference_id,
            "chunk_id": self.chunk_id,
            "revision": self.revision,
            "reviewer": self.reviewer,
            "source_sha256": self.source_sha256,
            "prior_revision_id": self.prior_revision_id,
            "review_status": self.review_status,
            "text": self.text,
            "speaker_streams": [stream.to_dict() for stream in self.speaker_streams],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceRevision:
        """Build a reference revision from JSON-compatible data."""

        payload = dict(value)
        payload["speaker_streams"] = tuple(
            SpeakerStream.from_dict(stream) for stream in payload.get("speaker_streams", ())
        )
        if "source_sha256" not in payload and "source_hash" in payload:
            payload["source_sha256"] = payload["source_hash"]
        if "prior_revision_id" not in payload and "prior_revision" in payload:
            payload["prior_revision_id"] = payload["prior_revision"]
        if "review_status" not in payload and "status" in payload:
            payload["review_status"] = payload["status"]
        payload.pop("status", None)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ChunkBenchmarkManifest:
    """Versioned manifest tying chunks, references and hypotheses together."""

    manifest_id: str
    schema_version: int = 1
    chunks: tuple[AudioChunk, ...] = ()
    source_manifest_sha256: str | None = None
    references: tuple[TranscriptReference, ...] = ()
    hypotheses: tuple[TranscriptionHypothesis, ...] = ()
    partitions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source_hashes: tuple[str, ...] = ()
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.manifest_id, "manifest_id")
        _require_positive_int(self.schema_version, "schema_version")
        if not self.chunks:
            raise ContractValidationError("benchmark manifest requires at least one chunk")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ContractValidationError("benchmark manifest chunk IDs must be unique")
        sources = tuple(sorted({chunk.source_sha256 for chunk in self.chunks}))
        if self.source_hashes:
            normalized_sources = tuple(self.source_hashes)
            for source_sha256 in normalized_sources:
                _require_sha256(source_sha256, "source_hashes")
            if set(normalized_sources) != set(sources):
                raise ContractValidationError("manifest source hashes do not match chunks")
        object.__setattr__(self, "source_hashes", sources)
        manifest_hash = self.source_manifest_sha256 or self.source_sha256
        if manifest_hash is not None:
            _require_sha256(manifest_hash, "source_manifest_sha256")
            if self.source_manifest_sha256 is not None and self.source_sha256 is not None:
                if self.source_manifest_sha256 != self.source_sha256:
                    raise ContractValidationError("manifest source hashes do not match")
        object.__setattr__(self, "source_manifest_sha256", manifest_hash)
        object.__setattr__(self, "source_sha256", manifest_hash)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        for reference in self.references:
            chunk = chunk_by_id.get(reference.chunk_id)
            if chunk is None:
                raise ContractValidationError("reference points to an unknown chunk")
            if reference.source_sha256 != chunk.source_sha256:
                raise ContractValidationError("reference source hash does not match its chunk")
        for hypothesis in self.hypotheses:
            if hypothesis.chunk_id not in chunk_by_id:
                raise ContractValidationError("hypothesis points to an unknown chunk")
        seen_partition_ids: set[str] = set()
        for partition, ids in self.partitions.items():
            _one_of(partition, {"development", "held_out"}, "partition")
            for chunk_id in ids:
                if chunk_id not in chunk_by_id:
                    raise ContractValidationError("partition points to an unknown chunk")
                if chunk_id in seen_partition_ids:
                    raise ContractValidationError("chunk occurs in multiple partitions")
                seen_partition_ids.add(chunk_id)

    @property
    def content_sha256(self) -> str:
        """Return a deterministic hash of the versioned manifest contents."""

        payload = self.to_dict()
        payload.pop("manifest_id", None)
        return _canonical_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete benchmark manifest and its provenance."""

        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_hashes": list(self.source_hashes),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "references": [reference.to_dict() for reference in self.references],
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "partitions": {key: list(value) for key, value in self.partitions.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChunkBenchmarkManifest:
        """Build a benchmark manifest from JSON-compatible data."""

        payload = dict(value)
        payload["chunks"] = tuple(
            AudioChunk.from_dict(chunk) for chunk in payload.get("chunks", ())
        )
        payload["references"] = tuple(
            TranscriptReference.from_dict(reference) for reference in payload.get("references", ())
        )
        payload["hypotheses"] = tuple(
            TranscriptionHypothesis.from_dict(hypothesis)
            for hypothesis in payload.get("hypotheses", ())
        )
        payload["partitions"] = {
            key: tuple(value) for key, value in payload.get("partitions", {}).items()
        }
        payload["source_hashes"] = tuple(payload.get("source_hashes", ()))
        if "source_manifest_sha256" not in payload and "source_sha256" in payload:
            payload["source_manifest_sha256"] = payload["source_sha256"]
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """ASR output whose status makes timestamp capability explicit."""

    artifact_id: str
    source_sha256: str
    model: ModelFingerprint
    text: str
    words: tuple[TimedWord, ...] = ()
    segments: tuple[Mapping[str, Any], ...] = ()
    status: ContractStatus = "timed"
    timestamp_granularities: tuple[TimestampGranularity, ...] | None = None
    request_id: str | None = None
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "artifact_id")
        _require_sha256(self.source_sha256, "source_sha256")
        if self.duration_ms is not None:
            _require_positive_int(self.duration_ms, "duration_ms")
        _one_of(self.status, {"timed", "text_only", "no_words", "missing_asr"}, "status")
        if not isinstance(self.text, str):
            raise ContractValidationError("text must be text, including an empty no-words result")
        granularities = self.timestamp_granularities
        if granularities is None:
            granularities = ("word",) if self.status == "timed" else ()
            object.__setattr__(self, "timestamp_granularities", granularities)
        if any(granularity not in {"word", "segment"} for granularity in granularities):
            raise ContractValidationError("timestamp granularities must be word or segment")
        if self.status == "timed" and "word" not in granularities:
            raise ContractValidationError("timed transcript results require word timestamps")
        if self.status != "timed" and self.words:
            raise ContractValidationError(
                "only timed transcript results may contain production words"
            )
        if self.status == "text_only" and granularities:
            raise ContractValidationError(
                "text-only transcript results cannot advertise timestamps"
            )
        if self.status in {"missing_asr", "no_words"} and self.text:
            raise ContractValidationError(f"{self.status} transcript results cannot contain text")
        if self.request_id is not None:
            _require_id(self.request_id, "request_id")
        word_ids = [word.word_id for word in self.words]
        if len(set(word_ids)) != len(word_ids):
            raise ContractValidationError("transcript word IDs must be unique within an artifact")
        for previous, current in zip(self.words, self.words[1:], strict=False):
            if current.start_ms < previous.start_ms:
                raise ContractValidationError("transcript words must be monotonic")
        if self.duration_ms is not None:
            for word in self.words:
                _interval(word.start_ms, word.end_ms, "word", self.duration_ms)

    @property
    def production_usable(self) -> bool:
        """Return whether the result can be used for timed attribution."""
        return self.status == "timed" and bool(self.words)

    def require_production(self) -> TranscriptResult:
        """Raise when this result lacks genuine word timing for production use."""
        if not self.production_usable:
            raise ContractValidationError(
                "production attribution requires a timed transcript with genuine word timestamps"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize ASR output while retaining explicit missing/timing status."""
        return {
            "artifact_id": self.artifact_id,
            "source_sha256": self.source_sha256,
            "model": self.model.to_dict(),
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "segments": [dict(segment) for segment in self.segments],
            "status": self.status,
            "timestamp_granularities": list(self.timestamp_granularities),
            "request_id": self.request_id,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranscriptResult:
        """Build a transcript result from JSON-compatible data."""
        payload = dict(value)
        payload["model"] = ModelFingerprint.from_dict(payload["model"])
        payload["words"] = tuple(TimedWord.from_dict(word) for word in payload.get("words", ()))
        payload["segments"] = tuple(payload.get("segments", ()))
        payload["timestamp_granularities"] = tuple(payload.get("timestamp_granularities", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    """Standard and exclusive diarization output with explicit overlap intervals."""

    artifact_id: str
    source_sha256: str
    duration_ms: int
    model: ModelFingerprint
    standard_turns: tuple[Turn, ...]
    exclusive_turns: tuple[Turn, ...]
    overlaps: tuple[Overlap, ...]
    request_id: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "artifact_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_positive_int(self.duration_ms, "duration_ms")
        if self.request_id is not None:
            _require_id(self.request_id, "request_id")
        for turn in (*self.standard_turns, *self.exclusive_turns):
            _interval(turn.start_ms, turn.end_ms, "turn", self.duration_ms)
        for overlap in self.overlaps:
            _interval(overlap.start_ms, overlap.end_ms, "overlap", self.duration_ms)

    def to_dict(self) -> dict[str, Any]:
        """Serialize both diarization views and every active overlap speaker."""
        return {
            "artifact_id": self.artifact_id,
            "source_sha256": self.source_sha256,
            "duration_ms": self.duration_ms,
            "model": self.model.to_dict(),
            "standard_turns": [turn.to_dict() for turn in self.standard_turns],
            "exclusive_turns": [turn.to_dict() for turn in self.exclusive_turns],
            "overlaps": [overlap.to_dict() for overlap in self.overlaps],
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiarizationResult:
        """Build a diarization result from JSON-compatible data."""
        payload = dict(value)
        payload["model"] = ModelFingerprint.from_dict(payload["model"])
        payload["standard_turns"] = tuple(
            Turn.from_dict(turn) for turn in payload["standard_turns"]
        )
        payload["exclusive_turns"] = tuple(
            Turn.from_dict(turn) for turn in payload["exclusive_turns"]
        )
        payload["overlaps"] = tuple(Overlap.from_dict(overlap) for overlap in payload["overlaps"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Ordered finite non-zero vectors tied to one immutable model fingerprint."""

    model: ModelFingerprint
    item_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    request_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.vectors):
            raise ContractValidationError("embedding item/vector count mismatch")
        _tuple_text(self.item_ids, "item_ids")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ContractValidationError("embedding item IDs must be unique and ordered")
        if self.model.dimensions is None:
            raise ContractValidationError("embedding model fingerprint must declare dimensions")
        for index, vector in enumerate(self.vectors):
            if len(vector) != self.model.dimensions:
                raise ContractValidationError(
                    f"embedding vector {index} has dimension {len(vector)}, "
                    f"expected {self.model.dimensions}"
                )
            norm = 0.0
            for component in vector:
                number = _finite_number(component, f"vectors[{index}]")
                norm += number * number
            if norm == 0.0:
                raise ContractValidationError(f"embedding vector {index} must be non-zero")
        if self.request_id is not None:
            _require_id(self.request_id, "request_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize vectors without changing order or normalizing them."""
        return {
            "model": self.model.to_dict(),
            "item_ids": list(self.item_ids),
            "vectors": [list(vector) for vector in self.vectors],
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EmbeddingBatch:
        """Build an embedding batch from JSON-compatible data."""
        payload = dict(value)
        payload["model"] = ModelFingerprint.from_dict(payload["model"])
        payload["item_ids"] = tuple(payload["item_ids"])
        payload["vectors"] = tuple(tuple(vector) for vector in payload["vectors"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Atomic stage-publication metadata with source and upstream hashes."""

    artifact_id: str
    source_sha256: str
    stage: str
    stage_key: str
    upstream_artifact_hashes: tuple[str, ...]
    file_checksums: Mapping[str, str]
    pipeline_version: str
    schema_version: int
    model_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "artifact_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_id(self.stage, "stage")
        _require_text(self.stage_key, "stage_key")
        _require_text(self.pipeline_version, "pipeline_version")
        _require_positive_int(self.schema_version, "schema_version")
        for index, digest in enumerate(self.upstream_artifact_hashes):
            _require_sha256(digest, f"upstream_artifact_hashes[{index}]")
        if not self.file_checksums:
            raise ContractValidationError(
                "file_checksums must contain the completed artifact files"
            )
        for path, digest in self.file_checksums.items():
            _require_text(path, "file_checksums path")
            if path.startswith("/") or ".." in path.split("/"):
                raise ContractValidationError("file checksum paths must remain inside the artifact")
            _require_sha256(digest, f"file_checksums[{path}]")
        if self.model_fingerprint_sha256 is not None:
            _require_sha256(self.model_fingerprint_sha256, "model_fingerprint_sha256")

    def to_dict(self) -> dict[str, Any]:
        """Serialize immutable publication metadata."""
        return {
            "artifact_id": self.artifact_id,
            "source_sha256": self.source_sha256,
            "stage": self.stage,
            "stage_key": self.stage_key,
            "upstream_artifact_hashes": list(self.upstream_artifact_hashes),
            "file_checksums": dict(self.file_checksums),
            "pipeline_version": self.pipeline_version,
            "schema_version": self.schema_version,
            "model_fingerprint_sha256": self.model_fingerprint_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        """Build publication metadata from JSON-compatible data."""
        payload = dict(value)
        payload["upstream_artifact_hashes"] = tuple(payload["upstream_artifact_hashes"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Durable worker state including fencing, lease, retry, and recovery data."""

    job_id: str
    stage: str
    status: JobState
    attempts: int = 0
    fencing_token: int = 0
    owner: str | None = None
    lease_expires_at: str | None = None
    error: ApiError | None = None
    recovery_action: str | None = None
    request_id: str | None = None
    paid: bool = False

    def __post_init__(self) -> None:
        _require_id(self.job_id, "job_id")
        _require_id(self.stage, "stage")
        _one_of(
            self.status,
            {"queued", "running", "retry_wait", "succeeded", "failed", "cancelled", "blocked"},
            "status",
        )
        _require_nonnegative_int(self.attempts, "attempts")
        _require_nonnegative_int(self.fencing_token, "fencing_token")
        if self.owner is not None:
            _require_id(self.owner, "owner")
        if self.request_id is not None:
            _require_id(self.request_id, "request_id")
        if not isinstance(self.paid, bool):
            raise ContractValidationError("paid must be a boolean")
        if self.lease_expires_at is not None:
            _require_text(self.lease_expires_at, "lease_expires_at")
        if self.status in {"failed", "blocked"} and self.error is None:
            raise ContractValidationError(f"{self.status} jobs require a typed error")
        if (
            self.error is not None
            and self.request_id is not None
            and self.error.request_id != self.request_id
        ):
            raise ContractValidationError("job error request_id must match request_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize durable worker state."""
        return {
            "job_id": self.job_id,
            "stage": self.stage,
            "status": self.status,
            "attempts": self.attempts,
            "fencing_token": self.fencing_token,
            "owner": self.owner,
            "lease_expires_at": self.lease_expires_at,
            "error": self.error.to_dict() if self.error else None,
            "recovery_action": self.recovery_action,
            "request_id": self.request_id,
            "paid": self.paid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobStatus:
        """Build worker state from JSON-compatible data."""
        payload = dict(value)
        if isinstance(payload.get("error"), Mapping):
            payload["error"] = ApiError.from_dict(payload["error"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """A candidate ranking with per-model scores and explicit calibration state."""

    candidate_id: str
    episode_speaker_id: str
    rank: int
    score: float
    model_scores: Mapping[str, float] = field(default_factory=dict)
    calibration_status: CalibrationStatus = "uncalibrated_rank_fusion"
    calibration_artifact_sha256: str | None = None
    provenance_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.candidate_id, "candidate_id")
        _require_id(self.episode_speaker_id, "episode_speaker_id")
        _one_of(
            self.calibration_status,
            {"uncalibrated_rank_fusion", "calibrated_estimate", "unavailable"},
            "calibration_status",
        )
        _require_positive_int(self.rank, "rank")
        _finite_number(self.score, "score")
        if not self.model_scores:
            raise ContractValidationError("candidate score requires per-model scores")
        for model_name, model_score in self.model_scores.items():
            _require_id(model_name, "model_scores key")
            _finite_number(model_score, f"model_scores[{model_name}]")
        if self.calibration_status == "calibrated_estimate":
            if self.calibration_artifact_sha256 is None:
                raise ContractValidationError(
                    "calibrated estimates require a calibration artifact hash"
                )
            _require_sha256(self.calibration_artifact_sha256, "calibration_artifact_sha256")
        elif self.calibration_artifact_sha256 is not None:
            _require_sha256(self.calibration_artifact_sha256, "calibration_artifact_sha256")
        for index, artifact_id in enumerate(self.provenance_artifact_ids):
            _require_id(artifact_id, f"provenance_artifact_ids[{index}]")

    @property
    def is_probability(self) -> bool:
        """Return whether the score is explicitly a calibrated estimate."""
        return self.calibration_status == "calibrated_estimate"

    def to_dict(self) -> dict[str, Any]:
        """Serialize ranking, per-model evidence, and calibration label."""
        return {
            "candidate_id": self.candidate_id,
            "episode_speaker_id": self.episode_speaker_id,
            "rank": self.rank,
            "score": self.score,
            "model_scores": dict(self.model_scores),
            "calibration_status": self.calibration_status,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "provenance_artifact_ids": list(self.provenance_artifact_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateScore:
        """Build a candidate score from JSON-compatible data."""
        payload = dict(value)
        payload["provenance_artifact_ids"] = tuple(payload.get("provenance_artifact_ids", ()))
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """Append-only human identity evidence between two episode-speaker records."""

    decision_id: str
    left_episode_speaker_id: str
    right_episode_speaker_id: str
    decision: IdentityAction
    reviewer: str
    evidence_artifact_ids: tuple[str, ...]
    created_at: str
    state: IdentityState = "active"
    revision: int = 1

    def __post_init__(self) -> None:
        _require_id(self.decision_id, "decision_id")
        _require_id(self.left_episode_speaker_id, "left_episode_speaker_id")
        _require_id(self.right_episode_speaker_id, "right_episode_speaker_id")
        _one_of(self.decision, {"same_person", "different_person", "uncertain"}, "decision")
        if self.left_episode_speaker_id == self.right_episode_speaker_id:
            raise ContractValidationError(
                "identity decisions require two distinct episode-speaker IDs"
            )
        _require_text(self.reviewer, "reviewer")
        if not self.evidence_artifact_ids:
            raise ContractValidationError("identity decisions require evidence artifact IDs")
        _tuple_text(self.evidence_artifact_ids, "evidence_artifact_ids")
        _require_text(self.created_at, "created_at")
        _one_of(self.state, {"active", "superseded"}, "state")
        _require_positive_int(self.revision, "revision")

    def to_dict(self) -> dict[str, Any]:
        """Serialize human decision history and evidence references."""
        return {
            "decision_id": self.decision_id,
            "left_episode_speaker_id": self.left_episode_speaker_id,
            "right_episode_speaker_id": self.right_episode_speaker_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "created_at": self.created_at,
            "state": self.state,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IdentityDecision:
        """Build a human decision from JSON-compatible data."""
        payload = dict(value)
        payload["evidence_artifact_ids"] = tuple(payload["evidence_artifact_ids"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate evaluation metadata tied to source, split, model, and code hashes."""

    report_id: str
    source_sha256: str
    split_sha256: str
    model_fingerprint_sha256: tuple[str, ...]
    configuration_sha256: str
    reviewed_commit: str
    verdict: Literal["pass", "blocked", "insufficient_evidence"]
    metrics: Mapping[str, float | int | str]
    limitations: tuple[str, ...] = ()
    generated_at: str | None = None

    def __post_init__(self) -> None:
        _require_id(self.report_id, "report_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.split_sha256, "split_sha256")
        if not self.model_fingerprint_sha256:
            raise ContractValidationError("evaluation reports require model fingerprint hashes")
        for index, digest in enumerate(self.model_fingerprint_sha256):
            _require_sha256(digest, f"model_fingerprint_sha256[{index}]")
        _require_sha256(self.configuration_sha256, "configuration_sha256")
        _require_text(self.reviewed_commit, "reviewed_commit")
        _one_of(self.verdict, {"pass", "blocked", "insufficient_evidence"}, "verdict")
        if self.generated_at is not None:
            _require_text(self.generated_at, "generated_at")
        _tuple_text(self.limitations, "limitations")

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate evidence without private per-item artifacts."""
        return {
            "report_id": self.report_id,
            "source_sha256": self.source_sha256,
            "split_sha256": self.split_sha256,
            "model_fingerprint_sha256": list(self.model_fingerprint_sha256),
            "configuration_sha256": self.configuration_sha256,
            "reviewed_commit": self.reviewed_commit,
            "verdict": self.verdict,
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationReport:
        """Build an evaluation report from JSON-compatible data."""
        payload = dict(value)
        payload["model_fingerprint_sha256"] = tuple(payload["model_fingerprint_sha256"])
        payload["limitations"] = tuple(payload.get("limitations", ()))
        return cls(**payload)


API_V1_ROUTES: Mapping[str, str] = {
    "annotations": "GET /api/v1/annotations/{episode_id}",
    "annotation_save": "POST /api/v1/annotations/{episode_id}",
    "annotation_seed": "POST /api/v1/annotations/{episode_id}/seed",
    "annotation_export": "GET /api/v1/annotations/{episode_id}/exports/{revision}/{filename}",
    "episodes": "GET /api/v1/episodes",
    "episode": "GET /api/v1/episodes/{id}",
    "transcript": "GET /api/v1/episodes/{id}/transcript",
    "media": "GET /api/v1/media/{episode_id}",
    "search": "GET /api/v1/search",
    "voice_search": "POST /api/v1/voice-search/jobs",
    "voice_search_speaker": "POST /api/v1/voice-search/jobs/{id}/speaker",
    "voice_search_cancel": "DELETE /api/v1/voice-search/jobs/{id}",
    "job": "GET /api/v1/jobs/{id}",
    "candidates": "GET /api/v1/voice-search/jobs/{id}/candidates",
    "speakers": "GET /api/v1/speakers",
    "speaker_appearances": "GET /api/v1/speakers/{id}/appearances",
    "speaker_candidates": "GET /api/v1/speakers/{id}/candidates",
    "identity_decision": "POST /api/v1/identity-decisions",
    "identity_undo": "POST /api/v1/identity-decisions/{id}/undo",
    "speaker_split": "POST /api/v1/speakers/{id}/split",
    "evaluation_reports": "GET /api/v1/evaluation/reports",
    "evaluation_report": "GET /api/v1/evaluation/reports/{id}",
}

SERVICE_ROUTES: Mapping[str, str] = {
    "health": "GET /health",
    "ready": "GET /ready",
    "transcribe": "POST /v1/audio/transcriptions",
    "diarize": "POST /v1/diarize",
    "embed_speakers": "POST /v1/embed-speakers",
    "embed_text": "POST /v1/embeddings",
}


class Transcriber(Protocol):
    """Protocol for local or cloud transcription adapters."""

    @property
    def capabilities(self) -> CapabilityDeclaration:
        """Return the model fingerprint and actual timing capabilities."""

    def transcribe(
        self,
        audio: AudioArtifact,
        *,
        request_id: str,
        language: str | None = None,
        timestamp_granularities: Sequence[TimestampGranularity] = (),
    ) -> TranscriptResult:
        """Transcribe registered audio, preserving genuine timing only."""


class Diarizer(Protocol):
    """Protocol for standard and exclusive speaker diarization adapters."""

    @property
    def capabilities(self) -> CapabilityDeclaration:
        """Return model and service capability metadata."""

    def diarize(self, audio: AudioArtifact, *, request_id: str) -> DiarizationResult:
        """Diarize registered audio and retain overlap intervals."""


class SpeakerEmbedder(Protocol):
    """Protocol for ordered speaker-excerpt embedding adapters."""

    @property
    def capabilities(self) -> CapabilityDeclaration:
        """Return model identity and vector dimension."""

    def embed(
        self,
        excerpts: Sequence[AudioArtifact],
        *,
        request_id: str,
    ) -> EmbeddingBatch:
        """Embed excerpts in input order with finite non-zero vectors."""


class TextEmbedder(Protocol):
    """Protocol for ordered text embedding adapters."""

    @property
    def capabilities(self) -> CapabilityDeclaration:
        """Return model identity and dense vector dimension."""

    def embed(self, texts: Sequence[str], *, request_id: str) -> EmbeddingBatch:
        """Embed texts in input order with finite non-zero vectors."""


def error_for_failure(
    code: str,
    message: str,
    *,
    request_id: str,
    retryable: bool,
) -> ApiError:
    """Create a typed API error for adapter failure paths."""
    return ApiError(code=code, message=message, retryable=retryable, request_id=request_id)


def json_round_trip(record: Any) -> Any:
    """Round-trip any contract record through JSON for fixture verification.

    The helper is intentionally small and is used by tests and synthetic tooling;
    callers should use each record's ``from_dict`` method to restore its type.
    """
    payload = record.to_dict()
    return json.loads(json.dumps(payload, ensure_ascii=False))


__all__ = [
    "API_V1_ROUTES",
    "AdapterCapabilities",
    "AdapterFailure",
    "ApiEnvelope",
    "ApiError",
    "ArtifactManifest",
    "AudioArtifact",
    "BudgetBlockedError",
    "CandidateScore",
    "CapabilityDeclaration",
    "ContractValidationError",
    "DiarizationResult",
    "Diarizer",
    "EmbeddingBatch",
    "EvaluationReport",
    "GenerationMismatchError",
    "IdentityDecision",
    "IdentityState",
    "InvalidAudioError",
    "JobStatus",
    "ModelFingerprint",
    "ModelUnavailableError",
    "Overlap",
    "RequestTimeoutError",
    "SERVICE_ROUTES",
    "SpeakerEmbedder",
    "TextEmbedder",
    "TimedWord",
    "TranscriptResult",
    "Transcriber",
    "Turn",
    "UnsupportedCapabilityError",
    "error_for_failure",
    "json_round_trip",
]

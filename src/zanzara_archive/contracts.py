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

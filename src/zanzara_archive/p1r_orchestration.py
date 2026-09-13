"""Durable multi-model dispatch for the P1R chunk benchmark.

The orchestrator owns dispatch identity and reconciliation, while the existing
model adapters own request/response validation.  A dispatch is keyed by the
frozen manifest, chunk and registered adapter; model failures therefore remain
visible without changing the successful peers or substituting another model.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from threading import Event, Thread
from typing import Any, Protocol

from .artifacts import ArtifactPublisher
from .contracts import (
    AdapterFailure,
    ApiError,
    AudioChunk,
    CapabilityDeclaration,
    ChunkBenchmarkManifest,
    ContractValidationError,
    ModelFingerprint,
    TranscriptionHypothesis,
)
from .inference import ParsedHypothesis
from .storage import SQLiteRepository, StorageConflictError

COMMON_CHUNK_PREPROCESSING: Mapping[str, object] = {
    "sample_rate_hz": 16_000,
    "channels": 1,
    "time_origin": "source chunk start",
}
PIPELINE_VERSION = "p1r-07-v1"
DISPATCH_STAGE = "chunk-inference"
DEFAULT_HEARTBEAT_SECONDS = 30.0
DEFAULT_LEASE_SECONDS = 120
DEFAULT_MAX_ATTEMPTS = 3


class ChunkInferenceAdapter(Protocol):
    """The minimal text-first adapter surface required for registration."""

    @property
    def capabilities(self) -> CapabilityDeclaration: ...

    def transcribe_bytes(
        self, chunk: AudioChunk, audio_bytes: bytes, *, request_id: str
    ) -> ParsedHypothesis: ...


class OrchestrationProvenanceError(RuntimeError):
    """Raised when a durable result cannot be reconciled to its dispatch."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"orchestration metadata is not JSON serializable: {exc}"
        ) from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class CommonChunkAudio:
    """One validated, common-preprocessed audio payload for a frozen chunk."""

    source_sha256: str
    start_ms: int
    end_ms: int
    audio_bytes: bytes
    audio_format: str = "wav"
    preprocessing: Mapping[str, object] = field(
        default_factory=lambda: dict(COMMON_CHUNK_PREPROCESSING)
    )

    @classmethod
    def for_chunk(
        cls,
        chunk: AudioChunk,
        audio_bytes: bytes,
        *,
        audio_format: str = "wav",
        preprocessing: Mapping[str, object] = COMMON_CHUNK_PREPROCESSING,
    ) -> CommonChunkAudio:
        return cls(
            source_sha256=chunk.source_sha256,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            preprocessing=preprocessing,
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise ContractValidationError("common audio source_sha256 must be a lowercase SHA-256")
        if (
            isinstance(self.start_ms, bool)
            or not isinstance(self.start_ms, int)
            or self.start_ms < 0
        ):
            raise ContractValidationError("common audio start_ms must be a non-negative integer")
        if (
            isinstance(self.end_ms, bool)
            or not isinstance(self.end_ms, int)
            or self.end_ms <= self.start_ms
        ):
            raise ContractValidationError("common audio end_ms must follow start_ms")
        if not isinstance(self.audio_bytes, bytes) or not self.audio_bytes:
            raise ContractValidationError("common audio must contain bytes")
        if self.audio_format != "wav":
            raise ContractValidationError("P1R common chunk audio must use WAV bytes")
        if not isinstance(self.preprocessing, Mapping):
            raise ContractValidationError("common audio preprocessing must be an object")
        for key, expected in COMMON_CHUNK_PREPROCESSING.items():
            if self.preprocessing.get(key) != expected:
                raise ContractValidationError(
                    f"common audio preprocessing does not match {key}={expected!r}"
                )
        _canonical_json(self.preprocessing)

    @property
    def audio_sha256(self) -> str:
        """Return the immutable identity of the bytes sent to every adapter."""

        return _sha256_bytes(self.audio_bytes)

    @property
    def preprocessing_sha256(self) -> str:
        """Return the deterministic common-preprocessing fingerprint."""

        return _sha256_json(self.preprocessing)


AudioLoader = Callable[[AudioChunk], bytes | CommonChunkAudio]


@dataclass(frozen=True, slots=True)
class AdapterRegistration:
    """A model adapter plus its stable, schema-independent registration key."""

    adapter_id: str
    adapter: ChunkInferenceAdapter
    configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.adapter_id, str)
            or not self.adapter_id
            or not self.adapter_id[0].isalnum()
            or any(
                character
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
                for character in self.adapter_id
            )
        ):
            raise ContractValidationError("adapter_id must be a simple stable identifier")
        if not isinstance(self.configuration, Mapping):
            raise ContractValidationError("adapter configuration must be an object")
        _canonical_json(self.configuration)
        capabilities = self.adapter.capabilities
        if not isinstance(capabilities, CapabilityDeclaration):
            raise ContractValidationError("adapter capabilities must be a CapabilityDeclaration")

    @property
    def capabilities(self) -> CapabilityDeclaration:
        return self.adapter.capabilities

    @property
    def model(self) -> ModelFingerprint:
        return self.capabilities.model


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """One visible model-specific failure in an orchestration report."""

    adapter_id: str
    chunk_id: str
    status: str
    attempts: int
    error: ApiError

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "chunk_id": self.chunk_id,
            "status": self.status,
            "attempts": self.attempts,
            "error": self.error.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrchestrationReport:
    """Reconciled counts and immutable result IDs for one manifest run."""

    manifest_id: str
    manifest_sha256: str
    counts_by_model: Mapping[str, Mapping[str, int]]
    hypothesis_ids: Mapping[str, tuple[str, ...]]
    failures: tuple[DispatchFailure, ...]

    @property
    def total_dispatches(self) -> int:
        return sum(sum(counts.values()) for counts in self.counts_by_model.values())

    @property
    def hypothesis_count(self) -> int:
        return sum(len(hypotheses) for hypotheses in self.hypothesis_ids.values())

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def complete(self) -> bool:
        return not self.failures and all(
            counts.get("succeeded", 0) == sum(counts.values())
            for counts in self.counts_by_model.values()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "counts_by_model": {key: dict(value) for key, value in self.counts_by_model.items()},
            "hypothesis_ids": {key: list(value) for key, value in self.hypothesis_ids.items()},
            "failures": [failure.to_dict() for failure in self.failures],
            "total_dispatches": self.total_dispatches,
            "hypothesis_count": self.hypothesis_count,
            "failure_count": self.failure_count,
            "complete": self.complete,
        }


def _dispatch_identity(
    manifest: ChunkBenchmarkManifest,
    chunk: AudioChunk,
    registration: AdapterRegistration,
) -> str:
    payload = {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.content_sha256,
        "chunk_id": chunk.chunk_id,
        "adapter_id": registration.adapter_id,
        "model_fingerprint_sha256": registration.model.fingerprint_sha256,
        "configuration": dict(registration.configuration),
        "preprocessing": dict(COMMON_CHUNK_PREPROCESSING),
    }
    return "dispatch-" + _sha256_json(payload)


def _as_error(value: object, *, request_id: str) -> ApiError:
    if isinstance(value, AdapterFailure):
        error = value.error
        return ApiError(error.code, error.message, error.retryable, request_id)
    if isinstance(value, TimeoutError):
        return ApiError("timeout", str(value) or "adapter request timed out", True, request_id)
    if isinstance(value, ContractValidationError):
        return ApiError("invalid_audio", str(value), False, request_id)
    return ApiError(
        "adapter_failed",
        str(value) or value.__class__.__name__,
        True,
        request_id,
    )


class ChunkInferenceOrchestrator:
    """Run registered chunk adapters with durable per-model reconciliation."""

    def __init__(
        self,
        repository: SQLiteRepository,
        artifact_publisher: ArtifactPublisher,
        adapters: Iterable[AdapterRegistration] = (),
        *,
        pipeline_version: str = PIPELINE_VERSION,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not pipeline_version:
            raise ValueError("pipeline_version must be non-empty")
        if heartbeat_seconds <= 0 or lease_seconds <= 0 or max_attempts <= 0:
            raise ValueError("worker lease and retry settings must be positive")
        self.repository = repository
        self.artifact_publisher = artifact_publisher
        self.pipeline_version = pipeline_version
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self._adapters: dict[str, AdapterRegistration] = {}
        for registration in adapters:
            self.register_adapter(registration)

    @property
    def adapters(self) -> tuple[AdapterRegistration, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))

    def register_adapter(
        self,
        registration: AdapterRegistration | str,
        adapter: ChunkInferenceAdapter | None = None,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> AdapterRegistration:
        """Register one model without changing the dispatch schema for new models."""

        if isinstance(registration, str):
            if adapter is None:
                raise ValueError("adapter is required when registration is an ID")
            registration = AdapterRegistration(registration, adapter, configuration or {})
        elif adapter is not None:
            raise ValueError("adapter cannot be supplied with an AdapterRegistration")
        if registration.adapter_id in self._adapters:
            raise StorageConflictError(f"adapter is already registered: {registration.adapter_id}")
        if any(
            item.model.fingerprint_sha256 == registration.model.fingerprint_sha256
            for item in self.adapters
        ):
            raise StorageConflictError(
                "one chunk cannot publish two hypotheses for the same model fingerprint"
            )
        self._adapters[registration.adapter_id] = registration
        return registration

    def run(
        self,
        manifest: ChunkBenchmarkManifest,
        audio_loader: AudioLoader,
        *,
        owner: str = "p1r-07",
        now: datetime | str | None = None,
        retry_failed: bool = False,
    ) -> OrchestrationReport:
        """Dispatch every chunk/model pair and reconcile its durable result.

        The loader is called once per chunk.  The returned bytes are passed to
        every adapter unchanged, so all model rows share one auditable audio
        identity.  ``retry_failed`` only retries the same registered adapter;
        it never substitutes a different provider or model.
        """

        if not self._adapters:
            raise ValueError("at least one chunk adapter must be registered")
        if not owner:
            raise ValueError("owner must be non-empty")
        self.repository.record_chunk_benchmark_manifest(manifest)
        registrations = self.adapters
        dispatches = self._ensure_dispatches(manifest, registrations)
        dispatch_by_key = {
            (str(row["chunk_id"]), str(row["adapter_id"])): row for row in dispatches
        }

        for chunk in manifest.chunks:
            chunk_dispatches = [
                dispatch_by_key[(chunk.chunk_id, registration.adapter_id)]
                for registration in registrations
            ]
            if all(str(dispatch["status"]) == "succeeded" for dispatch in chunk_dispatches):
                continue
            common_audio: CommonChunkAudio | None = None
            common_error: ApiError | None = None
            try:
                common_audio = self._load_common_audio(chunk, audio_loader(chunk))
            except Exception as exc:  # preserve the same typed failure for every peer
                common_error = _as_error(exc, request_id=f"audio-{chunk.chunk_id}")
            for registration in registrations:
                dispatch = dispatch_by_key[(chunk.chunk_id, registration.adapter_id)]
                self._run_dispatch(
                    manifest,
                    chunk,
                    registration,
                    dispatch,
                    common_audio,
                    common_error,
                    owner=owner,
                    now=now,
                    retry_failed=retry_failed,
                )

        return self._report(manifest)

    def _ensure_dispatches(
        self,
        manifest: ChunkBenchmarkManifest,
        registrations: tuple[AdapterRegistration, ...],
    ) -> tuple[dict[str, object], ...]:
        for chunk in manifest.chunks:
            for registration in registrations:
                dispatch_id = _dispatch_identity(manifest, chunk, registration)
                request_id = "request-" + dispatch_id.removeprefix("dispatch-")
                job_id = dispatch_id
                configuration = {
                    "manifest_id": manifest.manifest_id,
                    "chunk_id": chunk.chunk_id,
                    "adapter_id": registration.adapter_id,
                    **dict(registration.configuration),
                }
                self.repository.enqueue_job(
                    job_id=job_id,
                    stage=DISPATCH_STAGE,
                    source_sha256=chunk.source_sha256,
                    model_fingerprint_sha256=registration.model.fingerprint_sha256,
                    configuration=configuration,
                    pipeline_version=self.pipeline_version,
                    stage_key=dispatch_id,
                    request_id=request_id,
                )
                self.repository.register_chunk_inference_dispatch(
                    dispatch_id=dispatch_id,
                    manifest_id=manifest.manifest_id,
                    chunk_id=chunk.chunk_id,
                    adapter_id=registration.adapter_id,
                    job_id=job_id,
                    model_fingerprint_sha256=registration.model.fingerprint_sha256,
                    model_payload=registration.model.to_dict(),
                    configuration=configuration,
                    preprocessing=COMMON_CHUNK_PREPROCESSING,
                    preprocessing_sha256=_sha256_json(COMMON_CHUNK_PREPROCESSING),
                    source_sha256=chunk.source_sha256,
                    start_ms=chunk.start_ms,
                    end_ms=chunk.end_ms,
                    request_id=request_id,
                )
        return self.repository.list_chunk_inference_dispatches(manifest.manifest_id)

    @staticmethod
    def _load_common_audio(chunk: AudioChunk, value: bytes | CommonChunkAudio) -> CommonChunkAudio:
        if isinstance(value, CommonChunkAudio):
            audio = value
        elif isinstance(value, bytes):
            audio = CommonChunkAudio.for_chunk(chunk, value)
        else:
            raise ContractValidationError("audio loader must return bytes or CommonChunkAudio")
        if (
            audio.source_sha256 != chunk.source_sha256
            or audio.start_ms != chunk.start_ms
            or audio.end_ms != chunk.end_ms
        ):
            raise ContractValidationError("common audio identity does not match the frozen chunk")
        return audio

    def _run_dispatch(
        self,
        manifest: ChunkBenchmarkManifest,
        chunk: AudioChunk,
        registration: AdapterRegistration,
        dispatch: Mapping[str, object],
        common_audio: CommonChunkAudio | None,
        common_error: ApiError | None,
        *,
        owner: str,
        now: datetime | str | None,
        retry_failed: bool,
    ) -> None:
        dispatch_id = str(dispatch["dispatch_id"])
        job_id = str(dispatch["job_id"])
        current_job = self.repository.fetch_job(job_id)
        if current_job is None:
            raise OrchestrationProvenanceError(f"dispatch job disappeared: {job_id}")
        self._reconcile_existing_hypothesis(manifest, registration, dispatch)
        dispatch = self.repository.fetch_chunk_inference_dispatch(dispatch_id) or dispatch
        current_job = self.repository.fetch_job(job_id) or current_job
        if str(dispatch["status"]) == "succeeded":
            self._require_reconciled_success(dispatch, chunk)
            return
        if current_job.status == "succeeded":
            raise OrchestrationProvenanceError(
                f"job {job_id} succeeded without a reconciled hypothesis"
            )
        if current_job.status in {"failed"} and retry_failed:
            self.repository.retry_job(job_id, now=now)
            current_job = self.repository.fetch_job(job_id)
        if current_job.status in {"failed", "blocked", "cancelled"}:
            return
        try:
            claimed = self.repository.claim_job(
                job_id,
                owner,
                now=now,
                lease_seconds=self.lease_seconds,
                max_attempts=self.max_attempts,
            )
        except StorageConflictError:
            return
        input_audio_sha256 = common_audio.audio_sha256 if common_audio is not None else None
        self.repository.start_chunk_inference_dispatch(
            dispatch_id,
            claimed,
            input_audio_sha256=input_audio_sha256,
            now=now,
        )
        started = time.perf_counter()
        try:
            if common_error is not None:
                raise AdapterFailure(common_error)
            if common_audio is None:
                raise ContractValidationError("common chunk audio is unavailable")
            parsed = self._transcribe_with_heartbeat(
                registration,
                chunk,
                common_audio.audio_bytes,
                request_id=str(dispatch["request_id"]),
                job=claimed,
            )
            hypothesis = self._decorate_hypothesis(
                parsed,
                manifest,
                chunk,
                registration,
                dispatch_id,
                common_audio,
                pipeline_version=self.pipeline_version,
            )
            raw_response = parsed.raw_response
            raw_response_sha256 = _sha256_json(raw_response)
            raw_payload = {
                "dispatch_id": dispatch_id,
                "manifest_id": manifest.manifest_id,
                "manifest_sha256": manifest.content_sha256,
                "adapter_id": registration.adapter_id,
                "model": registration.model.to_dict(),
                "configuration": dict(registration.configuration),
                "request_id": str(dispatch["request_id"]),
                "chunk": chunk.to_dict(),
                "common_audio": {
                    "source_sha256": common_audio.source_sha256,
                    "start_ms": common_audio.start_ms,
                    "end_ms": common_audio.end_ms,
                    "audio_sha256": common_audio.audio_sha256,
                    "audio_format": common_audio.audio_format,
                    "preprocessing": dict(common_audio.preprocessing),
                },
                "raw_response_sha256": raw_response_sha256,
                "response": raw_response,
            }
            raw_bytes = _canonical_json(raw_payload).encode("utf-8")
            artifact_id = f"chunk-inference:{dispatch_id}"
            self.artifact_publisher.publish(
                source_sha256=chunk.source_sha256,
                stage=DISPATCH_STAGE,
                stage_key=dispatch_id,
                files={
                    "raw-response.json": raw_bytes,
                    "input-audio.sha256": common_audio.audio_sha256.encode("ascii"),
                },
                pipeline_version=self.pipeline_version,
                model_fingerprint_sha256=registration.model.fingerprint_sha256,
                artifact_id=artifact_id,
                provenance=raw_payload,
                job=claimed,
                now=now,
            )
            runtime = {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "model_runtime": dict(registration.model.runtime),
            }
            self.repository.publish_chunk_inference_success(
                dispatch_id,
                claimed,
                hypothesis,
                raw_artifact_id=artifact_id,
                input_audio_sha256=common_audio.audio_sha256,
                runtime=runtime,
                now=now,
            )
            self.repository.complete_job(
                job_id,
                owner=owner,
                fencing_token=claimed.fencing_token,
                now=now,
            )
        except (OrchestrationProvenanceError, StorageConflictError):
            raise
        except Exception as exc:
            error = _as_error(exc, request_id=str(dispatch["request_id"]))
            runtime = {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "model_runtime": dict(registration.model.runtime),
            }
            self.repository.record_chunk_inference_failure(
                dispatch_id,
                claimed,
                error,
                input_audio_sha256=input_audio_sha256,
                runtime=runtime,
                now=now,
            )
            failure_class = "transient" if error.retryable else "deterministic"
            self.repository.fail_job(
                job_id,
                owner=owner,
                fencing_token=claimed.fencing_token,
                error=error,
                failure_class=failure_class,
                now=now,
                max_attempts=self.max_attempts,
            )

    def _transcribe_with_heartbeat(
        self,
        registration: AdapterRegistration,
        chunk: AudioChunk,
        audio_bytes: bytes,
        *,
        request_id: str,
        job: Any,
    ) -> ParsedHypothesis:
        stopped = Event()
        lease_lost: list[StorageConflictError] = []
        heartbeat: Thread | None = None
        if getattr(self.repository, "_database_path", None) is not None:

            def renew() -> None:
                while not stopped.wait(self.heartbeat_seconds):
                    try:
                        self.repository.heartbeat(
                            job.job_id,
                            owner=str(job.owner),
                            fencing_token=job.fencing_token,
                            lease_seconds=self.lease_seconds,
                        )
                    except StorageConflictError as exc:
                        lease_lost.append(exc)
                        stopped.set()
                        return

            heartbeat = Thread(target=renew, name=f"p1r-heartbeat-{job.job_id}", daemon=True)
            heartbeat.start()
        try:
            parsed = registration.adapter.transcribe_bytes(
                chunk, audio_bytes, request_id=request_id
            )
        finally:
            stopped.set()
            if heartbeat is not None:
                heartbeat.join()
        if lease_lost:
            raise StorageConflictError("chunk inference worker lease was lost")
        if not isinstance(parsed, ParsedHypothesis):
            raise ContractValidationError("adapter did not return a ParsedHypothesis")
        return parsed

    @staticmethod
    def _decorate_hypothesis(
        parsed: ParsedHypothesis,
        manifest: ChunkBenchmarkManifest,
        chunk: AudioChunk,
        registration: AdapterRegistration,
        dispatch_id: str,
        common_audio: CommonChunkAudio,
        *,
        pipeline_version: str,
    ) -> TranscriptionHypothesis:
        result = parsed.result
        if result.chunk_id != chunk.chunk_id:
            raise OrchestrationProvenanceError("adapter returned a hypothesis for another chunk")
        if result.source_sha256 not in {None, chunk.source_sha256}:
            raise OrchestrationProvenanceError("adapter returned a hypothesis for another source")
        if result.model_fingerprint_hash != registration.model.fingerprint_sha256:
            raise OrchestrationProvenanceError("adapter returned a different model fingerprint")
        try:
            json.dumps(parsed.raw_response, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise OrchestrationProvenanceError(
                f"adapter raw response cannot be retained: {exc}"
            ) from exc
        metadata = dict(result.raw_metadata)
        metadata["orchestration"] = {
            "pipeline_version": pipeline_version,
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.content_sha256,
            "dispatch_id": dispatch_id,
            "adapter_id": registration.adapter_id,
            "model_fingerprint_sha256": registration.model.fingerprint_sha256,
            "raw_response_sha256": _sha256_json(parsed.raw_response),
            "common_audio_sha256": common_audio.audio_sha256,
            "common_preprocessing": dict(common_audio.preprocessing),
            "common_preprocessing_sha256": common_audio.preprocessing_sha256,
            "source_interval": {
                "source_sha256": chunk.source_sha256,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
            },
        }
        return replace(
            result,
            source_sha256=chunk.source_sha256,
            artifact_id=f"chunk-inference:{dispatch_id}",
            hypothesis_id=None,
            raw_metadata=metadata,
        )

    def _reconcile_existing_hypothesis(
        self,
        manifest: ChunkBenchmarkManifest,
        registration: AdapterRegistration,
        dispatch: Mapping[str, object],
    ) -> None:
        if str(dispatch["status"]) == "succeeded":
            return
        hypothesis = self.repository.find_transcription_hypothesis(
            chunk_id=str(dispatch["chunk_id"]),
            model_fingerprint_sha256=registration.model.fingerprint_sha256,
        )
        if hypothesis is None:
            return
        orchestration = hypothesis.raw_metadata.get("orchestration")
        if not isinstance(orchestration, Mapping):
            raise OrchestrationProvenanceError(
                f"existing hypothesis lacks orchestration provenance: {hypothesis.hypothesis_id}"
            )
        if (
            orchestration.get("manifest_id") != manifest.manifest_id
            or orchestration.get("dispatch_id") != dispatch["dispatch_id"]
        ):
            raise OrchestrationProvenanceError(
                f"existing hypothesis provenance does not match dispatch {dispatch['dispatch_id']}"
            )
        self.repository.reconcile_chunk_inference_success(
            str(dispatch["dispatch_id"]),
            hypothesis_id=str(hypothesis.hypothesis_id),
            raw_artifact_id=hypothesis.artifact_id,
            input_audio_sha256=orchestration.get("common_audio_sha256"),
        )

    def _require_reconciled_success(
        self, dispatch: Mapping[str, object], chunk: AudioChunk
    ) -> None:
        hypothesis_id = dispatch.get("hypothesis_id")
        if not isinstance(hypothesis_id, str):
            raise OrchestrationProvenanceError(
                f"successful dispatch has no hypothesis ID: {dispatch['dispatch_id']}"
            )
        hypothesis = self.repository.fetch_transcription_hypothesis(hypothesis_id)
        if hypothesis is None or hypothesis.chunk_id != chunk.chunk_id:
            raise OrchestrationProvenanceError(
                f"successful dispatch hypothesis is missing or mismatched: {hypothesis_id}"
            )

    def _report(self, manifest: ChunkBenchmarkManifest) -> OrchestrationReport:
        rows = self.repository.list_chunk_inference_dispatches(manifest.manifest_id)
        counts: dict[str, dict[str, int]] = {}
        hypothesis_ids: dict[str, list[str]] = {}
        failures: list[DispatchFailure] = []
        for row in rows:
            adapter_id = str(row["adapter_id"])
            status = str(row["status"])
            adapter_counts = counts.setdefault(adapter_id, {})
            adapter_counts[status] = adapter_counts.get(status, 0) + 1
            hypothesis_id = row.get("hypothesis_id")
            if status == "succeeded" and isinstance(hypothesis_id, str):
                hypothesis_ids.setdefault(adapter_id, []).append(hypothesis_id)
            error = row.get("error")
            if isinstance(error, ApiError):
                failures.append(
                    DispatchFailure(
                        adapter_id=adapter_id,
                        chunk_id=str(row["chunk_id"]),
                        status=status,
                        attempts=int(row["attempts"]),
                        error=error,
                    )
                )
        return OrchestrationReport(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest.content_sha256,
            counts_by_model={key: dict(value) for key, value in sorted(counts.items())},
            hypothesis_ids={key: tuple(value) for key, value in sorted(hypothesis_ids.items())},
            failures=tuple(failures),
        )


MultiModelChunkOrchestrator = ChunkInferenceOrchestrator


__all__ = [
    "AdapterRegistration",
    "COMMON_CHUNK_PREPROCESSING",
    "ChunkInferenceAdapter",
    "ChunkInferenceOrchestrator",
    "CommonChunkAudio",
    "DispatchFailure",
    "MultiModelChunkOrchestrator",
    "OrchestrationProvenanceError",
    "OrchestrationReport",
    "PIPELINE_VERSION",
]

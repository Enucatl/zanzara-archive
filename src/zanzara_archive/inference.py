"""HTTP adapters for the repository-owned inference services.

The application process stays free of ML dependencies.  This module only
serializes the constrained internal request and validates the response before
it can become a canonical transcript artifact.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from .contracts import (
    AdapterFailure,
    ApiError,
    AudioArtifact,
    CapabilityDeclaration,
    ContractValidationError,
    ModelFingerprint,
    RequestTimeoutError,
    TimedWord,
    TranscriptResult,
    UnsupportedCapabilityError,
)
from .model_locks import model_fingerprint_from_lock

PARAKEET_TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_AUDIO_PAYLOAD_BYTES = 25_000_000


def _request_id(value: str | None) -> str:
    return value or f"parakeet-{uuid.uuid4().hex}"


def _failure(
    failure_type: type[AdapterFailure],
    code: str,
    message: str,
    request_id: str,
    *,
    retryable: bool = False,
) -> AdapterFailure:
    return failure_type(ApiError(code, message, retryable, request_id))


def _as_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"transcription request is not JSON serializable: {exc}"
        ) from exc


def _seconds_to_ms(value: object, field_name: str) -> int:
    """Convert a service-local decimal second offset to milliseconds exactly once."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ContractValidationError(f"{field_name} must be a finite timestamp in seconds")
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be a finite timestamp in seconds"
        ) from exc
    if not seconds.is_finite() or seconds < 0:
        raise ContractValidationError(f"{field_name} must be a finite non-negative timestamp")
    return int((seconds * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def _word_offsets(item: Mapping[str, Any], index: int) -> tuple[int, int]:
    if "start_ms" in item or "end_ms" in item:
        start = item.get("start_ms")
        end = item.get("end_ms")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ContractValidationError(f"words[{index}] millisecond offsets must be integers")
        return start, end
    if "start" not in item or "end" not in item:
        raise ContractValidationError(f"words[{index}] is missing genuine start/end timestamps")
    return (
        _seconds_to_ms(item["start"], f"words[{index}].start"),
        _seconds_to_ms(item["end"], f"words[{index}].end"),
    )


def _unwrap_response(payload: Mapping[str, Any], request_id: str) -> Mapping[str, Any]:
    """Accept the versioned envelope and the equivalent verbose JSON subset."""

    if payload.get("status") == "error":
        raw_error = payload.get("error")
        if isinstance(raw_error, Mapping):
            try:
                error = ApiError.from_dict(raw_error)
            except (ContractValidationError, TypeError) as exc:
                raise _failure(
                    AdapterFailure,
                    "invalid_response",
                    f"Parakeet service returned an invalid error envelope: {exc}",
                    request_id,
                ) from exc
        else:
            error = ApiError(
                "service_error", "Parakeet service returned an invalid error", False, request_id
            )
        if error.request_id != request_id:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Parakeet service error request_id does not match the request",
                request_id,
            )
        if error.code == "unsupported_capability":
            raise UnsupportedCapabilityError(error)
        raise AdapterFailure(error)
    data = payload.get("data")
    if payload.get("status") == "ok":
        if not isinstance(data, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Parakeet service returned an invalid success envelope",
                request_id,
            )
        return data
    return payload


def parse_parakeet_response(
    payload: Mapping[str, Any],
    *,
    audio: AudioArtifact,
    model: ModelFingerprint,
    request_id: str,
) -> TranscriptResult:
    """Validate one verbose JSON response without manufacturing timestamps."""

    data = _unwrap_response(payload, request_id)
    raw_text = data.get("text")
    if not isinstance(raw_text, str):
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Parakeet response is missing transcript text",
            request_id,
        )
    raw_words = data.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise _failure(
            UnsupportedCapabilityError,
            "unsupported_capability",
            "word timestamps unavailable: Parakeet returned no word offsets",
            request_id,
        )

    words: list[TimedWord] = []
    for index, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Parakeet response words[{index}] is not an object",
                request_id,
            )
        text = raw_word.get("word", raw_word.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Parakeet response words[{index}] is missing text",
                request_id,
            )
        try:
            start_ms, end_ms = _word_offsets(raw_word, index)
        except ContractValidationError as exc:
            raise _failure(
                UnsupportedCapabilityError,
                "unsupported_capability",
                f"word timestamps unavailable: {exc}",
                request_id,
            ) from exc
        if start_ms < 0 or end_ms <= start_ms or end_ms > audio.duration_ms:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Parakeet response words[{index}] has an out-of-bounds interval "
                f"({start_ms},{end_ms}) for duration {audio.duration_ms} "
                f"in {request_id}",
                request_id,
            )
        confidence = raw_word.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Parakeet response words[{index}] has invalid confidence",
                request_id,
            )
        words.append(
            TimedWord(
                word_id=f"service-word-{index:06d}",
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    for previous, current in zip(words, words[1:], strict=False):
        if current.start_ms < previous.start_ms:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Parakeet response word timestamps are not monotonic",
                request_id,
            )

    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Parakeet response segments must be an array",
            request_id,
        )
    segments = tuple(item for item in raw_segments if isinstance(item, Mapping))
    granularities = data.get("timestamp_granularities", ("word",))
    if not isinstance(granularities, list) or "word" not in granularities:
        granularities = ["word"]
    return TranscriptResult(
        artifact_id=audio.artifact_id,
        source_sha256=audio.source_sha256,
        model=model,
        text=raw_text,
        words=tuple(words),
        segments=segments,
        status="timed",
        timestamp_granularities=tuple(granularities),
        request_id=request_id,
        duration_ms=audio.duration_ms,
    )


@dataclass(frozen=True, slots=True)
class ParsedTranscription:
    """The typed transcript plus the exact service response for private evidence."""

    result: TranscriptResult
    raw_response: Mapping[str, Any]


HttpPost = Callable[[str, bytes, float], bytes]


def _default_http_post(url: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(MAX_AUDIO_PAYLOAD_BYTES * 2)
    except urllib.error.HTTPError as exc:
        return exc.read(MAX_AUDIO_PAYLOAD_BYTES * 2)


class ParakeetAdapter:
    """Application-side adapter for the local Parakeet JSON endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: ModelFingerprint | str,
        audio_loader: Callable[[AudioArtifact], bytes] | None = None,
        *,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        http_post: HttpPost | None = None,
    ) -> None:
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be non-empty text")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.model = model_fingerprint_from_lock(model) if isinstance(model, str) else model
        self.audio_loader = audio_loader
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def capabilities(self) -> CapabilityDeclaration:
        return CapabilityDeclaration(
            model=self.model,
            supports_timestamps=True,
            timestamp_granularities=("word",),
            max_audio_ms=310_000,
            max_payload_bytes=MAX_AUDIO_PAYLOAD_BYTES,
        )

    def transcribe_bytes(
        self,
        audio: AudioArtifact,
        audio_bytes: bytes,
        *,
        request_id: str | None = None,
        language: str | None = None,
        timestamp_granularities: Sequence[str] = ("word", "segment"),
    ) -> ParsedTranscription:
        request_id = _request_id(request_id)
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                "audio payload must contain bytes",
                request_id,
            )
        if len(audio_bytes) > MAX_AUDIO_PAYLOAD_BYTES:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                f"audio payload exceeds {MAX_AUDIO_PAYLOAD_BYTES} bytes",
                request_id,
            )
        requested = tuple(timestamp_granularities) or ("word",)
        if "word" not in requested:
            raise _failure(
                UnsupportedCapabilityError,
                "unsupported_capability",
                "production transcription requires requested word timestamps",
                request_id,
            )
        payload: dict[str, Any] = {
            "request_id": request_id,
            "model": self.model.repository,
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": audio.audio_format,
            },
            "response_format": "verbose_json",
            "timestamp_granularities": list(requested),
        }
        if language is not None:
            payload["language"] = language
        body = _as_json_bytes(payload)
        try:
            response_bytes = self._http_post(
                f"{self.endpoint}{PARAKEET_TRANSCRIPTION_PATH}", body, self.timeout_seconds
            )
        except TimeoutError as exc:
            raise _failure(
                RequestTimeoutError,
                "timeout",
                "Parakeet transcription request timed out",
                request_id,
                retryable=True,
            ) from exc
        except OSError as exc:
            raise _failure(
                AdapterFailure,
                "model_unavailable",
                f"Parakeet endpoint is unavailable: {exc}",
                request_id,
                retryable=True,
            ) from exc
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Parakeet endpoint returned invalid JSON",
                request_id,
            ) from exc
        if not isinstance(response, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Parakeet endpoint returned a non-object JSON response",
                request_id,
            )
        result = parse_parakeet_response(
            response, audio=audio, model=self.model, request_id=request_id
        )
        return ParsedTranscription(result=result, raw_response=response)

    def transcribe(
        self,
        audio: AudioArtifact,
        *,
        request_id: str,
        language: str | None = None,
        timestamp_granularities: Sequence[str] = (),
    ) -> TranscriptResult:
        if self.audio_loader is None:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                "ParakeetAdapter requires an audio_loader for a registered artifact",
                request_id,
            )
        return self.transcribe_bytes(
            audio,
            self.audio_loader(audio),
            request_id=request_id,
            language=language,
            timestamp_granularities=timestamp_granularities,
        ).result


LocalParakeetAdapter = ParakeetAdapter

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "LocalParakeetAdapter",
    "MAX_AUDIO_PAYLOAD_BYTES",
    "PARAKEET_TRANSCRIPTION_PATH",
    "ParakeetAdapter",
    "ParsedTranscription",
    "parse_parakeet_response",
]

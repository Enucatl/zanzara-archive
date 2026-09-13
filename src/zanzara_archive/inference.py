"""HTTP adapters for the repository-owned inference services.

The application process stays free of ML dependencies.  This module only
serializes the constrained internal request and validates the response before
it can become a canonical transcript artifact.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

import niquests
from niquests.exceptions import Timeout as NiquestsTimeout

from .contracts import (
    AdapterFailure,
    ApiError,
    AudioArtifact,
    AudioChunk,
    CapabilityDeclaration,
    ContractValidationError,
    DiarizationResult,
    ModelFingerprint,
    Overlap,
    RequestTimeoutError,
    TimedWord,
    TranscriptionHypothesis,
    TranscriptResult,
    Turn,
    UnsupportedCapabilityError,
)
from .model_locks import model_fingerprint_from_lock

PARAKEET_TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
WHISPER_TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
DIARIZATION_PATH = "/v1/diarize"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_WHISPER_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_DIARIZATION_TIMEOUT_SECONDS = 600.0
MAX_AUDIO_PAYLOAD_BYTES = 25_000_000
MAX_DIARIZATION_PAYLOAD_BYTES = 50_000_000
MAX_WHISPER_AUDIO_MS = 30_000
WHISPER_LANGUAGE = "it"
WHISPER_TASK = "transcribe"
WHISPER_DECODING_SETTINGS: Mapping[str, Any] = {
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "do_sample": False,
    "logprob_threshold": -1.0,
    "max_new_tokens": 444,
    "no_speech_threshold": 0.6,
    "num_beams": 5,
    "return_timestamps": False,
    "temperature": 0.0,
}


def _request_id(value: str | None, prefix: str = "parakeet") -> str:
    return value or f"{prefix}-{uuid.uuid4().hex}"


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


def _unwrap_response(
    payload: Mapping[str, Any], request_id: str, *, service_name: str = "inference"
) -> Mapping[str, Any]:
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
                    f"{service_name} service returned an invalid error envelope: {exc}",
                    request_id,
                ) from exc
        else:
            error = ApiError(
                "service_error",
                f"{service_name} service returned an invalid error",
                False,
                request_id,
            )
        if error.request_id != request_id:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"{service_name} service error request_id does not match the request",
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
                f"{service_name} service returned an invalid success envelope",
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

    data = _unwrap_response(payload, request_id, service_name="Parakeet")
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


def parse_whisper_response(
    payload: Mapping[str, Any],
    *,
    chunk: AudioChunk,
    model: ModelFingerprint,
    request_id: str,
) -> TranscriptionHypothesis:
    """Validate one text-first Whisper response without inventing timing."""

    data = _unwrap_response(payload, request_id, service_name="Whisper")
    reported_model = data.get("model")
    if reported_model is not None and reported_model != model.repository:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response model does not match the locked model",
            request_id,
        )
    reported_revision = data.get("model_revision")
    if reported_revision is not None and reported_revision != model.revision:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response revision does not match the locked model",
            request_id,
        )
    language = data.get("language", WHISPER_LANGUAGE)
    if language != WHISPER_LANGUAGE:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response language is not the locked Italian setting",
            request_id,
        )
    task = data.get("task", WHISPER_TASK)
    if task != WHISPER_TASK:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response task is not the locked transcription setting",
            request_id,
        )
    raw_text = data.get("text")
    if not isinstance(raw_text, str):
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response is missing transcript text",
            request_id,
        )
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list) or any(
        not isinstance(segment, Mapping) for segment in raw_segments
    ):
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Whisper response segments must be an array of objects",
            request_id,
        )
    raw_metadata = {
        "model": data.get("model", model.repository),
        "model_revision": data.get("model_revision", model.revision),
        "language": language,
        "task": task,
        "preprocessing": data.get("preprocessing", {}),
        "decoding": data.get("decoding", dict(WHISPER_DECODING_SETTINGS)),
        "capabilities": data.get("capabilities", {"supports_timestamps": False}),
        "request_id": request_id,
    }
    try:
        json.dumps(raw_metadata, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            f"Whisper response provenance is not JSON serializable: {exc}",
            request_id,
        ) from exc
    return TranscriptionHypothesis(
        chunk_id=chunk.chunk_id,
        model_fingerprint=model,
        text=raw_text,
        segments=tuple(dict(segment) for segment in raw_segments),
        raw_metadata=raw_metadata,
        source_sha256=chunk.source_sha256,
    )


@dataclass(frozen=True, slots=True)
class ParsedTranscription:
    """The typed transcript plus the exact service response for private evidence."""

    result: TranscriptResult
    raw_response: Mapping[str, Any]


def _turn_offsets(item: Mapping[str, Any], field_name: str) -> tuple[int, int]:
    """Convert one service turn to integer milliseconds exactly once."""

    if "start_ms" in item or "end_ms" in item:
        start = item.get("start_ms")
        end = item.get("end_ms")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ContractValidationError(f"{field_name} millisecond offsets must be integers")
        return start, end
    if "start" not in item or "end" not in item:
        raise ContractValidationError(f"{field_name} is missing genuine start/end timestamps")
    return (
        _seconds_to_ms(item["start"], f"{field_name}.start"),
        _seconds_to_ms(item["end"], f"{field_name}.end"),
    )


def _parse_turns(
    data: Mapping[str, Any],
    field_name: str,
    *,
    audio: AudioArtifact,
    request_id: str,
) -> tuple[Turn, ...]:
    raw_turns = data.get(field_name)
    if not isinstance(raw_turns, list):
        raise _failure(
            AdapterFailure,
            "invalid_response",
            f"Community-1 response is missing {field_name}",
            request_id,
        )
    turns: list[Turn] = []
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Community-1 response {field_name}[{index}] is not an object",
                request_id,
            )
        speaker_id = raw_turn.get("speaker_id", raw_turn.get("speaker", raw_turn.get("label")))
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Community-1 response {field_name}[{index}] is missing speaker ID",
                request_id,
            )
        try:
            start_ms, end_ms = _turn_offsets(raw_turn, f"{field_name}[{index}]")
            if start_ms < 0 or end_ms > audio.duration_ms:
                raise ContractValidationError(
                    f"interval ({start_ms},{end_ms}) is outside audio duration {audio.duration_ms}"
                )
            confidence = raw_turn.get("confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise ContractValidationError("confidence must be between 0 and 1")
            turns.append(
                Turn(
                    speaker_id=speaker_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        except ContractValidationError as exc:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                f"Community-1 response {field_name}[{index}] is invalid: {exc}",
                request_id,
            ) from exc
    return tuple(turns)


def derive_overlap_intervals(turns: Sequence[Turn]) -> tuple[Overlap, ...]:
    """Derive merged half-open overlap intervals from standard turns."""

    boundaries = sorted({point for turn in turns for point in (turn.start_ms, turn.end_ms)})
    overlaps: list[Overlap] = []
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
        if len(active) < 2:
            continue
        current = Overlap(active, start_ms, end_ms)
        if overlaps and overlaps[-1].end_ms == start_ms and overlaps[-1].speaker_ids == active:
            overlaps[-1] = Overlap(active, overlaps[-1].start_ms, end_ms)
        else:
            overlaps.append(current)
    return tuple(overlaps)


def _rttm_seconds(milliseconds: int) -> str:
    return f"{milliseconds // 1000}.{milliseconds % 1000:03d}"


def render_rttm(turns: Sequence[Turn], *, file_id: str = "episode") -> str:
    """Render an evaluation-only RTTM view without changing canonical offsets."""

    if (
        not isinstance(file_id, str)
        or not file_id.strip()
        or any(character.isspace() for character in file_id)
    ):
        raise ValueError("RTTM file_id must be non-empty and contain no whitespace")
    lines = []
    for turn in sorted(turns, key=lambda item: (item.start_ms, item.end_ms, item.speaker_id)):
        duration_ms = turn.end_ms - turn.start_ms
        lines.append(
            "SPEAKER "
            f"{file_id} 1 {_rttm_seconds(turn.start_ms)} {_rttm_seconds(duration_ms)} "
            f"<NA> <NA> {turn.speaker_id} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_diarization_response(
    payload: Mapping[str, Any],
    *,
    audio: AudioArtifact,
    model: ModelFingerprint,
    request_id: str,
) -> DiarizationResult:
    """Validate Community-1 standard/exclusive output and derive overlaps."""

    data = _unwrap_response(payload, request_id, service_name="Community-1")
    reported_model = data.get("model")
    if reported_model is not None and reported_model != model.repository:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Community-1 response model does not match the locked model",
            request_id,
        )
    reported_revision = data.get("model_revision")
    if reported_revision is not None and reported_revision != model.revision:
        raise _failure(
            AdapterFailure,
            "invalid_response",
            "Community-1 response revision does not match the locked model",
            request_id,
        )
    standard_turns = _parse_turns(data, "standard_turns", audio=audio, request_id=request_id)
    exclusive_turns = _parse_turns(data, "exclusive_turns", audio=audio, request_id=request_id)
    return DiarizationResult(
        artifact_id=audio.artifact_id,
        source_sha256=audio.source_sha256,
        duration_ms=audio.duration_ms,
        model=model,
        standard_turns=standard_turns,
        exclusive_turns=exclusive_turns,
        overlaps=derive_overlap_intervals(standard_turns),
        request_id=request_id,
    )


@dataclass(frozen=True, slots=True)
class ParsedDiarization:
    """The typed diarization plus the exact service response for private evidence."""

    result: DiarizationResult
    raw_response: Mapping[str, Any]


HttpPost = Callable[[str, bytes, float], bytes]
MAX_RESPONSE_BYTES = MAX_DIARIZATION_PAYLOAD_BYTES
RESPONSE_CHUNK_BYTES = 64 * 1024


def _default_http_post(url: str, body: bytes, timeout: float) -> bytes:
    try:
        with niquests.post(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=timeout,
            allow_redirects=True,
            verify=True,
            stream=True,
            retries=0,
        ) as response:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=RESPONSE_CHUNK_BYTES):
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                remaining = MAX_RESPONSE_BYTES - size
                if remaining <= 0:
                    break
                chunks.append(chunk[:remaining])
                size += min(len(chunk), remaining)
                if size >= MAX_RESPONSE_BYTES:
                    break
            return b"".join(chunks)
    except NiquestsTimeout as exc:
        raise TimeoutError(str(exc)) from exc


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


@dataclass(frozen=True, slots=True)
class ParsedHypothesis:
    """The text-first chunk hypothesis plus the exact service response."""

    result: TranscriptionHypothesis
    raw_response: Mapping[str, Any]


class WhisperAdapter:
    """Application-side adapter for the local Whisper Large v3 service."""

    def __init__(
        self,
        endpoint: str,
        model: ModelFingerprint | str,
        audio_loader: Callable[[AudioChunk], bytes] | None = None,
        *,
        timeout_seconds: float = DEFAULT_WHISPER_REQUEST_TIMEOUT_SECONDS,
        http_post: HttpPost | None = None,
    ) -> None:
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be non-empty text")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.model = (
            model_fingerprint_from_lock(model, "whisper") if isinstance(model, str) else model
        )
        self.audio_loader = audio_loader
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def capabilities(self) -> CapabilityDeclaration:
        return CapabilityDeclaration(
            model=self.model,
            supports_timestamps=False,
            max_audio_ms=MAX_WHISPER_AUDIO_MS,
            max_payload_bytes=MAX_AUDIO_PAYLOAD_BYTES,
        )

    def transcribe_bytes(
        self,
        chunk: AudioChunk,
        audio_bytes: bytes,
        *,
        request_id: str | None = None,
        language: str = WHISPER_LANGUAGE,
    ) -> ParsedHypothesis:
        request_id = _request_id(request_id, "whisper")
        if language != WHISPER_LANGUAGE:
            raise _failure(
                AdapterFailure,
                "invalid_configuration",
                "Whisper language is fixed to Italian ('it')",
                request_id,
            )
        if chunk.end_ms - chunk.start_ms > MAX_WHISPER_AUDIO_MS:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                f"Whisper chunks cannot exceed {MAX_WHISPER_AUDIO_MS} ms",
                request_id,
            )
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
        payload: dict[str, Any] = {
            "request_id": request_id,
            "model": self.model.repository,
            "chunk_id": chunk.chunk_id,
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": "wav",
            },
            "response_format": "json",
            "language": WHISPER_LANGUAGE,
            "task": WHISPER_TASK,
            "decoding": dict(WHISPER_DECODING_SETTINGS),
        }
        try:
            response_bytes = self._http_post(
                f"{self.endpoint}{WHISPER_TRANSCRIPTION_PATH}",
                _as_json_bytes(payload),
                self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise _failure(
                RequestTimeoutError,
                "timeout",
                "Whisper transcription request timed out",
                request_id,
                retryable=True,
            ) from exc
        except OSError as exc:
            raise _failure(
                AdapterFailure,
                "model_unavailable",
                f"Whisper endpoint is unavailable: {exc}",
                request_id,
                retryable=True,
            ) from exc
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Whisper endpoint returned invalid JSON",
                request_id,
            ) from exc
        if not isinstance(response, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Whisper endpoint returned a non-object JSON response",
                request_id,
            )
        result = parse_whisper_response(
            response,
            chunk=chunk,
            model=self.model,
            request_id=request_id,
        )
        return ParsedHypothesis(result=result, raw_response=response)

    def transcribe(self, chunk: AudioChunk, *, request_id: str) -> TranscriptionHypothesis:
        if self.audio_loader is None:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                "WhisperAdapter requires an audio_loader for a validated chunk",
                request_id,
            )
        return self.transcribe_bytes(
            chunk,
            self.audio_loader(chunk),
            request_id=request_id,
        ).result


class DiarizerAdapter:
    """Application-side adapter for the local Community-1 JSON endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: ModelFingerprint | str,
        audio_loader: Callable[[AudioArtifact], bytes] | None = None,
        *,
        timeout_seconds: float = DEFAULT_DIARIZATION_TIMEOUT_SECONDS,
        http_post: HttpPost | None = None,
    ) -> None:
        if not endpoint or not isinstance(endpoint, str):
            raise ValueError("endpoint must be non-empty text")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.model = (
            model_fingerprint_from_lock(model, "diarization") if isinstance(model, str) else model
        )
        self.audio_loader = audio_loader
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def capabilities(self) -> CapabilityDeclaration:
        return CapabilityDeclaration(
            model=self.model,
            supports_timestamps=True,
            timestamp_granularities=("segment",),
            max_payload_bytes=MAX_DIARIZATION_PAYLOAD_BYTES,
        )

    def diarize_bytes(
        self,
        audio: AudioArtifact,
        audio_bytes: bytes,
        *,
        request_id: str | None = None,
    ) -> ParsedDiarization:
        request_id = request_id or f"diarization-{uuid.uuid4().hex}"
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                "audio payload must contain bytes",
                request_id,
            )
        if len(audio_bytes) > MAX_DIARIZATION_PAYLOAD_BYTES:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                f"audio payload exceeds {MAX_DIARIZATION_PAYLOAD_BYTES} bytes",
                request_id,
            )
        payload = {
            "request_id": request_id,
            "model": self.model.repository,
            "input_audio": {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "format": audio.audio_format,
            },
            "output": {"standard": True, "exclusive": True, "overlaps": True},
        }
        try:
            response_bytes = self._http_post(
                f"{self.endpoint}{DIARIZATION_PATH}",
                _as_json_bytes(payload),
                self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise _failure(
                RequestTimeoutError,
                "timeout",
                "Community-1 diarization request timed out",
                request_id,
                retryable=True,
            ) from exc
        except OSError as exc:
            raise _failure(
                AdapterFailure,
                "model_unavailable",
                f"Community-1 endpoint is unavailable: {exc}",
                request_id,
                retryable=True,
            ) from exc
        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Community-1 endpoint returned invalid JSON",
                request_id,
            ) from exc
        if not isinstance(response, Mapping):
            raise _failure(
                AdapterFailure,
                "invalid_response",
                "Community-1 endpoint returned a non-object JSON response",
                request_id,
            )
        result = parse_diarization_response(
            response,
            audio=audio,
            model=self.model,
            request_id=request_id,
        )
        return ParsedDiarization(result=result, raw_response=response)

    def diarize(self, audio: AudioArtifact, *, request_id: str) -> DiarizationResult:
        if self.audio_loader is None:
            raise _failure(
                AdapterFailure,
                "invalid_audio",
                "DiarizerAdapter requires an audio_loader for a registered artifact",
                request_id,
            )
        return self.diarize_bytes(
            audio,
            self.audio_loader(audio),
            request_id=request_id,
        ).result


Community1Adapter = DiarizerAdapter
LocalDiarizerAdapter = DiarizerAdapter
LocalParakeetAdapter = ParakeetAdapter

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_WHISPER_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_DIARIZATION_TIMEOUT_SECONDS",
    "DIARIZATION_PATH",
    "DiarizerAdapter",
    "Community1Adapter",
    "LocalParakeetAdapter",
    "LocalDiarizerAdapter",
    "MAX_AUDIO_PAYLOAD_BYTES",
    "MAX_DIARIZATION_PAYLOAD_BYTES",
    "MAX_WHISPER_AUDIO_MS",
    "PARAKEET_TRANSCRIPTION_PATH",
    "ParakeetAdapter",
    "ParsedHypothesis",
    "ParsedDiarization",
    "ParsedTranscription",
    "WHISPER_DECODING_SETTINGS",
    "WHISPER_LANGUAGE",
    "WHISPER_TASK",
    "WHISPER_TRANSCRIPTION_PATH",
    "WhisperAdapter",
    "derive_overlap_intervals",
    "parse_diarization_response",
    "parse_parakeet_response",
    "parse_whisper_response",
    "render_rttm",
]

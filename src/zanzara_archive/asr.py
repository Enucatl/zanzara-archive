"""Windowed ASR orchestration with source-relative timestamp ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import AudioArtifact, ContractValidationError, TimedWord, TranscriptResult
from .inference import ParakeetAdapter, ParsedTranscription

ASR_WINDOW_MS = 300_000
ASR_CONTEXT_MS = 5_000


@dataclass(frozen=True, slots=True)
class ASRWindow:
    """One half-open ownership interval and its expanded decode interval."""

    index: int
    ownership_start_ms: int
    ownership_end_ms: int
    decode_start_ms: int
    decode_end_ms: int

    @property
    def decode_duration_ms(self) -> int:
        return self.decode_end_ms - self.decode_start_ms

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "ownership_start_ms": self.ownership_start_ms,
            "ownership_end_ms": self.ownership_end_ms,
            "decode_start_ms": self.decode_start_ms,
            "decode_end_ms": self.decode_end_ms,
        }


def build_asr_windows(
    duration_ms: int,
    *,
    window_ms: int = ASR_WINDOW_MS,
    context_ms: int = ASR_CONTEXT_MS,
) -> tuple[ASRWindow, ...]:
    """Build clipped decode windows with deterministic ownership boundaries."""

    if duration_ms <= 0 or window_ms <= 0 or context_ms < 0:
        raise ValueError(
            "duration_ms and window_ms must be positive; context_ms cannot be negative"
        )
    windows: list[ASRWindow] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        windows.append(
            ASRWindow(
                index=index,
                ownership_start_ms=start,
                ownership_end_ms=end,
                decode_start_ms=max(0, start - context_ms),
                decode_end_ms=min(duration_ms, end + context_ms),
            )
        )
        start = end
        index += 1
    return tuple(windows)


def window_audio_artifact(audio: AudioArtifact, window: ASRWindow) -> AudioArtifact:
    """Describe a decoded window while retaining its original source identity."""

    transform = {
        "decoder": "ffmpeg",
        "format": "wav",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "decode_start_ms": window.decode_start_ms,
        "decode_end_ms": window.decode_end_ms,
    }
    transform_hash = hashlib.sha256(
        json.dumps(transform, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return AudioArtifact(
        artifact_id=f"{audio.artifact_id}:window-{window.index:04d}",
        source_sha256=audio.source_sha256,
        format="wav",
        duration_ms=window.decode_duration_ms,
        sample_rate_hz=16_000,
        channels=1,
        time_origin_ms=window.decode_start_ms,
        transform_hash=transform_hash,
        decoder_version="ffmpeg",
        preprocessing=transform,
    )


def _owned(word: TimedWord, window: ASRWindow) -> bool:
    """Use the mathematical midpoint, assigning an exact boundary forward."""

    midpoint_numerator = word.start_ms + word.end_ms
    return (
        2 * window.ownership_start_ms <= midpoint_numerator
        and midpoint_numerator < 2 * window.ownership_end_ms
    )


def assign_window_words(result: TranscriptResult, window: ASRWindow) -> tuple[TimedWord, ...]:
    """Translate local word offsets and retain only this window's owners."""

    if result.duration_ms != window.decode_duration_ms:
        raise ContractValidationError(
            "window transcript duration does not match its expanded decode interval"
        )
    assigned: list[TimedWord] = []
    for word in result.words:
        shifted_start = word.start_ms + window.decode_start_ms
        shifted_end = word.end_ms + window.decode_start_ms
        shifted = TimedWord(
            word_id=f"window-{window.index:04d}-{word.word_id}",
            text=word.text,
            start_ms=shifted_start,
            end_ms=shifted_end,
            confidence=word.confidence,
            speaker_id=word.speaker_id,
            overlap=word.overlap,
        )
        if _owned(shifted, window):
            assigned.append(shifted)
    return tuple(assigned)


def merge_window_results(
    audio: AudioArtifact,
    windows: Sequence[ASRWindow],
    results: Sequence[TranscriptResult],
    *,
    request_id: str,
) -> TranscriptResult:
    """Merge owned words without text-based deduplication."""

    if len(windows) != len(results):
        raise ValueError("every ASR window must have exactly one transcript result")
    model = results[0].model if results else None
    if model is None:
        raise ValueError("at least one ASR window is required")
    words = [
        word
        for window, result in zip(windows, results, strict=True)
        for word in assign_window_words(result, window)
    ]
    words.sort(key=lambda word: (word.start_ms, word.end_ms, word.word_id))
    if any(result.model.fingerprint_sha256 != model.fingerprint_sha256 for result in results):
        raise ContractValidationError("all ASR windows must use one model fingerprint")
    text = " ".join(word.text.strip() for word in words)
    return TranscriptResult(
        artifact_id=audio.artifact_id,
        source_sha256=audio.source_sha256,
        model=model,
        text=text,
        words=tuple(words),
        status="timed",
        timestamp_granularities=("word",),
        request_id=request_id,
        duration_ms=audio.duration_ms,
    )


WindowAudioLoader = Callable[[ASRWindow], bytes]


def transcribe_windowed(
    adapter: ParakeetAdapter,
    audio: AudioArtifact,
    audio_loader: WindowAudioLoader,
    *,
    request_id: str,
    language: str | None = None,
) -> tuple[TranscriptResult, tuple[ASRWindow, ...], tuple[Mapping[str, Any], ...]]:
    """Run every expanded window and return the merged result plus private responses."""

    windows = build_asr_windows(audio.duration_ms)
    parsed: list[ParsedTranscription] = []
    for window in windows:
        window_audio = window_audio_artifact(audio, window)
        parsed.append(
            adapter.transcribe_bytes(
                window_audio,
                audio_loader(window),
                request_id=f"{request_id}-window-{window.index:04d}",
                language=language,
                timestamp_granularities=("word", "segment"),
            )
        )
    merged = merge_window_results(
        audio,
        windows,
        tuple(item.result for item in parsed),
        request_id=request_id,
    )
    return merged, windows, tuple(item.raw_response for item in parsed)


__all__ = [
    "ASR_CONTEXT_MS",
    "ASR_WINDOW_MS",
    "ASRWindow",
    "assign_window_words",
    "build_asr_windows",
    "merge_window_results",
    "transcribe_windowed",
    "window_audio_artifact",
]

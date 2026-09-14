"""Community-1 standard/exclusive diarization service."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from smoke_server import run_service

MODEL_PATH = os.environ["MODEL_PATH"]
pipeline = Pipeline.from_pretrained(MODEL_PATH)
pipeline.to(torch.device("cuda"))


def _decode_audio(audio_bytes: bytes, _audio_format: str) -> tuple[np.ndarray, int]:
    """Decode request audio, normalizing compressed input through the shared FFmpeg runtime."""

    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValueError("decoded audio must have one channel")
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    except (OSError, RuntimeError, ValueError):
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"audio decoding failed: {detail}") from None
        return np.frombuffer(completed.stdout, dtype=np.float32), 16_000


def _annotation_turns(annotation: object) -> list[dict[str, Any]]:
    return [
        {
            "speaker_id": str(label),
            "start": float(segment.start),
            "end": float(segment.end),
        }
        for segment, _track, label in annotation.itertracks(yield_label=True)  # type: ignore[attr-defined]
    ]


def _native_activity(
    feature: object, *, duration_ms: int, source_sha256: str = "pending"
) -> dict[str, Any]:
    """Serialize the public speaker_counting feature and its source timeline."""

    data = np.asarray(getattr(feature, "data", None))
    window = getattr(feature, "sliding_window", None)
    if data.ndim != 2 or data.shape[1] != 1 or window is None:
        raise ValueError("Community-1 speaker_counting hook returned an unsupported feature")
    start_s = getattr(window, "start", None)
    step_s = getattr(window, "step", None)
    duration_s = getattr(window, "duration", None)
    if not all(isinstance(value, (int, float)) for value in (start_s, step_s, duration_s)):
        raise ValueError("Community-1 speaker_counting feature has no reproducible timeline")
    if float(start_s) != 0.0 or float(step_s) <= 0.0 or float(duration_s) <= 0.0:
        raise ValueError("Community-1 speaker_counting timeline is not source-relative")

    def to_ms(seconds: float) -> int:
        return int(np.floor(seconds * 1000.0 + 0.5))

    intervals: list[dict[str, int]] = []
    frame_count = int(data.shape[0])
    retained_frame_count = 0
    for index, raw_count in enumerate(data[:, 0].tolist()):
        count = int(raw_count)
        if count < 0 or float(raw_count) != count:
            raise ValueError("Community-1 speaker_counting returned a non-integral count")
        frame_start_ms = to_ms(float(start_s) + index * float(step_s))
        if frame_start_ms >= duration_ms:
            # pyannote may pad the public feature timeline beyond the decoded
            # source.  Those frames have no source-relative coverage.
            break
        retained_frame_count += 1
        # ``speaker_counting`` is a SlidingWindowFeature whose windows overlap
        # (duration is larger than step).  The public frame timeline therefore
        # represents samples at successive frame starts, not contiguous windows.
        # Use the next frame start as the run boundary and close the final frame
        # at the decoded source end so the persisted representation is contiguous
        # and source-relative without consulting private model internals.
        next_start_s = (
            float(start_s) + (index + 1) * float(step_s)
            if index + 1 < frame_count
            else duration_ms / 1000.0
        )
        frame_end_ms = to_ms(next_start_s)
        if frame_start_ms != (intervals[-1]["end_ms"] if intervals else 0):
            raise ValueError("Community-1 speaker_counting frames do not map contiguously")
        frame_end_ms = min(frame_end_ms, duration_ms)
        if frame_end_ms <= frame_start_ms:
            continue
        if intervals and intervals[-1]["speaker_count"] == count:
            intervals[-1]["end_ms"] = frame_end_ms
        else:
            intervals.append(
                {
                    "start_ms": frame_start_ms,
                    "end_ms": frame_end_ms,
                    "speaker_count": count,
                }
            )
    if not intervals or intervals[-1]["end_ms"] != duration_ms:
        raise ValueError("Community-1 speaker_counting frames do not cover the source duration")
    timeline = {
        "frame_count": retained_frame_count,
        "frame_start_ms": to_ms(float(start_s)),
        "frame_step_ms": to_ms(float(step_s)),
        "frame_duration_ms": to_ms(float(duration_s)),
        "time_origin": "original episode",
        "capture_version": "community1-speaker-count-snapshot-v1",
    }
    payload: dict[str, Any] = {
        "source_sha256": source_sha256,
        "duration_ms": duration_ms,
        "intervals": intervals,
        "timeline": timeline,
    }
    return payload


def _snapshot_speaker_counting(feature: object) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Detach the hook's public feature before Community-1 mutates it in place."""

    data = np.array(getattr(feature, "data", None), copy=True)
    window = getattr(feature, "sliding_window", None)
    metadata = tuple(getattr(window, name, None) for name in ("start", "step", "duration"))
    if (
        data.ndim != 2
        or len(metadata) != 3
        or not all(isinstance(value, (int, float)) for value in metadata)
    ):
        raise ValueError("Community-1 speaker_counting hook returned an unsupported feature")
    return data, tuple(float(value) for value in metadata)  # type: ignore[return-value]


def _diarize(
    audio: np.ndarray,
    sample_rate: int,
    *,
    source_sha256: str = "pending",
    duration_ms: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if audio.ndim != 1:
        raise ValueError("decoded audio must be mono after preprocessing")
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    captured: dict[str, object] = {}

    def hook(step_name: str, artifact: object, **_kwargs: object) -> None:
        if step_name == "speaker_counting":
            captured["snapshot"] = _snapshot_speaker_counting(artifact)

    with torch.inference_mode():
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    snapshot = captured.get("snapshot")
    if snapshot is None:
        raise ValueError("Community-1 speaker_counting hook did not expose an artifact")
    standard = result.speaker_diarization
    exclusive = result.exclusive_speaker_diarization
    data, (start, step, frame_duration) = snapshot  # type: ignore[misc]
    # _native_activity intentionally accepts only the public feature surface.
    feature = type(
        "SpeakerCountingSnapshot",
        (),
        {
            "data": data,
            "sliding_window": type(
                "SlidingWindow", (), {"start": start, "step": step, "duration": frame_duration}
            )(),
        },
    )()
    native = _native_activity(
        feature,
        duration_ms=duration_ms or round(len(audio) * 1000 / sample_rate),
        source_sha256=source_sha256,
    )
    return _annotation_turns(standard), _annotation_turns(exclusive), native


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    standard, exclusive, native = _diarize(np.asarray(audio, dtype=np.float32), int(sample_rate))
    return {
        "shape": [len(standard), len(exclusive)],
        "output": "standard turns, exclusive turns, and overlap-capable annotation",
        "standard_turns": len(standard),
        "exclusive_turns": len(exclusive),
        "speaker_count": len({turn["speaker_id"] for turn in standard}),
        "native_activity": native,
    }


def diarize_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one full audio request and retain both Community-1 representations."""

    input_audio = request.get("input_audio")
    if not isinstance(input_audio, Mapping):
        raise ValueError("input_audio must be an object")
    encoded = input_audio.get("data")
    audio_format = input_audio.get("format", "wav")
    if not isinstance(encoded, str) or not isinstance(audio_format, str) or not audio_format:
        raise ValueError("input_audio requires base64 data and a format")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("input_audio.data is not valid base64") from exc
    audio, sample_rate = _decode_audio(audio_bytes, audio_format)
    source_sha256 = request.get("source_sha256", "pending")
    if not isinstance(source_sha256, str):
        raise ValueError("source_sha256 must be text when provided")
    expected_duration_ms = request.get("duration_ms")
    if expected_duration_ms is not None and (
        isinstance(expected_duration_ms, bool)
        or not isinstance(expected_duration_ms, int)
        or expected_duration_ms <= 0
    ):
        raise ValueError("duration_ms must be a positive integer when provided")
    standard, exclusive, native = _diarize(
        audio,
        sample_rate,
        source_sha256=source_sha256,
        duration_ms=expected_duration_ms,
    )
    native_payload = json.dumps(native, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    native["artifact_id"] = (
        "native-activity-" + hashlib.sha256(native_payload.encode("utf-8")).hexdigest()
    )
    return {
        "model": os.environ.get("MODEL_SOURCE", "pyannote/speaker-diarization-community-1"),
        "model_revision": os.environ.get("MODEL_REVISION", "unknown"),
        "standard_turns": standard,
        "exclusive_turns": exclusive,
        "overlap_source": "all active standard turns; derived by the archive adapter",
        "episode_wide": True,
        "native_activity": native,
    }


if __name__ == "__main__":
    run_service(
        "diarization",
        infer,
        diarize_request,
        post_path="/v1/diarize",
        max_request_bytes=160_000_000,
    )

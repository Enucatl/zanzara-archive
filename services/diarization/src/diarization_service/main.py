"""Community-1 standard/exclusive diarization service."""

from __future__ import annotations

import base64
import binascii
import io
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


def _diarize(
    audio: np.ndarray, sample_rate: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if audio.ndim != 1:
        raise ValueError("decoded audio must be mono after preprocessing")
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    with torch.inference_mode():
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    standard = result.speaker_diarization
    exclusive = result.exclusive_speaker_diarization
    return _annotation_turns(standard), _annotation_turns(exclusive)


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    standard, exclusive = _diarize(np.asarray(audio, dtype=np.float32), int(sample_rate))
    return {
        "shape": [len(standard), len(exclusive)],
        "output": "standard turns, exclusive turns, and overlap-capable annotation",
        "standard_turns": len(standard),
        "exclusive_turns": len(exclusive),
        "speaker_count": len({turn["speaker_id"] for turn in standard}),
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
    standard, exclusive = _diarize(audio, sample_rate)
    return {
        "model": os.environ.get("MODEL_SOURCE", "pyannote/speaker-diarization-community-1"),
        "model_revision": os.environ.get("MODEL_REVISION", "unknown"),
        "standard_turns": standard,
        "exclusive_turns": exclusive,
        "overlap_source": "all active standard turns; derived by the archive adapter",
        "episode_wide": True,
    }


if __name__ == "__main__":
    run_service(
        "diarization",
        infer,
        diarize_request,
        post_path="/v1/diarize",
        max_request_bytes=160_000_000,
    )

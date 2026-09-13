"""Real Parakeet TDT inference smoke."""

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
from smoke_server import run_service
from transformers import AutoModelForTDT, AutoProcessor

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_REVISION = os.environ.get("MODEL_REVISION", "unknown")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "nvidia/parakeet-tdt-0.6b-v3")
SAMPLE_RATE_HZ = 16_000
MAX_CHUNK_AUDIO_SECONDS = 30
CHUNK_PREPROCESSING = {
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "channels": 1,
    "time_origin": "source chunk start",
}
CHUNK_CONFIGURATION = {
    "dtype": "bfloat16",
    "inference_mode": True,
    "decoder": "native Parakeet token spans grouped by decoder whitespace",
    "timestamp_interpolation": False,
}
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTDT.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")
model.eval()


def _ffmpeg_decode(audio_bytes: bytes) -> tuple[np.ndarray, int]:
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
            str(SAMPLE_RATE_HZ),
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
    return np.frombuffer(completed.stdout, dtype=np.float32), SAMPLE_RATE_HZ


def _decode_audio(audio_bytes: bytes, _audio_format: str) -> tuple[np.ndarray, int]:
    """Decode every request to the shared mono 16 kHz chunk representation."""

    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValueError("decoded audio must have one channel")
        if int(sample_rate) != SAMPLE_RATE_HZ:
            return _ffmpeg_decode(audio_bytes)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    except (OSError, RuntimeError, ValueError):
        return _ffmpeg_decode(audio_bytes)


def _decode_model_output(audio: np.ndarray, sample_rate: int) -> tuple[str, list[dict[str, Any]]]:
    inputs = processor(audio=audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {
        key: (
            value.to("cuda", dtype=torch.bfloat16)
            if value.is_floating_point()
            else value.to("cuda")
        )
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        generated = model.generate(**inputs)
    _decoded_text, timestamps = processor.decode(generated.sequences, durations=generated.durations)
    raw_words = timestamps[0] if timestamps and isinstance(timestamps[0], list) else timestamps
    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    output_tokens: list[str] = []
    duration_seconds = len(audio) / sample_rate
    for index, item in enumerate(raw_words or ()):
        if not isinstance(item, Mapping):
            raise ValueError(f"decoder returned a non-object timestamp at index {index}")
        token = item.get("token", item.get("word", item.get("text")))
        start = item.get("start")
        end = item.get("end")
        if not isinstance(token, str) or not token.strip():
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"decoder returned no genuine word timestamp at index {index}")
        if start < 0 or end < start:
            raise ValueError(f"decoder returned an invalid timestamp interval at index {index}")
        # TDT timestamps are quantized to encoder frames, so the final frame can
        # extend a few milliseconds past the actual waveform. Discard only that
        # native out-of-range emission; never clamp, shift, or interpolate it.
        if end > duration_seconds:
            continue
        output_tokens.append(token)
        starts_new_word = bool(current and token[0].isspace())
        if starts_new_word:
            words.append(current)
            current = None
        if current is None:
            current = {
                "word": token.strip(),
                "start": float(start),
                "end": float(end),
            }
        else:
            current["word"] += token
            current["end"] = max(float(current["end"]), float(end))
    if current is not None:
        words.append(current)
    words = [word for word in words if word["end"] > word["start"]]
    if not words:
        raise ValueError("decoder returned no genuine word timestamps")
    return "".join(output_tokens).strip(), words


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError("smoke audio must be mono after preprocessing")
    text, words = _decode_model_output(np.asarray(audio, dtype=np.float32), int(sample_rate))
    return {
        "shape": [len(text), len(words)],
        "output": "text plus genuine word/segment timestamp decode",
        "word_timestamp_count": len(words),
    }


def transcribe_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Implement the repository's small OpenAI-compatible transcription subset."""

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
    chunk_id = request.get("chunk_id")
    if chunk_id is not None:
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be non-empty text")
        chunk = request.get("chunk")
        if not isinstance(chunk, Mapping):
            raise ValueError("chunk must be an object for text-first requests")
        start_ms = chunk.get("start_ms")
        end_ms = chunk.get("end_ms")
        if (
            isinstance(start_ms, bool)
            or not isinstance(start_ms, int)
            or isinstance(end_ms, bool)
            or not isinstance(end_ms, int)
            or end_ms <= start_ms
            or end_ms - start_ms > MAX_CHUNK_AUDIO_SECONDS * 1000
        ):
            raise ValueError("chunk interval must be a positive interval no longer than 30 seconds")
    if request.get("model", MODEL_SOURCE) != MODEL_SOURCE:
        raise ValueError("Parakeet service model is locked to the configured checkpoint")
    text, words = _decode_model_output(audio, sample_rate)
    response: dict[str, Any] = {
        "model": MODEL_SOURCE,
        "text": text,
        "words": words,
        "segments": [],
        "timestamp_granularities": ["word"],
        "timestamp_source": "native Parakeet token spans grouped by decoder whitespace",
        "timestamp_interpolation": False,
        "model_revision": MODEL_REVISION,
        "preprocessing": CHUNK_PREPROCESSING,
        "configuration": CHUNK_CONFIGURATION,
        "capabilities": {
            "supports_timestamps": True,
            "timestamp_granularities": ["word"],
        },
    }
    if isinstance(chunk_id, str):
        response["chunk_id"] = chunk_id
    return response


if __name__ == "__main__":
    run_service("parakeet", infer, transcribe_request)

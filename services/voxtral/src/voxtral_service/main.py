"""Local Voxtral Mini 4B Realtime offline transcription service."""

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
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_REVISION = os.environ.get("MODEL_REVISION", "unknown")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "mistralai/Voxtral-Mini-4B-Realtime-2602")
LANGUAGE = "it"
SAMPLE_RATE_HZ = 16_000
MAX_AUDIO_SECONDS = 30
TRANSCRIPTION_DELAY_MS = 480
GENERATION_SETTINGS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 1024,
}

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
if processor.feature_extractor.sampling_rate != SAMPLE_RATE_HZ:
    raise RuntimeError("Voxtral processor does not use the locked 16 kHz sample rate")
if processor.num_delay_tokens * 80 != TRANSCRIPTION_DELAY_MS:
    raise RuntimeError(
        "Voxtral processor delay does not match the locked 480 ms accuracy configuration"
    )
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
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
    """Decode a chunk and normalize it to the locked mono 16 kHz input."""

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


def _decode_text(audio: np.ndarray, sample_rate: int) -> tuple[str, list[int]]:
    if audio.ndim != 1:
        raise ValueError("decoded audio must be mono after preprocessing")
    duration_seconds = len(audio) / sample_rate
    if duration_seconds <= 0 or duration_seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"audio duration must be between 0 and {MAX_AUDIO_SECONDS} seconds")
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        is_streaming=False,
        is_first_audio_chunk=True,
    )
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if value.is_floating_point():
            model_inputs[key] = value.to("cuda", dtype=torch.bfloat16)
        else:
            model_inputs[key] = value.to("cuda")
    with torch.inference_mode():
        generated = model.generate(**model_inputs, **GENERATION_SETTINGS)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    token_ids = sequences[0].detach().cpu().tolist()
    if not text:
        raise ValueError("Voxtral decoder returned empty transcript text")
    return text, [int(token_id) for token_id in token_ids]


def infer(audio_path: Path) -> dict[str, object]:
    audio_bytes = audio_path.read_bytes()
    audio, sample_rate = _decode_audio(audio_bytes, audio_path.suffix.lstrip("."))
    text, token_ids = _decode_text(audio, sample_rate)
    return {
        "shape": [len(text), len(token_ids)],
        "output": "text-first hypothesis; no timestamps",
        "language": LANGUAGE,
        "decoding": {
            **GENERATION_SETTINGS,
            "transcription_delay_ms": TRANSCRIPTION_DELAY_MS,
            "streaming": False,
        },
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "time_origin": "source chunk start",
        },
    }


def transcribe_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one validated text-first common-chunk request."""

    input_audio = request.get("input_audio")
    if not isinstance(input_audio, Mapping):
        raise ValueError("input_audio must be an object")
    encoded = input_audio.get("data")
    audio_format = input_audio.get("format", "wav")
    if not isinstance(encoded, str) or not isinstance(audio_format, str) or not audio_format:
        raise ValueError("input_audio requires base64 data and a format")
    if request.get("language", LANGUAGE) != LANGUAGE:
        raise ValueError("Voxtral service language is fixed to Italian ('it')")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("input_audio.data is not valid base64") from exc
    audio, sample_rate = _decode_audio(audio_bytes, audio_format)
    text, token_ids = _decode_text(audio, sample_rate)
    return {
        "model": MODEL_SOURCE,
        "model_revision": MODEL_REVISION,
        "text": text,
        "segments": [],
        "language": LANGUAGE,
        "decoding": {
            **GENERATION_SETTINGS,
            "transcription_delay_ms": TRANSCRIPTION_DELAY_MS,
            "streaming": False,
        },
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "time_origin": "source chunk start",
        },
        "capabilities": {
            "supports_timestamps": False,
            "timestamp_granularities": [],
        },
        "raw_generation": {"sequence_token_ids": token_ids},
    }


if __name__ == "__main__":
    run_service("voxtral", infer, transcribe_request)

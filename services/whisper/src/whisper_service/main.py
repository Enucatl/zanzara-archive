"""Local Whisper Large v3 text-first inference service."""

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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_REVISION = os.environ.get("MODEL_REVISION", "unknown")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "openai/whisper-large-v3")
LANGUAGE = "it"
TASK = "transcribe"
SAMPLE_RATE_HZ = 16_000
MAX_AUDIO_SECONDS = 30
DECODING_SETTINGS: dict[str, Any] = {
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

processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
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


def _decode_text(audio: np.ndarray, sample_rate: int) -> str:
    if audio.ndim != 1:
        raise ValueError("decoded audio must be mono after preprocessing")
    duration_seconds = len(audio) / sample_rate
    if duration_seconds <= 0 or duration_seconds > MAX_AUDIO_SECONDS:
        raise ValueError(f"audio duration must be between 0 and {MAX_AUDIO_SECONDS} seconds")
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        truncation=False,
        padding="longest",
        return_attention_mask=True,
    )
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if value.is_floating_point():
            model_inputs[key] = value.to("cuda", dtype=torch.float16)
        else:
            model_inputs[key] = value.to("cuda")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            forced_decoder_ids=forced_decoder_ids,
            **DECODING_SETTINGS,
        )
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = _decode_audio(audio_path.read_bytes(), audio_path.suffix.lstrip("."))
    text = _decode_text(audio, sample_rate)
    return {
        "shape": [len(text)],
        "text": text,
        "language": LANGUAGE,
        "task": TASK,
        "decoding": DECODING_SETTINGS,
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "time_origin": "source chunk start",
        },
    }


def transcribe_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one validated text-first chunk request with locked Italian decoding."""

    input_audio = request.get("input_audio")
    if not isinstance(input_audio, Mapping):
        raise ValueError("input_audio must be an object")
    encoded = input_audio.get("data")
    audio_format = input_audio.get("format", "wav")
    if not isinstance(encoded, str) or not isinstance(audio_format, str) or not audio_format:
        raise ValueError("input_audio requires base64 data and a format")
    if request.get("language", LANGUAGE) != LANGUAGE:
        raise ValueError("Whisper service language is fixed to Italian ('it')")
    if request.get("task", TASK) != TASK:
        raise ValueError("Whisper service task is fixed to transcription")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("input_audio.data is not valid base64") from exc
    audio, sample_rate = _decode_audio(audio_bytes, audio_format)
    text = _decode_text(audio, sample_rate)
    return {
        "model": MODEL_SOURCE,
        "model_revision": MODEL_REVISION,
        "text": text,
        "segments": [],
        "language": LANGUAGE,
        "task": TASK,
        "decoding": DECODING_SETTINGS,
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "time_origin": "source chunk start",
        },
        "capabilities": {
            "supports_timestamps": False,
            "timestamp_granularities": [],
        },
    }


if __name__ == "__main__":
    run_service("whisper", infer, transcribe_request)

"""Local AudioSet AST inference for non-speaker acoustic conditions."""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from smoke_server import run_service
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_REVISION = os.environ.get("MODEL_REVISION", "unknown")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "MIT/ast-finetuned-audioset-10-10-0.4593")
LABEL_MAP_SHA256 = os.environ.get("LABEL_MAP_SHA256", "unknown")
PREPROCESSING_SHA256 = os.environ.get("PREPROCESSING_SHA256", "unknown")
SAMPLE_RATE_HZ = 16_000
WINDOW_MS = 10_000
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_MS // 1000
MUSIC_TERMS = ("music", "musical instrument", "singing", "song")
SPEECH_TERMS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "babbling",
    "speech synthesizer",
)
ACTIVITY_TERMS = (
    "applause",
    "breathing",
    "cough",
    "crowd",
    "crying",
    "engine",
    "laughter",
    "noise",
    "rain",
    "singing",
    "vehicle",
    "wind",
)

processor = AutoFeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    local_files_only=True,
).to("cuda")
model.eval()
LABELS = {int(key): str(value) for key, value in model.config.id2label.items()}


def _contains(label: str, terms: tuple[str, ...]) -> bool:
    lowered = label.casefold()
    return any(term in lowered for term in terms)


RELEVANT_IDS = tuple(
    sorted(
        class_id
        for class_id, label in LABELS.items()
        if _contains(label, MUSIC_TERMS)
        or _contains(label, SPEECH_TERMS)
        or _contains(label, ACTIVITY_TERMS)
    )
)


def _ffmpeg_decode(audio_bytes: bytes) -> np.ndarray:
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
        raise ValueError(f"audio decoding failed: {detail}")
    return np.frombuffer(completed.stdout, dtype=np.float32)


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1:
            raise ValueError("decoded audio must have one channel")
        if int(sample_rate) != SAMPLE_RATE_HZ:
            return _ffmpeg_decode(audio_bytes)
        return np.asarray(audio, dtype=np.float32)
    except (OSError, RuntimeError, ValueError):
        return _ffmpeg_decode(audio_bytes)


def _signal_measurements(audio: np.ndarray) -> dict[str, float]:
    if audio.size == 0:
        return {
            "clipping_fraction": 0.0,
            "peak_dbfs": -120.0,
            "rms_dbfs": -120.0,
            "silence_fraction": 1.0,
        }
    values = audio.astype(np.float64, copy=False)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(values * values)))

    def to_dbfs(value: float) -> float:
        return 20.0 * math.log10(max(value, 1e-12))

    return {
        "clipping_fraction": float(np.mean(np.abs(values) >= 0.999)),
        "peak_dbfs": to_dbfs(peak),
        "rms_dbfs": to_dbfs(rms),
        "silence_fraction": float(np.mean(np.abs(values) < 1e-4)),
    }


def _windows(audio: np.ndarray, chunk_start_ms: int, chunk_end_ms: int) -> list[dict[str, Any]]:
    requested_duration_ms = chunk_end_ms - chunk_start_ms
    if requested_duration_ms <= 0 or not len(audio):
        raise ValueError("audio duration and requested chunk interval must be positive")
    windows: list[dict[str, Any]] = []
    offset_ms = 0
    while offset_ms < requested_duration_ms:
        end_ms = min(offset_ms + WINDOW_MS, requested_duration_ms)
        start_sample = offset_ms * SAMPLE_RATE_HZ // 1000
        end_sample = min(len(audio), end_ms * SAMPLE_RATE_HZ // 1000)
        values = audio[start_sample:end_sample]
        padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        padded[: min(len(values), WINDOW_SAMPLES)] = values[:WINDOW_SAMPLES]
        inputs = processor(
            padded,
            sampling_rate=SAMPLE_RATE_HZ,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        model_inputs = {
            key: value.to("cuda", dtype=torch.float16)
            if value.is_floating_point()
            else value.to("cuda")
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**model_inputs).logits).squeeze(0).float()
        if probabilities.ndim != 1 or probabilities.numel() != len(LABELS):
            raise ValueError("AudioSet AST returned an invalid class vector")
        if not torch.isfinite(probabilities).all():
            raise ValueError("AudioSet AST returned non-finite probabilities")
        windows.append(
            {
                "start_ms": chunk_start_ms + offset_ms,
                "end_ms": chunk_start_ms + end_ms,
                "scores": [
                    {
                        "class_id": class_id,
                        "label": LABELS[class_id],
                        "probability": float(probabilities[class_id]),
                    }
                    for class_id in RELEVANT_IDS
                ],
            }
        )
        offset_ms = end_ms
    return windows


def _classify(audio: np.ndarray, chunk_start_ms: int, chunk_end_ms: int) -> dict[str, Any]:
    windows = _windows(audio, chunk_start_ms, chunk_end_ms)
    return {
        "model": MODEL_SOURCE,
        "model_revision": MODEL_REVISION,
        "windows": windows,
        "signal_measurements": _signal_measurements(audio),
        "label_map_sha256": LABEL_MAP_SHA256,
        "label_map_version": f"config.json:id2label@{MODEL_REVISION}",
        "label_count": len(LABELS),
        "relevant_class_count": len(RELEVANT_IDS),
        "preprocessing_sha256": PREPROCESSING_SHA256,
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 1,
            "window_ms": WINDOW_MS,
            "padding": "zero-pad-final-window-to-10s",
            "feature_extractor": "ASTFeatureExtractor",
            "feature_extractor_max_length": 1024,
        },
    }


def infer(audio_path: Path) -> dict[str, object]:
    """Run one real Opus-derived smoke classification."""

    audio = _decode_audio(audio_path.read_bytes())
    result = _classify(audio, 0, len(audio) * 1000 // SAMPLE_RATE_HZ)
    return {
        "shape": [len(result["windows"]), result["relevant_class_count"]],
        "windows": len(result["windows"]),
        "label_count": result["label_count"],
        "relevant_class_count": result["relevant_class_count"],
        "signal_measurements": result["signal_measurements"],
        "preprocessing": result["preprocessing"],
    }


def classify_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Classify one canonical chunk through the internal JSON boundary."""

    input_audio = request.get("input_audio")
    chunk = request.get("chunk")
    if not isinstance(input_audio, Mapping) or not isinstance(chunk, Mapping):
        raise ValueError("input_audio and chunk must be objects")
    encoded = input_audio.get("data")
    if not isinstance(encoded, str):
        raise ValueError("input_audio.data must be base64 text")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("input_audio.data is not valid base64") from exc
    start_ms = chunk.get("start_ms")
    end_ms = chunk.get("end_ms")
    if (
        isinstance(start_ms, bool)
        or not isinstance(start_ms, int)
        or isinstance(end_ms, bool)
        or not isinstance(end_ms, int)
        or end_ms <= start_ms
    ):
        raise ValueError("chunk must contain valid start_ms/end_ms")
    return {
        "chunk_id": request.get("chunk_id"),
        **_classify(_decode_audio(audio_bytes), start_ms, end_ms),
    }


if __name__ == "__main__":
    run_service(
        "audioset_ast",
        infer,
        classify_request,
        post_path="/v1/audio/classify",
        max_request_bytes=40_000_000,
    )

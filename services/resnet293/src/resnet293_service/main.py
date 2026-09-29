"""Ordered WeSpeaker ResNet293-LM embedding service."""

from __future__ import annotations

import base64
import binascii
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import wespeaker
from smoke_server import run_service

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_SOURCE = os.environ["MODEL_SOURCE"]
MODEL_REVISION = os.environ["MODEL_REVISION"]
DIMENSIONS = 256
PREPROCESSING = "mono 16 kHz audio, 80-bin Kaldi fbank and cepstral mean normalization"
model = wespeaker.load_model(MODEL_PATH)
model.set_device("cuda:0")


def _vector(pcm: torch.Tensor, sample_rate: int) -> list[float]:
    """Return one original, validated model vector."""

    with torch.inference_mode():
        value = model.extract_embedding_from_pcm(pcm, sample_rate)
    tensor = torch.as_tensor(value).detach().cpu()
    if (
        tensor.shape != (DIMENSIONS,)
        or not torch.isfinite(tensor).all()
        or torch.linalg.vector_norm(tensor) == 0
    ):
        raise ValueError("ResNet293-LM returned an invalid 256-dimensional embedding")
    return tensor.tolist()


def _audio(encoded: str) -> tuple[torch.Tensor, int]:
    """Decode one mono WAV excerpt without changing its source samples."""

    try:
        data = base64.b64decode(encoded, validate=True)
        pcm, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except (binascii.Error, ValueError, OSError, RuntimeError) as exc:
        raise ValueError("excerpt must contain base64 WAV audio") from exc
    if pcm.ndim != 1 or not len(pcm) or not torch.isfinite(torch.from_numpy(pcm)).all():
        raise ValueError("excerpt must contain finite mono audio")
    return torch.from_numpy(pcm).unsqueeze(0), int(sample_rate)


def infer(audio_path: Path) -> dict[str, object]:
    """Smoke the locked model on one real clean excerpt."""

    pcm, sample_rate = sf.read(audio_path, dtype="float32")
    if pcm.ndim != 1:
        raise ValueError("smoke audio must be mono")
    vector = _vector(torch.from_numpy(pcm).unsqueeze(0), int(sample_rate))
    return {
        "shape": [len(vector)],
        "output": "original finite ResNet293-LM embedding",
        "preprocessing": PREPROCESSING,
        "embedding_norm": float(torch.linalg.vector_norm(torch.tensor(vector))),
    }


def embed_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Embed requested excerpts in order with the locked model identity."""

    if request.get("model") != MODEL_SOURCE or request.get("model_revision") != MODEL_REVISION:
        raise ValueError("speaker embedding model does not match the locked checkpoint")
    items = request.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty array")
    item_ids: list[str] = []
    vectors: list[list[float]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("each excerpt must be an object")
        item_id = item.get("item_id")
        input_audio = item.get("input_audio")
        if not isinstance(item_id, str) or not item_id or item_id in item_ids:
            raise ValueError("excerpt IDs must be non-empty and unique")
        if not isinstance(input_audio, Mapping) or input_audio.get("format") != "wav":
            raise ValueError("excerpt input_audio must be WAV")
        encoded = input_audio.get("data")
        if not isinstance(encoded, str):
            raise ValueError("excerpt input_audio.data must be base64 text")
        pcm, sample_rate = _audio(encoded)
        vectors.append(_vector(pcm, sample_rate))
        item_ids.append(item_id)
    return {
        "model": MODEL_SOURCE,
        "model_revision": MODEL_REVISION,
        "preprocessing": PREPROCESSING,
        "item_ids": item_ids,
        "vectors": vectors,
    }


if __name__ == "__main__":
    run_service("resnet293", infer, embed_request, post_path="/v1/embed-speakers")

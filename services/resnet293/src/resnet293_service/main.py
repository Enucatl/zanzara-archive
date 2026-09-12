"""Real WeSpeaker ResNet293-LM inference smoke."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import wespeaker
from smoke_server import run_service

MODEL_PATH = os.environ["MODEL_PATH"]
model = wespeaker.load_model(MODEL_PATH)
model.set_device("cuda:0")


def infer(audio_path: Path) -> dict[str, object]:
    embedding = model.extract_embedding(str(audio_path))
    tensor = torch.as_tensor(embedding)
    if (
        tensor.ndim != 1
        or not torch.isfinite(tensor).all()
        or torch.linalg.vector_norm(tensor) == 0
    ):
        raise ValueError("ResNet293-LM returned an invalid embedding")
    return {"shape": list(tensor.shape), "output": "L2-valid ResNet293-LM embedding"}


if __name__ == "__main__":
    run_service("resnet293", infer)

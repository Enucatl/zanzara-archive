"""Real Parakeet TDT inference smoke."""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from smoke_server import run_service
from transformers import AutoModelForTDT, AutoProcessor

MODEL_PATH = os.environ["MODEL_PATH"]
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForTDT.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")
model.eval()


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim != 1:
        raise ValueError("smoke audio must be mono after preprocessing")
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
    text, timestamps = processor.decode(generated.sequences, durations=generated.durations)
    words = timestamps[0] if timestamps and isinstance(timestamps[0], list) else timestamps
    return {
        "shape": [len(text), len(words)],
        "output": "text plus genuine word/segment timestamp decode",
        "word_timestamp_count": len(words),
    }


if __name__ == "__main__":
    run_service("parakeet", infer)

"""Real pyannote Community-1 standard and exclusive diarization smoke."""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from smoke_server import run_service

MODEL_PATH = os.environ["MODEL_PATH"]
pipeline = Pipeline.from_pretrained(MODEL_PATH)
pipeline.to(torch.device("cuda"))


def _turn_count(annotation: object) -> int:
    return sum(1 for _ in annotation.itertracks(yield_label=True))  # type: ignore[attr-defined]


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(audio).reshape(1, -1)
    with torch.inference_mode():
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    standard = result.speaker_diarization
    exclusive = result.exclusive_speaker_diarization
    return {
        "shape": [_turn_count(standard), _turn_count(exclusive)],
        "output": "standard turns, exclusive turns, and overlap-capable annotation",
        "standard_turns": _turn_count(standard),
        "exclusive_turns": _turn_count(exclusive),
        "speaker_count": len(set(label for _, _, label in standard.itertracks(yield_label=True))),
    }


if __name__ == "__main__":
    run_service("diarization", infer)

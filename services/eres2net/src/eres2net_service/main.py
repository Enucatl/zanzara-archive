"""Real ERes2Net v1.0.3 inference using the pinned 3D-Speaker source."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, "/opt/3D-Speaker")
from smoke_server import run_service
from speakerlab.models.eres2net.ERes2Net import ERes2Net  # noqa: E402
from speakerlab.process.processor import FBank  # noqa: E402

MODEL_PATH = Path(os.environ["MODEL_PATH"])
checkpoint = torch.load(
    MODEL_PATH / "pretrained_eres2net.ckpt",
    map_location="cpu",
    weights_only=False,
)
model = ERes2Net(feat_dim=80, embedding_size=192)
model.load_state_dict(checkpoint, strict=True)
model.to("cuda").eval()
feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)


def infer(audio_path: Path) -> dict[str, object]:
    wav, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != 16000:
        raise ValueError(f"ERes2Net smoke expects 16 kHz, got {sample_rate}")
    feature = feature_extractor(wav).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        embedding = model(feature).squeeze(0)
    if embedding.ndim != 1 or embedding.numel() != 192 or not torch.isfinite(embedding).all():
        raise ValueError("ERes2Net returned an invalid embedding")
    return {"shape": list(embedding.shape), "output": "finite 192-dimensional embedding"}


if __name__ == "__main__":
    run_service("eres2net", infer)

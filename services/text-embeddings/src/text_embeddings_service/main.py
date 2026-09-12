"""Real BGE-M3 dense embedding smoke."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from smoke_server import run_service
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.environ["MODEL_PATH"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")
model.eval()


def infer(_audio_path: Path) -> dict[str, object]:
    inputs = tokenizer(
        ["un esempio di ricerca testuale nell'archivio"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
        embedding = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze(0)
    if embedding.ndim != 1 or embedding.numel() != 1024 or not torch.isfinite(embedding).all():
        raise ValueError("BGE-M3 returned an invalid dense embedding")
    return {
        "shape": list(embedding.shape),
        "output": "finite normalized 1024-dimensional dense embedding",
    }


if __name__ == "__main__":
    run_service("text_embeddings", infer)

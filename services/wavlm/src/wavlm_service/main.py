"""Real UniSpeech fine-tuned WavLM Large plus verification-head smoke."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, "/opt/UniSpeech/WavLM")
from WavLM import WavLM, WavLMConfig  # noqa: E402

sys.path.insert(0, "/opt/wavlm-runtime")
from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # noqa: E402
from smoke_server import run_service

MODEL_PATH = Path(os.environ["MODEL_PATH"])


class LocalWavLMUpstream(torch.nn.Module):
    """Adapt the released standalone WavLM source to the UniSpeech head contract."""

    def __init__(self) -> None:
        super().__init__()
        config = WavLMConfig(
            {
                "extractor_mode": "layer_norm",
                "encoder_layers": 24,
                "encoder_embed_dim": 1024,
                "encoder_ffn_embed_dim": 4096,
                "encoder_attention_heads": 16,
                "relative_position_embedding": True,
                "gru_rel_pos": True,
            }
        )
        self.model = WavLM(config)

    def forward(self, wavs: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        lengths = torch.tensor([sample.numel() for sample in wavs], device=wavs[0].device)
        padded = pad_sequence(wavs, batch_first=True)
        padding_mask = ~torch.lt(
            torch.arange(padded.shape[1], device=padded.device).unsqueeze(0), lengths.unsqueeze(1)
        )
        (_, layer_results), _ = self.model.extract_features(
            padded,
            padding_mask=padding_mask,
            mask=False,
            output_layer=24,
            ret_layer_results=True,
        )
        return {"hidden_states": [result[0].transpose(0, 1) for result in layer_results]}


original_hub_load = torch.hub.load
torch.hub.load = lambda *_args, **_kwargs: LocalWavLMUpstream()
try:
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024,
        emb_dim=256,
        feat_type="wavlm_large",
        feature_selection="hidden_states",
        update_extract=False,
    )
finally:
    torch.hub.load = original_hub_load

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
checkpoint_state = checkpoint["model"]
model_state = model.state_dict()
missing = sorted(set(model_state) - set(checkpoint_state))
unexpected = sorted(set(checkpoint_state) - set(model_state))
if missing or unexpected != ["loss_calculator.projection.weight"]:
    raise RuntimeError(
        f"WavLM checkpoint did not load exactly: missing={missing}, unexpected={unexpected}"
    )
model.load_state_dict(
    {key: value for key, value in checkpoint_state.items() if key in model_state},
    strict=True,
)
model.to("cuda").eval()


def infer(audio_path: Path) -> dict[str, object]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"WavLM smoke expects 16 kHz, got {sample_rate}")
    waveform = torch.from_numpy(audio).to("cuda").unsqueeze(0)
    with torch.inference_mode():
        embedding = model(waveform).squeeze(0)
    if embedding.ndim != 1 or embedding.numel() != 256 or not torch.isfinite(embedding).all():
        raise ValueError("WavLM verification head returned an invalid embedding")
    return {
        "shape": list(embedding.shape),
        "output": "finite 256-dimensional verification-head embedding",
    }


if __name__ == "__main__":
    run_service("wavlm", infer)

"""Synthetic contract checks for the ordered ResNet293 adapter."""

from __future__ import annotations

import json

import pytest

from zanzara_archive.contracts import AdapterFailure, AudioArtifact, ModelFingerprint
from zanzara_archive.inference import SpeakerEmbeddingAdapter

MODEL = ModelFingerprint(
    name="resnet293",
    repository="Wespeaker/wespeaker-voxceleb-resnet293-LM",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    dimensions=256,
    preprocessing={
        "description": "mono 16 kHz audio, 80-bin Kaldi fbank and cepstral mean normalization"
    },
)
EXCERPTS = (
    AudioArtifact("first", "c" * 64, "wav", 4_000, 16_000, 1, time_origin_ms=7_000),
    AudioArtifact("second", "d" * 64, "wav", 4_000, 16_000, 1, time_origin_ms=12_000),
)


def test_adapter_preserves_order_offsets_and_original_vectors() -> None:
    """Map the ordered response to the request without normalizing values."""

    def respond(url: str, body: bytes, timeout: float) -> bytes:
        request = json.loads(body)
        assert url.endswith("/v1/embed-speakers")
        assert timeout > 0
        assert [item["item_id"] for item in request["items"]] == ["first", "second"]
        assert [item["time_origin_ms"] for item in request["items"]] == [7_000, 12_000]
        assert [item["source_sha256"] for item in request["items"]] == ["c" * 64, "d" * 64]
        return json.dumps(
            {
                "request_id": "req-1",
                "status": "ok",
                "data": {
                    "model": MODEL.repository,
                    "model_revision": MODEL.revision,
                    "preprocessing": MODEL.preprocessing["description"],
                    "item_ids": ["first", "second"],
                    "vectors": [[2.0] + [0.0] * 255, [0.0, 3.0] + [0.0] * 254],
                },
            }
        ).encode()

    adapter = SpeakerEmbeddingAdapter(
        "http://127.0.0.1:18084", MODEL, lambda _item: b"WAV", http_post=respond
    )
    batch = adapter.embed(EXCERPTS, request_id="req-1")
    assert batch.item_ids == ("first", "second")
    assert batch.vectors[0][0] == 2.0
    assert batch.vectors[1][1] == 3.0
    assert batch.model == MODEL


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model_revision": "wrong"}, "locked model"),
        ({"item_ids": ["second", "first"]}, "order"),
        ({"vectors": [[1.0] * 255, [1.0] * 256]}, "dimension"),
        ({"vectors": [[float("nan")] * 256, [1.0] * 256]}, "finite"),
        ({"vectors": [[0.0] * 256, [1.0] * 256]}, "non-zero"),
    ],
)
def test_adapter_rejects_invalid_response(change: dict[str, object], message: str) -> None:
    """Reject misidentified, reordered, or unusable service output."""

    response = {
        "model": MODEL.repository,
        "model_revision": MODEL.revision,
        "preprocessing": MODEL.preprocessing["description"],
        "item_ids": ["first", "second"],
        "vectors": [[1.0] * 256, [1.0] * 256],
    }
    response.update(change)
    adapter = SpeakerEmbeddingAdapter(
        "http://127.0.0.1:18084",
        MODEL,
        lambda _item: b"WAV",
        http_post=lambda *_args: json.dumps(response).encode(),
    )
    with pytest.raises(AdapterFailure, match=message):
        adapter.embed(EXCERPTS, request_id="req-1")

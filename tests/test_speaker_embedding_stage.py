"""Focused publication check for the immutable ResNet293 stage."""

from __future__ import annotations

import json
from types import SimpleNamespace

from zanzara_archive import cli, inference
from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.contracts import EmbeddingBatch, ModelFingerprint


def test_speaker_embedding_stage_preserves_order_and_vectors(tmp_path, monkeypatch) -> None:
    """Batch bounded WAVs and publish source offsets with raw model vectors."""

    source_sha256 = "a" * 64
    episode = SimpleNamespace(
        relative_filename="episode.wav", sha256=source_sha256, duration_ms=30_000
    )
    corpus = SimpleNamespace(episodes=(episode,), sha256="b" * 64)
    model = ModelFingerprint(
        name="resnet293",
        repository="example/resnet293",
        revision="c" * 40,
        checkpoint_sha256=("d" * 64,),
        dimensions=256,
        preprocessing={"description": "mono WAV"},
    )
    monkeypatch.setattr(cli, "load_manifest", lambda _path: corpus)
    monkeypatch.setattr(cli, "model_fingerprint_from_lock", lambda *_args: model)
    monkeypatch.setattr(inference, "MAX_SPEAKER_EMBEDDING_PAYLOAD_BYTES", 8)
    calls: list[list[str]] = []

    class Adapter:
        """Return recognizable, unnormalized vectors for each ordered batch."""

        def __init__(self, _endpoint, _model, audio_loader):
            self.audio_loader = audio_loader

        def embed(self, items, *, request_id):
            assert request_id.startswith("resnet293-")
            assert all(self.audio_loader(item) == b"WAVE" for item in items)
            calls.append([item.artifact_id for item in items])
            return EmbeddingBatch(
                model,
                tuple(item.artifact_id for item in items),
                tuple((float(item.time_origin_ms),) + (0.0,) * 255 for item in items),
                request_id,
            )

    monkeypatch.setattr(inference, "SpeakerEmbeddingAdapter", Adapter)
    root = tmp_path / "artifacts"
    publisher = ArtifactPublisher(root)
    exemplar_payload = {
        "episode_id": episode.relative_filename,
        "source_sha256": source_sha256,
        "speakers": {
            "SPEAKER_00": {
                "status": "voice_searchable",
                "excerpts": [
                    {"start_ms": 1_000, "end_ms": 5_000, "file": "audio/one.wav"},
                    {"start_ms": 6_000, "end_ms": 10_000, "file": "audio/two.wav"},
                    {"start_ms": 11_000, "end_ms": 15_000, "file": "audio/three.wav"},
                ],
            },
            "SPEAKER_01": {"status": "not_voice_searchable", "excerpts": []},
        },
    }
    upstream = publisher.publish(
        source_sha256=source_sha256,
        stage="exemplars",
        stage_key="exemplar-key",
        files={
            "exemplars.json": json.dumps(exemplar_payload).encode(),
            "audio/one.wav": b"WAVE",
            "audio/two.wav": b"WAVE",
            "audio/three.wav": b"WAVE",
        },
    )
    args = SimpleNamespace(
        manifest="unused.json",
        episode=episode.relative_filename,
        artifact_root=str(root),
        exemplars_artifact=str(publisher.artifact_path(source_sha256, "exemplars", "exemplar-key")),
        model_lock="unused.json",
        endpoint=None,
        speaker_embedding_endpoint="http://127.0.0.1:18084",
    )
    result = cli._process_speaker_embeddings_resnet293_stage(args)
    payload = json.loads(
        (
            publisher.root
            / source_sha256
            / result["stage"]
            / result["artifact"]["stage_key"]
            / "embeddings.json"
        ).read_text()
    )
    excerpts = payload["speakers"]["SPEAKER_00"]["excerpts"]
    assert [len(batch) for batch in calls] == [2, 1]
    assert [item["exemplar_id"] for item in excerpts] == calls[0] + calls[1]
    assert [item["start_ms"] for item in excerpts] == [1_000, 6_000, 11_000]
    assert [item["vector"][0] for item in excerpts] == [1_000.0, 6_000.0, 11_000.0]
    assert payload["model"] == model.to_dict()
    assert payload["exemplars_artifact_id"] == upstream.artifact_id
    assert payload["speakers"]["SPEAKER_01"]["excerpts"] == []
    assert result["batch_count"] == 2

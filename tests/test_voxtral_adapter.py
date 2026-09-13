"""CPU contract checks for the local Voxtral text-first chunk adapter."""

import base64
import json

import pytest

from zanzara_archive.contracts import AdapterFailure, AudioChunk, ModelFingerprint
from zanzara_archive.inference import (
    VOXTRAL_DECODING_SETTINGS,
    VoxtralAdapter,
    parse_voxtral_response,
)

MODEL = ModelFingerprint(
    name="voxtral",
    repository="mistralai/Voxtral-Mini-4B-Realtime-2602",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    precision="bfloat16",
    runtime={"transformers": "test"},
)
CHUNK = AudioChunk.create(
    episode_id="episode-p1r",
    source_sha256="c" * 64,
    start_ms=1_000,
    end_ms=11_000,
    segmentation_fingerprint="d" * 64,
    duration_ms=20_000,
    partition="development",
)


def test_parse_voxtral_response_preserves_text_first_provenance() -> None:
    result = parse_voxtral_response(
        {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "text": "ciao mondo",
            "segments": [],
            "language": "it",
            "decoding": dict(VOXTRAL_DECODING_SETTINGS),
            "preprocessing": {"sample_rate_hz": 16_000, "channels": 1},
            "raw_generation": {"sequence_token_ids": [1, 2, 3]},
        },
        chunk=CHUNK,
        model=MODEL,
        request_id="voxtral-request-1",
    )

    assert result.chunk_id == CHUNK.chunk_id
    assert result.source_sha256 == CHUNK.source_sha256
    assert result.text == "ciao mondo"
    assert result.words == ()
    assert result.timestamp_granularities == ()
    assert result.raw_metadata["language"] == "it"


def test_adapter_sends_identical_chunk_identity_and_locked_decoding() -> None:
    seen: dict[str, object] = {}

    def post(url: str, body: bytes, timeout: float) -> bytes:
        seen.update(url=url, timeout=timeout, payload=json.loads(body))
        return json.dumps(
            {
                "request_id": "voxtral-request-2",
                "status": "ok",
                "data": {
                    "model": MODEL.repository,
                    "model_revision": MODEL.revision,
                    "text": "ciao",
                    "segments": [],
                    "language": "it",
                    "decoding": dict(VOXTRAL_DECODING_SETTINGS),
                },
            }
        ).encode()

    parsed = VoxtralAdapter(
        "http://voxtral",
        MODEL,
        http_post=post,
        timeout_seconds=9,
    ).transcribe_bytes(CHUNK, b"wav", request_id="voxtral-request-2")

    payload = seen["payload"]
    assert seen["url"] == "http://voxtral/v1/audio/transcriptions"
    assert seen["timeout"] == 9
    assert isinstance(payload, dict)
    assert payload["model"] == MODEL.repository
    assert payload["chunk_id"] == CHUNK.chunk_id
    assert payload["language"] == "it"
    assert payload["decoding"] == dict(VOXTRAL_DECODING_SETTINGS)
    assert base64.b64decode(payload["input_audio"]["data"]) == b"wav"
    assert parsed.result.text == "ciao"


def test_adapter_rejects_non_italian_override() -> None:
    with pytest.raises(AdapterFailure, match="fixed to Italian"):
        VoxtralAdapter("http://voxtral", MODEL).transcribe_bytes(
            CHUNK,
            b"wav",
            language="en",
        )


def test_parser_rejects_model_revision_drift() -> None:
    with pytest.raises(AdapterFailure, match="revision"):
        parse_voxtral_response(
            {
                "model": MODEL.repository,
                "model_revision": "e" * 40,
                "text": "ciao",
            },
            chunk=CHUNK,
            model=MODEL,
            request_id="voxtral-request-3",
        )

"""CPU contract checks for the local Whisper Large v3 chunk adapter."""

import base64
import json

import pytest

from zanzara_archive.contracts import AdapterFailure, AudioChunk, ModelFingerprint
from zanzara_archive.inference import (
    WHISPER_DECODING_SETTINGS,
    WhisperAdapter,
    parse_whisper_response,
)

MODEL = ModelFingerprint(
    name="whisper",
    repository="openai/whisper-large-v3",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    precision="float16",
    runtime={"torch": "test"},
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


def test_parse_whisper_response_preserves_text_only_provenance() -> None:
    result = parse_whisper_response(
        {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "text": "ciao mondo",
            "segments": [],
            "language": "it",
            "task": "transcribe",
            "decoding": dict(WHISPER_DECODING_SETTINGS),
            "preprocessing": {"sample_rate_hz": 16_000, "channels": 1},
        },
        chunk=CHUNK,
        model=MODEL,
        request_id="whisper-request-1",
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
                "request_id": "whisper-request-2",
                "status": "ok",
                "data": {
                    "model": MODEL.repository,
                    "model_revision": MODEL.revision,
                    "text": "ciao",
                    "segments": [],
                    "language": "it",
                    "task": "transcribe",
                    "decoding": dict(WHISPER_DECODING_SETTINGS),
                },
            }
        ).encode()

    parsed = WhisperAdapter(
        "http://whisper",
        MODEL,
        http_post=post,
        timeout_seconds=9,
    ).transcribe_bytes(CHUNK, b"wav", request_id="whisper-request-2")

    payload = seen["payload"]
    assert seen["url"] == "http://whisper/v1/audio/transcriptions"
    assert seen["timeout"] == 9
    assert isinstance(payload, dict)
    assert payload["model"] == MODEL.repository
    assert payload["chunk_id"] == CHUNK.chunk_id
    assert payload["language"] == "it"
    assert payload["task"] == "transcribe"
    assert payload["decoding"] == dict(WHISPER_DECODING_SETTINGS)
    assert base64.b64decode(payload["input_audio"]["data"]) == b"wav"
    assert parsed.result.text == "ciao"


def test_adapter_rejects_non_italian_override() -> None:
    with pytest.raises(AdapterFailure, match="fixed to Italian"):
        WhisperAdapter("http://whisper", MODEL).transcribe_bytes(
            CHUNK,
            b"wav",
            language="en",
        )


def test_parser_rejects_model_revision_drift() -> None:
    with pytest.raises(AdapterFailure, match="revision"):
        parse_whisper_response(
            {
                "model": MODEL.repository,
                "model_revision": "e" * 40,
                "text": "ciao",
            },
            chunk=CHUNK,
            model=MODEL,
            request_id="whisper-request-3",
        )

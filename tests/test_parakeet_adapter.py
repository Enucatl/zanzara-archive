"""CPU contract checks for the local Parakeet adapter."""

import json

import niquests
import pytest

from zanzara_archive.contracts import (
    AudioArtifact,
    AudioChunk,
    ModelFingerprint,
    UnsupportedCapabilityError,
)
from zanzara_archive.inference import (
    PARAKEET_CHUNK_CONFIGURATION,
    PARAKEET_CHUNK_PREPROCESSING,
    ParakeetAdapter,
    ParakeetChunkAdapter,
    _default_http_post,
    parse_parakeet_chunk_response,
    parse_parakeet_response,
)
from zanzara_archive.p1r_asr import score_text_pair

MODEL = ModelFingerprint(
    name="parakeet",
    repository="nvidia/parakeet-tdt-0.6b-v3",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    runtime={"torch": "test"},
)
AUDIO = AudioArtifact(
    artifact_id="audio",
    source_sha256="c" * 64,
    format="wav",
    duration_ms=2_000,
    sample_rate_hz=16_000,
    channels=1,
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


def test_parse_converts_decimal_seconds_to_integer_milliseconds() -> None:
    parsed = parse_parakeet_response(
        {
            "text": "ciao mondo",
            "words": [
                {"word": "ciao", "start": "0.125", "end": "0.500"},
                {"word": "mondo", "start": 0.501, "end": 1.25, "confidence": 0.9},
            ],
        },
        audio=AUDIO,
        model=MODEL,
        request_id="request-1",
    )
    assert [(word.text, word.start_ms, word.end_ms) for word in parsed.words] == [
        ("ciao", 125, 500),
        ("mondo", 501, 1250),
    ]
    assert parsed.require_production() is parsed


def test_missing_word_timestamps_are_an_explicit_unsupported_capability() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="word timestamps unavailable"):
        parse_parakeet_response(
            {"text": "ciao", "words": []},
            audio=AUDIO,
            model=MODEL,
            request_id="request-2",
        )


def test_chunk_parser_accepts_text_without_timestamps_and_keeps_provenance() -> None:
    result = parse_parakeet_chunk_response(
        {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "chunk_id": CHUNK.chunk_id,
            "text": "ciao mondo",
            "words": [],
            "segments": [],
            "timestamp_granularities": [],
            "preprocessing": dict(PARAKEET_CHUNK_PREPROCESSING),
            "configuration": dict(PARAKEET_CHUNK_CONFIGURATION),
        },
        chunk=CHUNK,
        model=MODEL,
        request_id="chunk-request-1",
    )

    assert result.text == "ciao mondo"
    assert result.words == ()
    assert result.source_sha256 == CHUNK.source_sha256
    assert result.raw_metadata["chunk"]["start_ms"] == CHUNK.start_ms
    assert result.raw_metadata["preprocessing"] == dict(PARAKEET_CHUNK_PREPROCESSING)
    assert score_text_pair("ciao mondo", result.text)["normalized_wer"]["value"] == 0.0


def test_chunk_parser_retains_optional_native_words() -> None:
    result = parse_parakeet_chunk_response(
        {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "chunk_id": CHUNK.chunk_id,
            "text": "ciao mondo",
            "words": [
                {"word": "ciao", "start": 0.125, "end": 0.5},
                {"word": "mondo", "start": 0.501, "end": 1.25},
            ],
            "timestamp_granularities": ["word"],
            "preprocessing": dict(PARAKEET_CHUNK_PREPROCESSING),
            "configuration": dict(PARAKEET_CHUNK_CONFIGURATION),
        },
        chunk=CHUNK,
        model=MODEL,
        request_id="chunk-request-2",
    )

    assert [(word.text, word.start_ms, word.end_ms) for word in result.words] == [
        ("ciao", 125, 500),
        ("mondo", 501, 1250),
    ]
    assert result.timestamp_granularities == ("word",)
    assert result.raw_metadata["native_word_count"] == 2


def test_adapter_sends_the_bounded_local_json_subset() -> None:
    seen: dict[str, object] = {}

    def post(url: str, body: bytes, timeout: float) -> bytes:
        seen.update(url=url, timeout=timeout, payload=json.loads(body))
        return json.dumps(
            {
                "request_id": "request-3",
                "status": "ok",
                "data": {
                    "text": "ciao",
                    "words": [{"word": "ciao", "start": 0, "end": 0.25}],
                    "segments": [],
                    "timestamp_granularities": ["word"],
                },
            }
        ).encode()

    parsed = ParakeetAdapter(
        "http://parakeet",
        MODEL,
        http_post=post,
        timeout_seconds=7,
    ).transcribe_bytes(AUDIO, b"wav", request_id="request-3")
    payload = seen["payload"]
    assert seen["url"] == "http://parakeet/v1/audio/transcriptions"
    assert seen["timeout"] == 7
    assert isinstance(payload, dict)
    assert payload["model"] == MODEL.repository
    assert payload["response_format"] == "verbose_json"
    assert payload["timestamp_granularities"] == ["word", "segment"]
    assert parsed.result.words[0].end_ms == 250


def test_chunk_adapter_sends_interval_and_returns_raw_response() -> None:
    seen: dict[str, object] = {}

    def post(url: str, body: bytes, timeout: float) -> bytes:
        seen.update(url=url, timeout=timeout, payload=json.loads(body))
        return json.dumps(
            {
                "request_id": "chunk-request-3",
                "status": "ok",
                "data": {
                    "model": MODEL.repository,
                    "model_revision": MODEL.revision,
                    "chunk_id": CHUNK.chunk_id,
                    "text": "ciao",
                    "words": [],
                    "segments": [],
                    "timestamp_granularities": [],
                    "preprocessing": dict(PARAKEET_CHUNK_PREPROCESSING),
                    "configuration": dict(PARAKEET_CHUNK_CONFIGURATION),
                },
            }
        ).encode()

    parsed = ParakeetChunkAdapter(
        "http://parakeet",
        MODEL,
        http_post=post,
        timeout_seconds=9,
    ).transcribe_bytes(CHUNK, b"wav", request_id="chunk-request-3")

    payload = seen["payload"]
    assert seen["url"] == "http://parakeet/v1/audio/transcriptions"
    assert seen["timeout"] == 9
    assert isinstance(payload, dict)
    assert payload["model"] == MODEL.repository
    assert payload["chunk_id"] == CHUNK.chunk_id
    assert payload["chunk"]["start_ms"] == CHUNK.start_ms
    assert payload["chunk"]["end_ms"] == CHUNK.end_ms
    assert payload["preprocessing"] == dict(PARAKEET_CHUNK_PREPROCESSING)
    assert parsed.raw_response["status"] == "ok"
    assert parsed.result.text == "ciao"


def test_default_http_post_uses_niquests_streaming_and_preserves_error_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Response:
        status_code = 503

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            seen["closed"] = True

        def iter_content(self, *, chunk_size: int):
            seen["chunk_size"] = chunk_size
            yield b'{"status":"error"}'

    def post(url: str, **kwargs: object) -> Response:
        seen.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(niquests, "post", post)

    body = _default_http_post("http://inference/v1", b"{}", 7.5)

    assert body == b'{"status":"error"}'
    assert seen == {
        "url": "http://inference/v1",
        "data": b"{}",
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "timeout": 7.5,
        "allow_redirects": True,
        "verify": True,
        "stream": True,
        "retries": 0,
        "chunk_size": 64 * 1024,
        "closed": True,
    }


def test_default_http_post_maps_niquests_timeout_to_adapter_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(*_args: object, **_kwargs: object) -> None:
        raise niquests.exceptions.Timeout("synthetic timeout")

    monkeypatch.setattr(niquests, "post", post)

    with pytest.raises(TimeoutError, match="synthetic timeout"):
        _default_http_post("http://inference/v1", b"{}", 2)

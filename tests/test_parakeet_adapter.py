"""CPU contract checks for the local Parakeet adapter."""

import json

import niquests
import pytest

from zanzara_archive.contracts import AudioArtifact, ModelFingerprint, UnsupportedCapabilityError
from zanzara_archive.inference import (
    ParakeetAdapter,
    _default_http_post,
    parse_parakeet_response,
)

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

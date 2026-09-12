"""CPU contract checks for the Community-1 diarization adapter."""

from __future__ import annotations

import json

import pytest

from zanzara_archive.contracts import AdapterFailure, AudioArtifact, ModelFingerprint
from zanzara_archive.inference import (
    DiarizerAdapter,
    derive_overlap_intervals,
    parse_diarization_response,
    render_rttm,
)

MODEL = ModelFingerprint(
    name="diarization",
    repository="pyannote/speaker-diarization-community-1",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    runtime={"torch": "test"},
)
AUDIO = AudioArtifact(
    artifact_id="episode-diarization",
    source_sha256="c" * 64,
    format="wav",
    duration_ms=5_000,
    sample_rate_hz=16_000,
    channels=1,
)


def _response() -> dict[str, object]:
    return {
        "request_id": "request-1",
        "status": "ok",
        "data": {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "standard_turns": [
                {"speaker_id": "SPEAKER_00", "start": "0.000", "end": "1.000"},
                {"speaker_id": "SPEAKER_01", "start": "0.500", "end": "1.500"},
                {"speaker_id": "SPEAKER_00", "start": "2.000", "end": "3.000"},
            ],
            "exclusive_turns": [
                {"speaker_id": "SPEAKER_00", "start": "0.000", "end": "0.500"},
                {"speaker_id": "SPEAKER_01", "start": "0.500", "end": "1.500"},
                {"speaker_id": "SPEAKER_00", "start": "2.000", "end": "3.000"},
            ],
        },
    }


def test_parser_retains_both_views_and_derives_all_overlap_speakers() -> None:
    result = parse_diarization_response(
        _response(), audio=AUDIO, model=MODEL, request_id="request-1"
    )

    assert [(turn.speaker_id, turn.start_ms, turn.end_ms) for turn in result.standard_turns] == [
        ("SPEAKER_00", 0, 1_000),
        ("SPEAKER_01", 500, 1_500),
        ("SPEAKER_00", 2_000, 3_000),
    ]
    assert [(turn.speaker_id, turn.start_ms, turn.end_ms) for turn in result.exclusive_turns] == [
        ("SPEAKER_00", 0, 500),
        ("SPEAKER_01", 500, 1_500),
        ("SPEAKER_00", 2_000, 3_000),
    ]
    assert [overlap.to_dict() for overlap in result.overlaps] == [
        {"speaker_ids": ["SPEAKER_00", "SPEAKER_01"], "start_ms": 500, "end_ms": 1_000}
    ]


def test_synthetic_overlap_boundaries_and_rttm_use_millisecond_source_offsets() -> None:
    result = parse_diarization_response(
        {
            "standard_turns": [
                {"speaker_id": "A", "start_ms": 0, "end_ms": 1_001},
                {"speaker_id": "B", "start": "1.000", "end": "2.250"},
            ],
            "exclusive_turns": [
                {"speaker_id": "A", "start_ms": 0, "end_ms": 1_000},
                {"speaker_id": "B", "start_ms": 1_000, "end_ms": 2_250},
            ],
        },
        audio=AUDIO,
        model=MODEL,
        request_id="request-2",
    )

    assert [(item.start_ms, item.end_ms, item.speaker_ids) for item in result.overlaps] == [
        (1_000, 1_001, ("A", "B"))
    ]
    assert render_rttm(result.standard_turns, file_id="golden") == (
        "SPEAKER golden 1 0.000 1.001 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER golden 1 1.000 1.250 <NA> <NA> B <NA> <NA>\n"
    )


def test_adapter_sends_full_representation_request_and_validates_response() -> None:
    seen: dict[str, object] = {}

    def post(url: str, body: bytes, timeout: float) -> bytes:
        seen.update(url=url, timeout=timeout, payload=json.loads(body))
        return json.dumps(_response()).encode()

    parsed = DiarizerAdapter(
        "http://diarization",
        MODEL,
        http_post=post,
        timeout_seconds=17,
    ).diarize_bytes(AUDIO, b"wav", request_id="request-1")

    payload = seen["payload"]
    assert seen["url"] == "http://diarization/v1/diarize"
    assert seen["timeout"] == 17
    assert isinstance(payload, dict)
    assert payload["model"] == MODEL.repository
    assert payload["output"] == {"standard": True, "exclusive": True, "overlaps": True}
    assert len(parsed.result.standard_turns) == 3


def test_missing_representation_is_a_typed_invalid_response() -> None:
    with pytest.raises(AdapterFailure, match="missing exclusive_turns"):
        parse_diarization_response(
            {
                "standard_turns": [],
            },
            audio=AUDIO,
            model=MODEL,
            request_id="request-3",
        )


def test_repeated_local_speaker_ids_remain_separate_episode_scoped_turns() -> None:
    result = parse_diarization_response(
        {
            "standard_turns": [
                {"speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 100},
                {"speaker_id": "SPEAKER_01", "start_ms": 100, "end_ms": 200},
                {"speaker_id": "SPEAKER_00", "start_ms": 200, "end_ms": 300},
            ],
            "exclusive_turns": [
                {"speaker_id": "SPEAKER_00", "start_ms": 0, "end_ms": 100},
                {"speaker_id": "SPEAKER_01", "start_ms": 100, "end_ms": 200},
                {"speaker_id": "SPEAKER_00", "start_ms": 200, "end_ms": 300},
            ],
        },
        audio=AUDIO,
        model=MODEL,
        request_id="request-4",
    )

    assert [turn.speaker_id for turn in result.standard_turns] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert derive_overlap_intervals(result.standard_turns) == ()

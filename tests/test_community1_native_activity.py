"""Contracts for the supported Community-1 speaker-count hook artifact."""

from __future__ import annotations

import json

from zanzara_archive.contracts import (
    AudioArtifact,
    ModelFingerprint,
    NativeActivityArtifact,
    NativeActivityInterval,
)
from zanzara_archive.inference import parse_diarization_response

MODEL = ModelFingerprint(
    name="diarization",
    repository="pyannote/speaker-diarization-community-1",
    revision="a" * 40,
    checkpoint_sha256=("b" * 64,),
    runtime={"torch": "test"},
)
AUDIO = AudioArtifact(
    artifact_id="episode-native",
    source_sha256="c" * 64,
    format="wav",
    duration_ms=1_000,
    sample_rate_hz=16_000,
    channels=1,
)


def _native() -> dict[str, object]:
    return {
        "artifact_id": "native-test",
        "source_sha256": AUDIO.source_sha256,
        "duration_ms": 1_000,
        "intervals": [
            {"start_ms": 0, "end_ms": 400, "speaker_count": 0},
            {"start_ms": 400, "end_ms": 1_000, "speaker_count": 1},
        ],
        "timeline": {
            "frame_count": 10,
            "frame_start_ms": 0,
            "frame_step_ms": 100,
            "frame_duration_ms": 100,
            "time_origin": "original episode",
        },
    }


def test_native_activity_round_trips_and_merges_runs() -> None:
    artifact = NativeActivityArtifact.from_dict(_native())
    assert artifact.intervals == (
        NativeActivityInterval(0, 400, 0),
        NativeActivityInterval(400, 1_000, 1),
    )
    assert json.loads(json.dumps(artifact.to_dict())) == artifact.to_dict()


def test_parser_retains_native_activity_from_one_diarization_response() -> None:
    result = parse_diarization_response(
        {
            "model": MODEL.repository,
            "model_revision": MODEL.revision,
            "standard_turns": [{"speaker_id": "A", "start_ms": 0, "end_ms": 1_000}],
            "exclusive_turns": [{"speaker_id": "A", "start_ms": 0, "end_ms": 1_000}],
            "native_activity": _native(),
        },
        audio=AUDIO,
        model=MODEL,
        request_id="native-request",
    )
    assert result.native_activity is not None
    assert result.native_activity.intervals[0].speaker_count == 0
    assert result.native_activity.to_dict()["timeline"]["time_origin"] == "original episode"

"""CPU checks for deterministic stage keys and D4 invalidation boundaries."""

from __future__ import annotations

import pytest

from zanzara_archive.stages import (
    compute_stage_key,
    invalidated_stages,
    reusable_stages,
)


def test_stage_key_canonicalizes_configuration_and_preserves_inputs() -> None:
    values = dict(
        stage="asr",
        source_sha256="a" * 64,
        upstream_artifact_hashes=("b" * 64,),
        model_fingerprint_sha256="c" * 64,
        pipeline_version="pipeline-1",
    )
    first = compute_stage_key(configuration={"b": 2, "a": 1}, **values)
    second = compute_stage_key(configuration={"a": 1, "b": 2}, **values)
    assert first == second
    assert first != compute_stage_key(configuration={"a": 2, "b": 1}, **values)


def test_asr_change_does_not_invalidate_voice_and_diarization_change_does() -> None:
    asr_invalidated = set(invalidated_stages(("asr",)))
    assert {"attribution", "chunks", "text_index"} <= asr_invalidated
    assert "exemplars" not in asr_invalidated
    assert "voice_indexes" not in asr_invalidated

    diarization_invalidated = set(invalidated_stages(("diarization",)))
    assert {"attribution", "exemplars", "voice_indexes", "candidates"} <= diarization_invalidated
    assert "text_index" in diarization_invalidated


def test_unknown_stages_and_invalid_hashes_fail() -> None:
    with pytest.raises(ValueError, match="unknown archive stage"):
        invalidated_stages(("unknown",))
    with pytest.raises(ValueError, match="lowercase SHA"):
        compute_stage_key("asr", source_sha256="not-a-hash")
    assert reusable_stages(("asr",), ("voice_indexes", "text_index")) == ("voice_indexes",)

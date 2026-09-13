"""CPU fixtures for deterministic model-independent P1R segmentation."""

from __future__ import annotations

import pytest

from zanzara_archive.chunking import (
    AcousticBoundary,
    ChunkSegmentationConfig,
    segment_chunks,
    segment_chunks_with_metadata,
    validate_chunk_coverage,
)
from zanzara_archive.contracts import ChunkBenchmarkManifest, ContractValidationError, Turn

SOURCE_SHA256 = "a" * 64


def test_priority_prefers_pause_then_speaker_turn_then_acoustic() -> None:
    chunks = segment_chunks(
        "episode-p1r-chunking",
        SOURCE_SHA256,
        40_000,
        vad_boundaries=(AcousticBoundary(12_000, "silence"),),
        speaker_turns=(Turn("SPEAKER_00", 0, 11_000),),
        acoustic_boundaries=(AcousticBoundary(13_000, "acoustic"),),
    )
    assert [(chunk.start_ms, chunk.end_ms) for chunk in chunks] == [
        (0, 12_000),
        (12_000, 40_000),
    ]

    speaker_chunks = segment_chunks(
        "episode-p1r-chunking",
        SOURCE_SHA256,
        40_000,
        speaker_turns=(Turn("SPEAKER_00", 0, 12_000),),
        acoustic_boundaries=(AcousticBoundary(11_000, "acoustic"),),
    )
    assert speaker_chunks[0].end_ms == 12_000


def test_hard_maximum_and_selected_region_coverage_are_enforced() -> None:
    result = segment_chunks_with_metadata(
        "episode-p1r-coverage",
        SOURCE_SHA256,
        75_000,
        selected_regions=((5_000, 70_000),),
    )
    assert all(chunk.end_ms - chunk.start_ms <= 30_000 for chunk in result.chunks)
    assert [(chunk.start_ms, chunk.end_ms) for chunk in result.chunks] == [
        (5_000, 35_000),
        (35_000, 62_000),
        (62_000, 70_000),
    ]
    validate_chunk_coverage(
        result.chunks,
        ((5_000, 70_000),),
        duration_ms=75_000,
    )


def test_segmentation_is_reproducible_and_fingerprints_inputs_and_configuration() -> None:
    kwargs = {
        "vad_boundaries": (8_500,),
        "acoustic_boundaries": (17_500,),
        "speaker_turns": (Turn("SPEAKER_00", 0, 12_000),),
        "vad_fingerprint": "b" * 64,
        "diarization_fingerprint": "c" * 64,
    }
    first = segment_chunks_with_metadata(
        "episode-p1r-reproducible", SOURCE_SHA256, 40_000, **kwargs
    )
    second = segment_chunks_with_metadata(
        "episode-p1r-reproducible", SOURCE_SHA256, 40_000, **kwargs
    )
    assert first == second
    assert first.segmentation_fingerprint == first.chunks[0].segmentation_fingerprint
    assert first.manifest_metadata()["segmentation_version"] == "p1r-chunk-segmentation-v1"
    manifest = ChunkBenchmarkManifest(
        manifest_id="manifest-p1r-chunking",
        chunks=first.chunks,
        **first.manifest_metadata(),
    )
    assert manifest.to_dict()["segmentation_configuration"]["target_s"] == 12.0
    assert ChunkBenchmarkManifest.from_dict(manifest.to_dict()) == manifest
    changed = segment_chunks_with_metadata(
        "episode-p1r-reproducible",
        SOURCE_SHA256,
        40_000,
        **kwargs,
        config=ChunkSegmentationConfig(target_s=13),
    )
    assert changed.segmentation_fingerprint != first.segmentation_fingerprint


def test_invalid_bounds_fail_and_missing_diarization_degrades_to_acoustic() -> None:
    with pytest.raises(ContractValidationError, match="outside the source duration"):
        segment_chunks("episode-p1r-invalid", SOURCE_SHA256, 10_000, vad_boundaries=(10_001,))

    chunks = segment_chunks(
        "episode-p1r-no-diarization",
        SOURCE_SHA256,
        40_000,
        acoustic_boundaries=(12_000,),
    )
    assert chunks[0].end_ms == 12_000

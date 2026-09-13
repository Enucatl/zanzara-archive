"""CPU-only P1R chunk, hypothesis and reference contract checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zanzara_archive.contracts import (
    AudioChunk,
    ChunkBenchmarkManifest,
    ChunkCondition,
    ContractValidationError,
    ModelFingerprint,
    Overlap,
    ReferenceRevision,
    SpeakerStream,
    TimedWord,
    TranscriptionHypothesis,
    TranscriptReference,
    Turn,
    json_round_trip,
)
from zanzara_archive.storage import SCHEMA_VERSION, SQLiteRepository

SOURCE_SHA256 = "a" * 64
SEGMENTATION_SHA256 = "b" * 64
MODEL_SHA256 = "c" * 64


def _model() -> ModelFingerprint:
    return ModelFingerprint(
        name="synthetic-p1r-model",
        repository="local/p1r",
        revision="fixture-v1",
        checkpoint_sha256=(MODEL_SHA256,),
        dimensions=None,
        preprocessing={"sample_rate_hz": 16000},
        precision="float32",
        runtime={"python": "3.14"},
        terms_evidence="synthetic fixture",
    )


def _condition() -> ChunkCondition:
    return ChunkCondition(
        speaker_streams=(
            SpeakerStream("SPEAKER_A", (Turn("SPEAKER_A", 100, 2_000),)),
            SpeakerStream("SPEAKER_B", (Turn("SPEAKER_B", 1_500, 2_500),)),
        ),
        overlaps=(Overlap(("SPEAKER_A", "SPEAKER_B"), 1_500, 2_000),),
        acoustic_labels=("speech",),
        rapid_turn_taking=True,
    )


def _chunk() -> AudioChunk:
    return AudioChunk.create(
        episode_id="episode-p1r",
        source_sha256=SOURCE_SHA256,
        start_ms=0,
        end_ms=3_000,
        segmentation_fingerprint=SEGMENTATION_SHA256,
        duration_ms=10_000,
        condition=_condition(),
        partition="development",
    )


def test_chunk_and_hypothesis_variants_round_trip() -> None:
    chunk = _chunk()
    assert chunk.chunk_id == AudioChunk.deterministic_id(
        chunk.episode_id,
        chunk.source_sha256,
        chunk.start_ms,
        chunk.end_ms,
        chunk.segmentation_fingerprint,
    )
    assert AudioChunk.from_dict(json_round_trip(chunk)) == chunk

    hypotheses = (
        TranscriptionHypothesis(chunk.chunk_id, _model(), "testo senza tempi"),
        TranscriptionHypothesis(
            chunk.chunk_id,
            MODEL_SHA256,
            "testo con parole",
            words=(TimedWord("word-1", "testo", 100, 400),),
        ),
        TranscriptionHypothesis(
            chunk.chunk_id,
            "d" * 64,
            "testo con segmenti",
            segments=({"start_ms": 0, "end_ms": 800, "text": "testo"},),
        ),
    )
    for hypothesis in hypotheses:
        assert TranscriptionHypothesis.from_dict(json_round_trip(hypothesis)) == hypothesis


def test_reference_revision_round_trip_retains_review_chain() -> None:
    chunk = _chunk()
    reference = TranscriptReference(
        reference_id="reference-p1r",
        chunk_id=chunk.chunk_id,
        source_sha256=SOURCE_SHA256,
        text="testo verificato",
        speaker_streams=_condition().speaker_streams,
        review_status="human_truth",
        reviewer="reviewer-1",
        revision=1,
    )
    revision = ReferenceRevision(
        revision_id="revision-p1r-1",
        reference_id=reference.reference_id,
        chunk_id=chunk.chunk_id,
        revision=1,
        reviewer="reviewer-1",
        source_sha256=SOURCE_SHA256,
        review_status="human_truth",
        text=reference.text,
        speaker_streams=reference.speaker_streams,
    )
    assert TranscriptReference.from_dict(json_round_trip(reference)) == reference
    assert ReferenceRevision.from_dict(json_round_trip(revision)) == revision

    with pytest.raises(ContractValidationError, match="source hashes"):
        ReferenceRevision(
            revision_id="revision-bad",
            reference_id=reference.reference_id,
            chunk_id=chunk.chunk_id,
            revision=2,
            reviewer="reviewer-1",
            source_sha256=SOURCE_SHA256,
            source_hash="e" * 64,
            prior_revision_id=revision.revision_id,
        )


def test_manifest_and_sqlite_publication_preserve_legacy_timed_words(tmp_path: Path) -> None:
    chunk = _chunk()
    reference = TranscriptReference(
        reference_id="reference-p1r",
        chunk_id=chunk.chunk_id,
        source_sha256=SOURCE_SHA256,
        text="testo verificato",
        review_status="draft",
    )
    hypothesis = TranscriptionHypothesis(chunk.chunk_id, _model(), "testo senza tempi")
    manifest = ChunkBenchmarkManifest(
        manifest_id="manifest-p1r",
        schema_version=1,
        chunks=(chunk,),
        references=(reference,),
        hypotheses=(hypothesis,),
        partitions={"development": (chunk.chunk_id,)},
    )
    assert ChunkBenchmarkManifest.from_dict(json_round_trip(manifest)) == manifest

    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-episode", "f" * 64, "fixture", "episode.opus", "{}", "now"),
    )
    repository.connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk.episode_id,
            "manifest-episode",
            "episode.opus",
            "2026-09-13",
            SOURCE_SHA256,
            100,
            10_000,
            "opus",
            1,
            48_000,
        ),
    )
    repository.connection.commit()

    repository.record_chunk_benchmark_manifest(manifest)
    assert repository.fetch_chunk_benchmark_manifest(manifest.manifest_id) == manifest
    assert repository.fetch_audio_chunk(chunk.chunk_id) == chunk
    assert repository.fetch_transcription_hypothesis(hypothesis.hypothesis_id) == hypothesis
    assert repository.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert repository.connection.execute("SELECT COUNT(*) FROM transcript_words").fetchone()[0] == 0

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        repository.connection.execute(
            "UPDATE transcription_hypotheses SET text = 'mutated' WHERE hypothesis_id = ?",
            (hypothesis.hypothesis_id,),
        )
    repository.close()

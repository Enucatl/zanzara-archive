"""Canonical SQLite state and transactional publication helpers.

SQLite is the source of truth for mutable archive state.  This module keeps
filesystem and database checks close to the connection boundary so later
workers cannot accidentally place job state on the read-only archive or on a
network filesystem that does not provide the WAL guarantees we require.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import urllib.parse
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import get_ident

from .contracts import (
    AcousticConditionCorrection,
    ApiError,
    ArtifactManifest,
    AudioChunk,
    ChunkBenchmarkManifest,
    EvaluationReport,
    IdentityDecision,
    JobStatus,
    ReferenceRevision,
    TranscriptionHypothesis,
    TranscriptReference,
)
from .corpus import CorpusManifest
from .stages import stage_fingerprint

SCHEMA_VERSION = 9
DEFAULT_BUSY_TIMEOUT_MS = 5_000
RETRY_BACKOFF_SECONDS = (5, 30)


class StorageError(RuntimeError):
    """Base error for canonical storage failures."""


class StoragePathError(StorageError):
    """Raised when SQLite state cannot be proven to live on local storage."""


class StorageCapabilityError(StorageError):
    """Raised when the SQLite build lacks a required capability."""


class StorageConflictError(StorageError):
    """Raised when an immutable or optimistic publication conflicts."""


NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "fuse.sshfs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb3",
    }
)


def _decode_mount_field(value: str) -> str:
    """Decode the octal escapes used by ``/proc/self/mountinfo``."""

    result = value
    for escaped, character in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
        result = result.replace(escaped, character)
    return result


def _filesystem_type(path: Path) -> str | None:
    """Return the mounted filesystem type for *path*, when Linux exposes it."""

    try:
        resolved = path.resolve()
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, RuntimeError):
        return None
    best_mount = ""
    best_type: str | None = None
    for line in lines:
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields = left.split()
        if len(fields) < 5:
            continue
        mountpoint = Path(_decode_mount_field(fields[4]))
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        if len(str(mountpoint)) >= len(best_mount):
            right_fields = right.split()
            if right_fields:
                best_mount = str(mountpoint)
                best_type = right_fields[0]
    return best_type


def ensure_local_state_path(path: str | os.PathLike[str]) -> Path:
    """Validate and create a local path suitable for SQLite/job state."""

    raw = os.fspath(path)
    if raw == ":memory:":
        raise StoragePathError("SQLite WAL/job state requires a local filesystem path")
    if raw.startswith("//") or raw.startswith("\\\\"):
        raise StoragePathError("network paths are not valid SQLite/job-state paths")
    if raw.startswith("file:"):
        parsed = urllib.parse.urlparse(raw)
        if parsed.query or parsed.netloc:
            raise StoragePathError("SQLite URI paths with remote/query semantics are not allowed")
        raw = urllib.parse.unquote(parsed.path)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    parent = candidate.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        resolved_candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise StoragePathError(f"cannot create local SQLite parent {parent}: {exc}") from exc
    filesystem = _filesystem_type(resolved_candidate)
    if filesystem is None:
        raise StoragePathError(
            f"cannot establish local filesystem for SQLite/job-state path: {resolved_candidate}"
        )
    if filesystem in NETWORK_FILESYSTEMS:
        raise StoragePathError(
            f"SQLite/job-state path is on network filesystem {filesystem}: {resolved_candidate}"
        )
    return resolved_candidate


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS corpus_manifests (
        manifest_id TEXT PRIMARY KEY,
        sha256 TEXT NOT NULL UNIQUE,
        selection TEXT NOT NULL,
        golden_episode TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id TEXT PRIMARY KEY,
        manifest_id TEXT REFERENCES corpus_manifests(manifest_id) ON DELETE RESTRICT,
        relative_filename TEXT NOT NULL UNIQUE,
        episode_date TEXT NOT NULL,
        source_sha256 TEXT NOT NULL UNIQUE,
        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
        duration_ms INTEGER NOT NULL CHECK (duration_ms > 0),
        codec TEXT NOT NULL,
        channels INTEGER NOT NULL CHECK (channels > 0),
        sample_rate_hz INTEGER NOT NULL CHECK (sample_rate_hz > 0),
        source_read_only INTEGER NOT NULL DEFAULT 1 CHECK (source_read_only IN (0, 1))
    );
    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        fingerprint_sha256 TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        source_sha256 TEXT NOT NULL,
        stage TEXT NOT NULL,
        stage_key TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        manifest_sha256 TEXT NOT NULL UNIQUE,
        upstream_artifact_hashes_json TEXT NOT NULL,
        model_fingerprint_sha256 TEXT,
        preprocessing_json TEXT NOT NULL,
        pipeline_version TEXT NOT NULL,
        artifact_schema_version INTEGER NOT NULL CHECK (artifact_schema_version > 0),
        file_checksums_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (source_sha256, stage, stage_key)
    );
    CREATE TABLE IF NOT EXISTS processing_runs (
        run_id TEXT PRIMARY KEY,
        stage TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        status TEXT NOT NULL,
        configuration_json TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        stage TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('queued','running','retry_wait','succeeded','failed','cancelled','blocked')
        ),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        fencing_token INTEGER NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
        owner TEXT,
        lease_expires_at TEXT,
        request_id TEXT,
        error_json TEXT,
        recovery_action TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS episode_speakers (
        episode_speaker_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        diarization_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        local_speaker_id TEXT NOT NULL,
        voice_searchable TEXT NOT NULL DEFAULT 'unknown',
        UNIQUE (episode_id, diarization_artifact_id, local_speaker_id)
    );
    CREATE TABLE IF NOT EXISTS exemplars (
        exemplar_id TEXT PRIMARY KEY,
        episode_speaker_id TEXT NOT NULL REFERENCES episode_speakers
            (episode_speaker_id) ON DELETE RESTRICT,
        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        start_ms INTEGER NOT NULL,
        end_ms INTEGER NOT NULL,
        duration_ms INTEGER NOT NULL,
        clipping_fraction REAL NOT NULL,
        exclusion_reasons_json TEXT NOT NULL,
        UNIQUE (episode_speaker_id, artifact_id, start_ms, end_ms)
    );
    CREATE TABLE IF NOT EXISTS text_chunks (
        chunk_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        attribution_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        start_ms INTEGER NOT NULL,
        end_ms INTEGER NOT NULL,
        text TEXT NOT NULL,
        word_ids_json TEXT NOT NULL,
        overlap INTEGER NOT NULL DEFAULT 0 CHECK (overlap IN (0, 1))
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks_fts USING fts5(
        chunk_id UNINDEXED,
        text,
        tokenize = 'unicode61 remove_diacritics 0'
    );
    CREATE TABLE IF NOT EXISTS global_speakers (
        global_speaker_id TEXT PRIMARY KEY,
        display_name TEXT,
        state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active','superseded')),
        revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0)
    );
    CREATE TABLE IF NOT EXISTS identity_decisions (
        decision_id TEXT PRIMARY KEY,
        left_episode_speaker_id TEXT NOT NULL REFERENCES episode_speakers
            (episode_speaker_id) ON DELETE RESTRICT,
        right_episode_speaker_id TEXT NOT NULL REFERENCES episode_speakers
            (episode_speaker_id) ON DELETE RESTRICT,
        decision TEXT NOT NULL CHECK (decision IN ('same_person','different_person','uncertain')),
        reviewer TEXT NOT NULL,
        evidence_artifact_ids_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('active','superseded')),
        revision INTEGER NOT NULL CHECK (revision > 0),
        CHECK (left_episode_speaker_id <> right_episode_speaker_id)
    );
    CREATE TABLE IF NOT EXISTS identity_audit (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id TEXT NOT NULL REFERENCES identity_decisions(decision_id) ON DELETE RESTRICT,
        revision INTEGER NOT NULL CHECK (revision > 0),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (decision_id, revision)
    );
    CREATE TABLE IF NOT EXISTS index_generations (
        generation_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        generation_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK (status IN ('complete','abandoned')),
        expected_points INTEGER NOT NULL CHECK (expected_points >= 0),
        published_at TEXT
    );
    CREATE TABLE IF NOT EXISTS committed_pointers (
        pointer_name TEXT PRIMARY KEY,
        generation_id TEXT NOT NULL REFERENCES index_generations(generation_id) ON DELETE RESTRICT,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS cost_reservations (
        reservation_id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL UNIQUE,
        reserved_microusd INTEGER NOT NULL CHECK (reserved_microusd >= 0),
        spent_microusd INTEGER NOT NULL DEFAULT 0 CHECK (spent_microusd >= 0),
        status TEXT NOT NULL CHECK (status IN ('reserved','settled','released','ambiguous')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    2: """
    CREATE INDEX IF NOT EXISTS idx_artifacts_source_stage ON artifacts(source_sha256, stage);
    CREATE INDEX IF NOT EXISTS idx_jobs_eligible ON jobs(status, lease_expires_at);
    CREATE INDEX IF NOT EXISTS idx_identity_decisions_pair
      ON identity_decisions(left_episode_speaker_id, right_episode_speaker_id, state);
    """,
    3: """
    ALTER TABLE jobs ADD COLUMN stage_key TEXT NOT NULL DEFAULT '';
    ALTER TABLE jobs ADD COLUMN source_sha256 TEXT;
    ALTER TABLE jobs ADD COLUMN upstream_artifact_hashes_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE jobs ADD COLUMN model_fingerprint_sha256 TEXT;
    ALTER TABLE jobs ADD COLUMN configuration_json TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE jobs ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'dev';
    ALTER TABLE jobs ADD COLUMN paid INTEGER NOT NULL DEFAULT 0 CHECK (paid IN (0, 1));
    CREATE INDEX IF NOT EXISTS idx_jobs_stage_key ON jobs(source_sha256, stage, stage_key);
    """,
    4: """
    CREATE TABLE IF NOT EXISTS transcript_words (
        word_id TEXT PRIMARY KEY,
        transcript_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        model_fingerprint_sha256 TEXT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        text TEXT NOT NULL,
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        episode_speaker_id TEXT REFERENCES episode_speakers(episode_speaker_id) ON DELETE RESTRICT,
        overlap INTEGER NOT NULL DEFAULT 0 CHECK (overlap IN (0, 1)),
        UNIQUE (transcript_artifact_id, ordinal)
    );
    CREATE INDEX IF NOT EXISTS idx_transcript_words_episode_time
      ON transcript_words(episode_id, start_ms, end_ms);

    CREATE TABLE IF NOT EXISTS standard_turns (
        turn_id TEXT PRIMARY KEY,
        diarization_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        model_fingerprint_sha256 TEXT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        local_speaker_id TEXT NOT NULL,
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        UNIQUE (diarization_artifact_id, ordinal)
    );
    CREATE TABLE IF NOT EXISTS exclusive_turns (
        turn_id TEXT PRIMARY KEY,
        diarization_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        model_fingerprint_sha256 TEXT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        local_speaker_id TEXT NOT NULL,
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
        UNIQUE (diarization_artifact_id, ordinal)
    );
    CREATE INDEX IF NOT EXISTS idx_standard_turns_episode_time
      ON standard_turns(episode_id, start_ms, end_ms);
    CREATE INDEX IF NOT EXISTS idx_exclusive_turns_episode_time
      ON exclusive_turns(episode_id, start_ms, end_ms);

    CREATE TABLE IF NOT EXISTS overlap_intervals (
        overlap_id TEXT PRIMARY KEY,
        diarization_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        model_fingerprint_sha256 TEXT,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        speaker_ids_json TEXT NOT NULL,
        UNIQUE (diarization_artifact_id, ordinal)
    );
    CREATE INDEX IF NOT EXISTS idx_overlap_intervals_episode_time
      ON overlap_intervals(episode_id, start_ms, end_ms);

    CREATE TABLE IF NOT EXISTS annotation_revisions (
        annotation_revision_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        reviewer TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('draft','submitted','accepted','superseded')),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (episode_id, revision)
    );
    CREATE TABLE IF NOT EXISTS review_records (
        review_id TEXT PRIMARY KEY,
        annotation_revision_id TEXT NOT NULL REFERENCES annotation_revisions(annotation_revision_id)
            ON DELETE RESTRICT,
        reviewer TEXT NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        decision TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        state TEXT NOT NULL CHECK (state IN ('active','superseded')),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (annotation_revision_id, target_type, target_id, revision)
    );
    CREATE TABLE IF NOT EXISTS split_records (
        split_id TEXT PRIMARY KEY,
        episode_speaker_id TEXT NOT NULL REFERENCES episode_speakers(episode_speaker_id)
            ON DELETE RESTRICT,
        annotation_revision_id TEXT REFERENCES annotation_revisions(annotation_revision_id)
            ON DELETE RESTRICT,
        revision INTEGER NOT NULL CHECK (revision > 0),
        requested_memberships_json TEXT NOT NULL,
        reason TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('proposed','accepted','superseded','rejected')),
        created_at TEXT NOT NULL,
        UNIQUE (episode_speaker_id, revision)
    );

    CREATE TABLE IF NOT EXISTS upload_jobs (
        upload_job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE RESTRICT,
        request_id TEXT NOT NULL UNIQUE,
        owner TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('queued','running','blocked','succeeded','failed','cancelled','expired')
        ),
        upload_artifact_id TEXT REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
        selected_episode_speaker_id TEXT REFERENCES episode_speakers(episode_speaker_id)
            ON DELETE RESTRICT,
        expected_revision INTEGER CHECK (expected_revision IS NULL OR expected_revision > 0),
        payload_path TEXT,
        expires_at TEXT NOT NULL,
        error_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_upload_jobs_expiry ON upload_jobs(status, expires_at);

    CREATE TABLE IF NOT EXISTS evaluation_reports (
        report_id TEXT PRIMARY KEY,
        source_sha256 TEXT NOT NULL,
        split_sha256 TEXT NOT NULL,
        model_fingerprint_sha256_json TEXT NOT NULL,
        configuration_sha256 TEXT NOT NULL,
        reviewed_commit TEXT NOT NULL,
        verdict TEXT NOT NULL CHECK (verdict IN ('pass','blocked','insufficient_evidence')),
        metrics_json TEXT NOT NULL,
        limitations_json TEXT NOT NULL,
        generated_at TEXT,
        revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_evaluation_reports_verdict
      ON evaluation_reports(verdict, generated_at);
    """,
    5: """
    CREATE TRIGGER transcript_words_provenance_insert
    BEFORE INSERT ON transcript_words BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.transcript_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.transcript_artifact_id)
      THEN RAISE(ABORT, 'invalid transcript word interval or provenance') END;
    END;
    CREATE TRIGGER transcript_words_provenance_update
    BEFORE UPDATE ON transcript_words BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.transcript_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.transcript_artifact_id)
      THEN RAISE(ABORT, 'invalid transcript word interval or provenance') END;
    END;

    CREATE TRIGGER standard_turns_provenance_insert
    BEFORE INSERT ON standard_turns BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid standard turn interval or provenance') END;
    END;
    CREATE TRIGGER standard_turns_provenance_update
    BEFORE UPDATE ON standard_turns BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid standard turn interval or provenance') END;
    END;

    CREATE TRIGGER exclusive_turns_provenance_insert
    BEFORE INSERT ON exclusive_turns BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid exclusive turn interval or provenance') END;
    END;
    CREATE TRIGGER exclusive_turns_provenance_update
    BEFORE UPDATE ON exclusive_turns BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid exclusive turn interval or provenance') END;
    END;

    CREATE TRIGGER overlap_intervals_provenance_insert
    BEFORE INSERT ON overlap_intervals BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid overlap interval or provenance') END;
    END;
    CREATE TRIGGER overlap_intervals_provenance_update
    BEFORE UPDATE ON overlap_intervals BEGIN
      SELECT CASE WHEN
        typeof(NEW.start_ms) <> 'integer' OR typeof(NEW.end_ms) <> 'integer'
        OR NEW.end_ms > (SELECT duration_ms FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM episodes WHERE episode_id = NEW.episode_id)
        OR NEW.source_sha256 IS NOT
           (SELECT source_sha256 FROM artifacts WHERE artifact_id = NEW.diarization_artifact_id)
        OR NEW.model_fingerprint_sha256 IS NOT
           (SELECT model_fingerprint_sha256 FROM artifacts
            WHERE artifact_id = NEW.diarization_artifact_id)
      THEN RAISE(ABORT, 'invalid overlap interval or provenance') END;
    END;
    """,
    6: """
    CREATE TABLE IF NOT EXISTS chunk_benchmark_manifests (
        manifest_id TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        source_manifest_sha256 TEXT,
        content_sha256 TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS audio_chunks (
        chunk_id TEXT PRIMARY KEY,
        manifest_id TEXT REFERENCES chunk_benchmark_manifests(manifest_id) ON DELETE RESTRICT,
        episode_id TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms > 0),
        segmentation_fingerprint TEXT NOT NULL,
        partition TEXT CHECK (partition IS NULL OR partition IN ('development','held_out')),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audio_chunks_episode_time
      ON audio_chunks(episode_id, start_ms, end_ms);

    CREATE TABLE IF NOT EXISTS transcription_hypotheses (
        hypothesis_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        model_fingerprint_sha256 TEXT NOT NULL,
        text TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (chunk_id, model_fingerprint_sha256)
    );

    CREATE TABLE IF NOT EXISTS transcript_references (
        reference_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision > 0),
        review_status TEXT NOT NULL CHECK (
            review_status IN ('draft','human_truth','superseded','rejected')
        ),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS reference_revisions (
        revision_id TEXT PRIMARY KEY,
        reference_id TEXT NOT NULL REFERENCES transcript_references(reference_id)
            ON DELETE RESTRICT,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        revision INTEGER NOT NULL CHECK (revision > 0),
        reviewer TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        prior_revision_id TEXT REFERENCES reference_revisions(revision_id) ON DELETE RESTRICT,
        review_status TEXT NOT NULL CHECK (
            review_status IN ('draft','human_truth','superseded','rejected')
        ),
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (reference_id, revision)
    );

    CREATE TRIGGER transcription_hypotheses_immutable_update
    BEFORE UPDATE ON transcription_hypotheses BEGIN
      SELECT RAISE(ABORT, 'transcription hypotheses are immutable');
    END;
    CREATE TRIGGER transcription_hypotheses_immutable_delete
    BEFORE DELETE ON transcription_hypotheses BEGIN
      SELECT RAISE(ABORT, 'transcription hypotheses are immutable');
    END;
    CREATE TRIGGER audio_chunks_immutable_update
    BEFORE UPDATE ON audio_chunks BEGIN
      SELECT RAISE(ABORT, 'audio chunks are immutable');
    END;
    CREATE TRIGGER audio_chunks_immutable_delete
    BEFORE DELETE ON audio_chunks BEGIN
      SELECT RAISE(ABORT, 'audio chunks are immutable');
    END;
    CREATE TRIGGER chunk_benchmark_manifests_immutable_update
    BEFORE UPDATE ON chunk_benchmark_manifests BEGIN
      SELECT RAISE(ABORT, 'benchmark manifests are immutable');
    END;
    CREATE TRIGGER chunk_benchmark_manifests_immutable_delete
    BEFORE DELETE ON chunk_benchmark_manifests BEGIN
      SELECT RAISE(ABORT, 'benchmark manifests are immutable');
    END;
    CREATE TRIGGER transcript_references_immutable_update
    BEFORE UPDATE ON transcript_references BEGIN
      SELECT RAISE(ABORT, 'transcript references are append-only');
    END;
    CREATE TRIGGER transcript_references_immutable_delete
    BEFORE DELETE ON transcript_references BEGIN
      SELECT RAISE(ABORT, 'transcript references are append-only');
    END;
    CREATE TRIGGER reference_revisions_immutable_update
    BEFORE UPDATE ON reference_revisions BEGIN
      SELECT RAISE(ABORT, 'reference revisions are append-only');
    END;
    CREATE TRIGGER reference_revisions_immutable_delete
    BEFORE DELETE ON reference_revisions BEGIN
      SELECT RAISE(ABORT, 'reference revisions are append-only');
    END;
    """,
    7: """
    CREATE TABLE IF NOT EXISTS chunk_inference_dispatches (
        dispatch_id TEXT PRIMARY KEY,
        manifest_id TEXT NOT NULL REFERENCES chunk_benchmark_manifests(manifest_id)
            ON DELETE RESTRICT,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        adapter_id TEXT NOT NULL,
        job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id) ON DELETE RESTRICT,
        model_fingerprint_sha256 TEXT NOT NULL,
        model_payload_json TEXT NOT NULL,
        configuration_json TEXT NOT NULL,
        preprocessing_json TEXT NOT NULL,
        preprocessing_sha256 TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        start_ms INTEGER NOT NULL CHECK (start_ms >= 0),
        end_ms INTEGER NOT NULL CHECK (end_ms > start_ms),
        request_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        input_audio_sha256 TEXT,
        raw_artifact_id TEXT,
        hypothesis_id TEXT,
        runtime_json TEXT NOT NULL DEFAULT '{}',
        error_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (manifest_id, chunk_id, adapter_id)
    );
    CREATE INDEX IF NOT EXISTS idx_chunk_inference_dispatches_manifest
      ON chunk_inference_dispatches(manifest_id, status, adapter_id, chunk_id);
    """,
    8: """
    CREATE TABLE IF NOT EXISTS acoustic_condition_corrections (
        correction_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        seed_version TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_acoustic_condition_corrections_chunk
      ON acoustic_condition_corrections(chunk_id, created_at, correction_id);
    CREATE TRIGGER acoustic_condition_corrections_immutable_update
    BEFORE UPDATE ON acoustic_condition_corrections BEGIN
      SELECT RAISE(ABORT, 'acoustic condition corrections are append-only');
    END;
    CREATE TRIGGER acoustic_condition_corrections_immutable_delete
    BEFORE DELETE ON acoustic_condition_corrections BEGIN
      SELECT RAISE(ABORT, 'acoustic condition corrections are append-only');
    END;
    """,
    9: """
    CREATE TABLE IF NOT EXISTS annotation_assistance_drafts (
        draft_id TEXT PRIMARY KEY,
        chunk_id TEXT NOT NULL REFERENCES audio_chunks(chunk_id) ON DELETE RESTRICT,
        source_sha256 TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        prompt_sha256 TEXT NOT NULL,
        model_fingerprint_sha256 TEXT NOT NULL,
        model_fingerprint_json TEXT NOT NULL,
        candidate_order_json TEXT NOT NULL,
        input_payload_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('succeeded','unavailable')),
        draft_text TEXT,
        error_json TEXT,
        provenance_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (chunk_id, request_sha256, model_fingerprint_sha256)
    );
    CREATE INDEX IF NOT EXISTS idx_annotation_assistance_drafts_chunk
      ON annotation_assistance_drafts(chunk_id, created_at, draft_id);
    CREATE TRIGGER annotation_assistance_drafts_immutable_update
    BEFORE UPDATE ON annotation_assistance_drafts BEGIN
      SELECT RAISE(ABORT, 'annotation assistance drafts are append-only');
    END;
    CREATE TRIGGER annotation_assistance_drafts_immutable_delete
    BEFORE DELETE ON annotation_assistance_drafts BEGIN
      SELECT RAISE(ABORT, 'annotation assistance drafts are append-only');
    END;
    """,
}


def assert_sqlite_capabilities(connection: sqlite3.Connection) -> None:
    """Prove that the connection has foreign keys, WAL, and FTS5."""

    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise StorageCapabilityError("SQLite foreign-key enforcement is unavailable")
    try:
        mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    except sqlite3.DatabaseError as exc:
        raise StorageCapabilityError(f"SQLite WAL is unavailable: {exc}") from exc
    if mode != "wal":
        raise StorageCapabilityError(f"SQLite did not enable WAL (reported {mode!r})")
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.zanzara_fts_capability USING fts5(value)")
        connection.execute("DROP TABLE temp.zanzara_fts_capability")
    except sqlite3.DatabaseError as exc:
        raise StorageCapabilityError(f"SQLite FTS5 is unavailable: {exc}") from exc


def migrate(connection: sqlite3.Connection) -> int:
    """Apply all migrations atomically and return the schema version."""

    assert_sqlite_capabilities(connection)
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise StorageError(f"database schema {current} is newer than supported {SCHEMA_VERSION}")
    connection.execute("BEGIN IMMEDIATE")
    try:
        for version in range(current + 1, SCHEMA_VERSION + 1):
            # ``executescript`` commits any active transaction before running,
            # which would expose a partially applied migration on failure.
            # Execute complete statements inside the surrounding transaction.
            statement = ""
            for line in MIGRATIONS[version].splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    connection.execute(statement)
                    statement = ""
            if statement.strip():
                raise StorageError(f"migration {version} has an incomplete SQL statement")
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return SCHEMA_VERSION


def open_database(
    path: str | os.PathLike[str], *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS
) -> sqlite3.Connection:
    """Open, validate, and migrate a local WAL SQLite database."""

    if busy_timeout_ms <= 0:
        raise ValueError("busy_timeout_ms must be positive")
    database_path = ensure_local_state_path(path)
    try:
        connection = sqlite3.connect(database_path, timeout=busy_timeout_ms / 1000)
    except sqlite3.Error as exc:
        raise StorageError(f"cannot open SQLite database {database_path}: {exc}") from exc
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    connection.execute("PRAGMA synchronous = NORMAL")
    try:
        migrate(connection)
    except Exception:
        connection.close()
        raise
    return connection


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteRepository:
    """Repository for canonical records and committed projections."""

    def __init__(self, connection: sqlite3.Connection, database_path: Path | None = None):
        self.connection = connection
        self._database_path = database_path
        self._connection_thread_id = get_ident()
        assert_sqlite_capabilities(connection)

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> SQLiteRepository:
        database_path = ensure_local_state_path(path)
        return cls(open_database(database_path), database_path)

    def close(self) -> None:
        self.connection.close()

    def register_corpus_manifest(self, manifest: CorpusManifest) -> str:
        """Idempotently register a frozen manifest and its episode metadata."""

        manifest_id = f"manifest-{manifest.sha256}"
        payload = _json(
            {
                "schema_version": manifest.schema_version,
                "observed_at": manifest.observed_at.isoformat(),
                "selection": manifest.selection,
                "golden_episode": manifest.golden_episode,
                "episodes": [episode.as_json() for episode in manifest.episodes],
            }
        )
        episode_fields = (
            "episode_id",
            "manifest_id",
            "relative_filename",
            "episode_date",
            "source_sha256",
            "size_bytes",
            "duration_ms",
            "codec",
            "channels",
            "sample_rate_hz",
            "source_read_only",
        )
        with self.transaction() as connection:
            existing_manifest = connection.execute(
                "SELECT manifest_id, selection, golden_episode, payload_json "
                "FROM corpus_manifests WHERE sha256 = ?",
                (manifest.sha256,),
            ).fetchone()
            expected_manifest = (
                manifest_id,
                manifest.selection,
                manifest.golden_episode,
                payload,
            )
            if existing_manifest is None:
                connection.execute(
                    "INSERT INTO corpus_manifests "
                    "(manifest_id, sha256, selection, golden_episode, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        manifest_id,
                        manifest.sha256,
                        manifest.selection,
                        manifest.golden_episode,
                        payload,
                        _now(),
                    ),
                )
            elif tuple(existing_manifest) != expected_manifest:
                raise StorageConflictError(
                    "registered corpus manifest has conflicting metadata or payload"
                )

            for episode in manifest.episodes:
                expected_episode = (
                    episode.relative_filename,
                    manifest_id,
                    episode.relative_filename,
                    episode.episode_date.isoformat(),
                    episode.sha256,
                    episode.size_bytes,
                    episode.duration_ms,
                    episode.codec,
                    episode.channels,
                    episode.sample_rate_hz,
                    1,
                )
                existing_episode = connection.execute(
                    "SELECT "
                    + ", ".join(episode_fields)
                    + " FROM episodes WHERE episode_id = ? OR relative_filename = ? "
                    "OR source_sha256 = ?",
                    (episode.relative_filename, episode.relative_filename, episode.sha256),
                ).fetchone()
                if existing_episode is None:
                    connection.execute(
                        "INSERT INTO episodes ("
                        + ", ".join(episode_fields)
                        + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        expected_episode,
                    )
                elif tuple(existing_episode) != expected_episode:
                    raise StorageConflictError(
                        f"registered episode has conflicting metadata: {episode.relative_filename}"
                    )
        return manifest_id

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a short transaction; callers must not perform inference here."""

        try:
            with self.connection:
                yield self.connection
        except sqlite3.IntegrityError as exc:
            raise StorageConflictError(str(exc)) from exc

    def _begin_immediate(self) -> None:
        """Start a claim transaction even after a caller's direct read/write."""

        if self.connection.in_transaction:
            self.connection.commit()
        self.connection.execute("BEGIN IMMEDIATE")

    def record_artifact(
        self,
        manifest: ArtifactManifest,
        artifact_path: Path,
        *,
        job: JobStatus | None = None,
        now: datetime | str | None = None,
    ) -> None:
        """Record a complete artifact after its atomic filesystem publication."""

        manifest_payload = _json(manifest.to_dict())
        manifest_sha256 = hashlib.sha256(manifest_payload.encode()).hexdigest()
        with self.transaction() as connection:
            if job is not None:
                self._assert_job_fence(connection, job, now=now)
            existing = connection.execute(
                """SELECT manifest_sha256 FROM artifacts
                WHERE artifact_id = ? OR (source_sha256 = ? AND stage = ? AND stage_key = ?)""",
                (manifest.artifact_id, manifest.source_sha256, manifest.stage, manifest.stage_key),
            ).fetchone()
            if existing is not None:
                if existing[0] == manifest_sha256:
                    return
                raise StorageConflictError("artifact identity already refers to different content")
            connection.execute(
                """INSERT INTO artifacts (
                    artifact_id, source_sha256, stage, stage_key, artifact_path,
                    manifest_sha256, upstream_artifact_hashes_json,
                    model_fingerprint_sha256, preprocessing_json, pipeline_version,
                    artifact_schema_version, file_checksums_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    manifest.artifact_id,
                    manifest.source_sha256,
                    manifest.stage,
                    manifest.stage_key,
                    str(artifact_path),
                    manifest_sha256,
                    _json(manifest.upstream_artifact_hashes),
                    manifest.model_fingerprint_sha256,
                    "{}",
                    manifest.pipeline_version,
                    manifest.schema_version,
                    _json(manifest.file_checksums),
                    _now(),
                ),
            )

    @staticmethod
    def _chunk_episode_check(connection: sqlite3.Connection, chunk: AudioChunk) -> None:
        """Check a chunk against registered episode provenance before publication."""

        episode = connection.execute(
            "SELECT source_sha256, duration_ms FROM episodes WHERE episode_id = ?",
            (chunk.episode_id,),
        ).fetchone()
        if episode is None:
            raise StorageConflictError(f"chunk episode is not registered: {chunk.episode_id}")
        if episode["source_sha256"] != chunk.source_sha256:
            raise StorageConflictError("chunk source hash does not match its episode")
        if chunk.end_ms > episode["duration_ms"]:
            raise StorageConflictError("chunk interval exceeds its episode duration")
        if chunk.duration_ms is not None and chunk.duration_ms != episode["duration_ms"]:
            raise StorageConflictError("chunk duration does not match its episode")

    @staticmethod
    def _record_audio_chunk(
        connection: sqlite3.Connection, chunk: AudioChunk, manifest_id: str | None = None
    ) -> None:
        SQLiteRepository._chunk_episode_check(connection, chunk)
        payload = _json(chunk.to_dict())
        existing = connection.execute(
            "SELECT payload_json, manifest_id FROM audio_chunks WHERE chunk_id = ?",
            (chunk.chunk_id,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] == payload and (
                manifest_id is None or existing["manifest_id"] == manifest_id
            ):
                return
            raise StorageConflictError("chunk identity already refers to different content")
        connection.execute(
            """INSERT INTO audio_chunks (
                chunk_id, manifest_id, episode_id, source_sha256, start_ms, end_ms,
                duration_ms, segmentation_fingerprint, partition, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id,
                manifest_id,
                chunk.episode_id,
                chunk.source_sha256,
                chunk.start_ms,
                chunk.end_ms,
                chunk.duration_ms,
                chunk.segmentation_fingerprint,
                chunk.partition,
                payload,
                _now(),
            ),
        )

    def record_audio_chunk(self, chunk: AudioChunk, *, manifest_id: str | None = None) -> None:
        """Publish one immutable chunk after checking registered source provenance."""

        with self.transaction() as connection:
            self._record_audio_chunk(connection, chunk, manifest_id)

    def fetch_audio_chunk(self, chunk_id: str) -> AudioChunk | None:
        """Restore one canonical chunk contract."""

        row = self.connection.execute(
            "SELECT payload_json FROM audio_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        chunk = AudioChunk.from_dict(json.loads(row["payload_json"]))
        correction = self.fetch_acoustic_condition_correction(chunk_id)
        if correction is None or chunk.condition is None:
            return chunk
        return replace(
            chunk,
            condition=replace(chunk.condition, acoustic_correction=correction),
        )

    def record_acoustic_condition_correction(
        self,
        chunk_id: str,
        correction: AcousticConditionCorrection,
        *,
        correction_id: str | None = None,
    ) -> str:
        """Append an identified human correction without mutating the AST seed."""

        row = self.connection.execute(
            "SELECT payload_json, source_sha256 FROM audio_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise StorageConflictError(f"acoustic condition chunk is not registered: {chunk_id}")
        chunk = AudioChunk.from_dict(json.loads(row["payload_json"]))
        metadata = chunk.condition.acoustic_metadata if chunk.condition is not None else None
        if metadata is None:
            raise StorageConflictError("acoustic correction requires a stored machine seed")
        if correction.seed_version != metadata.version:
            raise StorageConflictError("acoustic correction seed version does not match chunk seed")
        payload = _json(correction.to_dict())
        resolved_id = correction_id or (
            "acoustic-correction-" + hashlib.sha256(f"{chunk_id}:{payload}".encode()).hexdigest()
        )
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM acoustic_condition_corrections WHERE correction_id = ?",
                (resolved_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] == payload:
                    return resolved_id
                raise StorageConflictError(
                    "acoustic correction identity already refers to different content"
                )
            connection.execute(
                """INSERT INTO acoustic_condition_corrections (
                    correction_id, chunk_id, source_sha256, seed_version, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    resolved_id,
                    chunk_id,
                    row["source_sha256"],
                    correction.seed_version,
                    payload,
                    _now(),
                ),
            )
        return resolved_id

    def fetch_acoustic_condition_correction(
        self, chunk_id: str
    ) -> AcousticConditionCorrection | None:
        """Return the latest append-only human condition correction for a chunk."""

        row = self.connection.execute(
            """SELECT payload_json FROM acoustic_condition_corrections
            WHERE chunk_id = ? ORDER BY created_at DESC, correction_id DESC LIMIT 1""",
            (chunk_id,),
        ).fetchone()
        return (
            AcousticConditionCorrection.from_dict(json.loads(row["payload_json"])) if row else None
        )

    def list_acoustic_condition_corrections(
        self, chunk_id: str
    ) -> tuple[AcousticConditionCorrection, ...]:
        """Return all condition corrections in stable append order."""

        rows = self.connection.execute(
            """SELECT payload_json FROM acoustic_condition_corrections
            WHERE chunk_id = ? ORDER BY created_at, correction_id""",
            (chunk_id,),
        ).fetchall()
        return tuple(
            AcousticConditionCorrection.from_dict(json.loads(row["payload_json"])) for row in rows
        )

    @staticmethod
    def _record_transcription_hypothesis(
        connection: sqlite3.Connection, hypothesis: TranscriptionHypothesis
    ) -> None:
        if (
            connection.execute(
                "SELECT 1 FROM audio_chunks WHERE chunk_id = ?", (hypothesis.chunk_id,)
            ).fetchone()
            is None
        ):
            raise StorageConflictError(f"hypothesis chunk is not registered: {hypothesis.chunk_id}")
        payload = _json(hypothesis.to_dict())
        existing = connection.execute(
            """SELECT payload_json FROM transcription_hypotheses
            WHERE hypothesis_id = ? OR (chunk_id = ? AND model_fingerprint_sha256 = ?)""",
            (hypothesis.hypothesis_id, hypothesis.chunk_id, hypothesis.model_fingerprint_hash),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] == payload:
                return
            raise StorageConflictError("hypothesis identity already refers to different content")
        connection.execute(
            """INSERT INTO transcription_hypotheses (
                hypothesis_id, chunk_id, model_fingerprint_sha256, text,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                hypothesis.hypothesis_id,
                hypothesis.chunk_id,
                hypothesis.model_fingerprint_hash,
                hypothesis.text,
                payload,
                _now(),
            ),
        )

    def record_transcription_hypothesis(self, hypothesis: TranscriptionHypothesis) -> None:
        """Publish a hypothesis once; later writes cannot mutate the candidate."""

        with self.transaction() as connection:
            self._record_transcription_hypothesis(connection, hypothesis)

    def record_hypothesis(self, hypothesis: TranscriptionHypothesis) -> None:
        """Compatibility alias for the canonical hypothesis publisher."""

        self.record_transcription_hypothesis(hypothesis)

    def fetch_transcription_hypothesis(self, hypothesis_id: str) -> TranscriptionHypothesis | None:
        """Restore one immutable hypothesis contract."""

        row = self.connection.execute(
            "SELECT payload_json FROM transcription_hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        return (
            TranscriptionHypothesis.from_dict(json.loads(row["payload_json"]))
            if row is not None
            else None
        )

    def find_transcription_hypothesis(
        self, *, chunk_id: str, model_fingerprint_sha256: str
    ) -> TranscriptionHypothesis | None:
        """Find the immutable hypothesis for one chunk/model pair."""

        row = self.connection.execute(
            """SELECT payload_json FROM transcription_hypotheses
            WHERE chunk_id = ? AND model_fingerprint_sha256 = ?""",
            (chunk_id, model_fingerprint_sha256),
        ).fetchone()
        return (
            TranscriptionHypothesis.from_dict(json.loads(row["payload_json"]))
            if row is not None
            else None
        )

    @staticmethod
    def _annotation_assistance_draft_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "draft_id": row["draft_id"],
            "chunk_id": row["chunk_id"],
            "source_sha256": row["source_sha256"],
            "request_sha256": row["request_sha256"],
            "prompt_version": row["prompt_version"],
            "prompt_sha256": row["prompt_sha256"],
            "model_fingerprint_sha256": row["model_fingerprint_sha256"],
            "model_fingerprint": json.loads(row["model_fingerprint_json"]),
            "candidate_order": json.loads(row["candidate_order_json"]),
            "input_payload": json.loads(row["input_payload_json"]),
            "status": row["status"],
            "draft_text": row["draft_text"],
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
            "provenance": json.loads(row["provenance_json"]),
            "created_at": row["created_at"],
        }

    def record_annotation_assistance_draft(self, draft: Mapping[str, object]) -> None:
        """Append one helper result without touching canonical human references."""

        required = (
            "draft_id",
            "chunk_id",
            "source_sha256",
            "request_sha256",
            "prompt_version",
            "prompt_sha256",
            "model_fingerprint_sha256",
            "model_fingerprint",
            "candidate_order",
            "input_payload",
            "status",
            "draft_text",
            "error",
            "provenance",
        )
        missing = [key for key in required if key not in draft]
        if missing:
            raise StorageConflictError(
                "annotation assistance draft is missing fields: " + ", ".join(missing)
            )
        draft_id = draft["draft_id"]
        chunk_id = draft["chunk_id"]
        status = draft["status"]
        if not isinstance(draft_id, str) or not isinstance(chunk_id, str):
            raise StorageConflictError("annotation assistance draft IDs must be text")
        if status not in {"succeeded", "unavailable"}:
            raise StorageConflictError("annotation assistance draft has an invalid status")
        source_sha256 = draft["source_sha256"]
        request_sha256 = draft["request_sha256"]
        prompt_sha256 = draft["prompt_sha256"]
        model_hash = draft["model_fingerprint_sha256"]
        if not all(
            isinstance(value, str) and len(value) == 64
            for value in (source_sha256, request_sha256, prompt_sha256, model_hash)
        ):
            raise StorageConflictError("annotation assistance hashes must be SHA-256 text")
        if status == "succeeded" and not isinstance(draft["draft_text"], str):
            raise StorageConflictError("successful assistance drafts require draft_text")
        if status == "unavailable" and not isinstance(draft["error"], Mapping):
            raise StorageConflictError("unavailable assistance drafts require an error")
        chunk = self.connection.execute(
            "SELECT source_sha256 FROM audio_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if chunk is None:
            raise StorageConflictError(f"assistance chunk is not registered: {chunk_id}")
        if chunk["source_sha256"] != source_sha256:
            raise StorageConflictError("assistance source hash does not match its chunk")
        fields = {
            "draft_id": draft_id,
            "chunk_id": chunk_id,
            "source_sha256": source_sha256,
            "request_sha256": request_sha256,
            "prompt_version": draft["prompt_version"],
            "prompt_sha256": prompt_sha256,
            "model_fingerprint_sha256": model_hash,
            "model_fingerprint_json": _json(draft["model_fingerprint"]),
            "candidate_order_json": _json(draft["candidate_order"]),
            "input_payload_json": _json(draft["input_payload"]),
            "status": status,
            "draft_text": draft["draft_text"],
            "error_json": _json(draft["error"]) if draft["error"] is not None else None,
            "provenance_json": _json(draft["provenance"]),
        }
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM annotation_assistance_drafts WHERE draft_id = ?", (draft_id,)
            ).fetchone()
            if existing is not None:
                persisted = self._annotation_assistance_draft_from_row(existing)
                if all(
                    persisted[key] == draft[key]
                    for key in (
                        "draft_id",
                        "chunk_id",
                        "source_sha256",
                        "request_sha256",
                        "prompt_version",
                        "prompt_sha256",
                        "model_fingerprint_sha256",
                        "status",
                        "draft_text",
                    )
                ):
                    return
                raise StorageConflictError(
                    "annotation assistance draft identity already refers to different content"
                )
            try:
                connection.execute(
                    """INSERT INTO annotation_assistance_drafts (
                        draft_id, chunk_id, source_sha256, request_sha256,
                        prompt_version, prompt_sha256, model_fingerprint_sha256,
                        model_fingerprint_json, candidate_order_json, input_payload_json,
                        status, draft_text, error_json, provenance_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fields["draft_id"],
                        fields["chunk_id"],
                        fields["source_sha256"],
                        fields["request_sha256"],
                        fields["prompt_version"],
                        fields["prompt_sha256"],
                        fields["model_fingerprint_sha256"],
                        fields["model_fingerprint_json"],
                        fields["candidate_order_json"],
                        fields["input_payload_json"],
                        fields["status"],
                        fields["draft_text"],
                        fields["error_json"],
                        fields["provenance_json"],
                        _now(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StorageConflictError(
                    "annotation assistance draft conflicts with an existing request"
                ) from exc

    def fetch_annotation_assistance_draft(self, draft_id: str) -> dict[str, object] | None:
        """Fetch one immutable helper result."""

        row = self.connection.execute(
            "SELECT * FROM annotation_assistance_drafts WHERE draft_id = ?", (draft_id,)
        ).fetchone()
        return self._annotation_assistance_draft_from_row(row) if row is not None else None

    def find_annotation_assistance_draft(
        self, *, chunk_id: str, request_sha256: str, model_hash: str
    ) -> dict[str, object] | None:
        """Find the immutable result for one chunk/input/model tuple."""

        row = self.connection.execute(
            """SELECT * FROM annotation_assistance_drafts
            WHERE chunk_id = ? AND request_sha256 = ? AND model_fingerprint_sha256 = ?
            ORDER BY created_at DESC, draft_id DESC LIMIT 1""",
            (chunk_id, request_sha256, model_hash),
        ).fetchone()
        return self._annotation_assistance_draft_from_row(row) if row is not None else None

    @staticmethod
    def _chunk_inference_dispatch_from_row(row: sqlite3.Row) -> dict[str, object]:
        error_payload = row["error_json"]
        return {
            "dispatch_id": row["dispatch_id"],
            "manifest_id": row["manifest_id"],
            "chunk_id": row["chunk_id"],
            "adapter_id": row["adapter_id"],
            "job_id": row["job_id"],
            "model_fingerprint_sha256": row["model_fingerprint_sha256"],
            "model_payload": json.loads(row["model_payload_json"]),
            "configuration": json.loads(row["configuration_json"]),
            "preprocessing": json.loads(row["preprocessing_json"]),
            "preprocessing_sha256": row["preprocessing_sha256"],
            "source_sha256": row["source_sha256"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "request_id": row["request_id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "input_audio_sha256": row["input_audio_sha256"],
            "raw_artifact_id": row["raw_artifact_id"],
            "hypothesis_id": row["hypothesis_id"],
            "runtime": json.loads(row["runtime_json"]),
            "error": ApiError.from_dict(json.loads(error_payload)) if error_payload else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def register_chunk_inference_dispatch(
        self,
        *,
        dispatch_id: str,
        manifest_id: str,
        chunk_id: str,
        adapter_id: str,
        job_id: str,
        model_fingerprint_sha256: str,
        model_payload: Mapping[str, object],
        configuration: Mapping[str, object],
        preprocessing: Mapping[str, object],
        preprocessing_sha256: str,
        source_sha256: str,
        start_ms: int,
        end_ms: int,
        request_id: str,
    ) -> dict[str, object]:
        """Register one idempotent per-adapter chunk dispatch."""

        model_json = _json(dict(model_payload))
        configuration_json = _json(dict(configuration))
        preprocessing_json = _json(dict(preprocessing))
        now = _now()
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT * FROM chunk_inference_dispatches
                WHERE dispatch_id = ? OR (manifest_id = ? AND chunk_id = ? AND adapter_id = ?)""",
                (dispatch_id, manifest_id, chunk_id, adapter_id),
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["dispatch_id"],
                    existing["manifest_id"],
                    existing["chunk_id"],
                    existing["adapter_id"],
                    existing["job_id"],
                    existing["model_fingerprint_sha256"],
                    existing["model_payload_json"],
                    existing["configuration_json"],
                    existing["preprocessing_json"],
                    existing["preprocessing_sha256"],
                    existing["source_sha256"],
                    existing["start_ms"],
                    existing["end_ms"],
                    existing["request_id"],
                )
                expected = (
                    dispatch_id,
                    manifest_id,
                    chunk_id,
                    adapter_id,
                    job_id,
                    model_fingerprint_sha256,
                    model_json,
                    configuration_json,
                    preprocessing_json,
                    preprocessing_sha256,
                    source_sha256,
                    start_ms,
                    end_ms,
                    request_id,
                )
                if immutable != expected:
                    raise StorageConflictError(
                        "chunk inference dispatch identity already refers to different content"
                    )
            else:
                connection.execute(
                    """INSERT INTO chunk_inference_dispatches (
                        dispatch_id, manifest_id, chunk_id, adapter_id, job_id,
                        model_fingerprint_sha256, model_payload_json, configuration_json,
                        preprocessing_json, preprocessing_sha256, source_sha256,
                        start_ms, end_ms, request_id, status, attempts,
                        input_audio_sha256, raw_artifact_id, hypothesis_id, runtime_json,
                        error_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0,
                        NULL, NULL, NULL, '{}', NULL, ?, ?)""",
                    (
                        dispatch_id,
                        manifest_id,
                        chunk_id,
                        adapter_id,
                        job_id,
                        model_fingerprint_sha256,
                        model_json,
                        configuration_json,
                        preprocessing_json,
                        preprocessing_sha256,
                        source_sha256,
                        start_ms,
                        end_ms,
                        request_id,
                        now,
                        now,
                    ),
                )
            row = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return self._chunk_inference_dispatch_from_row(row)

    def fetch_chunk_inference_dispatch(self, dispatch_id: str) -> dict[str, object] | None:
        """Read one dispatch row, including its typed visible failure if present."""

        row = self.connection.execute(
            "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        return self._chunk_inference_dispatch_from_row(row) if row is not None else None

    def list_chunk_inference_dispatches(self, manifest_id: str) -> tuple[dict[str, object], ...]:
        """Read all dispatches for a manifest in stable model/chunk order."""

        rows = self.connection.execute(
            """SELECT * FROM chunk_inference_dispatches
            WHERE manifest_id = ? ORDER BY adapter_id, chunk_id""",
            (manifest_id,),
        ).fetchall()
        return tuple(self._chunk_inference_dispatch_from_row(row) for row in rows)

    def start_chunk_inference_dispatch(
        self,
        dispatch_id: str,
        job: JobStatus,
        *,
        input_audio_sha256: str | None,
        now: datetime | str | None = None,
    ) -> dict[str, object]:
        """Fence and mark a dispatch running after common audio identity is known."""

        with self.transaction() as connection:
            self._assert_job_fence(connection, job, now=now)
            row = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
            if row is None:
                raise StorageConflictError(f"unknown chunk inference dispatch: {dispatch_id}")
            previous_audio_hash = row["input_audio_sha256"]
            if previous_audio_hash is not None and previous_audio_hash != input_audio_sha256:
                raise StorageConflictError("common chunk audio identity changed across reruns")
            connection.execute(
                """UPDATE chunk_inference_dispatches
                SET status='running', attempts=?, input_audio_sha256=?, updated_at=?
                WHERE dispatch_id = ?""",
                (job.attempts, input_audio_sha256, _now(), dispatch_id),
            )
            updated = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return self._chunk_inference_dispatch_from_row(updated)

    def publish_chunk_inference_success(
        self,
        dispatch_id: str,
        job: JobStatus,
        hypothesis: TranscriptionHypothesis,
        *,
        raw_artifact_id: str,
        input_audio_sha256: str | None,
        runtime: Mapping[str, object],
        now: datetime | str | None = None,
    ) -> dict[str, object]:
        """Publish a hypothesis and its dispatch result under one worker fence."""

        with self.transaction() as connection:
            self._assert_job_fence(connection, job, now=now)
            row = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
            if row is None:
                raise StorageConflictError(f"unknown chunk inference dispatch: {dispatch_id}")
            if row["input_audio_sha256"] not in {None, input_audio_sha256}:
                raise StorageConflictError("common chunk audio identity changed across reruns")
            self._record_transcription_hypothesis(connection, hypothesis)
            if row["status"] == "succeeded" and row["hypothesis_id"] not in {
                None,
                hypothesis.hypothesis_id,
            }:
                raise StorageConflictError("dispatch already has a different hypothesis")
            connection.execute(
                """UPDATE chunk_inference_dispatches SET
                    status='succeeded', attempts=?, input_audio_sha256=?,
                    raw_artifact_id=?, hypothesis_id=?, runtime_json=?,
                    error_json=NULL, updated_at=?
                WHERE dispatch_id = ?""",
                (
                    job.attempts,
                    input_audio_sha256,
                    raw_artifact_id,
                    hypothesis.hypothesis_id,
                    _json(dict(runtime)),
                    _now(),
                    dispatch_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return self._chunk_inference_dispatch_from_row(updated)

    def record_chunk_inference_failure(
        self,
        dispatch_id: str,
        job: JobStatus,
        error: ApiError,
        *,
        input_audio_sha256: str | None,
        runtime: Mapping[str, object],
        now: datetime | str | None = None,
    ) -> dict[str, object]:
        """Persist a typed model-specific failure without affecting peer dispatches."""

        with self.transaction() as connection:
            self._assert_job_fence(connection, job, now=now)
            row = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
            if row is None:
                raise StorageConflictError(f"unknown chunk inference dispatch: {dispatch_id}")
            if row["input_audio_sha256"] not in {None, input_audio_sha256}:
                raise StorageConflictError("common chunk audio identity changed across reruns")
            connection.execute(
                """UPDATE chunk_inference_dispatches SET
                    status='failed', attempts=?, input_audio_sha256=?, runtime_json=?,
                    error_json=?, updated_at=? WHERE dispatch_id = ?""",
                (
                    job.attempts,
                    input_audio_sha256,
                    _json(dict(runtime)),
                    _json(error.to_dict()),
                    _now(),
                    dispatch_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return self._chunk_inference_dispatch_from_row(updated)

    def reconcile_chunk_inference_success(
        self,
        dispatch_id: str,
        *,
        hypothesis_id: str,
        raw_artifact_id: str | None,
        input_audio_sha256: str | None,
    ) -> dict[str, object]:
        """Repair a durable dispatch after a crash between publication steps."""

        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
            if row is None:
                raise StorageConflictError(f"unknown chunk inference dispatch: {dispatch_id}")
            if row["hypothesis_id"] not in {None, hypothesis_id}:
                raise StorageConflictError("dispatch reconciliation found a different hypothesis")
            if row["input_audio_sha256"] not in {None, input_audio_sha256}:
                raise StorageConflictError("dispatch reconciliation found different audio bytes")
            connection.execute(
                """UPDATE chunk_inference_dispatches SET status='succeeded',
                    hypothesis_id=?, raw_artifact_id=COALESCE(?, raw_artifact_id),
                    input_audio_sha256=COALESCE(?, input_audio_sha256),
                    error_json=NULL, updated_at=? WHERE dispatch_id=?""",
                (hypothesis_id, raw_artifact_id, input_audio_sha256, _now(), dispatch_id),
            )
            updated = connection.execute(
                "SELECT * FROM chunk_inference_dispatches WHERE dispatch_id = ?",
                (dispatch_id,),
            ).fetchone()
        return self._chunk_inference_dispatch_from_row(updated)

    @staticmethod
    def _record_transcript_reference(
        connection: sqlite3.Connection, reference: TranscriptReference
    ) -> None:
        chunk = connection.execute(
            "SELECT source_sha256 FROM audio_chunks WHERE chunk_id = ?", (reference.chunk_id,)
        ).fetchone()
        if chunk is None:
            raise StorageConflictError(f"reference chunk is not registered: {reference.chunk_id}")
        if chunk["source_sha256"] != reference.source_sha256:
            raise StorageConflictError("reference source hash does not match its chunk")
        payload = _json(reference.to_dict())
        existing = connection.execute(
            "SELECT payload_json FROM transcript_references WHERE reference_id = ?",
            (reference.reference_id,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] == payload:
                return
            raise StorageConflictError("reference identity already refers to different content")
        connection.execute(
            """INSERT INTO transcript_references (
                reference_id, chunk_id, source_sha256, revision, review_status,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                reference.reference_id,
                reference.chunk_id,
                reference.source_sha256,
                reference.revision,
                reference.review_status,
                payload,
                reference.created_at or _now(),
            ),
        )

    def record_transcript_reference(self, reference: TranscriptReference) -> None:
        """Publish the immutable logical reference record."""

        with self.transaction() as connection:
            self._record_transcript_reference(connection, reference)

    @staticmethod
    def _record_reference_revision(
        connection: sqlite3.Connection, revision: ReferenceRevision
    ) -> None:
        reference = connection.execute(
            """SELECT chunk_id FROM transcript_references WHERE reference_id = ?""",
            (revision.reference_id,),
        ).fetchone()
        if reference is None:
            raise StorageConflictError(f"reference is not registered: {revision.reference_id}")
        if reference["chunk_id"] != revision.chunk_id:
            raise StorageConflictError("reference revision chunk does not match reference")
        chunk = connection.execute(
            "SELECT source_sha256 FROM audio_chunks WHERE chunk_id = ?", (revision.chunk_id,)
        ).fetchone()
        if chunk is None or chunk["source_sha256"] != revision.source_sha256:
            raise StorageConflictError("reference revision source hash does not match its chunk")
        latest = connection.execute(
            """SELECT revision, revision_id FROM reference_revisions
            WHERE reference_id = ? ORDER BY revision DESC LIMIT 1""",
            (revision.reference_id,),
        ).fetchone()
        expected_revision = 1 if latest is None else latest["revision"] + 1
        if revision.revision != expected_revision:
            raise StorageConflictError(
                "reference revision conflict: "
                f"expected {expected_revision}, got {revision.revision}"
            )
        expected_prior = None if latest is None else latest["revision_id"]
        if revision.prior_revision_id != expected_prior:
            raise StorageConflictError("reference revision predecessor does not match history")
        payload = _json(revision.to_dict())
        existing = connection.execute(
            "SELECT payload_json FROM reference_revisions WHERE revision_id = ?",
            (revision.revision_id,),
        ).fetchone()
        if existing is not None:
            if existing["payload_json"] == payload:
                return
            raise StorageConflictError(
                "reference revision identity already refers to different content"
            )
        connection.execute(
            """INSERT INTO reference_revisions (
                revision_id, reference_id, chunk_id, revision, reviewer, source_sha256,
                prior_revision_id, review_status, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                revision.revision_id,
                revision.reference_id,
                revision.chunk_id,
                revision.revision,
                revision.reviewer,
                revision.source_sha256,
                revision.prior_revision_id,
                revision.review_status,
                payload,
                revision.created_at or _now(),
            ),
        )

    def append_reference_revision(self, revision: ReferenceRevision) -> None:
        """Append a reference revision without updating or deleting history."""

        with self.transaction() as connection:
            self._record_reference_revision(connection, revision)

    def fetch_reference_revision(self, revision_id: str) -> ReferenceRevision | None:
        """Restore one append-only reference revision."""

        row = self.connection.execute(
            "SELECT payload_json FROM reference_revisions WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()
        return ReferenceRevision.from_dict(json.loads(row["payload_json"])) if row else None

    def record_chunk_benchmark_manifest(self, manifest: ChunkBenchmarkManifest) -> None:
        """Publish a manifest and its immutable chunk/hypothesis/reference records."""

        payload = _json(manifest.to_dict())
        content_sha256 = manifest.content_sha256
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM chunk_benchmark_manifests WHERE manifest_id = ?",
                (manifest.manifest_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise StorageConflictError(
                        "benchmark manifest identity already refers to different content"
                    )
            else:
                connection.execute(
                    """INSERT INTO chunk_benchmark_manifests (
                        manifest_id, schema_version, source_manifest_sha256,
                        content_sha256, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        manifest.manifest_id,
                        manifest.schema_version,
                        manifest.source_manifest_sha256,
                        content_sha256,
                        payload,
                        _now(),
                    ),
                )
            for chunk in manifest.chunks:
                self._record_audio_chunk(connection, chunk, manifest.manifest_id)
            for reference in manifest.references:
                self._record_transcript_reference(connection, reference)
            for hypothesis in manifest.hypotheses:
                self._record_transcription_hypothesis(connection, hypothesis)

    def fetch_chunk_benchmark_manifest(self, manifest_id: str) -> ChunkBenchmarkManifest | None:
        """Restore one versioned benchmark manifest."""

        row = self.connection.execute(
            "SELECT payload_json FROM chunk_benchmark_manifests WHERE manifest_id = ?",
            (manifest_id,),
        ).fetchone()
        return (
            ChunkBenchmarkManifest.from_dict(json.loads(row["payload_json"]))
            if row is not None
            else None
        )

    def enqueue_job(
        self,
        *,
        job_id: str,
        stage: str,
        source_sha256: str | None = None,
        upstream_artifact_hashes: tuple[str, ...] = (),
        model_fingerprint_sha256: str | None = None,
        configuration: dict[str, object] | None = None,
        pipeline_version: str = "dev",
        stage_key: str | None = None,
        request_id: str | None = None,
        paid: bool = False,
    ) -> JobStatus:
        """Insert one queued job, idempotently when its immutable input matches."""

        if not job_id or not stage:
            raise ValueError("job_id and stage must be non-empty")
        if source_sha256 is None:
            source_sha256 = "0" * 64
        key = stage_key or stage_fingerprint(
            stage,
            source_sha256=source_sha256,
            upstream_artifact_hashes=upstream_artifact_hashes,
            model_fingerprint_sha256=model_fingerprint_sha256,
            configuration=configuration,
            pipeline_version=pipeline_version,
        )
        config_json = _json(configuration or {})
        upstream_json = _json(upstream_artifact_hashes)
        now = _now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["stage"],
                    existing["stage_key"],
                    existing["source_sha256"],
                    existing["upstream_artifact_hashes_json"],
                    existing["model_fingerprint_sha256"],
                    existing["configuration_json"],
                    existing["pipeline_version"],
                    existing["paid"],
                )
                expected = (
                    stage,
                    key,
                    source_sha256,
                    upstream_json,
                    model_fingerprint_sha256,
                    config_json,
                    pipeline_version,
                    int(paid),
                )
                if immutable != expected:
                    raise StorageConflictError("job identity already refers to different content")
                return self._job_from_row(existing)
            connection.execute(
                """INSERT INTO jobs (
                    job_id, stage, status, attempts, fencing_token, owner,
                    lease_expires_at, request_id, error_json, recovery_action,
                    created_at, updated_at, stage_key, source_sha256,
                    upstream_artifact_hashes_json, model_fingerprint_sha256,
                    configuration_json, pipeline_version, paid
                ) VALUES (
                    ?, ?, 'queued', 0, 0, NULL, NULL, ?, NULL, NULL,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    job_id,
                    stage,
                    request_id,
                    now,
                    now,
                    key,
                    source_sha256,
                    upstream_json,
                    model_fingerprint_sha256,
                    config_json,
                    pipeline_version,
                    int(paid),
                ),
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_from_row(row)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobStatus:
        error_payload = row["error_json"]
        error = ApiError.from_dict(json.loads(error_payload)) if error_payload else None
        return JobStatus(
            job_id=row["job_id"],
            stage=row["stage"],
            status=row["status"],
            attempts=row["attempts"],
            fencing_token=row["fencing_token"],
            owner=row["owner"],
            lease_expires_at=row["lease_expires_at"],
            error=error,
            recovery_action=row["recovery_action"],
            request_id=row["request_id"],
            paid=bool(row["paid"]),
        )

    def fetch_job(self, job_id: str) -> JobStatus | None:
        """Read a durable job as its typed contract."""

        row = self.connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._job_from_row(row) if row is not None else None

    def assert_job_fence(
        self,
        job_id: str,
        *,
        owner: str,
        fencing_token: int,
        now: datetime | str | None = None,
    ) -> JobStatus:
        """Verify a worker still owns its lease before publishing stage output."""

        return self._assert_job_fence(
            self.connection,
            JobStatus(
                job_id=job_id,
                stage="fence",
                status="running",
                owner=owner,
                fencing_token=fencing_token,
            ),
            now=now,
        )

    def _assert_job_fence(
        self,
        connection: sqlite3.Connection,
        job: JobStatus,
        *,
        now: datetime | str | None = None,
    ) -> JobStatus:
        """Verify a claimed job fence using the publication transaction."""

        current_row = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)
        ).fetchone()
        current = self._job_from_row(current_row) if current_row is not None else None
        stamp = self._timestamp(self._instant(now))
        if (
            current is None
            or current.status != "running"
            or current.owner != job.owner
            or current.fencing_token != job.fencing_token
            or current.lease_expires_at is None
            or current.lease_expires_at <= stamp
        ):
            raise StorageConflictError("stale worker cannot publish this job output")
        return current

    @staticmethod
    def _instant(value: datetime | str | None) -> datetime:
        if value is None:
            return datetime.now(UTC)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="seconds")

    def claim_job(
        self,
        job_id: str,
        owner: str,
        *,
        now: datetime | str | None = None,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ) -> JobStatus:
        """Claim a specific eligible job with a new fencing token."""

        if not owner or lease_seconds <= 0 or max_attempts <= 0:
            raise ValueError("owner, lease_seconds and max_attempts must be valid")
        instant = self._instant(now)
        stamp = self._timestamp(instant)
        expiry = self._timestamp(instant + timedelta(seconds=lease_seconds))
        blocked_paid = False
        with self.connection:
            self._begin_immediate()
            row = self.connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job {job_id}")
            eligible = row["status"] in {"queued", "retry_wait"}
            if row["status"] == "retry_wait" and (
                row["lease_expires_at"] is not None and row["lease_expires_at"] > stamp
            ):
                eligible = False
            expired = row["status"] == "running" and (
                row["lease_expires_at"] is None or row["lease_expires_at"] <= stamp
            )
            if not eligible and not expired:
                raise StorageConflictError(f"job {job_id} is not eligible for claim")
            if row["attempts"] >= max_attempts:
                raise StorageConflictError(f"job {job_id} exhausted its retry budget")
            if row["paid"] and not self._paid_attempt_is_dispatchable(self.connection, row):
                self._block_paid_attempt(self.connection, row, stamp)
                blocked_paid = True
            else:
                connection = self.connection
                connection.execute(
                    """UPDATE jobs SET status='running', attempts=attempts+1,
                    fencing_token=fencing_token+1, owner=?, lease_expires_at=?,
                    recovery_action=?, updated_at=? WHERE job_id=?""",
                    (
                        owner,
                        expiry,
                        "reclaimed_expired_lease" if expired else row["recovery_action"],
                        stamp,
                        job_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        if blocked_paid:
            raise StorageConflictError(
                f"paid job {job_id} requires a new reserved attempt before dispatch"
            )
        return self._job_from_row(updated)

    @staticmethod
    def _paid_attempt_is_dispatchable(connection: sqlite3.Connection, row: sqlite3.Row) -> bool:
        """Allow only a fresh paid job backed by its own active reservation."""

        if (
            row["status"] != "queued"
            or row["attempts"] != 0
            or row["recovery_action"] is not None
            or not row["request_id"]
        ):
            return False
        reservation = connection.execute(
            "SELECT status FROM cost_reservations WHERE request_id=?",
            (row["request_id"],),
        ).fetchone()
        prior_dispatch = connection.execute(
            """SELECT 1 FROM jobs WHERE request_id=? AND job_id<>? AND attempts>0
            LIMIT 1""",
            (row["request_id"], row["job_id"]),
        ).fetchone()
        return (
            reservation is not None
            and reservation["status"] == "reserved"
            and prior_dispatch is None
        )

    @staticmethod
    def _block_paid_attempt(connection: sqlite3.Connection, row: sqlite3.Row, stamp: str) -> None:
        """Block a paid retry/reclaim and retain an unknown charge for reconciliation."""

        request_id = row["request_id"] or f"job-{row['job_id']}"
        ambiguous = row["status"] == "running" or row["attempts"] > 0
        code = "paid_outcome_ambiguous" if ambiguous else "budget_blocked"
        message = (
            "paid attempt outcome must be reconciled before another dispatch"
            if ambiguous
            else "paid job requires an active reservation for this attempt"
        )
        error = ApiError(code, message, False, request_id)
        connection.execute(
            """UPDATE jobs SET status='blocked', owner=NULL, lease_expires_at=NULL,
            error_json=?, recovery_action=?, updated_at=? WHERE job_id=?""",
            (
                _json(error.to_dict()),
                "reconcile_paid_request" if ambiguous else "reserve_paid_attempt",
                stamp,
                row["job_id"],
            ),
        )
        if ambiguous and row["request_id"]:
            connection.execute(
                """UPDATE cost_reservations SET status='ambiguous', updated_at=?
                WHERE request_id=? AND status='reserved'""",
                (stamp, row["request_id"]),
            )

    def claim_next_job(
        self,
        owner: str,
        *,
        now: datetime | str | None = None,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ) -> JobStatus | None:
        """Claim the oldest eligible job, serializing selection with BEGIN IMMEDIATE."""

        instant = self._instant(now)
        stamp = self._timestamp(instant)
        with self.connection:
            self._begin_immediate()
            while True:
                row = self.connection.execute(
                    """SELECT * FROM jobs WHERE attempts < ? AND
                ((status IN ('queued','retry_wait') AND
                    (status='queued' OR lease_expires_at IS NULL OR lease_expires_at <= ?))
                OR (status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))
                ORDER BY created_at, job_id LIMIT 1""",
                    (max_attempts, stamp, stamp),
                ).fetchone()
                if row is None:
                    return None
                if row["paid"] and not self._paid_attempt_is_dispatchable(self.connection, row):
                    self._block_paid_attempt(self.connection, row, stamp)
                    continue
                break
            job_id = row["job_id"]
            current = self.connection.execute(
                "SELECT attempts FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if current["attempts"] >= max_attempts:
                error = ApiError(
                    "retry_exhausted", "maximum attempts exhausted", False, f"job-{job_id}"
                )
                self.connection.execute(
                    """UPDATE jobs SET status='failed', error_json=?,
                    recovery_action='retry_exhausted', updated_at=? WHERE job_id=?""",
                    (_json(error.to_dict()), stamp, job_id),
                )
                return None
            expiry = self._timestamp(instant + timedelta(seconds=lease_seconds))
            old = self.connection.execute(
                "SELECT status FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            self.connection.execute(
                """UPDATE jobs SET status='running', attempts=attempts+1,
                fencing_token=fencing_token+1, owner=?, lease_expires_at=?,
                recovery_action=?, updated_at=? WHERE job_id=?""",
                (
                    owner,
                    expiry,
                    "reclaimed_expired_lease" if old["status"] == "running" else None,
                    stamp,
                    job_id,
                ),
            )
            claimed = self.connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._job_from_row(claimed)

    def heartbeat(
        self,
        job_id: str,
        *,
        owner: str,
        fencing_token: int,
        now: datetime | str | None = None,
        lease_seconds: int = 120,
    ) -> JobStatus:
        """Extend a live lease only for its current owner and fencing token."""

        instant = self._instant(now)
        stamp = self._timestamp(instant)
        expiry = self._timestamp(instant + timedelta(seconds=lease_seconds))
        connection = self.connection
        separate_connection = False
        if self._database_path is not None and get_ident() != self._connection_thread_id:
            connection = open_database(self._database_path)
            separate_connection = True
        try:
            with connection:
                result = connection.execute(
                    """UPDATE jobs SET lease_expires_at=?, updated_at=?
                    WHERE job_id=? AND status='running' AND owner=? AND fencing_token=?
                    AND lease_expires_at > ?""",
                    (expiry, stamp, job_id, owner, fencing_token, stamp),
                )
                if result.rowcount != 1:
                    raise StorageConflictError(
                        "lease heartbeat rejected by fencing token or expiry"
                    )
                row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            if separate_connection:
                connection.close()
        return self._job_from_row(row)

    def complete_job(
        self,
        job_id: str,
        *,
        owner: str,
        fencing_token: int,
        now: datetime | str | None = None,
    ) -> JobStatus:
        """Mark a job succeeded only from its current fenced lease."""

        stamp = self._timestamp(self._instant(now))
        with self.transaction() as connection:
            result = connection.execute(
                """UPDATE jobs SET status='succeeded', owner=NULL, lease_expires_at=NULL,
                recovery_action=NULL, updated_at=? WHERE job_id=? AND status='running'
                AND owner=? AND fencing_token=? AND lease_expires_at > ?""",
                (stamp, job_id, owner, fencing_token, stamp),
            )
            if result.rowcount != 1:
                raise StorageConflictError("stale worker cannot complete this job")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job_from_row(row)

    def fail_job(
        self,
        job_id: str,
        *,
        owner: str,
        fencing_token: int,
        error: ApiError,
        failure_class: str = "transient",
        now: datetime | str | None = None,
        max_attempts: int = 3,
    ) -> JobStatus:
        """Persist a typed failure with bounded retry/backoff semantics."""

        if failure_class not in {"transient", "deterministic", "ambiguous"}:
            raise ValueError("unknown failure class")
        instant = self._instant(now)
        stamp = self._timestamp(instant)
        if failure_class in {"deterministic", "ambiguous"}:
            status = "blocked"
            recovery = (
                "reconcile_paid_request"
                if failure_class == "ambiguous"
                else "fix_deterministic_failure"
            )
            retry_at = None
        elif self.fetch_job(job_id) and self.fetch_job(job_id).attempts >= max_attempts:
            status = "failed"
            recovery = "retry_exhausted"
            retry_at = None
        else:
            status = "retry_wait"
            index = max(0, (self.fetch_job(job_id).attempts if self.fetch_job(job_id) else 1) - 1)
            delay = RETRY_BACKOFF_SECONDS[min(index, len(RETRY_BACKOFF_SECONDS) - 1)]
            recovery = f"retry_after_{delay}s"
            retry_at = self._timestamp(instant + timedelta(seconds=delay))
        with self.transaction() as connection:
            result = connection.execute(
                """UPDATE jobs SET status=?, owner=NULL, lease_expires_at=?, error_json=?,
                recovery_action=?, updated_at=? WHERE job_id=? AND status='running'
                AND owner=? AND fencing_token=?""",
                (
                    status,
                    retry_at,
                    _json(error.to_dict()),
                    recovery,
                    stamp,
                    job_id,
                    owner,
                    fencing_token,
                ),
            )
            if result.rowcount != 1:
                raise StorageConflictError("stale worker cannot record this failure")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if failure_class == "ambiguous" and row["request_id"]:
                connection.execute(
                    """UPDATE cost_reservations SET status='ambiguous', updated_at=?
                    WHERE request_id=? AND status='reserved'""",
                    (stamp, row["request_id"]),
                )
        return self._job_from_row(row)

    def cancel_job(
        self,
        job_id: str,
        *,
        now: datetime | str | None = None,
        owner: str | None = None,
        fencing_token: int | None = None,
    ) -> JobStatus:
        """Cancel a queued job or revoke a running lease by advancing its fence."""

        stamp = self._timestamp(self._instant(now))
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown job {job_id}")
            if row["status"] in {"succeeded", "failed", "blocked", "cancelled"}:
                raise StorageConflictError("terminal jobs cannot be cancelled")
            if row["status"] == "running" and (owner is not None or fencing_token is not None):
                if row["owner"] != owner or row["fencing_token"] != fencing_token:
                    raise StorageConflictError("stale worker cannot cancel this job")
            connection.execute(
                """UPDATE jobs SET status='cancelled', owner=NULL, lease_expires_at=NULL,
                fencing_token=fencing_token+1, recovery_action='cancelled_by_operator', updated_at=?
                WHERE job_id=?""",
                (stamp, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job_from_row(updated)

    def retry_job(self, job_id: str, *, now: datetime | str | None = None) -> JobStatus:
        """Queue a failed job for an explicit operator retry."""

        stamp = self._timestamp(self._instant(now))
        with self.transaction() as connection:
            result = connection.execute(
                """UPDATE jobs SET status='queued', attempts=0, error_json=NULL,
                lease_expires_at=NULL, owner=NULL, recovery_action='manual_retry', updated_at=?
                WHERE job_id=? AND status='failed'""",
                (stamp, job_id),
            )
            if result.rowcount != 1:
                raise StorageConflictError("only exhausted failed jobs can be retried")
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job_from_row(row)

    def recover_expired_leases(
        self, *, now: datetime | str | None = None, max_attempts: int = 3
    ) -> tuple[str, ...]:
        """Requeue expired jobs, or persist retry exhaustion, in one short transaction."""

        stamp = self._timestamp(self._instant(now))
        recovered: list[str] = []
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM jobs WHERE status='running' AND lease_expires_at IS NOT NULL
                AND lease_expires_at <= ? ORDER BY job_id""",
                (stamp,),
            ).fetchall()
            for row in rows:
                if row["paid"]:
                    self._block_paid_attempt(connection, row, stamp)
                elif row["attempts"] >= max_attempts:
                    error = ApiError(
                        "retry_exhausted",
                        "maximum attempts exhausted",
                        False,
                        row["request_id"] or f"job-{row['job_id']}",
                    )
                    connection.execute(
                        """UPDATE jobs SET status='failed', owner=NULL, lease_expires_at=NULL,
                        error_json=?, recovery_action='retry_exhausted',
                        updated_at=? WHERE job_id=?""",
                        (_json(error.to_dict()), stamp, row["job_id"]),
                    )
                else:
                    connection.execute(
                        """UPDATE jobs SET status='queued', owner=NULL, lease_expires_at=NULL,
                        recovery_action='requeued_after_expired_lease',
                        updated_at=? WHERE job_id=?""",
                        (stamp, row["job_id"]),
                    )
                recovered.append(row["job_id"])
        return tuple(recovered)

    def _update_reservation(
        self, request_id: str, status: str, *, spent_microusd: int | None = None
    ) -> None:
        if status not in {"settled", "released", "ambiguous"}:
            raise ValueError("unsupported reservation status")
        with self.transaction() as connection:
            result = connection.execute(
                """UPDATE cost_reservations SET status=?,
                spent_microusd=COALESCE(?, spent_microusd), updated_at=?
                WHERE request_id=? AND status IN ('reserved','ambiguous')""",
                (status, spent_microusd, _now(), request_id),
            )
            if result.rowcount != 1:
                raise StorageConflictError("cost reservation is missing or already finalized")

    def mark_cost_ambiguous(self, request_id: str) -> None:
        """Keep an unknown paid attempt reserved until it is reconciled."""

        self._update_reservation(request_id, "ambiguous")

    def settle_cost(self, request_id: str, spent_microusd: int) -> None:
        if spent_microusd < 0:
            raise ValueError("spent_microusd must be non-negative")
        self._update_reservation(request_id, "settled", spent_microusd=spent_microusd)

    def release_cost(self, request_id: str) -> None:
        self._update_reservation(request_id, "released")

    def reconcile_artifact(self, manifest: ArtifactManifest, artifact_path: Path) -> bool:
        """Insert a complete orphan manifest, returning whether it was new."""

        row = self.connection.execute(
            "SELECT artifact_id, manifest_sha256 FROM artifacts WHERE artifact_id = ?",
            (manifest.artifact_id,),
        ).fetchone()
        if row is not None:
            return False
        self.record_artifact(manifest, artifact_path)
        return True

    def publish_generation(
        self,
        *,
        generation_id: str,
        kind: str,
        generation_key: str,
        pointer_name: str,
        expected_points: int,
        job: JobStatus | None = None,
        now: datetime | str | None = None,
    ) -> None:
        """Publish a validated index generation and its pointer atomically."""

        if expected_points < 0:
            raise ValueError("expected_points must be non-negative")
        published_at = self._timestamp(self._instant(now))
        with self.transaction() as connection:
            if job is not None:
                self._assert_job_fence(connection, job, now=now)
            connection.execute(
                """INSERT INTO index_generations
                (generation_id, kind, generation_key, status, expected_points, published_at)
                VALUES (?, ?, ?, 'complete', ?, ?)""",
                (generation_id, kind, generation_key, expected_points, published_at),
            )
            connection.execute(
                """INSERT INTO committed_pointers(pointer_name, generation_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(pointer_name) DO UPDATE SET
                    generation_id = excluded.generation_id, updated_at = excluded.updated_at""",
                (pointer_name, generation_id, published_at),
            )

    def append_identity_decision(self, decision: IdentityDecision) -> None:
        """Append a human decision and its audit revision in one transaction."""

        payload = _json(decision.to_dict())
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO identity_decisions (
                    decision_id, left_episode_speaker_id, right_episode_speaker_id,
                    decision, reviewer, evidence_artifact_ids_json, created_at, state, revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.decision_id,
                    decision.left_episode_speaker_id,
                    decision.right_episode_speaker_id,
                    decision.decision,
                    decision.reviewer,
                    _json(decision.evidence_artifact_ids),
                    decision.created_at,
                    decision.state,
                    decision.revision,
                ),
            )
            connection.execute(
                """INSERT INTO identity_audit(decision_id, revision, payload_json, created_at)
                VALUES (?, ?, ?, ?)""",
                (decision.decision_id, decision.revision, payload, _now()),
            )

    def artifact_exists(self, artifact_id: str) -> bool:
        """Return whether an artifact is registered in canonical SQLite state."""

        return (
            self.connection.execute(
                "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
            is not None
        )

    def append_annotation_revision(
        self,
        *,
        episode_id: str,
        source_artifact_id: str | None,
        source_sha256: str,
        reviewer: str,
        status: str,
        payload: dict[str, object],
        expected_revision: int,
        created_at: str | None = None,
    ) -> dict[str, object]:
        """Append one annotation revision without overwriting an earlier review."""

        if status not in {"draft", "accepted"}:
            raise ValueError("annotation status must be draft or accepted")
        if expected_revision < 0:
            raise ValueError("expected_revision must be non-negative")
        if not reviewer:
            raise ValueError("annotation reviewer must be non-empty")
        stamp = created_at or _now()
        with self.transaction() as connection:
            current_row = connection.execute(
                """SELECT COALESCE(MAX(revision), 0) AS revision
                FROM annotation_revisions WHERE episode_id = ?""",
                (episode_id,),
            ).fetchone()
            current_revision = int(current_row["revision"])
            if current_revision != expected_revision:
                raise StorageConflictError(
                    f"annotation revision conflict: expected {expected_revision}, "
                    f"current {current_revision}"
                )
            revision = current_revision + 1
            persisted = dict(payload)
            persisted.update(
                {
                    "episode_id": episode_id,
                    "source_sha256": source_sha256,
                    "reviewer": reviewer,
                    "status": "reviewed" if status == "accepted" else "draft",
                    "revision": revision,
                    "created_at": stamp,
                }
            )
            if status == "accepted":
                persisted["reviewed_at"] = persisted.get("reviewed_at") or stamp
            annotation_revision_id = str(
                persisted.get("annotation_revision_id")
                or "annotation-"
                + hashlib.sha256(
                    _json(
                        {
                            "episode_id": episode_id,
                            "revision": revision,
                            "payload": persisted,
                        }
                    ).encode("utf-8")
                ).hexdigest()[:24]
            )
            persisted["annotation_revision_id"] = annotation_revision_id
            connection.execute(
                """INSERT INTO annotation_revisions (
                    annotation_revision_id, episode_id, source_artifact_id, source_sha256,
                    revision, reviewer, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    annotation_revision_id,
                    episode_id,
                    source_artifact_id,
                    source_sha256,
                    revision,
                    reviewer,
                    status,
                    _json(persisted),
                    stamp,
                ),
            )
            connection.execute(
                """INSERT INTO review_records (
                    review_id, annotation_revision_id, reviewer, target_type, target_id,
                    decision, revision, state, payload_json, created_at
                ) VALUES (?, ?, ?, 'annotation', ?, ?, ?, 'active', ?, ?)""",
                (
                    f"review-{annotation_revision_id}",
                    annotation_revision_id,
                    reviewer,
                    episode_id,
                    "reviewed" if status == "accepted" else "draft",
                    revision,
                    _json(persisted),
                    stamp,
                ),
            )
        return persisted

    @staticmethod
    def _annotation_from_row(row: sqlite3.Row) -> dict[str, object]:
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            raise StorageError("annotation revision payload must be a JSON object")
        payload.setdefault("annotation_revision_id", row["annotation_revision_id"])
        payload.setdefault("episode_id", row["episode_id"])
        payload.setdefault("source_sha256", row["source_sha256"])
        payload.setdefault("revision", row["revision"])
        payload.setdefault("reviewer", row["reviewer"])
        payload.setdefault("status", "reviewed" if row["status"] == "accepted" else "draft")
        payload.setdefault("created_at", row["created_at"])
        return payload

    def fetch_annotation_revision(
        self, episode_id: str, revision: int | None = None
    ) -> dict[str, object] | None:
        """Read one immutable annotation revision or the latest revision."""

        if revision is None:
            row = self.connection.execute(
                """SELECT * FROM annotation_revisions
                WHERE episode_id = ? ORDER BY revision DESC LIMIT 1""",
                (episode_id,),
            ).fetchone()
        else:
            row = self.connection.execute(
                """SELECT * FROM annotation_revisions
                WHERE episode_id = ? AND revision = ?""",
                (episode_id, revision),
            ).fetchone()
        return self._annotation_from_row(row) if row is not None else None

    def list_annotation_revisions(self, episode_id: str) -> tuple[dict[str, object], ...]:
        """Return append-only annotation history from newest to oldest."""

        rows = self.connection.execute(
            """SELECT * FROM annotation_revisions
            WHERE episode_id = ? ORDER BY revision DESC""",
            (episode_id,),
        ).fetchall()
        return tuple(self._annotation_from_row(row) for row in rows)

    def record_evaluation_report(self, report: EvaluationReport) -> None:
        """Persist one immutable aggregate report with its provenance hashes."""

        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO evaluation_reports (
                    report_id, source_sha256, split_sha256,
                    model_fingerprint_sha256_json, configuration_sha256,
                    reviewed_commit, verdict, metrics_json, limitations_json,
                    generated_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.report_id,
                    report.source_sha256,
                    report.split_sha256,
                    _json(report.model_fingerprint_sha256),
                    report.configuration_sha256,
                    report.reviewed_commit,
                    report.verdict,
                    _json(report.metrics),
                    _json(report.limitations),
                    report.generated_at,
                    _now(),
                ),
            )

    def fetch_evaluation_report(self, report_id: str) -> EvaluationReport | None:
        """Restore a report without losing its model and split provenance."""

        row = self.connection.execute(
            "SELECT * FROM evaluation_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if row is None:
            return None
        return EvaluationReport(
            report_id=row["report_id"],
            source_sha256=row["source_sha256"],
            split_sha256=row["split_sha256"],
            model_fingerprint_sha256=tuple(json.loads(row["model_fingerprint_sha256_json"])),
            configuration_sha256=row["configuration_sha256"],
            reviewed_commit=row["reviewed_commit"],
            verdict=row["verdict"],
            metrics=json.loads(row["metrics_json"]),
            limitations=tuple(json.loads(row["limitations_json"])),
            generated_at=row["generated_at"],
        )

    def reserve_cost(
        self,
        *,
        reservation_id: str,
        request_id: str,
        amount_microusd: int,
        budget_microusd: int = 10_000_000,
    ) -> None:
        """Reserve an upper cost bound without exceeding the configured budget."""

        if amount_microusd < 0 or budget_microusd < 0:
            raise ValueError("cost amounts must be non-negative")
        now = _now()
        with self.connection:
            self._begin_immediate()
            total = self.connection.execute(
                """SELECT COALESCE(SUM(
                    CASE status
                        WHEN 'settled' THEN spent_microusd
                        WHEN 'reserved' THEN reserved_microusd
                        WHEN 'ambiguous' THEN reserved_microusd
                        ELSE 0
                    END
                ), 0)
                FROM cost_reservations"""
            ).fetchone()[0]
            if total + amount_microusd > budget_microusd:
                raise StorageConflictError("cost reservation exceeds the configured budget")
            self.connection.execute(
                """INSERT INTO cost_reservations
                (reservation_id, request_id, reserved_microusd, status, created_at, updated_at)
                VALUES (?, ?, ?, 'reserved', ?, ?)""",
                (reservation_id, request_id, amount_microusd, now, now),
            )

    def fetch_pointer(self, pointer_name: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """SELECT p.pointer_name, p.generation_id, g.kind, g.generation_key, g.status
            FROM committed_pointers p JOIN index_generations g ON g.generation_id = p.generation_id
            WHERE p.pointer_name = ?""",
            (pointer_name,),
        ).fetchone()


# Compatibility aliases for callers that prefer a storage-oriented name.
Database = SQLiteRepository
StorageRepository = SQLiteRepository

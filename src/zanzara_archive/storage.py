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
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import get_ident

from .contracts import (
    ApiError,
    ArtifactManifest,
    EvaluationReport,
    IdentityDecision,
    JobStatus,
)
from .stages import stage_fingerprint

SCHEMA_VERSION = 5
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
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise StoragePathError(f"cannot create local SQLite parent {parent}: {exc}") from exc
    filesystem = _filesystem_type(resolved_parent)
    if filesystem in NETWORK_FILESYSTEMS:
        raise StoragePathError(
            f"SQLite/job-state path is on network filesystem {filesystem}: {resolved_parent}"
        )
    return (resolved_parent / candidate.name).resolve()


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

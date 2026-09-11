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
from datetime import UTC, datetime
from pathlib import Path

from .contracts import ArtifactManifest, IdentityDecision

SCHEMA_VERSION = 2
DEFAULT_BUSY_TIMEOUT_MS = 5_000


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
    with connection:
        for version in range(current + 1, SCHEMA_VERSION + 1):
            connection.executescript(MIGRATIONS[version])
            connection.execute(f"PRAGMA user_version = {version}")
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

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        assert_sqlite_capabilities(connection)

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> SQLiteRepository:
        return cls(open_database(path))

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

    def record_artifact(self, manifest: ArtifactManifest, artifact_path: Path) -> None:
        """Record a complete artifact after its atomic filesystem publication."""

        manifest_payload = _json(manifest.to_dict())
        manifest_sha256 = hashlib.sha256(manifest_payload.encode()).hexdigest()
        existing = self.connection.execute(
            """SELECT manifest_sha256 FROM artifacts
            WHERE artifact_id = ? OR (source_sha256 = ? AND stage = ? AND stage_key = ?)""",
            (manifest.artifact_id, manifest.source_sha256, manifest.stage, manifest.stage_key),
        ).fetchone()
        if existing is not None:
            if existing[0] == manifest_sha256:
                return
            raise StorageConflictError("artifact identity already refers to different content")
        with self.transaction() as connection:
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
    ) -> None:
        """Publish a validated index generation and its pointer atomically."""

        if expected_points < 0:
            raise ValueError("expected_points must be non-negative")
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO index_generations
                (generation_id, kind, generation_key, status, expected_points, published_at)
                VALUES (?, ?, ?, 'complete', ?, ?)""",
                (generation_id, kind, generation_key, expected_points, now),
            )
            connection.execute(
                """INSERT INTO committed_pointers(pointer_name, generation_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(pointer_name) DO UPDATE SET
                    generation_id = excluded.generation_id, updated_at = excluded.updated_at""",
                (pointer_name, generation_id, now),
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
        with self.transaction() as connection:
            total = connection.execute(
                """SELECT COALESCE(SUM(reserved_microusd), 0)
                FROM cost_reservations WHERE status IN ('reserved', 'ambiguous')"""
            ).fetchone()[0]
            if total + amount_microusd > budget_microusd:
                raise StorageConflictError("cost reservation exceeds the configured budget")
            connection.execute(
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

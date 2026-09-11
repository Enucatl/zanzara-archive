"""CPU-only checks for canonical SQLite state and migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zanzara_archive.storage import (
    MIGRATIONS,
    SCHEMA_VERSION,
    SQLiteRepository,
    StorageConflictError,
    StoragePathError,
    ensure_local_state_path,
    open_database,
)


def test_fresh_database_enables_wal_fts_and_foreign_keys(tmp_path: Path) -> None:
    connection = open_database(tmp_path / "state.db")
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'text_chunks_fts'"
    ).fetchone()
    connection.close()


def test_upgrade_preserves_fixture_data_and_enforces_fk(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATIONS[1])
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-1", "a" * 64, "fixture", "golden.opus", "{}", "now"),
    )
    connection.commit()
    connection.close()

    upgraded = open_database(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert upgraded.execute("SELECT selection FROM corpus_manifests").fetchone()[0] == "fixture"
    with pytest.raises(sqlite3.IntegrityError):
        upgraded.execute(
            """INSERT INTO episode_speakers
            VALUES ('speaker-1', 'missing-episode', 'missing-artifact', 'SPEAKER_00', 'unknown')"""
        )
    upgraded.close()


def test_generation_pointer_identity_audit_and_budget_are_relational(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.publish_generation(
        generation_id="generation-1",
        kind="speaker",
        generation_key="speaker-key",
        pointer_name="active-speaker",
        expected_points=2,
    )
    pointer = repository.fetch_pointer("active-speaker")
    assert pointer is not None
    assert pointer["generation_id"] == "generation-1"
    repository.reserve_cost(
        reservation_id="reservation-1",
        request_id="request-1",
        amount_microusd=5_000_000,
    )
    with pytest.raises(StorageConflictError):
        repository.reserve_cost(
            reservation_id="reservation-2",
            request_id="request-2",
            amount_microusd=5_000_001,
        )
    repository.close()


def test_network_filesystem_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", lambda _: "nfs")
    with pytest.raises(StoragePathError, match="network filesystem"):
        ensure_local_state_path(tmp_path / "state.db")

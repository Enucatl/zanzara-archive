"""CPU-only checks for canonical SQLite state and migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from zanzara_archive.contracts import EvaluationReport
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


def test_fresh_database_has_all_d3_canonical_tables(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    required = {
        "corpus_manifests",
        "episodes",
        "models",
        "artifacts",
        "processing_runs",
        "jobs",
        "transcript_words",
        "standard_turns",
        "exclusive_turns",
        "overlap_intervals",
        "episode_speakers",
        "exemplars",
        "text_chunks",
        "annotation_revisions",
        "review_records",
        "split_records",
        "identity_decisions",
        "global_speakers",
        "upload_jobs",
        "cost_reservations",
        "evaluation_reports",
    }
    actual = {
        row[0]
        for row in repository.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert required <= actual
    repository.close()


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


def test_upgrade_from_v3_adds_tables_without_losing_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATIONS[1])
    connection.execute("PRAGMA user_version = 1")
    connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-1", "b" * 64, "fixture", "golden.opus", "{}", "now"),
    )
    connection.commit()
    connection.close()

    upgraded = open_database(path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert (
        upgraded.execute("SELECT manifest_id FROM corpus_manifests").fetchone()[0] == "manifest-1"
    )
    assert upgraded.execute("SELECT COUNT(*) FROM transcript_words").fetchone()[0] == 0
    upgraded.close()


def test_evaluation_report_round_trip_preserves_provenance(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    report = EvaluationReport(
        report_id="report-1",
        source_sha256="a" * 64,
        split_sha256="b" * 64,
        model_fingerprint_sha256=("c" * 64, "d" * 64),
        configuration_sha256="e" * 64,
        reviewed_commit="commit-sha",
        verdict="pass",
        metrics={"wer": 0.12, "count": 20},
        limitations=("synthetic fixture",),
        generated_at="2026-09-12T00:00:00+00:00",
    )
    repository.record_evaluation_report(report)
    assert repository.fetch_evaluation_report("report-1") == report
    repository.close()


def test_canonical_interval_and_status_constraints_are_enforced(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    with pytest.raises(sqlite3.IntegrityError):
        repository.connection.execute(
            """INSERT INTO evaluation_reports (
                report_id, source_sha256, split_sha256,
                model_fingerprint_sha256_json, configuration_sha256,
                reviewed_commit, verdict, metrics_json, limitations_json, created_at
            ) VALUES ('report-2', ?, ?, ?, ?, 'commit', 'unknown', '{}', '[]', 'now')""",
            ("a" * 64, "b" * 64, "[]", "c" * 64),
        )
    with pytest.raises(sqlite3.IntegrityError):
        repository.connection.execute(
            """INSERT INTO overlap_intervals (
                overlap_id, diarization_artifact_id, episode_id, source_sha256,
                ordinal, start_ms, end_ms, speaker_ids_json
            ) VALUES ('overlap-1', 'missing', 'missing', ?, 0, 100, 100, '[]')""",
            ("a" * 64,),
        )
    repository.close()


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

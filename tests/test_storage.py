"""CPU-only checks for canonical SQLite state and migrations."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from zanzara_archive.contracts import EvaluationReport
from zanzara_archive.storage import (
    MIGRATIONS,
    SCHEMA_VERSION,
    SQLiteRepository,
    StorageConflictError,
    StoragePathError,
    ensure_local_state_path,
    migrate,
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
        "annotation_assistance_drafts",
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
    for version in range(1, 4):
        connection.executescript(MIGRATIONS[version])
    connection.execute("PRAGMA user_version = 3")
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


def test_migration_failure_rolls_back_schema_and_version(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    for version in range(1, 5):
        connection.executescript(MIGRATIONS[version])
    connection.execute("PRAGMA user_version = 4")
    connection.commit()
    monkeypatch.setitem(
        MIGRATIONS,
        5,
        "CREATE TABLE migration_must_rollback (id INTEGER);\nINVALID SQL;\n",
    )

    with pytest.raises(sqlite3.OperationalError):
        migrate(connection)

    assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'migration_must_rollback'"
        ).fetchone()[0]
        == 0
    )
    connection.close()


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


@pytest.mark.parametrize("start_ms,end_ms", [(0.5, 500), (0, 1001)])
def test_transcript_word_rejects_non_integer_or_episode_overrun(
    tmp_path: Path, start_ms: float, end_ms: int
) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    _insert_interval_provenance(repository.connection)
    with pytest.raises(sqlite3.IntegrityError, match="interval or provenance"):
        repository.connection.execute(
            """INSERT INTO transcript_words (
                word_id, transcript_artifact_id, episode_id, source_sha256,
                model_fingerprint_sha256, ordinal, text, start_ms, end_ms
            ) VALUES ('word-1', 'artifact-1', 'episode-1', ?, ?, 0, 'word', ?, ?)""",
            ("a" * 64, "b" * 64, start_ms, end_ms),
        )
    repository.close()


def test_transcript_word_rejects_cross_record_provenance(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    _insert_interval_provenance(repository.connection)
    with pytest.raises(sqlite3.IntegrityError, match="interval or provenance"):
        repository.connection.execute(
            """INSERT INTO transcript_words (
                word_id, transcript_artifact_id, episode_id, source_sha256,
                model_fingerprint_sha256, ordinal, text, start_ms, end_ms
            ) VALUES ('word-1', 'artifact-1', 'episode-1', ?, ?, 0, 'word', 0, 500)""",
            ("c" * 64, "b" * 64),
        )
    repository.close()


def _insert_interval_provenance(connection: sqlite3.Connection) -> None:
    connection.execute(
        "INSERT INTO corpus_manifests VALUES (?, ?, ?, ?, ?, ?)",
        ("manifest-1", "d" * 64, "fixture", "golden.opus", "{}", "now"),
    )
    connection.execute(
        """INSERT INTO episodes (
            episode_id, manifest_id, relative_filename, episode_date, source_sha256,
            size_bytes, duration_ms, codec, channels, sample_rate_hz
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "episode-1",
            "manifest-1",
            "episode.opus",
            "2026-09-10",
            "a" * 64,
            100,
            1000,
            "opus",
            1,
            48000,
        ),
    )
    connection.execute(
        """INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "artifact-1",
            "a" * 64,
            "asr",
            "key",
            "/artifact",
            "e" * 64,
            "[]",
            "b" * 64,
            "{}",
            "test",
            1,
            "{}",
            "now",
        ),
    )


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


def test_cost_ledger_accounts_for_settled_spend_and_state_transitions(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")

    repository.reserve_cost(
        reservation_id="reservation-1", request_id="request-1", amount_microusd=6_000_000
    )
    repository.settle_cost("request-1", spent_microusd=4_000_000)
    assert tuple(
        repository.connection.execute(
            "SELECT reserved_microusd, spent_microusd, status "
            "FROM cost_reservations WHERE request_id='request-1'"
        ).fetchone()
    ) == (6_000_000, 4_000_000, "settled")

    with pytest.raises(StorageConflictError):
        repository.reserve_cost(
            reservation_id="reservation-2",
            request_id="request-2",
            amount_microusd=7_000_000,
        )

    repository.reserve_cost(
        reservation_id="reservation-2", request_id="request-2", amount_microusd=6_000_000
    )
    repository.release_cost("request-2")
    repository.reserve_cost(
        reservation_id="reservation-3", request_id="request-3", amount_microusd=5_000_000
    )
    repository.mark_cost_ambiguous("request-3")
    with pytest.raises(StorageConflictError):
        repository.reserve_cost(
            reservation_id="reservation-4",
            request_id="request-4",
            amount_microusd=2_000_000,
        )

    repository.settle_cost("request-3", spent_microusd=3_000_000)
    repository.reserve_cost(
        reservation_id="reservation-4", request_id="request-4", amount_microusd=3_000_000
    )
    rows = repository.connection.execute(
        "SELECT request_id, reserved_microusd, spent_microusd, status "
        "FROM cost_reservations ORDER BY request_id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("request-1", 6_000_000, 4_000_000, "settled"),
        ("request-2", 6_000_000, 0, "released"),
        ("request-3", 5_000_000, 3_000_000, "settled"),
        ("request-4", 3_000_000, 0, "reserved"),
    ]
    repository.close()


@pytest.mark.parametrize("spent_microusd", [6_000_000, 8_000_000])
def test_cost_ledger_uses_actual_settled_spend_above_or_at_reservation(
    tmp_path: Path, spent_microusd: int
) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.reserve_cost(
        reservation_id="reservation-1", request_id="request-1", amount_microusd=6_000_000
    )
    repository.settle_cost("request-1", spent_microusd=spent_microusd)

    with pytest.raises(StorageConflictError):
        repository.reserve_cost(
            reservation_id="reservation-2",
            request_id="request-2",
            amount_microusd=10_000_000 - spent_microusd + 1,
        )
    assert (
        repository.connection.execute(
            "SELECT spent_microusd FROM cost_reservations WHERE request_id='request-1'"
        ).fetchone()[0]
        == spent_microusd
    )
    repository.close()


def test_cost_reservations_are_atomic_across_competing_connections(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    SQLiteRepository.open(path).close()
    start = Barrier(2)

    def reserve(request_id: str) -> str:
        repository = SQLiteRepository.open(path)
        try:
            start.wait()
            repository.reserve_cost(
                reservation_id=f"reservation-{request_id}",
                request_id=request_id,
                amount_microusd=6_000_000,
            )
            return "reserved"
        except StorageConflictError:
            return "blocked"
        finally:
            repository.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, ("request-1", "request-2")))

    assert sorted(results) == ["blocked", "reserved"]
    repository = SQLiteRepository.open(path)
    assert tuple(
        repository.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(reserved_microusd), 0) "
            "FROM cost_reservations WHERE status='reserved'"
        ).fetchone()
    ) == (1, 6_000_000)
    repository.close()


def test_network_filesystem_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", lambda _: "nfs")
    with pytest.raises(StoragePathError, match="network filesystem"):
        ensure_local_state_path(tmp_path / "state.db")


def test_database_file_symlink_cannot_bypass_network_rejection(tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "local"
    network = tmp_path / "network"
    local.mkdir()
    network.mkdir()
    database_path = local / "state.db"
    network_target = network / "state.db"
    database_path.symlink_to(network_target)

    def filesystem_type(path: Path) -> str:
        return "nfs" if path.is_relative_to(network) else "ext4"

    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", filesystem_type)
    with pytest.raises(StoragePathError, match="network filesystem nfs"):
        ensure_local_state_path(database_path)


def test_database_parent_symlink_cannot_bypass_network_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    network = tmp_path / "network"
    network.mkdir()
    linked_parent = tmp_path / "state"
    linked_parent.symlink_to(network, target_is_directory=True)

    def filesystem_type(path: Path) -> str:
        return "nfs" if path.is_relative_to(network) else "ext4"

    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", filesystem_type)
    with pytest.raises(StoragePathError, match="network filesystem nfs"):
        ensure_local_state_path(linked_parent / "state.db")


def test_unknown_filesystem_locality_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", lambda _: None)
    with pytest.raises(StoragePathError, match="cannot establish local filesystem"):
        ensure_local_state_path(tmp_path / "state.db")


def test_local_database_path_resolves_to_validated_backing_storage(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "local"
    local.mkdir()
    database_path = local / "state.db"
    observed: list[Path] = []

    def filesystem_type(path: Path) -> str:
        observed.append(path)
        return "ext4"

    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", filesystem_type)
    resolved = ensure_local_state_path(database_path)

    assert resolved == database_path.resolve()
    assert observed == [resolved]


def test_local_file_symlink_keeps_database_wal_and_shm_on_validated_target(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "local"
    target = tmp_path / "target"
    local.mkdir()
    target.mkdir()
    database_path = local / "state.db"
    database_target = target / "state.db"
    database_path.symlink_to(database_target)
    monkeypatch.setattr("zanzara_archive.storage._filesystem_type", lambda _: "ext4")

    connection = open_database(database_path)
    connection.execute("CREATE TABLE sidecar_fixture (id INTEGER)")

    assert database_target.is_file()
    assert database_target.with_name("state.db-wal").is_file()
    assert database_target.with_name("state.db-shm").is_file()
    assert not database_path.with_name("state.db-wal").exists()
    assert not database_path.with_name("state.db-shm").exists()
    connection.close()

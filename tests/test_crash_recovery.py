"""CPU crash-recovery checks for leases and durable artifact reuse."""

from __future__ import annotations

from pathlib import Path

import pytest

from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.jobs import DurableWorker, synthetic_runner
from zanzara_archive.storage import SQLiteRepository, StorageConflictError


def test_expired_lease_is_recovered_and_worker_finishes_once(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="job-1", stage="decode", source_sha256="e" * 64)
    abandoned = repository.claim_job("job-1", "dead-worker", now="2026-01-01T00:00:00+00:00")
    assert repository.recover_expired_leases(now="2026-01-01T00:02:01+00:00") == ("job-1",)
    worker = DurableWorker(repository, "live-worker", synthetic_runner())
    result = worker.run_once(now="2026-01-01T00:03:00+00:00")
    assert result.completed
    assert result.job.attempts == abandoned.attempts + 1
    assert repository.fetch_job("job-1").status == "succeeded"
    repository.close()


def test_complete_artifact_is_reused_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    first = ArtifactPublisher(root).publish(
        source_sha256="f" * 64,
        stage="decode",
        stage_key="stage-key",
        files={"output": b"synthetic"},
    )
    second = ArtifactPublisher(root).publish(
        source_sha256="f" * 64,
        stage="decode",
        stage_key="stage-key",
        files={"output": b"synthetic"},
    )
    assert second == first
    assert len(list((root / ("f" * 64) / "decode" / "stage-key").iterdir())) == 2


def test_paid_expiry_recovery_blocks_and_marks_reservation_ambiguous(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.reserve_cost(reservation_id="reservation", request_id="request", amount_microusd=100)
    repository.enqueue_job(job_id="paid", stage="asr", request_id="request", paid=True)
    repository.claim_job("paid", "dead-worker", now="2026-01-01T00:00:00+00:00")

    assert repository.recover_expired_leases(now="2026-01-01T00:02:01+00:00") == ("paid",)
    job = repository.fetch_job("paid")
    assert job.status == "blocked"
    assert job.recovery_action == "reconcile_paid_request"
    assert (
        repository.connection.execute(
            "SELECT status FROM cost_reservations WHERE request_id='request'"
        ).fetchone()[0]
        == "ambiguous"
    )
    assert (
        DurableWorker(repository, "live-worker", synthetic_runner())
        .run_once(now="2026-01-01T00:03:00+00:00")
        .job
        is None
    )
    repository.close()


def test_direct_and_normal_reclaim_cannot_bypass_paid_reconciliation(tmp_path: Path) -> None:
    for direct in (True, False):
        repository = SQLiteRepository.open(tmp_path / f"state-{direct}.db")
        repository.reserve_cost(
            reservation_id="reservation", request_id="request", amount_microusd=100
        )
        repository.enqueue_job(job_id="paid", stage="asr", request_id="request", paid=True)
        repository.claim_job("paid", "dead-worker", now="2026-01-01T00:00:00+00:00")
        if direct:
            with pytest.raises(StorageConflictError, match="requires a new reserved attempt"):
                repository.claim_job("paid", "live-worker", now="2026-01-01T00:02:01+00:00")
        else:
            assert repository.claim_next_job("live-worker", now="2026-01-01T00:02:01+00:00") is None
        assert repository.fetch_job("paid").status == "blocked"
        assert (
            repository.connection.execute(
                "SELECT status FROM cost_reservations WHERE request_id='request'"
            ).fetchone()[0]
            == "ambiguous"
        )
        repository.close()

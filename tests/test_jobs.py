"""CPU checks for durable claims, fencing, retries and cancellation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from zanzara_archive.contracts import ApiError
from zanzara_archive.jobs import DurableWorker
from zanzara_archive.storage import SQLiteRepository, StorageConflictError


def test_long_callback_is_renewed_and_cannot_be_reclaimed(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="long", stage="decode", source_sha256="a" * 64)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    heartbeat_seen = Event()
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        calls += 1
        return start + timedelta(seconds=31 * calls)

    original_heartbeat = repository.heartbeat

    def heartbeat(*args: object, **kwargs: object) -> object:
        result = original_heartbeat(*args, **kwargs)
        heartbeat_seen.set()
        return result

    repository.heartbeat = heartbeat  # type: ignore[method-assign]

    def runner(_: object) -> None:
        assert heartbeat_seen.wait(1)
        with pytest.raises(StorageConflictError, match="not eligible"):
            repository.claim_job("long", "worker-b", now=start + timedelta(seconds=121))

    result = DurableWorker(
        repository, "worker-a", runner, clock=clock, heartbeat_interval=0.01
    ).run_once(now=start)
    assert result.completed
    assert result.job is not None
    assert result.job.status == "succeeded"
    repository.close()


def test_lease_loss_prevents_successful_completion(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="lost", stage="decode", source_sha256="b" * 64)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    heartbeat_attempted = Event()

    def clock() -> datetime:
        heartbeat_attempted.set()
        return start + timedelta(seconds=121)

    def runner(_: object) -> None:
        assert heartbeat_attempted.wait(1)

    result = DurableWorker(
        repository, "worker-a", runner, clock=clock, heartbeat_interval=0.01
    ).run_once(now=start)
    assert not result.completed
    assert result.error is not None
    assert result.error.code == "lease_lost"
    assert repository.fetch_job("lost").status == "running"
    repository.close()


def test_claims_are_fenced_and_stale_workers_cannot_complete(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(
        job_id="job-1", stage="decode", source_sha256="a" * 64, request_id="request-1"
    )
    first = repository.claim_job("job-1", "worker-a", now="2026-01-01T00:00:00+00:00")
    second = repository.claim_job("job-1", "worker-b", now="2026-01-01T00:02:01+00:00")
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(StorageConflictError, match="stale worker"):
        repository.assert_job_fence("job-1", owner="worker-a", fencing_token=first.fencing_token)
    with pytest.raises(StorageConflictError, match="stale worker"):
        repository.complete_job("job-1", owner="worker-a", fencing_token=first.fencing_token)
    repository.complete_job(
        "job-1",
        owner="worker-b",
        fencing_token=second.fencing_token,
        now="2026-01-01T00:02:02+00:00",
    )
    assert repository.fetch_job("job-1").status == "succeeded"
    repository.close()


def test_expired_owner_cannot_complete_without_reclaim(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="expired", stage="decode", source_sha256="a" * 64)
    claimed = repository.claim_job("expired", "worker-a", now="2026-01-01T00:00:00+00:00")
    with pytest.raises(StorageConflictError, match="stale worker"):
        repository.complete_job(
            "expired",
            owner="worker-a",
            fencing_token=claimed.fencing_token,
            now="2026-01-01T00:02:01+00:00",
        )
    assert repository.fetch_job("expired").status == "running"
    repository.close()


def test_transient_retry_backoff_and_terminal_failure_are_bounded(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="job-2", stage="asr", source_sha256="b" * 64)
    error = ApiError("timeout", "synthetic timeout", True, "request-2")
    for attempt, expected_status in ((0, "retry_wait"), (1, "retry_wait"), (2, "failed")):
        claimed = repository.claim_job(
            "job-2", f"worker-{attempt}", now=f"2026-01-01T00:0{attempt}:00+00:00"
        )
        result = repository.fail_job(
            "job-2",
            owner=f"worker-{attempt}",
            fencing_token=claimed.fencing_token,
            error=error,
            now=f"2026-01-01T00:0{attempt}:00+00:00",
        )
        assert result.status == expected_status
        if result.status == "retry_wait":
            repository.connection.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=?",
                (f"2026-01-01T00:0{attempt}:01+00:00", "job-2"),
            )
    assert repository.fetch_job("job-2").recovery_action == "retry_exhausted"
    assert repository.retry_job("job-2").status == "queued"
    repository.close()


def test_deterministic_failure_and_cancel_are_terminal(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="job-3", stage="diarization", source_sha256="c" * 64)
    claimed = repository.claim_job("job-3", "worker", now="2026-01-01T00:00:00+00:00")
    failed = repository.fail_job(
        "job-3",
        owner="worker",
        fencing_token=claimed.fencing_token,
        error=ApiError("invalid_audio", "bad fixture", False, "job-3"),
        failure_class="deterministic",
    )
    assert failed.status == "blocked"
    with pytest.raises(StorageConflictError):
        repository.retry_job("job-3")
    repository.enqueue_job(job_id="job-4", stage="decode", source_sha256="d" * 64)
    assert repository.cancel_job("job-4").status == "cancelled"
    repository.close()


def test_ambiguous_paid_failure_keeps_ledger_reserved(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.reserve_cost(
        reservation_id="reservation-1", request_id="request-paid", amount_microusd=100
    )
    repository.enqueue_job(
        job_id="job-paid",
        stage="transcription",
        source_sha256="e" * 64,
        request_id="request-paid",
        paid=True,
    )
    claimed = repository.claim_job("job-paid", "worker", now="2026-01-01T00:00:00+00:00")
    result = repository.fail_job(
        "job-paid",
        owner="worker",
        fencing_token=claimed.fencing_token,
        error=ApiError("timeout", "unknown charge", True, "request-paid"),
        failure_class="ambiguous",
    )
    assert result.status == "blocked"
    assert (
        repository.connection.execute(
            "SELECT status FROM cost_reservations WHERE request_id='request-paid'"
        ).fetchone()[0]
        == "ambiguous"
    )
    repository.close()

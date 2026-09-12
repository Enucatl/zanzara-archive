"""CPU checks for durable claims, fencing, retries and cancellation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from zanzara_archive.contracts import (
    ApiError,
    InvalidAudioError,
    ModelUnavailableError,
    RequestTimeoutError,
)
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


@pytest.mark.parametrize(
    ("failure_type", "code"),
    [(InvalidAudioError, "invalid_audio"), (ModelUnavailableError, "model_unavailable")],
)
def test_worker_preserves_deterministic_adapter_failures(
    tmp_path: Path, failure_type: type[Exception], code: str
) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.enqueue_job(job_id="typed", stage="asr", request_id="typed-request")

    def runner(_: object) -> None:
        raise failure_type(ApiError(code, "synthetic adapter failure", False, "typed-request"))

    result = DurableWorker(repository, "worker", runner).run_once(now="2026-01-01T00:00:00+00:00")
    assert result.error == ApiError(code, "synthetic adapter failure", False, "typed-request")
    assert result.job is not None
    assert result.job.status == "blocked"
    assert result.job.recovery_action == "fix_deterministic_failure"
    repository.close()


def test_paid_timeout_blocks_callback_redispatch_and_retains_ledger(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.reserve_cost(
        reservation_id="reservation", request_id="paid-request", amount_microusd=100
    )
    repository.enqueue_job(job_id="paid", stage="asr", request_id="paid-request", paid=True)
    calls = 0

    def runner(_: object) -> None:
        nonlocal calls
        calls += 1
        raise RequestTimeoutError(
            ApiError("timeout", "synthetic unknown outcome", True, "paid-request")
        )

    worker = DurableWorker(repository, "worker", runner)
    result = worker.run_once(now="2026-01-01T00:00:00+00:00")
    assert result.job is not None
    assert result.job.status == "blocked"
    assert result.job.error == ApiError(
        "timeout", "synthetic unknown outcome", True, "paid-request"
    )
    assert result.job.recovery_action == "reconcile_paid_request"
    assert worker.run_once(now="2026-01-01T00:00:06+00:00").job is None
    assert calls == 1
    reservation = repository.connection.execute(
        "SELECT reserved_microusd, status FROM cost_reservations WHERE request_id=?",
        ("paid-request",),
    ).fetchone()
    assert tuple(reservation) == (100, "ambiguous")
    repository.close()


@pytest.mark.parametrize("reservation_status", [None, "settled", "released", "ambiguous"])
def test_paid_claim_requires_a_fresh_active_reservation(
    tmp_path: Path, reservation_status: str | None
) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    if reservation_status is not None:
        repository.reserve_cost(
            reservation_id="reservation", request_id="request", amount_microusd=100
        )
        if reservation_status == "settled":
            repository.settle_cost("request", 50)
        elif reservation_status == "released":
            repository.release_cost("request")
        elif reservation_status == "ambiguous":
            repository.mark_cost_ambiguous("request")
    repository.enqueue_job(job_id="paid", stage="asr", request_id="request", paid=True)

    with pytest.raises(StorageConflictError, match="requires a new reserved attempt"):
        repository.claim_job("paid", "worker", now="2026-01-01T00:00:00+00:00")
    assert repository.fetch_job("paid").status == "blocked"
    repository.close()


def test_paid_reservation_cannot_be_reused_by_another_job(tmp_path: Path) -> None:
    repository = SQLiteRepository.open(tmp_path / "state.db")
    repository.reserve_cost(reservation_id="reservation", request_id="request", amount_microusd=100)
    repository.enqueue_job(job_id="first", stage="asr", request_id="request", paid=True)
    repository.enqueue_job(job_id="second", stage="asr", request_id="request", paid=True)

    repository.claim_job("first", "worker", now="2026-01-01T00:00:00+00:00")
    with pytest.raises(StorageConflictError, match="requires a new reserved attempt"):
        repository.claim_job("second", "worker", now="2026-01-01T00:00:00+00:00")
    assert repository.fetch_job("second").status == "blocked"
    repository.close()

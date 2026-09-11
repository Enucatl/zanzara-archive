"""Single-worker durable job operations and stage execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .contracts import ApiError, JobStatus
from .storage import SQLiteRepository

LEASE_SECONDS = 120
HEARTBEAT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (5, 30)


class StageRunner(Protocol):
    """A CPU or service-backed stage callback used by the durable worker."""

    def __call__(self, job: JobStatus) -> Mapping[str, Any] | None:
        """Run one claimed job without holding a database transaction."""


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Outcome of one worker poll."""

    job: JobStatus | None
    completed: bool = False
    error: ApiError | None = None


def _as_error(value: ApiError | Exception, *, request_id: str) -> ApiError:
    if isinstance(value, ApiError):
        return value
    return ApiError(
        code="stage_failed",
        message=str(value) or value.__class__.__name__,
        retryable=True,
        request_id=request_id,
    )


class DurableWorker:
    """Execute one claimed job at a time with fenced completion."""

    def __init__(
        self,
        repository: SQLiteRepository,
        owner: str,
        runner: StageRunner,
        *,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        if not owner:
            raise ValueError("owner must be non-empty")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.repository = repository
        self.owner = owner
        self.runner = runner
        self.max_attempts = max_attempts

    def run_once(self, *, now: datetime | str | None = None) -> WorkerResult:
        claimed = self.repository.claim_next_job(
            self.owner, now=now, max_attempts=self.max_attempts
        )
        if claimed is None:
            return WorkerResult(None)
        try:
            self.runner(claimed)
        except Exception as exc:  # callback failures become typed durable state
            error = _as_error(exc, request_id=claimed.request_id or f"job-{claimed.job_id}")
            final = self.repository.fail_job(
                claimed.job_id,
                owner=self.owner,
                fencing_token=claimed.fencing_token,
                error=error,
                failure_class="transient" if error.retryable else "deterministic",
                now=now,
                max_attempts=self.max_attempts,
            )
            return WorkerResult(final, error=error)
        final = self.repository.complete_job(
            claimed.job_id,
            owner=self.owner,
            fencing_token=claimed.fencing_token,
            now=now,
        )
        return WorkerResult(final, completed=True)


def synthetic_runner(payload: Mapping[str, Any] | None = None) -> StageRunner:
    """Return a deterministic no-op runner for CPU-only CLI smoke checks."""

    result = dict(payload or {})

    def run(_: JobStatus) -> Mapping[str, Any]:
        return result

    return run


__all__ = [
    "HEARTBEAT_SECONDS",
    "LEASE_SECONDS",
    "MAX_ATTEMPTS",
    "RETRY_BACKOFF_SECONDS",
    "DurableWorker",
    "StageRunner",
    "WorkerResult",
    "synthetic_runner",
]

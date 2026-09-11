# Resumable worker and stage invalidation

`zanzara_archive.stages` computes a SHA-256 stage key from the source hash,
ordered upstream artifact hashes, model fingerprint, canonical configuration,
pipeline version and stage name. The D4 graph is explicit: ASR changes flow
through attribution, chunks and text indexing; diarization changes additionally
invalidate exemplars, speaker indexes and candidates. Voice artifacts remain
reusable when only ASR changes.

`SQLiteRepository` owns one durable worker queue in the local WAL database. A
claim uses `BEGIN IMMEDIATE`, a 120-second lease, a monotonically increasing
fencing token and an attempt count. Workers should heartbeat every 30 seconds.
Expired leases are requeued on recovery, transient failures back off for five
then thirty seconds, and three attempts is the default limit. Deterministic
failures become `blocked`; exhausted transient failures become `failed`;
cancellation becomes `cancelled`. Completion and failure updates verify both
owner and fencing token, so a stale worker cannot mark a job complete.

Unknown paid requests use the `ambiguous` failure class and remain reserved in
the cost ledger until an operator reconciles them. No network or paid calls are
made by the synthetic worker.

CPU-only commands use a local database path:

```bash
uv run zanzara jobs enqueue --database state/state.db --job-id decode-1 --stage decode --source-sha256 <sha256>
uv run zanzara jobs status --database state/state.db decode-1
uv run zanzara jobs recover --database state/state.db
uv run zanzara worker run --database state/state.db --owner worker-1
```

Focused checks:

```bash
uv run pytest tests/test_jobs.py tests/test_stage_invalidation.py tests/test_crash_recovery.py
```

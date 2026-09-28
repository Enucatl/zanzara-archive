# Zanzara Archive

The application has durable jobs, transcript/diarization artifacts, local model
services, annotation pages and chunk evaluation tools. Search and cross-episode
voice discovery are the next product work. The revised
[working plan](planning/README.md) prioritizes FTS5 search/playback and one voice
encoder, with focused checks and a small human-reviewed ASR pilot.

The binding implementation specification is in [`planning/README.md`](planning/README.md),
with the architecture in [`planning/SYSTEM-DESIGN.md`](planning/SYSTEM-DESIGN.md)
and evaluation gates in [`planning/EVALUATION.md`](planning/EVALUATION.md). The
original [`plan.md`](plan.md) is retained as background; the planning package
records the current decisions.

## Developer setup

The application targets Python 3.14 and uses `uv` with a committed `uv.lock`:

```bash
uv sync --frozen
uv run zanzara --help
uv run zanzara --version
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/test_package.py
```

The canonical archive is mounted read-only and remains outside the repository;
processing uses the persistent manifest and source paths without recurring
source hashing or media verification. See
[`docs/corpus-verification.md`](docs/corpus-verification.md) for the source
mount and artifact-path layout.

The top-level package intentionally has no ML dependencies. Inference services
live under `services/` with isolated packages and a shared GPU runtime, so CPU
application checks remain usable without CUDA,
model credentials, network inference or private archive data.

Tests that need resources beyond the application process declare one of the
following markers: `gpu`, `integration`, `browser` or `paid`. Paid tests are
opt-in and must never run from the default CPU check:

```bash
uv run pytest -m 'paid'  # only after explicit budget authorization
```

Synthetic fixtures belong in `tests/fixtures/` and must contain no private
audio, credentials, model weights, identities or local database state. Private
artifacts and generated processing output stay in ignored local directories.

The durable single-worker queue, stage fingerprints and crash recovery contract
are documented in [`docs/jobs.md`](docs/jobs.md). Jobs and stage state belong on
the local SQLite `state` volume; the canonical archive remains read-only.

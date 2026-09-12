# Zanzara Archive

Foundation status: the Python 3.14 application skeleton and CPU-only developer
Foundation status: the Python 3.14 application skeleton and CPU-only developer
checks are in place. The frozen 20-episode manifest and read-only source paths
are available; the canonical archive is a persistent trusted input and is not
reverified during routine commits, reviews or code runs. Model services, the
web application and search features remain planned work.

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

The top-level package intentionally has no ML dependencies. Future Python 3.11
inference services will live under `services/`, each with its own package,
Dockerfile and lock, so CPU application checks remain usable without CUDA,
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

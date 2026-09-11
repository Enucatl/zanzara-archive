# Zanzara Archive

Foundation status: the Python 3.14 application skeleton and CPU-only developer
checks are in place. The frozen 20-episode manifest and read-only source
verification are available; model services, the web application and search
features remain planned work.

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

To verify the frozen archive sources without changing them, run
`uv run zanzara corpus verify --manifest planning/corpus-20.json
--archive-root /export/scratch/archive/zanzara`. The verifier checks every
manifest hash and `ffprobe` media field, rejects traversal and escaping
symlinks, and records a JSON report. See
[`docs/corpus-verification.md`](docs/corpus-verification.md) for the required
read-only container mount and artifact paths.

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

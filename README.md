# Zanzara Archive

The application has durable jobs, transcript/diarization artifacts, local model
services, annotation pages and chunk evaluation tools. Search and cross-episode
voice discovery are the next product work. The revised
[working plan](planning/README.md) prioritizes FTS5 search/playback and one voice
encoder, with focused checks and a small human-reviewed ASR pilot.
The [ASR pilot report](docs/asr-pilot.md) records the measured Parakeet default
and its limitations.

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
Start and recover the LAN app, Qdrant, and serial GPU jobs with the
[`workstation runbook`](docs/workstation.md).

## ResNet voice index

Start Qdrant with `COMPOSE_ENV_FILES=/dev/null docker compose up -d qdrant`.
For each episode with a completed exemplar artifact, retain the original ResNet
vectors:

```bash
uv run zanzara process --manifest planning/corpus-20.json --episode EPISODE.opus \
  --stage speaker_embeddings_resnet293 --exemplars-artifact EXEMPLAR_ARTIFACT_DIR
```

Build or rebuild both voice collections from the returned embedding artifact
directories, repeating `--embedding-artifact` for each episode:

```bash
uv run python -m zanzara_archive.voice_index \
  --embedding-artifact EMBEDDING_ARTIFACT_DIR
```

The command validates the retained vectors and expected point counts before it
switches both active Qdrant aliases together. The collections can be rebuilt
from the private embedding artifacts; no audio inference is needed for a rebuild.

Find cross-episode voice candidates for one episode-local speaker:

```bash
uv run python -m zanzara_archive.voice_search \
  --embedding-artifact EMBEDDING_ARTIFACT_DIR --speaker SPEAKER_ID
```

The JSON lists up to 50 centroid candidates reranked by median matched-exemplar
cosine similarity, with original offsets and LAN playback links. Scores are
uncalibrated; candidates are not identity decisions. Speakers without selected
excerpts return `not_voice_searchable` with an empty candidate list.

The LAN [ASR review editor](http://complex.home.arpa:8000/chunk-review) lists registered
chunks, immutable ASR candidates, and separate editable human references. Enter a
reviewer name, edit the transcript and per-speaker text, then save a draft or
explicitly approve human truth. Condition corrections and prior revisions remain
in the database; missing AST/Qwen seeds do not prevent review. Save before moving
to another chunk. Ctrl/⌘+S saves; Alt+P plays, Alt+L loops, Alt+←/→ seeks,
Alt+N/B navigates, and Alt+A approves. Stale saves retain local edits and return a
conflict; copy those edits before reloading the latest revision.

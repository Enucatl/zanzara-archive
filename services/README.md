# P1 shared GPU model services

`services/shared-base/Dockerfile` is the repository-owned, digest-pinned GPU runtime.
It installs managed Python 3.14, the single `torch==2.14.0+cu130` /
`torchaudio==2.11.0+cu130` pair, `transformers==5.17.0`, and common runtime
packages exactly once. Each directory is an independent Python 3.14 package with its own `pyproject.toml`,
`uv.lock`, Dockerfile, and smoke entry point. The application environment remains
Python 3.14 and has no ML dependency. The derived Dockerfiles inherit from the
promoted `zanzara-archive/shared-base:py314-cu130` image and use `uv sync --inexact`
with the shared packages excluded from installation. Model weights are never copied into this
repository: Compose mounts the operator-provisioned read-only caches.

The six P1 services load the fixed checkpoints and expose only internal `/health`,
`/ready`, and `/v1/smoke` endpoints. `model-smoke` starts them through Compose,
waits for readiness, and reports the six real inference results. The smoke input is
the public ERes2Net ModelScope example, not archive audio; the result is a runtime
compatibility check and does not claim archive quality.

```bash
docker compose --profile processing build --with-dependencies
docker compose --profile processing run --rm model-smoke
```

The lock records the observed artifact checksums, preprocessing, output dimensions,
runtime, image/base digests, timing/VRAM aggregate, and private evidence checksum.
The private P1-09 disk report at
`planning/evidence/P1-09-disk-report.md` records total image size, Docker-store
shared/unique bytes, every measured layer category, and each derived image's
unique dependency/code layer.
After all measurements and smokes pass, the cleanup runbook removes only explicitly
named superseded Zanzara tags and unreferenced cache IDs; it never invokes
`docker system prune`, `docker builder prune`, or deletes model cache paths.

For a routine rebuild, capture `docker system df -v` and `docker buildx du` first,
then use an age-bounded BuildKit cleanup only after confirming the final images are
tagged and running:

```bash
docker buildx prune --force --filter 'until=24h'
```

Do not add `--all` or remove the external model-cache directories as part of this
runbook. Superseded image tags must be named explicitly and checked with
`docker image inspect` before `docker image rm`.

P1R adds a separate `whisper` service for text-first benchmark hypotheses. It
uses the frozen `openai/whisper-large-v3` checkpoint, forced Italian
transcription and beam-search decoding, and writes private startup smoke
evidence under `.git/zanzara-evidence/P1R-05/`:

```bash
docker compose --profile processing build whisper
docker compose --profile processing up whisper
```

The service accepts the same constrained `/v1/audio/transcriptions` JSON
shape, but deliberately returns no word timestamps. Its output is comparable
to Parakeet only on the shared chunk interval and text hypothesis fields.

Set
`HF_HOME`, `ZANZARA_MODEL_CACHE`, `SMOKE_AUDIO_HOST`, and `EVIDENCE_DIR` only when
the operator uses non-default cache locations.

P1R also adds a separate `voxtral` service for accuracy-oriented offline
benchmark hypotheses. It uses the pinned
`mistralai/Voxtral-Mini-4B-Realtime-2602` checkpoint, Transformers' native
Voxtral implementation, forced Italian request metadata, non-streaming
processing, and the model-card-recommended 480 ms delay configuration. The
service writes private startup smoke evidence under
`.git/zanzara-evidence/P1R-06/`:

```bash
docker compose --profile processing build voxtral
docker compose --profile processing up voxtral
```

It accepts the same constrained `/v1/audio/transcriptions` JSON shape as
Whisper and returns a text-first hypothesis with raw generated token IDs. It
does not synthesize word timestamps or provide a remote fallback. The smoke
audio and model cache are mounted read-only.

# Workstation operation

The app runs on this host at <http://complex.home.arpa:8000/>. SQLite state is
`.git/zanzara-state/state.db`, immutable outputs are in `.git/zanzara-artifacts`,
private run logs are in `.git/zanzara-evidence`, and Qdrant's rebuildable index
is in `.git/zanzara-qdrant`. The canonical audio archive is
`/export/scratch/archive/zanzara`; processing reads it without writing to it.
All state paths are on the workstation's local disk.

## Start, stop, restart

From the repository root, link the user service once and start the state service
and app. This host's `COMPOSE_ENV_FILES` points at a nonexistent `.env`, so clear
it for Compose commands.

```bash
systemctl --user link "$PWD/ops/zanzara-web.service"
env -u COMPOSE_ENV_FILES docker compose up -d qdrant
systemctl --user enable --now zanzara-web.service
curl --retry 10 --retry-connrefused --retry-delay 1 -f http://127.0.0.1:8000/healthz
```

```bash
systemctl --user restart zanzara-web.service
env -u COMPOSE_ENV_FILES docker compose restart qdrant
```

```bash
systemctl --user stop zanzara-web.service
env -u COMPOSE_ENV_FILES docker compose stop qdrant
```

Check app errors with `tail -n 50 .git/zanzara-state/web.log`;
check Qdrant with `env -u COMPOSE_ENV_FILES docker compose logs --tail=50 qdrant`.

## Process one queued stage

Start only the GPU model needed for the stage. `asr` uses `parakeet`,
`diarization` uses `diarization`, and `speaker_embeddings_resnet293` uses
`resnet293`; `attribution` needs no model service. Stop the previous GPU service
before starting a different one. Compose caps each model container at 20 GiB of
host RAM; running one model and one worker at a time leaves room for app/state.

```bash
env -u COMPOSE_ENV_FILES docker compose --profile processing up -d --no-deps --wait resnet293
uv run zanzara jobs enqueue --database .git/zanzara-state/state.db \
  --job-id voice-EPISODE-1 --stage speaker_embeddings_resnet293 \
  --source-sha256 SOURCE_SHA256_FROM_planning/corpus-20.json
systemd-run --user --scope -p MemoryMax=8G \
  uv run zanzara worker run --process --database .git/zanzara-state/state.db --owner workstation-1
uv run zanzara jobs status --database .git/zanzara-state/state.db voice-EPISODE-1
env -u COMPOSE_ENV_FILES docker compose stop resnet293
```

`worker run --process` claims one queued job, publishes through the existing
fenced stage path, prints the stage artifact path or typed error, then exits.
The command accepts the four stages above; `text_index` and `exemplars` retain
their existing direct `zanzara process` commands. A local lock permits only one
real worker per database at a time. A model service binds only to host loopback.

For a failed job, inspect its status and the worker output. After correcting the
cause, retry a terminal failure, or recover an expired lease after interruption:

```bash
uv run zanzara jobs status --database .git/zanzara-state/state.db JOB_ID
uv run zanzara jobs retry --database .git/zanzara-state/state.db JOB_ID
uv run zanzara jobs recover --database .git/zanzara-state/state.db
```

The worker has three attempts for transient failures and blocks deterministic
input failures. `jobs recover` requeues only expired leases; it does not publish
partial output. Check the completed artifact directory and its recorded job
status before rerunning a stage.

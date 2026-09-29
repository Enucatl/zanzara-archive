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

## Browse and review

Open <http://complex.home.arpa:8000/> on the trusted LAN. Search indexed words
or quoted phrases, filter by episode/date, and select a result to play its
original audio at the recorded time. Open
<http://complex.home.arpa:8000/speakers> to listen to anonymous voice excerpts,
request cross-episode candidates, and record a human same-person or
different-person decision. The page also supports undo, split, and naming
confirmed speakers. Similarity scores only rank candidates. The separate
<http://complex.home.arpa:8000/chunk-review> page collects human ASR references.
Only episodes with completed text or voice stages appear in those searches.

## Process one queued stage

For a new frozen episode, run `asr` and `diarization` with their respective
model services, then `attribution` and `text_index` to make transcript search
available. For voice search, run `exemplars` after diarization, then
`speaker_embeddings_resnet293` with its model service and publish the retained
vectors with the [voice index command](../README.md#resnet-voice-index).
`uv run zanzara process --manifest planning/corpus-20.json --episode EPISODE.opus
--stage STAGE` runs one stage directly; its JSON output names the artifact.
Repeat a completed stage only when its inputs or configuration change.

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

## Back up and restore

Back up the SQLite database and immutable artifacts together. Stop the app and
wait for any one-shot worker to finish so the snapshot includes matching
identity decisions, transcripts, and artifact records. The source audio stays
in the read-only archive and must be available after restore. Replace the
snapshot name with a new local directory each time:

```bash
set -e
snapshot=.git/zanzara-backups/2026-09-29-example
mkdir -p .git/zanzara-backups
mkdir "$snapshot"
systemctl --user stop zanzara-web.service
uv run python -c 'import sqlite3,sys; source=sqlite3.connect(".git/zanzara-state/state.db"); target=sqlite3.connect(sys.argv[1]); source.backup(target); target.close(); source.close()' "$snapshot/state.db"
rsync -a .git/zanzara-artifacts/ "$snapshot/artifacts/"
cp models.lock.json planning/corpus-20.json "$snapshot/"
systemctl --user start zanzara-web.service
```

Restore on the same checkout path, with the same source archive and model lock.
The database stores some absolute artifact paths, so a different checkout path
needs path remapping before use. Keep the current database and artifacts aside
until the restored state is verified. Qdrant is a rebuildable voice-search
projection, not the identity record:

```bash
set -e
snapshot=.git/zanzara-backups/2026-09-29-example
cmp "$snapshot/models.lock.json" models.lock.json
cmp "$snapshot/corpus-20.json" planning/corpus-20.json
systemctl --user stop zanzara-web.service
saved=.git/zanzara-state/before-restore-2026-09-29-example
mkdir "$saved"
for file in .git/zanzara-state/state.db*; do
  [ ! -e "$file" ] || mv "$file" "$saved/"
done
mv .git/zanzara-artifacts "$saved/artifacts"
cp "$snapshot/state.db" .git/zanzara-state/state.db
cp -a "$snapshot/artifacts" .git/zanzara-artifacts
uv run python -c 'import pathlib,sqlite3; db=sqlite3.connect(".git/zanzara-state/state.db"); assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"; assert db.execute("PRAGMA foreign_key_check").fetchone() is None; assert all(pathlib.Path(row[0]).is_dir() for row in db.execute("SELECT artifact_path FROM artifacts")); print("SQLite and artifact paths OK")'
systemctl --user start zanzara-web.service
curl -f http://127.0.0.1:8000/healthz
```

Check a transcript hit and a reviewed identity on the LAN after restore. If
Qdrant's index is missing or older than the restored database, rebuild it from
the retained `speaker_embeddings_resnet293` artifact directories using the
[`voice index command`](../README.md#resnet-voice-index), repeating
`--embedding-artifact` for each episode. Then check a voice candidate. Recover
an interrupted job with `jobs recover` above only after its lease expires;
inspect the job status before retrying a terminal failure. The
[isolated recovery drill](recovery-drill.md) records a completed restore check.

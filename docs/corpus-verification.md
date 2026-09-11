# Frozen corpus verification

P0-02 verifies the exact 20 filenames in `planning/corpus-20.json`. It never
selects a latest-20 set and it never writes to the archive. Each source is
resolved below the supplied archive root, including a check that symlinks do
not escape that root. The verifier hashes the bytes with SHA-256, checks that
the file did not change while it was read, and compares `ffprobe` codec,
channel, sample-rate and duration metadata. Durations are rounded half-up to
integer milliseconds.

The production container must mount the source read-only and keep generated
state elsewhere. For example:

```bash
docker run --read-only \
  --mount type=bind,src=/export/scratch/archive/zanzara,dst=/archive,readonly \
  --mount type=volume,src=zanzara-artifacts,dst=/artifacts \
  zanzara:local corpus verify --manifest /app/planning/corpus-20.json \
    --archive-root /archive --output /artifacts/corpus-verification.json
```

For local development, use the same separation of paths:

```bash
uv run zanzara corpus verify \
  --manifest planning/corpus-20.json \
  --archive-root /export/scratch/archive/zanzara \
  --output /tmp/zanzara-corpus-verification.json
```

The report records the manifest hash, every file result, the golden episode,
the original episode time origin and that no archive writes were performed.
The golden source is `260910-lazanzara.opus` with SHA-256
`06de18da0691a19738bcda30dace2d5be87c8be651e9bd9ed8e56b5828f64536`.
Persisted intervals remain integer millisecond half-open ranges relative to
the original episode start; decoding or later processing must not shift that
origin.

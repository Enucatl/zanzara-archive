# Frozen corpus source layout

P0-02 records the exact 20 filenames in `planning/corpus-20.json`. The
canonical archive is persistent and trusted; routine commits, reviews and code
runs do not hash or re-probe these source files. Processing resolves the named
source below the archive root and opens it when a stage needs it. The archive
is never modified.

The production container must mount the source read-only and keep generated
state elsewhere. For example:

```bash
docker run --read-only \
  --mount type=bind,src=/export/scratch/archive/zanzara,dst=/archive,readonly \
  --mount type=volume,src=zanzara-artifacts,dst=/artifacts \
  zanzara:local
```

Keep generated artifacts and state on `/artifacts` or another local writable
volume. Persisted intervals remain integer millisecond half-open ranges
relative to the original episode start; decoding or later processing must not
shift that origin.

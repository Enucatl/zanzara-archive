# Local annotation and reviewed references

P1-05 provides a loopback-only FastAPI/Jinja editor. Start it with a local
SQLite state database:

```bash
uv run zanzara web \
  --database .git/zanzara-state/state.db \
  --artifact-root .git/zanzara-artifacts \
  --manifest planning/corpus-20.json \
  --archive-root /export/scratch/archive/zanzara
```

The web command rejects non-loopback hosts. Open
`http://127.0.0.1:8000/annotations/260910-lazanzara.opus`. The page plays the
registered frozen source, draws source-relative millisecond timings, and lets
the reviewer edit words, standard/exclusive turns, overlaps, and
unintelligible spans. A P1-04 `attributed.json` can be imported locally into
the editor, or posted to the seed endpoint as a machine draft:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/annotations/260910-lazanzara.opus/seed \
  -H 'content-type: application/json' \
  --data-binary @attributed.json
```

Annotation writes are append-only revisions in SQLite. Each write requires an
`expected_revision`; a stale write returns HTTP 409 with the preserved current
revision. Machine writes are forced to `draft`/`machine` and cannot become
reviewed. Only a named human reviewer can save `status=reviewed`; the saved
revision records that reviewer and `reviewed_at`, and reviewed payloads must
list every reviewed word. Private exports are atomically written
under the ignored artifact root with `reference.json`, exact transcript views,
manual timing, and the deterministic five-block split manifest.

P1-06 can validate a private revision without validating human judgment:

```bash
uv run zanzara evaluation validate-reference \
  --corpus planning/corpus-20.json \
  --reference .git/zanzara-artifacts/<source-sha256>/annotation/<revision-key>
```

The validator checks the frozen golden source, source-relative integer timing,
human-reviewed/full-word coverage, E1 split bounds, provenance, and at least
200 distinct manually timed words. It reports block counts and exits non-zero
for insufficient evidence. Audio, annotations, identity data, and generated
exports remain private and are not committed to Git.

# Local annotation and reviewed references

P1-05 provides a FastAPI/Jinja editor with loopback access by default. For the
LAN-only review workflow on `complex.home.arpa`, start it with a local SQLite
state database and explicitly enable the network bind:

```bash
uv run zanzara web \
  --database .git/zanzara-state/state.db \
  --artifact-root .git/zanzara-artifacts \
  --manifest planning/corpus-20.json \
  --archive-root /export/scratch/archive/zanzara \
  --host 0.0.0.0 \
  --allow-network
```

The editor is unauthenticated, so only use this on the trusted LAN and stop it
when the review is complete. The DNS record for `complex.home.arpa` must point
to this machine; the application does not create DNS records. Open
`http://complex.home.arpa:8000/annotations/260910-lazanzara.opus`. The page
plays the registered frozen source, draws source-relative millisecond timings,
and lets the reviewer edit words, standard/exclusive turns, overlaps, and
unintelligible spans. A P1-04 `attributed.json` can be imported locally into the
editor, or posted to the seed endpoint as a machine draft:

Large annotation sections are paginated at 100 rows per page so the browser
does not create a DOM node for the entire episode at once. Page changes and
draft saves preserve edits made on other pages.

The supplied frozen manifest is registered idempotently in the local SQLite
state database when the application starts; no manual database import is
needed.

Check name resolution and service reachability from the reviewing computer
before opening the editor:

```bash
getent hosts complex.home.arpa
curl -fsS http://complex.home.arpa:8000/healthz
```

The health response should report `"access": "network-enabled"`.

```bash
curl -sS -X POST http://complex.home.arpa:8000/api/v1/annotations/260910-lazanzara.opus/seed \
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

The original P1 editor also exported a five-block split and a 200-word timing
sample. Those fields remain readable for historical P1 artifacts, but they are
not current transcription-quality release requirements. P1R uses frozen audio
chunks and human-reviewed chunk references; word timing is optional benchmark
metadata. P1-06 can validate a private legacy revision without validating
human judgment:

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

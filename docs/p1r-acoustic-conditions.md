# P1R acoustic-condition metadata

P1R-08A runs the locked `MIT/ast-finetuned-audioset-10-10-0.4593` AudioSet
AST checkpoint locally on the same source-relative chunk interval sent to the
ASR services. The application owns deterministic windows and condition
mapping; the AST service owns only AudioSet event probabilities. It never
infers speaker count or overlap and never reads ASR output.

The model is pinned to revision
`f826b80d28226b62986cc218e5cec390b1096902`, its `config.json:id2label` map is
retained by SHA-256, and ten-second mono 16 kHz windows are zero-padded only
when the final window is shorter. Window probabilities are mean-aggregated by
AudioSet class. Each relevant class ID/name/probability is retained both per
window and in the aggregate seed.

`p1r-acoustic-thresholds-v1` maps music probabilities to `none`, `background`,
`dominant`, or `uncertain` using provisional thresholds. These defaults are
not calibrated until reviewed development evidence is recorded. Speech
presence and non-speech activity are separate fields. Clipping fraction, peak
dBFS, RMS dBFS and silence fraction are deterministic signal measurements;
`audio_quality` is a `clean`/`degraded`/`uncertain` machine seed from those
measurements, not an AST claim.

The machine seed is immutable inside the chunk payload. Human `music_level` and
`audio_quality` corrections are append-only SQLite records and are exposed by
the local condition API/UI without replacing raw probabilities or the seed.
Speaker and overlap metadata remains owned exclusively by P1R-08.

## Operator calibration handoff

### Human listening task

For each prepared development clip, listen to the whole interval at a
comfortable, consistent playback volume, then select one music label based
on what you hear:

| Label | Listening rule |
| --- | --- |
| `none` | No audible music or singing; speech, laughter and noise alone do not count as music. |
| `background` | Audible music accompanies speech, while speech remains the foreground content. |
| `dominant` | Music or singing is the foreground content, including music-only clips. |
| `uncertain` | You cannot confidently choose, including ambiguous sounds or a transition that makes a single label misleading. |

These are listening guidelines, not numerical loudness thresholds. Replay
ambiguous clips and add a short note such as "music enters near the end" or
"faint tone, possibly music". Do not force a clear label to fill a class quota.
Judge before inspecting the AST prediction. Record an explicit reviewed label
even when it agrees with the machine seed.

No transcription, word timing, speaker identification or audio-quality label
is needed for this music calibration. The existing `zanzara web` command
launches the legacy P1 word-level editor. Its condition-correction panel is
not a prepared P1R calibration workflow: there is no calibration batch page,
automatic chunk playback/navigation or prediction-blind labeling flow.
A dedicated calibration review page and development batch must be prepared
before directing the operator to begin. This bounded prerequisite cannot wait
for the full P1R-10 transcription UI, which is itself blocked by P1R-08A.

Save decisions under the human reviewer's name. Preparation, provenance checks, score
extraction, threshold fitting and report generation can be performed by the
agent; human listening and the final review of ambiguous examples cannot be
replaced by model predictions.

### Evidence preparation and completion

The dedicated P1R calibration server is separate from the legacy word editor.
After preparing a private batch JSON, start it on the trusted operator LAN:

```bash
uv run zanzara calibration prepare \
  --batch .git/zanzara-evidence/P1R-08B/development-batch.json \
  --manifest planning/corpus-20.json \
  --database .git/zanzara-state/state.db
uv run zanzara web \
  --database .git/zanzara-state/state.db \
  --artifact-root .git/zanzara-artifacts \
  --manifest planning/corpus-20.json \
  --archive-root /export/scratch/archive/zanzara \
  --host 0.0.0.0 --allow-network --port 8000
```

Open `http://<operator-host>:8000/calibration/<batch_id>`. The page has no
manual chunk-ID entry: it plays one registered interval at a time, stops at
the interval end, hides AST scores, records the reviewer's name and optional
note, and advances after saving. Reloading resumes the latest saved decision.
The batch JSON must contain `batch_id`, the exact frozen corpus
`manifest_sha256`, `partition: "development"`, `segmentation_version`, and a
`chunks` array. Each chunk is the serialized `AudioChunk` with
`partition: "development"`, canonical episode/source hash, source-relative
integer bounds and segmentation fingerprint. The preparation command rejects
held-out/unknown episodes, mismatched hashes, invalid bounds and duplicate
chunks before writing any batch state.

For a browser or API client, the same preparation is available as
`POST /api/v1/calibration/batches`; decisions are
`POST /api/v1/calibration/<batch_id>/decisions` with `chunk_id`, one of the four
music labels, `reviewer`, `expected_revision`, and optional `note`. Export
decisions after review with:

```bash
uv run zanzara calibration export \
  --batch-id <batch_id> --database .git/zanzara-state/state.db \
  --output .git/zanzara-evidence/P1R-08B/review.json
```

The export reports reviewed and total counts. It is a review artifact only;
threshold fitting and the reviewed confusion summary remain the P1R-08A gate.

[P1R-08A](../planning/issues/P1R-08A.md) remains blocked until its reviewed
development-only calibration artifact exists, even when implementation checks
and the real RTX 5090 smoke pass. P1R-10 and dependent benchmark work cannot
proceed through this gate on smoke evidence alone.

1. Record and hash an episode-level development/held-out assignment before
   selecting calibration chunks or tuning thresholds. Use only development
   episodes from the frozen corpus, with canonical chunk IDs, source hashes,
   intervals and segmentation fingerprints. This assignment must carry forward
   into P1R-11; its later benchmark freeze must not move calibration episodes
   into held-out. Historical P1 within-episode splits are not eligible.
2. Have an identified human listen to development chunks and record
   `none`, `background`, `dominant`, or `uncertain`, with reviewer, review time
   and revision. Preserve the machine seed separately. Cover all three required
   music levels and report missing coverage; do not manufacture labels from
   AST predictions or replace uncertain judgments with `none`.
3. Retain the locked AST per-window and aggregated class IDs, names and
   probabilities for those exact chunks. Record model/runtime/image,
   label-map and preprocessing hashes. Derive music and speech scores using
   the same keyword groups and aggregation as `acoustic_conditions.py`.
   The GPU smoke summary alone does not contain these calibration inputs.
4. Select thresholds using only the reviewed development examples. Record the
   selection procedure, candidate settings and limitations; freeze the selected
   `AcousticThresholds.to_dict()` with an explicit version. The existing
   `development_confusion(examples, thresholds=...)` helper expects rows with
   explicit `partition: "development"`, `music_level`, `music_score` and
   `speech_score`. It computes counts but does not verify human review,
   episode membership or provenance; validate those against the source records
   before calling it.
5. Save an immutable private artifact containing the split/chunk manifest and
   reviewed-label hashes, score artifact hashes, selected thresholds, code
   commit, selection procedure, and confusion matrix. Rows are reviewed labels
   and columns are predictions; retain the `uncertain` row and column. Include
   per-class chunk counts and durations, total counts, review identity/time,
   and explicit coverage limitations. These are development calibration
   results, not held-out performance estimates.
6. Record the artifact path and SHA-256 in acceptance evidence, review the
   confusion and selected mapping, and rerun P1R-08A acceptance checks against
   the frozen configuration. Only after all acceptance items pass may #99 be
   marked Done and P1R-10 proceed. Public evidence contains sanitized aggregate
   results; audio, detailed labels and chunk-level evidence remain private.

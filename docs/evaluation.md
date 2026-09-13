# Legacy P1 ASR, diarization, attribution and timing evaluation

This document describes the retained P1-07/P1-08 implementation and historical
artifacts. P1R is the current transcription-quality methodology: it evaluates
human-reviewed, model-independent audio chunks with episode-level partitions
and reports ASR, diarization and integrated attribution separately. The old
golden-episode, five-block, 200-manually-timed-word and timing-error clauses
below are not active release gates. See `planning/P1R-01-MIGRATION.md` and the
P1R issue bodies for the current contracts.

The independent chunk-level lexical scorer and its versioned private/sanitized
report contract are documented in [p1r-asr-scoring.md](p1r-asr-scoring.md).

P1-07 provides a deterministic CPU-only scorer for the JSON reference and
hypothesis shape exported by P1-05. It has no model, network, credential, or
paid-provider dependency:

```bash
uv run zanzara evaluation score \
  --reference .git/zanzara-artifacts/<reference>/reference.json \
  --hypothesis .git/zanzara-artifacts/<hypothesis>/hypothesis.json \
  --output .git/zanzara-evidence/evaluation-<run-id>
```

The output directory is a new private E6 run containing `run.json`,
`metrics.json`, `coverage.json`, `errors.json`, `results.json`, and
`report.html`. The aggregate JSON/HTML includes source, reference, hypothesis,
split, configuration, and available model hashes, denominators, slices, metric
definitions, and pass/block logic. `results.json` and `errors.json` contain
private per-match timing data. Output is atomic and existing run directories
are never overwritten.

## Real golden baseline

P1-08 validates the reviewed human reference, the supplied E1 split, the
frozen corpus manifest and `models.lock.json`, then executes ASR, diarization
and attribution as fenced `DurableWorker` jobs against the local inference
services before scoring the resulting attribution. The required invocation is:

```bash
uv run zanzara evaluation run \
  --suite golden \
  --reference "$ZANZARA_REFERENCE" \
  --split "$ZANZARA_GOLDEN_SPLIT" \
  --output "$ZANZARA_RESULTS"
```

The command accepts optional `--corpus`, `--archive-root`, `--artifact-root`,
`--database`, `--model-lock`, endpoint, decoder and language overrides. It
does not publish an evaluation directory until input validation, all worker
stages and scoring succeed. The private run records immutable source,
reference, split, model-lock and frozen-configuration hashes, worker job
states, development/held-out/full-episode slices, hardware context, sampled
peak process RSS/VRAM, input/output bytes, wall time, real-time factor, retry
count and zero paid cost. Public issue comments contain only sanitized
aggregates; transcript and detailed timing artifacts remain private.

Stage wall time is the measured warm end-to-end worker time. The local service
contract does not expose model-only, decode-only, queue or network timing, and
cold-start timing is unavailable when the locked services are already loaded;
those fields remain explicitly null rather than being inferred.

Raw WER preserves case and punctuation. Normalized WER and CER use the frozen
Italian `it-v1` rule from E1. DER uses standard overlap-inclusive turns with a
zero-collar primary score and a separately reported 250 ms reference-boundary
collar score. JER uses the primary DER mapping. SA-WER uses the maximum-overlap
one-to-one mapping, chronological normalized per-speaker alignment, and counts
unmapped words as insertions. Timing quantiles use linear interpolation and
are reported in seconds; input and persisted interval records remain integer
milliseconds.

An unreviewed or synthetic reference produces `insufficient_evidence`, even
when known-answer metrics are perfect. Zero reference denominators are also
`insufficient_evidence`, never a pass. The scorer does not fabricate human
references or report a model baseline verdict. The initial E2 quality checks
are limited to the specified held-out normalized WER and overlap-inclusive DER
thresholds; no threshold is invented for other metrics.

# ASR, diarization, attribution and timing evaluation

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

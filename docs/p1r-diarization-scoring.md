# P1R independent diarization scoring

`zanzara_archive.p1r_diarization` is the word-blind Community-1 evaluation
harness. It accepts source-relative, integer-millisecond, half-open speaker
turns and optional explicit overlap intervals. It does not parse, validate or
consult `words`, ASR hypotheses, word timestamps or lexical alignment.

## Locked evaluator

- Evaluator: `native-interval-evaluator-v1`.
- Metric definitions: `p1r-diarization-v1`.
- Primary DER: zero-collar, standard overlap-inclusive turns. Hypothesis
  speaker IDs are mapped one-to-one to reference IDs by maximum scored
  speaker-time overlap, with deterministic lexical tie-breaking. Reported
  components are missed speech, false alarm and speaker confusion, divided by
  reference speaker-time.
- Optional collar: `collar_ms` excludes windows around reference turn
  boundaries before DER scoring. The configured value is recorded in every
  report; the default is zero.
- JER: mean per-reference-speaker `(1 - intersection / union)` using the DER
  mapping. An unmapped reference speaker has zero intersection over its scored
  reference time.
- Speaker-count error: unique active hypothesis speakers minus unique active
  reference speakers within each aggregate or condition slice; both signed and
  absolute values are retained.
- Overlap detection: duration-based precision, recall and F1 over the unions
  of explicit reference and hypothesis overlap intervals. Metrics are
  `not_available` when reference overlap truth is absent and
  `insufficient_evidence` when explicit truth contains no positive reference
  duration.

All interval calculations preserve half-open geometry and use milliseconds;
only derived DER/JER durations are represented in seconds where the report
names the value as seconds. Condition slices may contain one interval or
multiple `ranges_ms` intervals, so disjoint benchmark chunks can be reported
as one condition aggregate.

Missing reference or hypothesis interval streams produce `unscorable`; the
harness never derives diarization truth from transcript words. The known-answer
fixtures in `tests/fixtures/p1r_diarization_known_answers.json` cover missed
overlap speech, speaker confusion, and partial overlap detection. They are
synthetic CPU cross-checks and do not claim real Community-1 quality.

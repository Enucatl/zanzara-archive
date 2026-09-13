# P1R chunk ASR scoring

`zanzara_archive.p1r_asr` is the CPU-only lexical scorer for the P1R benchmark.
It consumes the same immutable chunk reference and model hypothesis for every
model. It does not load a model, call an inference service, use word timing for
alignment, or choose a production model.

## Locked metrics

- `raw_wer` uses whitespace tokens and preserves case and punctuation.
- `normalized_wer` uses Italian `it-v1`: Unicode NFC, casefold, punctuation
  categories replaced with spaces, then whitespace collapse.
- `normalized_cer` scores Unicode code points of the normalized text,
  including the single inter-word spaces.
- `orc_wer` is the locked `orc-equivalent-interleaving-v1` metric. Reference
  speaker streams keep their within-stream order; the scorer chooses the
  lowest-error interleaving against the hypothesis token stream(s), ignoring
  speaker labels. This is speaker-independent overlap-aware lexical scoring,
  not cpWER, diarization scoring, or attribution scoring.

All metrics retain edit counts and denominators. Aggregates sum counts before
dividing, rather than averaging per-chunk percentages. Every reported slice
contains chunk count, duration in milliseconds and reference-word denominator.
Slices with no chunks are omitted; zero reference denominators are explicitly
`not_applicable`.

## Unintelligible masking

Text-only masks are safe only when explicit reference and hypothesis token-index
masks are both supplied. The paired indices are removed before raw WER,
normalized WER and CER. A one-sided or invalid mask is `unscorable`; the scorer
never guesses an alignment or silently rewards a hypothesis for omitted audio.
Speaker-stream masks use the same explicit token-index rule.

## Reports

`score_chunks` returns a versioned report with configuration and source/model
hashes. `write_asr_score_report` atomically creates a new private directory:

- `metrics.json` is the sanitized aggregate and contains no transcript text;
- `details.json` is private per-chunk metric detail;
- `report.html` is a sanitized aggregate view;
- `run.json` records artifact checksums and metric/version provenance.

Synthetic fixtures live in `tests/fixtures/p1r_asr_known_answers.json` and are
not evidence of real archive quality. Human audio-reviewed references, frozen
episode partitions and real model provenance remain release requirements.

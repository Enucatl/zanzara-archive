# P1R integrated speaker-attributed scoring

`zanzara_archive.p1r_integrated` is the separate CPU-only scorer for the
combined "who said what" question. It does not relabel lexical WER or reuse
interval DER/JER as an attribution metric.

## Locked cpWER contract

- `cpwer-v1` consumes explicit reference and hypothesis speaker text streams.
  Reference streams are human-reviewed channels; hypothesis streams are the
  model output after its diarization/channel assignment. A text-only stream
  is valid input, but it must not be inferred from a flat transcript.
- Speaker IDs are sorted for deterministic processing. The scorer chooses the
  one-to-one assignment with the minimum sum of normalized Italian `it-v1`
  Levenshtein edit counts. Equal-cost assignments use the lexicographically
  smallest sequence of reference IDs, with `~unmapped` as the dummy channel.
  Missing reference or extra hypothesis channels are represented by dummy
  channels and contribute deletions or insertions respectively.
- The denominator is every normalized reference token across all explicit
  reference streams. Reference and hypothesis channel text is aligned in its
  own chronological stream order. A stream is not concatenated with another
  stream, so words in explicit overlapping channels are retained once per
  channel and are not deduplicated.
- Optional `scored_ranges_ms` are sorted, non-overlapping, half-open integer
  millisecond ranges. Optional `Turn` and `Overlap` records are validated
  against the supplied duration and channel IDs. Text streams are caller-
  clipped; the scorer does not invent word timing or clip text from intervals.

The raw assignment, including dummy channels and per-channel edit counts, is
kept in `details.json` by `write_integrated_score_report`. `metrics.json` and
`report.html` contain only sanitized aggregate data. The report type and
metric names identify this output as integrated speaker-attributed scoring.

## tcpWER decision

tcpWER is explicitly deferred (`status: deferred`). P1R permits text-only
hypotheses and does not make word-timing gold or a time-constrained
speaker-attributed alignment contract a release requirement. Implementing a
time-constrained variant would therefore make non-reproducible assumptions;
it is not silently substituted by cpWER.

Synthetic fixtures in `tests/fixtures/p1r_integrated_known_answers.json` cover
correct words/wrong speaker, wrong words/correct speaker, global label swaps
and preserved overlap channels. They establish deterministic implementation
behavior only, not real archive or model quality.

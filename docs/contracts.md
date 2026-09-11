# Contract boundaries

`zanzara_archive.contracts` is the CPU-only boundary for pipeline stages and
internal services. It contains immutable records, adapter protocols, and typed
validation failures. It does not load models, access the network, or accept
arbitrary filesystem paths.

All persisted intervals use integer milliseconds and half-open `[start_ms,
end_ms)` bounds. `TranscriptResult(status="timed")` is the only transcript
status that can carry production `TimedWord` records. A `text_only`,
`missing_asr`, or `no_words` result stays explicit and fails
`require_production()`; text-only cloud comparisons therefore cannot be used
for attribution.

`EmbeddingBatch` validates equal item/vector counts, input ordering, declared
dimensions, finite values, and non-zero vectors. Every result carries source
and model provenance. `CandidateScore.calibration_status` is
`uncalibrated_rank_fusion` until a frozen calibration artifact exists; that
label is never a probability.

The JSON representation is [schemas/contracts.json](../schemas/contracts.json)
and synthetic, non-private examples live in
[tests/fixtures/contracts.json](../tests/fixtures/contracts.json). The public
route names from D10 are exported as `API_V1_ROUTES`; API responses use
`ApiEnvelope` and always carry a request ID.

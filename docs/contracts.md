# Contract boundaries

`zanzara_archive.contracts` is the CPU-only boundary for pipeline stages and
internal services. It contains immutable records, adapter protocols, and typed
validation failures. It does not load models, access the network, or accept
arbitrary filesystem paths.

All persisted intervals use integer milliseconds and half-open `[start_ms,
end_ms)` bounds. `TranscriptResult(status="timed")` is the only transcript
status that can carry production `TimedWord` records. P1R benchmark hypotheses
may be `text_only`; their optional words/segments are immutable metadata, not
human gold or a release prerequisite. A `text_only`,
`missing_asr`, or `no_words` result stays explicit and fails
`require_production()`; text-only cloud comparisons therefore cannot be used
for attribution.

`EmbeddingBatch` validates equal item/vector counts, input ordering, declared
dimensions, finite values, and non-zero vectors. Every result carries source
and model provenance. `CandidateScore.calibration_status` is
`uncalibrated_rank_fusion` until a frozen calibration artifact exists; that
label is never a probability.

P1R benchmark contracts are separate from production transcript timing:
`AudioChunk` uses a deterministic source interval and segmentation fingerprint,
`TranscriptionHypothesis` permits text-only, word-timed or segment-timed model
artifacts, and `ChunkCondition` retains speaker streams and genuine overlap
without requiring word timing. `ChunkCondition.acoustic_metadata` retains the
AudioSet AST label-map probabilities, deterministic window aggregation and
signal measurements separately from speaker metadata; its correction is an
append-only human view. `TranscriptReference` and `ReferenceRevision`
are append-only records; only a reviewed `human_truth` revision is reference
truth. `ChunkBenchmarkManifest` ties the versioned chunks, references,
hypotheses, episode partitions and source hashes together.

The P1R chunker records its version, canonical duration configuration and
boundary-input fingerprints on `ChunkBenchmarkManifest`. Each `AudioChunk`
also carries the resulting segmentation fingerprint. Chunk boundaries are
source-relative and are selected from silence/VAD, speaker-turn and acoustic
evidence only; ASR text and model fingerprints are not inputs to segmentation.

The JSON representation is [schemas/contracts.json](../schemas/contracts.json)
and synthetic, non-private examples live in
[tests/fixtures/contracts.json](../tests/fixtures/contracts.json). The public
route names from D10 are exported as `API_V1_ROUTES`; API responses use
`ApiEnvelope` and always carry a request ID.

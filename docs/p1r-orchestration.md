# P1R chunk inference orchestration

`ChunkInferenceOrchestrator` runs every registered text-first adapter against
the same common-preprocessed WAV payload for each immutable `AudioChunk`.
Registration is keyed by a stable adapter ID and stores the complete
`ModelFingerprint`; adding another adapter does not add a database enum or
schema column.

For each manifest/chunk/adapter pair, SQLite stores one durable
`chunk_inference_dispatches` row and one existing worker job. The dispatch ID
is derived from the manifest content, chunk identity, adapter ID, model
fingerprint, configuration and common preprocessing. A successful response
publishes its raw JSON response as an immutable `chunk-inference` artifact,
then records the immutable `TranscriptionHypothesis` with the raw artifact ID,
source interval, common audio hash, preprocessing hash and runtime metadata.

The orchestrator claims exact jobs with the existing 120-second lease and
fencing contract, heartbeats long requests, and resumes only the same adapter
after a retryable failure. A deterministic or exhausted model failure remains
visible in its dispatch row as an `ApiError`; successful peers are retained.
Reruns reconcile an already-persisted hypothesis before making another model
request. A success without matching hypothesis provenance, or a hypothesis
whose chunk/source/model does not match its dispatch, stops orchestration as a
provenance error.

The report exposes `counts_by_model`, immutable hypothesis IDs, and typed
failure records. It is an orchestration and provenance result only: it does
not rank candidates, mutate human references, promote gold, score transcripts,
or substitute a missing model/provider.

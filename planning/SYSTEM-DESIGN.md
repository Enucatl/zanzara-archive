# System design

This is the binding implementation specification. [README](README.md) records decisions that supersede `plan.md`. Section IDs D1–D12 are stable issue references. Model and API evidence is in [SOURCES](SOURCES.md); measured quality is a release requirement, not an architectural assumption.

## D1 — Packages and deployment boundary

The application is typed Python 3.14, FastAPI, Jinja server-rendered pages, and modest vanilla JavaScript for playback, annotation, uploads and polling. Use `uv`, `uv_build`, `src/zanzara_archive/`, a root `pyproject.toml` and committed `uv.lock`; pytest and Ruff run in CPU-only CI. Expose a `zanzara` CLI. Separate Python 3.11 packages under `services/{parakeet,diarization,resnet293,eres2net,wavlm,text_embeddings}/` each have their own `pyproject.toml`, lock, `src/` package and Dockerfile using `uv sync --frozen`. No ML dependency belongs in the application's interpreter. GPU imports and gated models must not be required by CPU tests.

The only persistent services are the web application, one SQLite-backed worker, a dedicated Qdrant container, and inference services selected by the measured Compose profile. No Redis, Celery, Kubernetes, graph database or reuse of an unrelated Qdrant deployment. Internal HTTP adapters are the compatibility boundary; Docker is sufficient and llama.cpp is not required. Begin with loopback-only development access; public exposure is eligible only after D11 is enforced.

Proposed modules: `contracts`, `storage`, `jobs`, `audio`, `inference`, `transcripts`, `voice`, `identity`, `search`, `evaluation`, `web`, `cli`. These are ownership boundaries, not a mandate for abstractions beyond the concrete interfaces below.

## D2 — Corpus and audio

[corpus-20.json](corpus-20.json) fixes exactly 20 relative filenames from 2026-07-01 through 2026-09-10, with recorded source identity, duration in milliseconds, codec, channel count and sample rate. The golden episode is `260910-lazanzara.opus`. P0-02 imports this manifest and establishes safe source paths; it never selects the latest files again. A source that cannot be opened when a stage needs it stops that stage, while persistent source content is not reverified. Archive-wide counts and durations in the original plan are estimates, not measured capacity.

Mount `/export/scratch/archive/zanzara` read-only at `/archive`. Resolve manifest relative filenames beneath that root and reject traversal and escaping symlinks. The canonical archive is a persistent trusted input: do not re-hash or re-probe its files as a routine precondition for commits, reviews or processing runs. Check that a source can be opened when a stage actually needs it, but use the manifest's recorded source identity and media metadata without recurring verification. Never alter canonical Opus sources. FFmpeg decodes locally into temporary model-specific PCM, normally 16 kHz mono. Record the decoder version, resampling/channel policy, original duration and transform hash. A model needing another format gets an explicit transform fingerprint. Preserve the time origin; silence removal must not shift persisted timestamps.

All persisted media intervals use integer milliseconds and half-open `[start_ms,end_ms)` bounds relative to the original episode. Require `0 <= start_ms < end_ms <= duration_ms`; word IDs are stable within an ASR artifact. Boundary conversions round once at the adapter, retain raw model output privately, and validate monotonicity without inventing timing. Invalid or missing timed words make a production ASR artifact unusable.

## D3 — Canonical data and publication

SQLite owns episodes, corpus manifests, model records, artifact manifests, processing runs, jobs, transcript words, standard/exclusive turns, overlap intervals, episode speakers, exemplars, text chunks/word references, annotation revisions, review/split records, identity decisions and names, upload jobs, cost reservations and evaluation report metadata. Use foreign keys, WAL, FTS5, schema migrations and bounded busy timeouts. Episode-speaker keys include episode and diarization artifact ID; local labels such as `SPEAKER_00` never identify a person across recordings. Reprocessing diarization must mark old identity mappings stale for human reconciliation rather than silently attaching decisions to new turns.

Local Docker volumes: `state` for SQLite including WAL/SHM, `artifacts` for immutable stage output and indexes' source vectors, `qdrant` for Qdrant storage, `model-cache`, and `scratch` for decoded audio/uploads. Resolve the volume backing filesystem and reject network storage for SQLite/job state. The archive is on NFS; WAL requires local shared-memory semantics ([SQLite WAL](https://sqlite.org/wal.html)).

Artifact layout: `/artifacts/<source_sha256>/<stage>/<stage_key>/`. Write output files and checksums to a sibling temporary directory, fsync, then atomically rename a completed manifest on the same local filesystem. Only afterward commit the SQLite reference in a short transaction. On restart, reconcile complete unreferenced artifacts; never treat a partial directory as complete. Readers follow a committed generation pointer, never directory listings. Store an artifact's source hash, ordered upstream artifact hashes, model fingerprint, preprocessing configuration, pipeline version, schema version and file checksums. Keys hash canonical sorted JSON; recording wall-clock time does not change content identity.

Qdrant is a rebuildable retrieval projection. Keep original vectors with immutable artifacts, even if Qdrant's cosine storage normalizes them. Collections are `speaker_exemplars_<generation>`, `speaker_centroids_<generation>`, and `text_chunks_<generation>`. Exemplar and centroid points have named vectors `resnet293`, `eres2net`, `wavlm`; text uses `dense` (1,024 dimensions). Fingerprints fix each dimension and cosine metric before collection creation. Use deterministic UUID point IDs and indexed episode/date/episode-speaker payloads. SQLite is authoritative for mutable global identity filters. A generation is published only after all expected points and payloads validate; atomically update the SQLite bundle pointer for both speaker collections. Do not rely on cross-database transactions or expose partial indexes.

## D4 — Worker and invalidation

Pipeline graph:

```text
manifest -> decode -> ASR words -------------------> attribution -> chunks -> text index
                 \-> diarization turns/overlaps ---/
                              \-> clean exemplars -> three encoders -> voice indexes
                                                                          -> candidates
                                                                          -> human decisions
```

Diarization and speaker indexing run without ASR. ASR changes invalidate attribution/chunks/text vectors, not speaker vectors. Diarization changes invalidate attribution, exemplars, dependent voice vectors and candidate results. Encoder changes invalidate only that model's vectors and derived speaker generation, not ASR. Identity/name changes affect relational membership and search filters without recomputing acoustic vectors.

Persist job states `queued`, `running`, `retry_wait`, `succeeded`, `failed`, `cancelled`, and `blocked`. In a short `BEGIN IMMEDIATE` transaction claim an eligible job, increment its fencing token and persist owner/lease expiry. Initial lease 120 seconds, heartbeat every 30 seconds, at most three attempts; transient failures back off 5 then 30 seconds. Never hold a database transaction during inference. Publication verifies the current owner and fencing token, so a stale worker cannot publish. Reclaim expired leases on restart. Deterministic validation/model-access failures block immediately; exhausted transient failures become failed. Retry of an ambiguous paid request needs D8 reconciliation before any resend. Persist errors, stage duration, attempts and recovery action. Serving and processing share this one worker queue; measure contention before adding concurrency.

## D5 — Models and adapters

| Interface | Fixed initial candidate | Required output |
|---|---|---|
| `Transcriber.transcribe` | `nvidia/parakeet-tdt-0.6b-v3` through NeMo | Text, real word timing, optional segments and actual confidence if supplied |
| `Diarizer.diarize` | `pyannote/speaker-diarization-community-1` | Standard turns, exclusive turns, explicit overlap intervals |
| `SpeakerEmbedder.embed` A | `Wespeaker/wespeaker-voxceleb-resnet293-LM` | Ordered exemplar vectors, locked dimension/fingerprint |
| `SpeakerEmbedder.embed` B | `iic/speech_eres2net_sv_en_voxceleb_16k`, revision `v1.0.3` | Same contract, separate vector space |
| `SpeakerEmbedder.embed` C | Microsoft UniSpeech released **fine-tuned WavLM Large verification checkpoint, Fix pre-train = No**, plus trained verification head | Verification vectors, not mean-pooled base WavLM features |
| `TextEmbedder.embed` | `BAAI/bge-m3`, dense output | Ordered strings in, ordered 1,024-dimensional vectors out |

Model lock records registry/repository, immutable revision, every checkpoint checksum including heads, preprocessing, precision, dimensions, license/terms evidence, Python/runtime versions, container image digest and hardware smoke result. P1-01 acquires and smoke-tests all six candidates on the RTX 5090 before adapters integrate them. Human model-access acceptance is P1-H01. A broken download, unavailable checkpoint, incompatible CUDA kernel or unverified license is a blocker; no silent architecture substitution. Published benchmark scores do not establish archive performance.

Define immutable typed records in P0-03: `AudioArtifact`, `ModelFingerprint`, `TimedWord`, `Turn`, `Overlap`, `TranscriptResult`, `DiarizationResult`, `EmbeddingBatch`, `ArtifactManifest`, `JobStatus`, `CandidateScore`, `IdentityDecision`, `EvaluationReport`. Audio artifacts carry checksum, format and time origin; internal service access uses a constrained shared artifact ID or validated audio bytes, never an arbitrary URL/path. Vectors must be finite, nonzero, dimension-correct, and preserve input ordering. Return typed failures such as `unsupported_capability`, `invalid_audio`, `model_unavailable`, `timeout`, `budget_blocked`, `generation_mismatch`; API errors include code, message, retryable and request ID.

A service exposes internal `/health`, `/ready`, and its typed `/v1/diarize` or `/v1/embed-speakers` endpoint; text uses OpenAI-compatible `/v1/embeddings`, and local transcription matches D8's `/v1/audio/transcriptions`. Fingerprint/capability metadata accompanies responses. Requests have bounded timeouts and payloads. Health distinguishes process liveness from loaded-model readiness. Genuine GPU fixtures are private artifacts; CPU contract tests use synthetic vectors and handcrafted timing, without claiming model quality.

## D6 — Transcript production

Start Parakeet with 300-second ownership windows and five seconds of left/right context, clipped at episode boundaries. Decode/infer each expanded interval, convert model-local times to original offsets, and retain only words whose midpoint belongs to that window's half-open ownership interval. Assign a midpoint exactly at a boundary to the later window. Persist window offsets and settings. Adjacent repeated words may be real speech: do not deduplicate by text alone. Test both duplicated boundary output and omitted words with known fixtures; benchmark changes on development data only. Window parameters are fingerprints, not silent runtime tuning.

Run Community-1 with episode-wide clustering. Internal segmentation is allowed, but never concatenate independently numbered chunk speakers. Preserve standard and exclusive outputs separately. Derive intervals with at least two active standard speakers as overlap; retain all active IDs.

Attribute each word to the exclusive turn with the greatest temporal intersection. On equal intersection prefer the turn containing the word midpoint; otherwise use stable speaker ID. With no intersection, leave unassigned. Attach overlap flags from standard diarization independently of exclusive attribution. Retain word references in JSON, text chunks and exports. TXT/SRT/VTT are derived views; no renderer changes canonical times or speaker identities.

## D7 — Voice retrieval and identity

Subtract overlaps and 250 ms on both sides of speaker transitions from eligible turns. In remaining clean runs greedily choose non-overlapping excerpts up to 15 seconds, preferring 8–15 seconds; accept 3–8 seconds only when insufficient preferred excerpts exist. Rank deterministically by preferred duration class, lower clipping fraction, longer duration, then start time; retain at most ten per episode speaker. Store clipping fraction, RMS, duration and exclusion reasons as measurements, never invented model confidence. A speaker with no qualifying 3-second segment remains visible as `not_voice_searchable`.

L2-normalize each exemplar per model and use the normalized arithmetic mean as its episode-speaker centroid; retain all valid exemplars, without an unspecified outlier threshold. A zero-norm mean is not indexable. For each encoder retrieve top 50 centroids, union by episode-speaker ID and exclude the query episode by default before limiting. Explicit same-episode mode must be labelled. Uploaded queries have no episode exclusion. Verify every candidate under all three models.

For each model independently, compute the cosine pair matrix, sort pairs by descending similarity then query/candidate exemplar IDs, and greedily accept pairs whose endpoints are unused until `min(n,m)` are matched. The model score is the median of those matches. This specifies deterministic one-to-one matching, not maximum-single-pair scoring. Expose matched pairs and all per-model scores. Missing a model makes the full ensemble unavailable; do not silently score with two.

Until reviewed calibration data pass E3, combine each model's verified candidate rank by `sum(1/(60+rank))`, with stable candidate-ID tie breaking, and label it **uncalibrated rank fusion**, never a probability. Conditional logistic fusion uses only the three median cosine scores, standardization fitted on calibration data, L2 regularization and identity-grouped validation; the resulting model includes calibration/split fingerprints. E3 fixes training and held-out reporting rules. Return candidate appearances, representative excerpts, provenance, model scores, coverage and calibration status.

SQLite stores append-only human decisions: `same_person`, `different_person`, `uncertain`, reviewer, timestamp, evidence artifact IDs and active/superseded status. A proposed merge must check all active different-person edges across both clusters transactionally, including transitive contradictions. Only a human-confirmed same-person operation changes membership. Names are separate manual labels, never inferred from voices. Undo supersedes the decision and recomputes affected membership; splitting records desired memberships, retracts conflicting same-person edges explicitly and preserves history. Test merge → contradiction refusal → undo → split. A stale UI revision returns conflict rather than overwriting another review. Acoustic reprocessing does not erase decision evidence.

## D8 — Shared inference, provider changes and money

Inspect `shared-inference` at `04c2eddcdf75e7893dc7648134ef02ebbf9dda9a`, then re-read upstream before editing. Reuse its `InferenceClient.embed` for text. P3-01 adds typed `InferenceClient.transcribe(model, audio, audio_format, language, timestamp_requirements, provider_options)` with text, optional timed words/segments, usage, request ID and raw response. Preserve typed errors and tracing conventions; keep audio/text payload traces local and out of public CI/log uploads. Test upstream independently and pin the archive to the merged commit, never a floating branch. If merge is pending, the integration issue remains open.

OpenRouter's JSON `/api/v1/audio/transcriptions` takes `model`, base64 `input_audio.data`, `input_audio.format`, optional `language`, `response_format: verbose_json`, `timestamp_granularities: [word, segment]`, and provider options under `provider.options`. The local Parakeet endpoint implements this subset. Probe model/provider combinations for Italian and actual returned timing; a requested field is not proof of capability. Text-only results are valid for WER comparison and invalid for production attribution. No interpolated timestamps. Local embedding inference uses the same OpenAI-compatible request/response shape used by shared-inference.

Model or provider changes create a new index generation by default. Any reuse requires an explicit equivalence report: matching immutable weights/tokenization/preprocessing/dimension, normalized-vector cosine >=0.9999 on every frozen probe and identical ordered top-10 results on frozen retrieval probes. If weights cannot be identified, the check fails. Record the report before opting into reuse; matching dimensions alone never suffices. Arbitrary audio embedding endpoints are not speaker-verification replacements.

Initial paid benchmark total is US$10, including paid capability probes, cloud text probes and retries. Human P3-H01 supplies an OpenRouter key with a non-resetting US$10 credit limit and confirms balance; never publish secrets. A local SQLite ledger atomically reserves a conservative upper-bound USD cost for every attempt before network I/O and enforces `spent + reserved <= 10`. Unknown pricing, missing credentials, insufficient balance or an unbounded cost estimate blocks dispatch. Reconcile usage and failed-request charges when exposed. Keep unknown/ambiguous charges reserved at their upper bound until reconciled; a timeout cannot free budget and trigger a free retry. Disable opaque SDK retries or route every retry through the ledger. P6–P8 are local by default; this budget does not authorize later spending.

## D9 — Text retrieval

Make chunks from consecutive attributed words, breaking at episode-speaker changes, 200 words, or a 60-second span, whichever comes first. Unassigned runs stay distinct. Keep exact ordered word IDs and original times, and preserve overlap flags. Derive a stable chunk ID from attribution artifact and word range. Do not silently truncate text at the embedding tokenizer; reject/report a model-limit violation.

SQLite FTS5 with `unicode61 remove_diacritics 0` and BM25 provides lexical ranking (lower FTS5 BM25 is better). Use bound parameters and safely parsed user terms; malformed FTS syntax is a validation error, not executable SQL. Dense BGE-M3 vectors live in Qdrant. Apply episode/date/speaker filters before each top-k retrieval; resolve global speaker membership from canonical SQLite. For hybrid mode take up to 100 lexical and 100 dense candidates and sum reciprocal ranks with `k=60`, deduplicate by chunk ID and break ties by chunk ID. Return excerpts, word references, timestamped media link, mode and generation.

Expose lexical, semantic and hybrid modes. When embeddings/Qdrant are unavailable or the active text generation is incomplete, return lexical results with a visible degraded flag and reason; report requested/effective modes. Keep FTS/chunk publication transactional and validate vector generation consistency before dense queries. No reranker or sparse BGE output in v1.

## D10 — Website and API

Pages: search with modes/filters; episode transcript with word-linked playback and overlap indicators; archive-speaker voice query; upload query with selected-speaker preview; candidate audio comparison; identity review/history; global-speaker appearances, anonymous recurrence and chronology; evaluation dashboard and job status. Human annotation is built in P1 as local tooling and later integrated into the private application. Show ASR-missing historical appearances with playable audio and clear transcript-unavailable state.

| Route under `/api/v1` | Behavior |
|---|---|
| `GET /episodes`, `/episodes/{id}`, `/episodes/{id}/transcript` | Cursor-paginated summaries, detail, attributed words; missing ASR is explicit |
| `GET /media/{episode_id}?start_ms=&end_ms=` | Validate registered ID and bounds; locally serve/transcode the segment, support browser seeking/range behavior |
| `GET /search?q=&mode=&episode_id=&date_from=&date_to=&speaker_id=` | Bounded lexical/semantic/hybrid result page |
| `POST /voice-search/jobs` | JSON archive speaker or multipart upload; return 202 with job ID |
| `POST /voice-search/jobs/{id}/speaker`, `DELETE /voice-search/jobs/{id}` | Select one detected upload speaker with expected revision, or cancel the query; enforce expiry and ownership |
| `GET /jobs/{id}`, `GET /voice-search/jobs/{id}/candidates` | State, retry action, then ranked candidates |
| `GET /speakers`, `/speakers/{id}/appearances`, `/speakers/{id}/candidates` | Anonymous/global discovery and filtered chronology |
| `POST /identity-decisions`, `POST /identity-decisions/{id}/undo`, `POST /speakers/{id}/split` | Human decision with expected revision; 409 on stale/contradictory state |
| `GET /evaluation/reports`, `/evaluation/reports/{id}` | Aggregate report metadata and protected report content |

Use 400/422 for invalid requests, 404 for unknown IDs, 409 for conflicts and 503 for temporary service failures; return D5's error envelope. Never take arbitrary file paths or remote media URLs. Media/transcript/report endpoints enforce the same authentication as HTML.

Upload limits: WAV, FLAC, MP3, Opus/Ogg and WebM; maximum 25,000,000 bytes and 60,000 ms decoded duration. Check bytes while streaming and probe content rather than trusting extension/MIME. Decode in a resource-bounded local process, diarize, require a single selected speaker (for multiple, use job state `blocked` with reason `needs_speaker_selection` and detected speaker IDs/preview excerpts), then apply D7 cleanliness rules and all three encoders. The selection endpoint takes `episode_speaker_id` from that upload and `expected_revision`, validates ownership/expiry, and resumes the job; a sole detected speaker may be selected automatically. Less than three seconds of clean selected speech is rejected. Start asynchronous work with 202 and pollable states; store temporary query vectors outside archive collections. Expire upload bytes, decoded audio, query vectors and candidate payloads one hour after receipt, including failures and abandoned selections. Recovery sweep removes expired data before resuming jobs. Cancellation/expiry wins over later publication. Uploads never enroll identities. Keep only non-content operational counters afterward.

## D11 — Access and operational profiles

Compose profiles: `processing` (worker plus processing services), `serving` (web, worker, interactive encoders/text embeddings and Qdrant), `exposure` (cloudflared, used with serving). Shared services listed in multiple profiles still have one container; benchmark actual simultaneous use. Qdrant and model services have no public host ports. Only web is routed through the tunnel. Local development binds loopback and uses an explicit test-only auth fixture; production has no bypass.

At the origin validate the Cloudflare Access JWT from `Cf-Access-Jwt-Assertion`: approved asymmetric algorithm/signature against cached rotating JWKS, exact configured issuer and audience, expiry and not-before. Missing/invalid credentials fail closed. Unknown key triggers a bounded refresh; unavailable keys never disable validation. Protect HTML, static resources, APIs, audio and reports. Internal container health checks may use a separate network-only endpoint. CSRF tokens and same-origin checks protect all mutation routes; secure same-site cookies complement them. Test forged/expired/wrong-issuer/wrong-audience JWTs, direct-origin access, CSRF rejection and valid media range playback. Operator supplies domain, Access organization/audience, allowed policy, tunnel credential and local paths in P5-H01; Luna writes the runbook and validation code.

Measure GPU residency and concurrent processing/search on the actual RTX 5090. Reserve at least 4 GiB of measured available VRAM; record total, used and headroom, including warm/cold loads. If exceeded, serialize heavy models and publish measured serving/processing profiles with wait states. Never remove an encoder to fit. Record stage real-time factor, peak VRAM/RAM, cold/warm latency, failures, retries and cost.

Back up SQLite consistently using its backup API, immutable artifacts and model locks; Qdrant snapshots are optional acceleration because vectors can rebuild it. Restore to fresh local volumes, validate hashes/generation counts, restart interrupted jobs and verify identity history/search results. Document recovery commands and observed timings. Keep audio, human annotations, identities, credentials, embeddings and sensitive traces out of the public repository; only synthetic fixtures and aggregate evidence belong there.

## D12 — Expansion and documentation

P5 releases the full frozen 20-episode feature set. P6 freezes 400 recent episodes, computes incremental resources excluding reusable artifacts, obtains human approval, runs a **40-episode total** canary, and gates the remaining bulk run on existing quality and drift checks. If a later 400 selection excludes some original 20, preserve their benchmark membership and explicitly report the union; never silently replace the benchmark. Prefer freezing the expansion relative to the initial 2026-09-10 cutoff to make a nested 20 → 40 → 400 cohort.

P7 freezes the remaining historical manifest, executes a stratified 20-episode voice-only canary across years/quality before human cross-year annotation and evaluation, obtains release, then indexes the remainder without ASR. P8 queues missing transcription by explicit operator priorities, then reviewed recurring-speaker appearances, then newest date/filename; estimates resources and obtains release before bounded batches (initially ten episodes). Each batch validates artifacts before updating searchable generations. Canary failures or regressions require remediation, never relaxed thresholds.

P5-05 produces the public README with actual feature status, synthetic/redacted screenshots, architecture, quick start, local/cloud configuration, measured results, resource requirements, recovery and limitations. It also creates `docs/architecture/{models,runtime,evaluation}/ARCHITECTURE-PROMPT.md`: three distinct diagram-generation prompts based on the implemented reviewed commit. Each specifies layout, visual hierarchy, exact labels/arrows, legend and synthetic example. Models covers every stage and encoder; runtime covers browser/Access/tunnel/app/worker/databases/inference; evaluation covers annotation/splits/experiments/reports/review/release. Solid arrows are default local paths, dashed arrows optional cloud paths. Do not portray planned capabilities as implemented.

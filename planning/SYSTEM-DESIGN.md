# System design

Revised 2026-09-28 for a single operator and fixed workstation. [README](README.md)
sets current scope and delivery order; [EVALUATION](EVALUATION.md) sets assessment
size and limits. D1–D12 remain stable issue references. Reuse implemented contracts
and artifacts; these sections describe intended behavior without claiming every
planned feature already exists.

## D1 — Packages and deployment boundary

Keep the typed Python 3.14 application, `uv`/`uv_build`, `src/zanzara_archive`,
FastAPI, Jinja, modest vanilla JavaScript, and the `zanzara` CLI. The existing ML
services use Python 3.14 and the shared CUDA runtime in `services/shared-base`;
reuse their packages, locks and Compose definitions. ML dependencies remain in
service environments, and ordinary application tests need no GPU model loading.

Use SQLite, one durable worker, dedicated Qdrant for voice vectors, and inference
services needed for the active stage. Serialize GPU work initially. The web app
runs on this host at `0.0.0.0:8000` for trusted LAN access; user links use
`http://complex.home.arpa:8000/`. Model-service host ports stay loopback-only where
needed by local CLI clients. Internet exposure is deferred under D11.

## D2 — Corpus and audio

[corpus-20.json](corpus-20.json) fixes the initial 20 episodes from 2026-07-01
through 2026-09-10, including `260910-lazanzara.opus`. Reuse recorded source
identities and media metadata. The archive at `/export/scratch/archive/zanzara`
is read-only and may be mounted as `/archive`; resolve registered relative paths
beneath it, rejecting traversal and escaping symlinks. Open sources when needed;
do not routinely rehash or reprobe the trusted archive.

Decode local temporary PCM with the existing FFmpeg path and model-specific
format. Preserve original time origin; silence removal must not shift persisted
times. Record relevant transform configuration. Media intervals are integer
milliseconds, half-open `[start_ms,end_ms)`, with
`0 <= start_ms < end_ms <= duration_ms`. Convert model-local offsets once and
preserve real timing. Missing or invalid timed words cannot serve production
consumers that require them. Canonical Opus files are never rewritten.

## D3 — Canonical data and publication

SQLite owns episodes, artifacts/runs/jobs, transcripts and turns, episode
speakers, exemplars, annotation revisions, identity decisions, and names. Keep
foreign keys, migrations, WAL, FTS5 and bounded busy timeouts. SQLite and job state
use local storage, not the NFS archive. Episode-speaker identity includes the
diarization artifact; `SPEAKER_00` is never a cross-episode person identifier.
New diarization marks affected identity mappings stale for human reconciliation.

Keep the implemented `/artifacts/<source_sha256>/<stage>/<stage_key>/` layout,
cheap canonical configuration fingerprints and hash-on-write metadata. Write a
complete temporary generation, fsync and atomically publish before recording the
SQLite reference. Reconcile complete orphan artifacts after interrupted work;
partial outputs never become searchable. Existing reader integrity checks stay
in place. This plan does not require replacing stable IDs or removing their
hash fields.

Qdrant is a rebuildable index. Retain original vectors and canonical metadata
locally; separate model/configuration generations and publish only complete
generations. Identity names/membership remain in SQLite. A new encoder does not
reuse another encoder's vectors merely because dimensions match.

## D4 — Worker and invalidation

```text
manifest -> decode -> ASR words -------------------> attribution -> chunks -> FTS5
                 \-> diarization turns/overlaps ---/
                              \-> clean exemplars -> ResNet -> voice index
                                                               -> candidates
                                                               -> human decisions
```

Voice indexing needs diarization and excerpts, independently of ASR or text
search. ASR changes invalidate attribution and text outputs; diarization changes
also invalidate affected excerpts/voice outputs. Encoder changes affect its own
vectors and retrieval generation. Identity/name edits update membership and
filters without rerunning acoustic inference.

Reuse durable job states, leases, heartbeats, fencing and bounded retries. Claim
and publish in short transactions; never hold a transaction during inference.
A stale worker cannot publish after its lease is reclaimed. Deterministic input
or access failures block with a useful reason; exhausted transient retries fail
with a recovery action. Serialize heavy model stages through the existing worker
and load only needed services. Add concurrency after actual queue waiting
justifies it. Ambiguous paid requests retain their reservation under D8.

## D5 — Models and adapters

| Role | Current path | Scope |
|---|---|---|
| Timed transcription | Parakeet v3 | Working production baseline |
| ASR comparison | Existing Parakeet, Whisper Large v3, Voxtral Mini 4B services | Same small reviewed P1R pilot |
| Diarization | Community-1 | Standard/exclusive turns and overlaps |
| Initial voice retrieval | ResNet293-LM | Clean exemplar vectors and candidates |
| Optional voice comparison | Existing ERes2Net and fine-tuned WavLM verification artifacts | Activate after useful ResNet misses |
| Optional text embeddings | Existing dense BGE-M3 service | Activate after useful lexical misses |
| Review aids | Existing Qwen/AudioSet AST paths | Drafts/condition assistance, never human truth |

Keep model revisions, preprocessing, precision, dimensions and runtime records
in the existing lock/fingerprints. Routine validation is structural. Verify
checkpoint bytes on acquisition/replacement, suspected corruption or recovery;
do not rescan all caches on ordinary tests or commits. Smoke affected services
after relevant model/runtime changes, reusing unchanged results for other models.
Human model terms/access remain human actions. A missing optional model does not
block an active baseline; a missing required model gets an explicit error.

Reuse typed contracts and internal HTTP adapters. Inputs are validated audio
bytes or constrained artifact references, never arbitrary URLs/filesystem paths.
Vectors must be finite, nonzero, dimension-correct and preserve input ordering.
Keep bounded requests/timeouts, readiness distinct from process liveness, model
metadata in responses, and typed failures. CPU fixtures check contracts; actual
GPU runs establish inference behavior. P1R hypotheses may be text-only, while
production timing requirements remain explicit.

## D6 — Transcript production

Reuse Parakeet's 300-second ownership windows with five seconds of context on
each side, clipped at episode boundaries. Convert local times to original
offsets and retain words whose midpoint belongs to the window's half-open
ownership interval. A boundary midpoint belongs to the later window. Do not
deduplicate repeated words by text alone. Record settings when changed and check
boundary word loss/duplication on relevant fixtures and audio.

Run Community-1 with episode-wide clustering; independently numbered chunks do
not define episode identities. Retain standard turns, exclusive turns and all
active IDs in overlap intervals. Attribute each word to the exclusive turn with
greatest temporal intersection, then the turn containing its midpoint, then a
stable speaker-ID tie-break. Leave nonintersecting words unassigned. Attach
standard-diarization overlap flags independently. Exports preserve canonical
word references/times and speaker identity.

P1R uses completed P1R-03D audio chunking and the 80-chunk pilot in E1. Human
listening creates text/speaker truth; timestamps do not define gold. Retain the
timed production baseline until any selected alternative has a supported timing
path. The competing P1R-03C approach is retired.

## D7 — Voice retrieval and identity

Subtract overlap and 250 ms around speaker transitions from eligible turns.
Select clean nonoverlapping excerpts, preferring 8–15 seconds, permitting 3–8
seconds when needed, and keeping at most ten per episode speaker. Use stable
quality/duration/start-time ordering and measured clipping/RMS metadata. Speakers
without three seconds of usable speech remain visible as `not_voice_searchable`.

Start with ResNet only. L2-normalize valid exemplars and their arithmetic-mean
centroid; reject zero-norm means. Retrieve up to 50 centroid candidates, excluding
the query episode by default. Rerank with exemplar cosine comparisons: sort pairs
by descending similarity and stable exemplar IDs, greedily accept unused
endpoints, and take the median matched score. Return matched excerpts, scores,
appearances and generation IDs. Scores rank candidates; they are not probabilities.

ERes2Net/WavLM, multi-model candidate unions and rank/calibrated fusion are
optional experiments after reviewed misses justify them. Keep separate model
spaces, evaluate the same reviewed queries and document the chosen active
configuration. A one-model index does not wait for all three models or calibration.

Only human-confirmed same-person decisions change global membership. Names are
manual. Store same/different/uncertain decisions, evidence references, reviewer,
time and superseded status. Before merging, transactionally reject all active
different-person contradictions across both clusters, including transitive ones.
Undo/split preserve decision history and recompute affected membership. A stale
revision returns conflict, never overwrites human work. Keep the meaningful
merge → contradiction refusal → undo → split regression.

## D8 — Shared inference, provider changes and money

Local inference is the active path. Shared-inference transcription extension and
paid provider comparisons are deferred; their absence does not block text search
or the local pilot. When activated, inspect current upstream/provider contracts,
reuse existing client conventions and pin the integrated revision. Requested
timing is not evidence of supported timing; never interpolate it to satisfy a
production contract.

Provider/model changes use separate generations by default. Reuse needs evidence
that weights, preprocessing, output ordering and retrieval behavior are
compatible; dimension equality alone is insufficient. No routine provider
equivalence audit is required while the local model is unchanged.

Paid work still needs explicit authorization and the existing US$10 total cap,
including probes/retries, capped credential and local reservation ledger.
Reserve a conservative upper cost before dispatch. Unknown pricing or insufficient
budget blocks calls. Timeout/unknown charges retain their reservation until
reconciled; no opaque unaccounted retries. Keep private payloads and credentials
out of public logs. This plan authorizes no paid calls or later expansion spend.

## D9 — Text retrieval

Build searchable chunks from existing attributed timed words. Break at episode
speaker changes, 200 words or 60 seconds; keep unassigned runs distinct, exact
word references, original offsets and overlap flags. Chunk IDs identify the
attribution artifact and word range. These text chunks are separate from P1R
benchmark chunks.

Ship SQLite FTS5 with `unicode61 remove_diacritics 0`, BM25 ranking and stable
ID tie-breaks. Use bound parameters and safe query parsing. Apply episode/date
filters during retrieval; global-speaker filters follow confirmed membership.
Return text, episode, word references and timestamped playback. Publish FTS/chunk
changes transactionally; missing ASR is an explicit state.

P3-03 first indexes the existing attributed episode. P5-06 then processes the
remaining frozen episodes through the timed Parakeet/Community-1/attribution
stages and indexes them in resumable batches, alongside search API and browser
work. Initial production does not wait for the separate ASR comparison; a later
model decision reprocesses only outputs actually affected.

Dense/hybrid retrieval is deferred until real saved queries justify it. If
activated, retain lexical fallback and keep vectors/generation/filter semantics
consistent. Existing BGE-M3 is the first candidate; do not truncate texts silently
at model limits. A simple reciprocal-rank merge can combine lexical and dense
results; compare against saved judgments before making it the default. No
semantic-mode API, embedding job or Qdrant dependency is required for initial
lexical search.

## D10 — Website and API

First deliver LAN search, episode/date filters, transcript and original-audio
playback. Then add archive-speaker candidates, side-by-side listening, human
identity confirmation/undo/split, and anonymous/global appearance lists. Retain
existing chunk-review tools, including `/p1r-03d`. Show missing transcripts and
unsearchable voices honestly. Basic job/error status suffices; a separate
evaluation dashboard is deferred.

Reuse existing routes and contract conventions; implement only endpoints needed
by these journeys. Media accepts registered episode IDs and bounded intervals,
with browser seeking support. Lists/results are bounded. Identity mutations use
expected revisions and return conflicts for stale or contradictory decisions.
Reject arbitrary file paths/remote media URLs. Maintain clear validation,
not-found, conflict and temporary-service errors and existing web protections.

Uploaded voice queries are deferred. If activated, bound streamed input to
25,000,000 bytes and decoded duration to 60 seconds; validate content, isolate
resource-bounded decoding and require one selected speaker with at least three
seconds of clean speech. Query data never enroll identities. Keep temporary
bytes/vectors outside archive collections, enforce ownership/revision checks,
expire all query content after one hour and let cancellation/expiry prevent
late publication. The active encoder suffices; uploads do not mandate an ensemble.

## D11 — Access and operational profiles

The current target is the trusted LAN web app on `0.0.0.0`, linked as
`http://complex.home.arpa:8000/`. Keep model/Qdrant access internal or host-loopback
as required. Retain existing authentication/CSRF/media protections; this scope
change does not request weakening implemented checks. Remote access and tunnel
setup are deferred until explicitly needed and authorized.

Before Internet exposure, enforce Cloudflare Access at the origin for HTML,
static assets, API, media and reports. Validate JWT signature/allowed algorithm,
issuer, audience, expiry and not-before with bounded rotating-key refresh; fail
closed on missing/invalid credentials. Protect mutations with CSRF/same-origin
checks. Verify direct-origin rejection and media playback. Operator-provided
credentials/policy are actual prerequisites for exposure, not for LAN work.

On the fixed RTX 5090, initially serialize model stages. Record useful elapsed
time, memory and failures for affected runs; do not require concurrent residency,
every-model audits or a fixed headroom target before delivery. Introduce measured
serving/concurrency profiles only when needed.

Back up SQLite consistently through its backup API together with immutable
artifacts and model/configuration records. Qdrant snapshots may accelerate
recovery; retained vectors must support rebuilding indexes. Demonstrate restore,
job restart, identity history and search once, and rerun after relevant recovery
changes. Full byte scans are for recovery/corruption diagnosis, not each code
change. Keep private audio, annotations, identities, embeddings, credentials and
sensitive traces outside public GitHub content.

## D12 — Expansion and documentation

Make the initial 20 useful before bulk expansion. P6 freezes 400 recent episodes
relative to the original 2026-09-10 cutoff, estimates incremental work and obtains
resource authorization. Use a nested 40-total canary before the remaining cohort;
preserve the original benchmark membership and account for all retained episodes.

P7 begins with a stratified historical voice-only canary and cross-year listening
before approved remaining indexing; ASR is not required for playable appearances.
P8 prioritizes missing transcription by operator priorities, reviewed recurrence,
then date/filename, with estimates and bounded batches, initially ten episodes.
Investigate canary failures before scaling; account for incomplete/unsearchable
items. One explicit approval covers its cohort/resources; larger scope or new
spend needs authorization. Phase labels require no separate release ceremony.

Keep practical documentation of actual features, quick start/LAN URL, selected
models, useful measurements and limitations, processing/recovery commands and
known failure states. Synthetic/redacted illustrations are optional. Architecture
prompt packages, release dashboards and repeated historical evidence packets are
not deliverables. Never present deferred capabilities as implemented.

# Zanzara Archive working plan

Revised 2026-09-28 for one operator on one fixed workstation. This plan and the
updated issue bodies supersede earlier phase-release, exhaustive provenance,
and mandatory ensemble/cloud requirements, including historical `plan.md` and
closed issues. GitHub issues track the work; `manifest.json` mirrors their scope
and direct dependencies. Phase labels organize work, not execution order.

## Deliver something useful first

| Order | Outcome | Existing issues |
|---|---|---|
| 1 | Build the FTS5 index, produce and index the initial 20 transcripts, then search and play original audio on the trusted LAN | P3-03, P5-06, P3-04, P4-01, P4-02 (#32, #51, #33, #38, #39) |
| 2, alongside product work | Finish a small comparison of the three implemented ASR candidates | P1R-10, P1R-11, P1R-H01, P1R-16, P1R-17 (#90–#92, #97–#98) |
| 3 | Find recurring anonymous voices, listen, confirm or undo identity links | P2-01, P2-02, P2-05–P2-09, P4-04 (#19, #20, #23–#27, #41) |
| 4 | Check search/voice quality and recoverability of the initial 20 | P3-06–P3-07, P4-06, P5-01, P5-03, P5-05 |
| 5 | Expand recent coverage, historical voice discovery, then remaining transcription | P6–P8, with bounded canaries and resource authorization |

One episode already has Parakeet/Community-1 attribution artifacts, and the
initial FTS5 index is empty. P3-03 indexes that real example; P5-06 then runs
the existing per-episode stages for the remaining frozen episodes in resumable
batches and fills the index alongside API/UI work. This uses the timed Parakeet
baseline without waiting for the separate ASR report. Voice indexing needs
diarization and clean exemplars, independently of ASR and text search.

## Architecture decisions

- Keep Python/uv, FastAPI/Jinja/vanilla JS, SQLite WAL/FTS5, one durable worker,
  isolated model services and Qdrant. Reuse working code.
- Start text retrieval with FTS5, episode/date filters and timestamped playback.
  Global-speaker filters follow identity membership. Add dense/hybrid search
  when real queries expose useful gaps.
- Start voice retrieval with ResNet293, clean excerpts, centroid candidates and
  exemplar reranking. Keep original vectors and separate model generations.
  Scores rank candidates; they are not probabilities or identity decisions.
  ERes2Net, WavLM and fusion remain optional measured comparisons.
- Use the frozen **80-chunk** pilot as the sampling pool. Human review is capped
  at a fixed **30-chunk** cohort: 15 per episode partition, each with ten
  representative and five difficult clips. Compare Parakeet, Whisper and
  Voxtral on those same reviewed clips; report the other 50 as unreviewed.
  Human listening remains required. Expand only with operator authorization.
  This pilot cannot establish archive-wide accuracy.
- Use completed P1R-03D chunking (#105); retire competing P1R-03C (#104).
  Qwen/AST are available aids; calibration refinement is not a product gate.
- Serialize GPU work initially and load the services needed for the current
  stage. Add concurrent serving when observed waiting warrants the work.
- Serve on the trusted LAN at `http://complex.home.arpa:8000/`, bound to
  `0.0.0.0`. Internet exposure is deferred; Cloudflare Access, origin enforcement
  and existing web protections remain prerequisites before exposure.

## Proportionate checks and provenance

Run focused checks for changed behavior once. Reuse passing results until their
code, inputs or configuration materially change. Documentation edits do not
require inference, the full test suite, checkpoint scans or corpus probes.
Smoke an affected inference service after relevant model/runtime changes;
do not smoke every model after unrelated changes.

Keep cheap stage/configuration fingerprints, model revisions, artifact IDs,
hashes computed during publication, and checks that prevent model mixing or
partial output. Do not migrate stable identifiers merely to remove hash fields.
Routine model-lock validation is structural; checkpoint byte verification is
explicit for acquisition, replacement, suspected corruption or recovery.
Do not repeatedly hash the trusted archive, model cache, reports or screenshots
as an implementation/review prerequisite. Existing artifact-reader integrity
checks remain; this revision does not claim every runtime checksum is removed.

Completion normally needs the commit, relevant command/result and a run ID or
artifact path, plus model/configuration/reference versions when material.
A rebuild warrants a changed-service smoke, not every historical benchmark.
Report actual defects or missing human truth; omit speculative evidence packets
and Project-metadata-only blockers.

Keep safe paths/media bounds, genuine timestamps, valid vectors, atomic
publication, resumability, backups, human identity confirmation/undo, privacy,
and paid-call limits. These protect actual data and work.

## Execution

Select eligible active work by numeric Order and direct unfinished prerequisites.
The manifest flags optional work `deferred: true` and retired work `retired: true`;
neither is auto-selected. Deferred issues stay open in Backlog and do not block
core parents. Remove or explicitly replace retired prerequisites; not-planned
closure never claims an implementation exists.

Continue across phase labels and batch closely related tasks when authorized.
Parents summarize core outcomes; no separate review invocation, exact-commit
PASS or `Release Pn` comment is mandatory. Review remains useful on request or
for material integration risk. Ordinary body/status changes do not require a
Project audit. Read back changed bodies/dependencies/fields once when updating
planning. Closed implementation records are not retroactively re-certified.

Human annotation, identity decisions and model terms remain human actions.
This revision authorizes no paid calls, public deployment or bulk expansion.
P6/P7/P8 retain concrete resource decisions before large runs; one approval
covers its stated scope without a repeated phase-boundary ceremony.

## Deferred work and return conditions

| Work | Return when |
|---|---|
| ERes2Net/WavLM and calibrated fusion | Reviewed ResNet misses justify another encoder |
| Dense/hybrid retrieval | Saved queries show useful paraphrases/topics are missed |
| Cloud comparison and upstream transcription extension | Local quality is inadequate and paid comparison is authorized |
| Uploaded voice queries | Archive-speaker search works and external samples are needed |
| Evaluation dashboard / assistance study | Existing reports and basic errors impede actual use |
| Cloudflare deployment | Remote access is needed |
| GPU concurrency tuning | Serial execution causes unacceptable waiting |

## Supporting contracts

[SYSTEM-DESIGN](SYSTEM-DESIGN.md) retains D1–D12 implementation details;
[EVALUATION](EVALUATION.md) defines smaller assessments and their limits.
[BACKLOG](BACKLOG.md), [manifest.json](manifest.json), and issue bodies carry
revised dependencies. [GITHUB-SETUP](GITHUB-SETUP.md) applies to structural changes,
not every coding task. [REVIEW-SKILL](REVIEW-SKILL.md) describes optional review.

Run `node planning/validate.mjs` when changing this package. It checks recorded
plan data, not archive bytes or model quality. Historical evidence remains in
Git and private run directories.

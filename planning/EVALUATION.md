# Evaluation

Revised 2026-09-28; supersedes earlier mandatory sample sizes, report packets
and phase-release ceremonies. E1–E7 remain stable references. Measure enough for
the next decision and disclose uncertainty. Fixtures establish code behavior;
human-reviewed real audio establishes quality.

## E1 — Golden reference and transcription split

Reuse completed P1R-03D chunking and existing annotation/scoring contracts.
The frozen 80-chunk pool has 60 representative and 20 difficult clips from
multiple episodes. Human review is capped at a fixed 30-clip subset: 15 per
development/held-out partition, each with ten representative and five difficult.
The partitions were assigned by episode before sampling. Report coverage gaps.

Select without ASR-score filtering. Include rapid turns, overlap, music and
degraded audio where present; manual tags suffice. Keep held-out episodes out of
prompt/threshold/model-setting tuning. Freeze membership once with a manifest
version/run ID. Expand only for an unresolved decision or important failure.

The operator listens and corrects text and speaker/condition truth for the 30
selected clips, recording reviewed or unresolved status for each. The other 50
remain unreviewed and outside human-scored denominators. Candidate consensus is not truth.
Unintelligible/unscorable content retains its reason and denominator. No complete
golden episode or manually timed words. Preserve existing review revisions and
human work when selecting the smaller sample.

## E2 — ASR, diarization, attribution and timing metrics

Run existing Parakeet, Whisper and Voxtral on the same reviewed cohort. Use existing Italian
WER/CER scoring; report representative/difficult samples separately with counts,
words, duration, failures and unsupported cases. Use overlap-aware ASR,
independent DER/JER and integrated attribution metrics where reviewed references
support them. Missing truth is a limitation, never invented zero error. Do not
relabel unsupported overlap as clean speech to improve scores.

Recommend an operating default using quality, failures, throughput and memory.
A close result permits retaining the working baseline and reporting uncertainty.
This pilot does not guarantee quality across thousands of episodes. Serious
observed word loss, bad offsets or systematic wrong-speaker attribution needs a
bounded fix. Historical P1 numeric thresholds are not current release gates.

Production still requires genuine timing for playback and word attribution.
A text-only benchmark winner is not automatically a production replacement:
retain the timed baseline until a supported timing path is implemented and
checked. Reference/normalizer changes require consistent rescoring of compared
outputs; unrelated documentation/code changes do not invalidate evidence.

## E3 — Voice labels, splits and scoring

Start with ResNet293 candidates and human comparison. Review recurring speakers
and difficult different-speaker pairs across episodes, including non-hosts where
present. Record same/different/uncertain, appearance IDs and excerpt times.
Product decisions and benchmark labels stay distinct; names remain manual.

Report Recall@1/5/10 and MRR for queries with a reviewed positive in another
episode, counts, host/non-host coverage, failures and exclusions. Exclude the
query episode by default; report no-match queries separately. With no recurring
non-host labels, permit exploratory browsing but make no quality or historical
bulk-readiness claim from host-only results.

Use uncalibrated similarities. The first interface needs no logistic calibration,
probability claim, large label quota or bootstrap suite. Compare additional
encoders on the same reviewed queries if activated; keep tuning/evaluation
identity groups separate. The former Recall@10 >=80% is an improvement target,
not a universal gate for a small pilot. Show false matches honestly for review.

Keep a regression for merge → contradiction refusal → undo → split. Only human
confirmation changes membership; zero contradictory fixture merges is required.
Historical expansion needs cross-year listening, not extrapolated host results.

## E4 — Text relevance and retrieval

Ship FTS5 first. Save about ten real Italian queries covering names, phrases,
date/episode filters and no-result behavior. The operator checks useful results
and playback offsets; distinguish transcription failures from retrieval failures.
This is a usability check, not a representative statistical benchmark.

If lexical misses justify dense retrieval, activate P3-02 and compare lexical,
dense/hybrid against the same generation, filters and frozen judgments. Expand
toward 40 queries only when comparison requires it. Report nDCG@10, Recall@10 and
MRR with denominators, no-result cases separately. Retain lexical fallback;
do not promote hybrid based only on queries used to tune it.

## E5 — Local/cloud comparisons and budget

Cloud comparison is deferred. If activated, use the same development subset and
normalizer, leaving held-out episodes untouched. Verify current provider support
and prices before paid calls. Existing US$10 total, capped credential, ledger
and explicit authorization still apply, including retries and uncertain charges.
Routine tests and this revision trigger no paid calls.

## E6 — Operational measurements and report format

Record run ID/path, code revision, relevant model/config/reference versions,
command, counts, results, failures, wall time and approximate peak memory.
Existing machine-readable outputs plus one concise summary suffice. No mandatory
HTML dashboard, every-report checksums or six-model smoke packet. Reuse recorded
source/model identity instead of scanning stored bytes.

Smoke only changed inference services. CPU checks target affected behavior and
shared contracts; broad changes/failures justify wider runs, not every closure.
Keep paid/GPU/human evidence distinct from fixtures. Private audio, transcripts,
identities and secrets stay outside public issues.

Serialize GPU jobs initially; measure concurrent residency when needed. Before
relying on durable reviewed data, demonstrate one SQLite/artifact restore and
retrieval rebuild from stored vectors. Reuse results until recovery changes.

## E7 — Gates and expansion checks

| Milestone | Sufficient evidence |
|---|---|
| P1 | Usable model services and timed transcript/diarization artifacts |
| P1R | Small reviewed pilot, honest comparison and operating choice |
| P2 | ResNet candidates, reviewed examples, identity reversal safeguards |
| P3 | FTS5 queries and useful results on reviewed examples |
| P4 | Search/playback and voice/compare/confirm/undo journeys on the LAN |
| P5 | Initial 20 accounted for, quality limitations, restart/restore, runbook |
| P6 | Resource approval, 40-total canary and new audio review, then 400 coverage |
| P7 | Historical canary, cross-year listening, resource approval, coverage |
| P8 | Approved remaining queue, bounded transcription, coverage/sample quality |

Parents summarize core deliverables. Deferred children do not block them. No
separate `Release Pn` ceremony or quality rerun at every commit. Reviews inspect
material risk/behavior and can be requested independently of milestones.

Expansion remains a resource/quality decision. Reuse unchanged reference results;
review new canary audio for year/channel/quality drift and investigate failures.
Report missing/unsearchable speakers and incomplete episodes. One explicit
approval covers its cohort/resources and bounded batches; new spend, exposure
or a larger cohort needs its own authorization.

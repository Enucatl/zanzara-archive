# P1R-01 migration record

<!-- zanzara-plan:P1R-01-migration -->

This record is the migration authority for the legacy P1 transcription-quality
contracts. It preserves the old issue bodies, implementation artifacts and
comments while making P1R the current quality methodology.

## Scope and authority

P1R supersedes only the P1 transcription-evaluation release semantics. The P1
services, production `TimedWord` artifacts, diarization outputs, attribution
pipeline, annotation editor and historical reports remain valid reusable
artifacts. P1 remains an operational/reusable-services phase; P1R owns
transcription-quality evidence and its release gate.

The authoritative order is:

1. P1R chunk contracts and benchmark issues (`P1R-02`, `P1R-11`, `P1R-H01`,
   `P1R-12`–`P1R-17`).
2. This migration record and the updated planning package.
3. The retained P1 issue bodies, reports and artifacts as explicitly marked
   historical evidence.

## Before and after

| Legacy clause | Replacement | Disposition |
|---|---|---|
| Complete `260910-lazanzara.opus` golden transcript as the transcription-quality reference | Frozen multi-episode chunk manifests with human audio-reviewed chunk references | P1R-10, P1R-11, P1R-H01 |
| Five contiguous 80/20 time blocks | Episode-level development/held-out partitions with no episode leakage | P1R-01, P1R-11, P1R-16 |
| At least 200 manually timed words and a timing-error gate | Optional immutable word/segment metadata; lexical, overlap-aware ASR, diarization and integrated reports are separate | P1R-02, P1R-12, P1R-13, P1R-14 |
| P1 E2 WER/DER/timing thresholds as the release gate | P1R benchmark report and current P1R release checklist; no inherited timing threshold | P1R-16, P1R-17 |
| Timestamp-capable ASR required to compare transcription quality | Text-only hypotheses are valid for ASR scoring; timing is required only by production attribution/playback consumers | P1R-02, P1R-04, P3-01 |
| P1-06/P1-07/P1-08 old evaluation path | Retain completed implementation and reports as historical, non-authoritative evidence | P1 issue history; P1R-01 |

`TimedWord`, real Parakeet word timing, standard/exclusive diarization and
production attribution are not removed. They remain required where the
production transcript/playback contract needs them, but they do not become
human gold or a release prerequisite for chunk-level text scoring.

## Live P1 review follow-ups

The following changes are authorized only after this replacement map was
recorded. Closing an issue as `not_planned` preserves its body and comments;
the replacement links are recorded in the close comment.

| Issue | Decision | P1R replacement |
|---|---|---|
| [#75](https://github.com/Enucatl/zanzara-archive/issues/75) | Superseded as a P1 release blocker; close `not_planned` | P1R-10, P1R-11, P1R-H01 |
| [#76](https://github.com/Enucatl/zanzara-archive/issues/76) | Superseded as a P1 release blocker; close `not_planned` | P1R-12, P1R-13, P1R-14, P1R-16, P1R-17 |
| [#77](https://github.com/Enucatl/zanzara-archive/issues/77) | Remains active; model/runtime provenance is not a transcription-methodology clause | P1 operational phase |
| [#78](https://github.com/Enucatl/zanzara-archive/issues/78) | Remains active; live GitHub metadata reconciliation is not a transcription-methodology clause | P1 operational phase |
| [#79](https://github.com/Enucatl/zanzara-archive/issues/79) | Superseded as a P1 release blocker; close `not_planned` | P1R-16, P1R-17 |

Completed P1 children (#9–#17, #73 and #74) remain closed with their original
evidence. No issue body, comment, artifact or historical report is deleted.

## Downstream dependency map

| Consumer | Before | After | Reason |
|---|---|---|---|
| P2 voice retrieval and identity | P1 | P1 | P2 explicitly runs without ASR/transcription evidence and still needs P1's operational voice/diarization services. |
| P3-07 cloud/local ASR report | P2, P3-05, P3-06 | P1R, P2, P3-05, P3-06 | Its ASR comparison must use the authoritative P1R methodology. |
| P5-06 20-episode release packet | P4, P5-05 | P1R, P4, P5-05 | Its ASR/diarization quality evidence must use current P1R reports. |
| P4/P8 production transcript consumers | P3/P7 phase chain | Unchanged phase chain | They consume production artifacts and inherit the applicable P1R quality evidence through their release packets; they do not need a duplicate native edge. |

No P1R edge is added to P2 merely to force a transcript gate onto voice-only
work. P2's retained P1 edge is operational, not an obsolete P1 transcription
quality gate.

## Exact legacy clauses superseded

The following authoritative clauses are now historical/non-authoritative:

- E1's complete-golden-episode, five-block, 200-manually-timed-word and
  timing-reference requirements.
- E2's timing-error summaries and P1 release thresholds as requirements for
  the transcription benchmark.
- E7's old P1 row requiring the human golden reference, fixed split and real
  baseline for P1 release.
- P1 review follow-ups #75, #76 and #79, after their P1R replacements were
  linked and their bodies/comments preserved.

The frozen corpus's golden filename remains authoritative corpus metadata. It
is not, by itself, a transcription-quality gold requirement.

## Read-back evidence

The final issue/project read-back for this migration is recorded on P1R-01's
completion comment. It includes the three closed-not-planned follow-ups, the
two retained P1 follow-ups, P2/P3-07/P5-06 native blockers, Project statuses,
the planning commit and the remote SHA.

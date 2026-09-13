<!-- zanzara-plan:P1R -->
# [P1R] Chunk-based multi-model transcription benchmark

- Stable ID: P1R
- Executor: Human
- Kind: Phase
- Parent: none
- Blocked by: P0, P1R-01, P1R-02, P1R-03, P1R-04, P1R-05, P1R-06, P1R-07, P1R-08, P1R-08A, P1R-09, P1R-10, P1R-11, P1R-H01, P1R-12, P1R-13, P1R-14, P1R-16, P1R-17

## Outcome and requirement

Replace the word-timestamp-centric P1 evaluation gate with a frozen, multi-episode, model-independent audio-chunk benchmark. Compare Parakeet, Whisper Large v3 and Voxtral Mini 4B against human-reviewed truth; report lexical, diarization and integrated speaker-attributed quality separately. Community-1 owns speaker/overlap metadata; local AudioSet AST separately owns acoustic/music evidence.

## Design and interfaces

Chunks are adaptive 8–18 second intervals (30 second hard maximum), derived from acoustic/diarization signals rather than ASR output. Human listening, not candidates or Qwen, establishes truth. Word timestamps remain optional immutable model metadata. See [P1R-01](P1R-01.md) for migration and [P1R-02](P1R-02.md) for canonical contracts.

## Bounded steps

1. Complete the listed release-blocking children and preserve private reference artifacts.
2. Obtain a current Sol review of the integrated commit and evidence.
3. Request explicit `Release P1R at <reviewed-commit>` from the user before closing this parent.

## Inputs, outputs and failure behavior

Inputs are released P0 capabilities, completed P1 reusable artifacts where valid, and the P1R child evidence. Output is a human release record tied to a current Sol PASS. Missing gold, model evidence, slice denominators, or an explicit release keeps this parent open.

## Exclusions

No archive-wide processing, word-boundary gold, automatic gold promotion, cloud ASR spend, source separation, identity clustering, or automatic phase closure.

## Acceptance checklist

- [ ] Legacy P1 transcription-evaluation semantics are formally superseded without deleting history.
- [ ] Frozen representative and stress manifests, reviewed chunk references, and all three local hypotheses exist.
- [ ] WER/CER, overlap-aware ASR, diarization, and integrated attribution reports are reproducible and separately reported.
- [ ] A current Sol PASS and the user's explicit release identify the reviewed commit.

## Verification commands

Run the focused CPU checks named by children, `node planning/validate.mjs`, and the private RTX 5090 evidence commands from P1R-05, P1R-06 and P1R-16. Invoke `$review-phase <this-issue-number>` under Sol; no command replaces human release.

## Evidence and documentation

Record manifest/reference/model/configuration/scoring hashes, sanitized aggregate reports, hardware/runtime evidence, Sol report and release comment. Keep audio and detailed references private.

## Stop conditions and completion rule

Stop if a release-blocking child, independent gold, real GPU evidence, current review, or explicit user release is missing. P1R-15 is an optional ergonomics study and does not block release. Completion requires every checked release criterion and user release.

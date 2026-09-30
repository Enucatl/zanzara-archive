We have a working archive application and a completed initial processing cohort. We’re ready for a bounded expansion study; the evidence is still too limited to commit confidently to archive-wide processing.

I checked the current plans, recorded evaluations, local database and active voice index. Some older reports understate today’s voice coverage.

 Area                Where we are
━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Transcription       All initial 20 episodes / 33.6 audio hours processed, with timed transcripts and speaker attribution
──────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Text search         15,563 searchable passages, with episode/date filters and original-audio playback
──────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Voice embeddings    ResNet293 artifacts for all 20 episodes; active index has 510 centroids and 2,826 excerpts
──────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Identity review     Listen, compare, confirm, name, undo and split are implemented; only two human pair decisions recorded
──────────────────  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Operations          Durable jobs, retries, atomic artifacts and serial GPU processing; an isolated backup/restore and vector-index rebuild
                     succeeded

The largest remaining questions concern the usefulness and quality of the outputs.

1. Is transcription good enough for the searches we actually want?

The completed comparison supports keeping Parakeet: it supplies word timestamps and is faster than the alternatives, with essentially tied transcription accuracy. But its reviewed normalized word error rate was 35.8% overall, 26.0% on representative clips, and 50.8% on difficult clips. All three models exceeded 56% on the heavy-overlap subset.

That sample contains only 30 clips—about 7.4 minutes—and cannot establish whole-archive accuracy. We should check names, caller speech, omitted turns and playback offsets in full-episode outputs, then save roughly ten practical searches and listen to the results. This establishes whether imperfect transcripts still deliver useful discovery. ASR report (docs/asr-pilot.md)

2. Are the speaker excerpts clean, and do voice candidates find recurring callers?

This is the biggest evidence gap before large-scale voice embedding. Diarization and embedding code work, but reviewed speaker boundaries are missing, so diarization and attribution accuracy remain unmeasured. An embedding can faithfully represent an excerpt containing mixed speakers or background speech.

We need a small human-reviewed set containing recurring non-hosts, difficult different-person pairs and voices with no known match. Listen for speaker merges, splits and contaminated excerpts; then measure Recall@1/5/10 and MRR with the query episode excluded. Two confirmed pairs cannot establish this. Additional encoders should follow observed ResNet failures. Evaluation guidance (planning/EVALUATION.md)

3. Which kind of embedding is worth scaling?

Voice embeddings already support candidate discovery. Dense text embeddings remain deferred: text retrieval currently uses FTS5.

For semantic text search, first record real queries where lexical search misses useful paraphrases or topics. Compare dense/hybrid retrieval on those same judged examples. Embeddings cannot recover words the transcription omitted, so distinguish transcription failures from retrieval failures.

4. Does quality survive older audio and recording conditions?

Recent episodes don’t establish performance across years, telephone channels, compression changes or aging voices. The plan calls for a stratified 20-episode historical voice canary, with cross-year listening. Historical voice discovery can run independently of transcription, allowing us to prioritize later transcription by usefulness. Historical canary (planning/issues/P7-01.md)

5. What does the complete pipeline actually cost?

The frozen 400-episode cohort contains 635.2 audio hours. For the additional 380 episodes, the current estimate is approximately:

42 hours for ASR, diarization and attribution.
4 GB for those artifacts and the text index, before backups and temporary files.

Those estimates exclude voice stages and extrapolate from a warm baseline. One initial episode needed shorter ASR windows after timeouts. The next canary should measure complete processing time, peak memory, voice storage, retries and speakers left unsearchable. Resource estimate (planning/P6-01-estimates.md)

My recommendation is to use the existing 20 episodes for the practical search and voice-quality checks, address observed no-speech and long-window failures, then authorize the planned 40-total recent canary. Its new audio review and measured resource use should inform the 400-episode decision. A historical canary should precede archive-wide expansion.

Recovery already has useful evidence; the next investment should be human listening and measured usefulness. No new processing or repository changes were made for this overview.

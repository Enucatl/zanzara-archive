# P1R speaker-condition metadata

P1R-08 runs Community-1 once over an entire episode and intersects its
episode-wide standard and exclusive turn streams with each frozen benchmark
chunk. It never invokes diarization on a chunk and never reads ASR output.

The machine seed is `p1r-speaker-condition-v1`. Its frozen labeling policy is:

- `single_speaker`: one standard-track speaker and no overlap;
- `multi_speaker_no_overlap`: at least two standard-track speakers and zero
  simultaneous speech;
- `partial_overlap`: overlap fraction greater than zero and below `0.5`;
- `heavy_overlap`: overlap fraction at least `0.5`;
- `uncertain`: no standard speech or missing episode-wide diarization.

`speech_ms` is the union of standard-track speech inside the chunk. `overlap_ms`
is the duration whose active standard-track speaker set has at least two IDs.
`overlap_fraction` is `overlap_ms / speech_ms`, and is `null` when
`speech_ms == 0`. `max_simultaneous_speakers` counts distinct active standard
tracks at the busiest interval. All intervals remain integer-millisecond,
source-relative half-open ranges.

The chunk condition persists clipped standard streams, clipped exclusive
streams, overlap intervals, the versioned `SpeakerConditionMetadata` machine
seed, and optional `SpeakerConditionCorrection`. A correction records an
identified reviewer, timestamp, reason and seed version; it does not replace
or mutate the machine seed. Missing diarization creates an explicit
`unknown`/`uncertain` seed with empty speaker IDs rather than inferred labels.

The derivation fingerprint is the Community-1 model fingerprint when available;
synthetic/contract inputs use a deterministic hash of the episode-wide turn
streams. Public tests use synthetic intervals only. Real archive diarization,
GPU inference and human corrections remain private acceptance work.

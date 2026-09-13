# P1R Community-1 chunk segmentation

`segment_community1_chunks` in `zanzara_archive.chunking` builds source-relative
benchmark intervals from Community-1 standard diarization. The input may be a
`DiarizationResult` or an explicit `standard_turns` sequence with its
`diarization_artifact_id`:

```python
from zanzara_archive.chunking import segment_chunks_with_metadata

result = segment_chunks_with_metadata(
    episode_id,
    source_sha256,
    duration_ms,
    algorithm="community1-adaptive-v1",
    standard_turns=diarization.standard_turns,
    diarization_artifact_id=diarization.artifact_id,
)
```

`Community1AdaptiveConfig` freezes the 8, 12, 18 and 30 second duration
limits plus the 600 ms strong-gap, 250 ms short-gap, ±250 ms speaker-change,
250 ms overlap-margin and 4 second minimum values. The configuration hash and
artifact ID are retained on every generated `AudioChunk`; the result also
contains deterministic boundary diagnostics. ASR, transcript text, exclusive
diarization and an additional VAD are not inputs to this path.

The existing `segment_chunks` API remains available for legacy P1R manifests.
Supplying Community-1 standard turns or `algorithm="community1-adaptive-v1"`
selects the new policy and rejects an artifact without `standard_turns`.

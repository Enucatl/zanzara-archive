# Attributed transcripts

`zanzara process --stage attribution` consumes the complete P1-02 ASR and
P1-03 diarization artifacts for one frozen episode. By default it resolves the
locked prerequisite generation keys; `--asr-artifact` and
`--diarization-artifact` select explicit complete artifact directories for
fixture or recovery work. Both inputs must carry the episode source hash and
the model fingerprint recorded in their payloads.

Attribution uses only exclusive turns. For each timed ASR word it chooses the
turn with the greatest positive half-open interval intersection. Equal
intersections prefer a turn containing the word midpoint, then the smallest
stable local speaker ID. A word with no intersection keeps `speaker_id: null`.
Overlap is calculated separately from standard turns and is exposed as the
word's `overlap` flag; the complete standard turns, exclusive turns and
overlap intervals remain in the structured output.

The command publishes an immutable `attribution` artifact containing:

- `attributed.json`: D10 success data with source/model/input provenance,
  exact word IDs and millisecond offsets, both diarization views, and the
  original-episode time origin.
- `transcript.txt`: one timestamped labelled line per word.
- `transcript.srt` and `transcript.vtt`: one cue per word, preserving the
  original interval. Unassigned words use `UNASSIGNED`; standard-turn overlap
  is marked `OVERLAP`.

The input ASR and diarization artifacts are checksummed before reading and are
never rewritten. Validation failures occur before publication, so no partial
attribution artifact is exposed. The output offsets are relative to the
original episode and can be passed directly to the later media playback route.

Example:

```bash
uv run zanzara process \
  --manifest planning/corpus-20.json \
  --episode 260910-lazanzara.opus \
  --stage attribution
```

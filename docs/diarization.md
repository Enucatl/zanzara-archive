# Community-1 diarization

The local Community-1 service runs one episode-wide clustering pass and exposes
`POST /v1/diarize`. The request uses the same bounded JSON envelope as the other
local model services and carries the source audio bytes. The service returns
both `standard_turns` and `exclusive_turns`, plus the locked model
repository/revision. It decodes compressed input through its local FFmpeg
runtime into mono 16 kHz model audio. It does not depend on ASR or speaker
identity state.

The application adapter converts service seconds to integer millisecond,
half-open offsets exactly once. It derives overlap intervals from every active
standard turn, retaining all active local speaker IDs. Local IDs such as
`SPEAKER_00` are scoped to the episode artifact and are never global identities.

The processing command publishes an immutable artifact under the usual hashed
artifact layout:

```bash
uv run zanzara process \
  --manifest planning/corpus-20.json \
  --episode 260910-lazanzara.opus \
  --stage diarization
```

The artifact contains `diarization.json` (canonical typed output),
`standard.rttm` and `exclusive.rttm` (evaluation-only seconds views), and the
private raw service response. Its provenance records the source, full-episode
decode configuration, model fingerprint, episode-wide scope and all output
counts. RTTM rendering never changes canonical millisecond offsets.

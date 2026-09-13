# P1R acoustic-condition metadata

P1R-08A runs the locked `MIT/ast-finetuned-audioset-10-10-0.4593` AudioSet
AST checkpoint locally on the same source-relative chunk interval sent to the
ASR services. The application owns deterministic windows and condition
mapping; the AST service owns only AudioSet event probabilities. It never
infers speaker count or overlap and never reads ASR output.

The model is pinned to revision
`f826b80d28226b62986cc218e5cec390b1096902`, its `config.json:id2label` map is
retained by SHA-256, and ten-second mono 16 kHz windows are zero-padded only
when the final window is shorter. Window probabilities are mean-aggregated by
AudioSet class. Each relevant class ID/name/probability is retained both per
window and in the aggregate seed.

`p1r-acoustic-thresholds-v1` maps music probabilities to `none`, `background`,
`dominant`, or `uncertain` using development-only frozen thresholds. Speech
presence and non-speech activity are separate fields. Clipping fraction, peak
dBFS, RMS dBFS and silence fraction are deterministic signal measurements;
`audio_quality` is a `clean`/`degraded`/`uncertain` machine seed from those
measurements, not an AST claim.

The machine seed is immutable inside the chunk payload. Human `music_level` and
`audio_quality` corrections are append-only SQLite records and are exposed by
the local condition API/UI without replacing raw probabilities or the seed.
Speaker and overlap metadata remains owned exclusively by P1R-08.

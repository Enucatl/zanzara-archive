# ASR pilot and operating choice

The 29 September 2026 P1R-16 comparison keeps **Parakeet TDT 0.6B v3** as the
production ASR default. Its normalized word error rate was effectively tied
with Whisper large-v3 and Voxtral Mini 4B, while it was faster, used less GPU
memory, and supplied the genuine word timestamps required by playback and
speaker attribution. No production model switch is needed for P1R-17.

## Reviewed cohort and results

The fixed cohort contains 30 of the 80 frozen pilot chunks: 15 development and
15 held-out, with 20 representative and 10 difficult. It spans 446.7 seconds.
The other 50 chunks have no human review and are excluded from every quality
denominator. One reviewed, music-dominated chunk has **no words**; the other
29 contain 1,203 normalized reference words. Each model attempted all 30
identical mono 16 kHz excerpts. No result was substituted for a failed model.

| Model | Hypotheses / attempts | Normalized WER | Normalized CER | Representative WER | Difficult WER | Request time / audio | Startup smoke peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Parakeet | 29 / 30 | 431 / 1,203 = 35.8% | 1,838 / 6,725 = 27.3% | 189 / 727 = 26.0% | 242 / 476 = 50.8% | 8.9 / 446.7 s (50.4×) | 1,270 MiB |
| Whisper | 30 / 30 | 430 / 1,203 = 35.7% | 1,902 / 6,725 = 28.3% | 173 / 727 = 23.8% | 257 / 476 = 54.0% | 31.7 / 446.7 s (14.1×) | 4,421 MiB |
| Voxtral | 29 / 30 | 429 / 1,203 = 35.7% | 1,777 / 6,725 = 26.4% | 191 / 727 = 26.3% | 238 / 476 = 50.0% | 128.8 / 446.7 s (3.5×) | 8,649 MiB |

All 90 dispatches reached a terminal state. Parakeet and Voxtral each failed
on the same no-words chunk because their decoder responses could not satisfy
the nonempty hypothesis contract. Whisper returned **222 words** on that
chunk. Its zero-word reference makes WER/CER inapplicable there, so the
hallucination is disclosed separately. The 29 scorable chunks give all three
models the same 1,203-word denominator. The representative score uses 727
words from 19 scorable chunks; the difficult score uses 476 words from 10.

The overlap-aware ORC-WER equivalent is 35.8% for Parakeet, 30.8% for Whisper,
and 32.9% for Voxtral on the 947 reference words with usable speaker text
streams. It does not measure diarization. Ten human-labeled heavy-overlap
chunks alone have 504 words and normalized WER above 56% for every model.
Reviewed word and speaker-boundary timing is unavailable, so DER/JER and
integrated attribution scores are not reported.
The pilot does not justify an archive-wide accuracy claim. Request time sums
the 30 persisted dispatch times, including service requests and publication;
it excludes service startup and source decoding. VRAM is the current service
startup smoke peak on the RTX 5090, an approximate workload proxy rather than
a measured batch maximum.

## Operating settings and evidence

Keep the locked `nvidia/parakeet-tdt-0.6b-v3` revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` and the existing Parakeet
production stage. It decodes mono 16 kHz audio with bfloat16 inference and
native decoder word spans; the episode stage uses windows of at most 300,000 ms
with 5,000 ms context. The service is on host loopback port `18080` by default.
Whisper (`06f233fe06e710322aca913c1bc4249a0d71fce1`) and Voxtral
(`2769294da9567371363522aac9bbcfdd19447add`) remain text-only benchmark
options. Their current outputs cannot provide production word playback or
attribution timing. Exact decoding settings and model fingerprints are in
[`inference.py`](../src/zanzara_archive/inference.py) and
[`models.lock.json`](../models.lock.json).

On this workstation, start only the selected service, then run the existing
episode stage with a filename from the frozen corpus manifest:

```bash
env -u COMPOSE_ENV_FILES docker compose --profile processing up -d --no-deps --wait parakeet
uv run zanzara process --stage asr --manifest "$MANIFEST" --episode "$EPISODE"
```

`MANIFEST` is the local frozen corpus manifest path; `EPISODE` is its relative
audio filename. The stage writes private timed artifacts through the existing
artifact publisher.

The private local run is `.git/zanzara-evidence/P1R-16/report-v1/`; its
`metrics.json` contains sanitized aggregates and its `details.json` remains
private. The immutable cohort is
`.git/zanzara-evidence/P1R-11/review-cohort-30.json`, under parent manifest
`p1r11-4b3205b1adb139dad9c1a14f`. Model dispatches and the 30 current
human reference revisions are in the local SQLite state. The report uses
`p1r-asr-v1` scoring and `it-v1` normalization at code revision `605e283`.
The focused orchestration tests passed (5 tests), and the GitHub lint/format
pipeline passed for that revision. This comparison does not test long episode
windows or solve the no-speech response contract; investigate that case before
relying on an empty music-only production window.

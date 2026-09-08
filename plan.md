Below is the draft I’d use as the working technical design. I’ve made the **three-embedding-model ensemble** a first-class part of the architecture rather than an optional experiment.

# Podcast Archive Transcription, Diarization, and Cross-Episode Speaker Identification

## 1. Objective

Build a local-first processing system for a large archive of Italian-language podcast episodes.

The archive consists of approximately:

* **4,000 episodes**
* approximately **1.5 hours per episode**
* approximately **6,000 total audio hours**
* mono Opus source files
* usually multiple speakers
* recurring hosts and guests
* occasional simultaneous/overlapping speech
* recordings spanning many years and potentially varying significantly in audio quality

An initial high-value subset consists of approximately:

* **400 recent episodes**
* approximately **600 audio hours**

The system should provide two complementary capabilities:

1. **High-quality searchable transcripts**, with words attributed to individual speakers.
2. **Cross-episode speaker discovery**, allowing an unidentified speaker in one episode to be matched against appearances in other episodes, including very old and forgotten episodes.

The second capability is particularly important. The system should not require that a speaker's identity be known in advance. It should be capable of discovering recurring anonymous speakers and assigning names later.

---

# 2. Design principles

The processing system should keep the following tasks logically separate:

1. audio preparation
2. automatic speech recognition
3. speaker diarization
4. overlap detection
5. transcript/speaker alignment
6. extraction of clean speaker exemplars
7. speaker embedding
8. cross-episode speaker retrieval
9. global speaker clustering
10. human identity assignment

This separation is important because improvements to one component should not require recomputing every other component.

For example:

* a future ASR model can replace Parakeet without rebuilding speaker embeddings;
* a future speaker-recognition model can be added without retranscribing the archive;
* diarization can be reprocessed independently;
* global identity assignments can be corrected without touching the source analysis.

The system should retain rich intermediate outputs rather than producing only final `.txt` transcripts.

---

# 3. Proposed high-level architecture

```text
                         episode.opus
                              │
                              ▼
                       FFmpeg decode
                              │
                    normalized PCM audio
                              │
             ┌────────────────┴─────────────────┐
             │                                  │
             ▼                                  ▼
        ASR pipeline                     diarization pipeline
      NVIDIA Parakeet                 pyannote Community-1
             │                                  │
       words + timestamps             speaker turns + overlap
             │                                  │
             └────────────────┬─────────────────┘
                              │
                              ▼
                 speaker-attributed transcript
                              │
                              ▼
                   clean-segment selection
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ResNet293-LM       ERes2Net family      WavLM-based model
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                         embeddings
                              │
                              ▼
                            Qdrant
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          nearest-neighbor           speaker graph /
             retrieval               identity clustering
                  │                       │
                  └───────────┬───────────┘
                              ▼
                      global speaker IDs
                              │
                              ▼
                    optional human names
```

---

# 4. Audio preparation

## Input

Original mono Opus files should remain the canonical archival source.

They should not be permanently converted simply for convenience.

For processing, FFmpeg should decode each source into a representation suitable for the model being executed.

A common working representation will be:

* mono
* 16 kHz
* PCM float or PCM16

Example conceptual pipeline:

```text
episode.opus
    ↓
FFmpeg
    ↓
16 kHz mono waveform
```

Temporary WAV files should not necessarily be retained permanently. Where practical, decoding can occur to temporary storage or through pipes.

The original Opus files should always remain untouched.

---

# 5. Automatic speech recognition

## Primary model

**NVIDIA Parakeet TDT 0.6B v3**

Parakeet v3 is the proposed production baseline because the workload is offline, extremely large, and Italian-language transcription is supported.

The model is small enough that an RTX 5090 should have abundant inference capacity, making local processing economically attractive.

The main required ASR outputs are:

```text
word
start_time
end_time
confidence, if available
```

The ASR stage does **not** need to determine persistent speaker identity.

Its responsibility is simply:

> What appears to have been said, and when?

The exact transcription model should remain replaceable.

### Initial ASR benchmark

Before committing the entire archive, approximately 3–5 representative hours should be compared against several cloud frontier models.

Suggested benchmark:

* Parakeet TDT 0.6B v3
* ElevenLabs Scribe v2
* Mistral Voxtral Transcribe
* Microsoft MAI-Transcribe-2

The benchmark should concentrate on actual podcast characteristics:

* natural Italian conversation
* regional accents
* rapid exchanges
* interruptions
* proper names
* English terms
* numbers and dates
* degraded older audio
* multiple speakers
* background music or effects

If Parakeet is sufficiently close to the cloud models, local Parakeet processing should become the default.

---

# 6. Speaker diarization

## Primary model

**pyannote Community-1**

Community-1 is proposed as the primary local diarization system.

It accepts mono 16-kHz speech and supports completely local inference after model acquisition. It provides both ordinary diarization and an **exclusive diarization** representation designed to simplify reconciliation with speech-recognition timestamps. ([Hugging Face][1])

Two representations should be retained.

### Standard diarization

This preserves genuine overlaps:

```text
00:10.000 – 00:18.500  SPEAKER_00
00:17.300 – 00:22.100  SPEAKER_01
```

Therefore:

```text
17.300 – 18.500
```

is recognized as simultaneous speech.

### Exclusive diarization

This assigns a single dominant speaker to each interval and is useful for attributing ASR words to speakers.

The two outputs serve different purposes:

```text
standard diarization
    → acoustic truth / overlap analysis

exclusive diarization
    → transcript speaker attribution
```

Community-1 is explicitly designed around this reconciliation workflow. ([pyannote.ai][2])

---

# 7. Overlapping speech

Overlapping speech should be represented explicitly rather than silently discarded.

Example:

```json
{
  "start": 517.30,
  "end": 518.50,
  "speakers": [
    "episode_speaker_00",
    "episode_speaker_01"
  ],
  "overlap": true
}
```

The transcript may still contain only one successfully decoded utterance during such a region. This is acceptable as long as the system knows the audio itself contains overlapping speakers.

Overlap regions are particularly important for speaker embeddings.

## Rule

**Speech containing detected overlap must not normally be used to construct speaker-identity embeddings.**

A mixed waveform:

```text
speaker A + speaker B
```

can produce an embedding that represents neither speaker cleanly and can contaminate the cross-episode identity index.

The local Community-1 configuration itself uses overlap exclusion for its speaker embedding stage, reinforcing this principle. ([Hugging Face][3])

---

# 8. Speaker hierarchy

Speaker identity should be represented at several distinct levels.

## 8.1 Turn

A single continuous interval of speech.

```text
episode 1842
13:31.4 – 13:46.8
speaker_02
```

## 8.2 Episode speaker

All speech believed to belong to one speaker within an individual episode.

```text
episode_1842:speaker_02
```

This identity is local to one recording.

It must never imply that:

```text
episode_1842:speaker_02
```

and:

```text
episode_1843:speaker_02
```

are the same person.

## 8.3 Global speaker

A persistent voice identity inferred across recordings.

Example:

```text
global_speaker_00172
```

This may link:

```text
episode_3821:speaker_02
episode_2911:speaker_01
episode_0482:speaker_04
```

## 8.4 Human identity

A manually or externally established real-world identity.

```text
global_speaker_00172
        ↓
"Mario Rossi"
```

This final mapping may be absent indefinitely.

Therefore the system can discover recurrence without knowing a name.

---

# 9. Clean speaker exemplar extraction

Speaker embeddings should not be generated indiscriminately from every diarized segment.

For each episode-speaker, candidate excerpts should be ranked according to quality.

Useful criteria include:

* no overlapping speaker
* sufficiently long continuous speech
* strong diarization confidence where available
* little music
* little background noise
* minimal clipping
* limited reverberation
* no obvious speaker transition close to the segment boundary

Initial duration policy:

```text
< 2 seconds       reject
3–8 seconds       acceptable
8–20 seconds      preferred
> 30 seconds      normally split
```

A useful initial target is approximately:

**5–10 independent exemplars per episode-speaker**

with approximately:

**8–15 seconds per exemplar**

where enough clean speech exists.

A guest speaking for only a short period may naturally produce fewer samples.

The actual audio segments should remain reproducible from timestamps rather than requiring thousands of permanently exported WAV files.

---

# 10. Three-model speaker embedding ensemble

Speaker identity will intentionally use **three independent embedding models**.

The objective is not simply to increase average benchmark accuracy.

The objective is to reduce catastrophic **false identity merges** by obtaining agreement from models with different architectures and learned representations.

## Model A — WeSpeaker ResNet293-LM

Proposed model:

**Wespeaker/wespeaker-voxceleb-resnet293-LM**

This is a dedicated speaker-verification model trained on VoxCeleb2.

It produces 256-dimensional speaker embeddings and supports CUDA inference.

The published model card reports approximately 0.447% EER on VoxCeleb1-O-clean when using its large-margin and AS-Norm recipe. ([Hugging Face][4])

Role:

> robust conventional speaker-recognition embedding and primary retrieval representation

Advantages:

* mature
* relatively lightweight
* explicitly optimized for speaker verification
* straightforward embedding extraction
* good operational tooling
* strong benchmark performance

---

# 11. Model B — ERes2Net family

The second model should come from the modern ERes2Net/3D-Speaker family.

The exact released checkpoint should be selected after a small empirical benchmark, with preference toward a strong pretrained checkpoint that generalizes well to European conversational audio.

Likely candidates include:

```text
ERes2Net-large
ERes2NetV2
```

Role:

> independent convolutional/multiscale speaker representation

The important objective is diversity relative to ResNet293 rather than simply selecting another nearly identical encoder.

This second model should be independently calibrated on the podcast archive.

---

# 12. Model C — WavLM-based speaker representation

The third model should be based on **WavLM Large** or an equivalent high-capacity self-supervised speech encoder with a speaker-verification head.

Role:

> high-capacity second-stage speaker verification and robustness across recording conditions

This model is expected to be substantially heavier than ResNet293 or ERes2Net.

That is acceptable because it does not need to process the entire 6,000-hour archive continuously.

It will process only selected clean speaker excerpts.

The high-capacity model can therefore be used either:

1. for every exemplar, or
2. as a second-stage verifier for uncertain candidate matches.

The initial implementation should benchmark both strategies.

---

# 13. Why three embeddings?

A single embedding model may produce a false high similarity due to:

* similar pitch
* similar accent
* microphone characteristics
* room acoustics
* compression artifacts
* speaking style
* insufficient sample duration

If three different representations independently conclude that two voices are similar, the evidence is substantially stronger.

The system should therefore retain all three representations rather than averaging them into a single vector.

Conceptually:

```text
voice exemplar
     │
     ├── ResNet293 → vector A
     │
     ├── ERes2Net  → vector B
     │
     └── WavLM     → vector C
```

---

# 14. Qdrant architecture

The existing Qdrant deployment should be used as the speaker-vector store.

Qdrant supports multiple **named vectors** of different dimensions within the same point, making it a natural fit for the three-model representation. ([Qdrant][5])

Conceptually:

```json
{
  "id": "speaker-exemplar-uuid",

  "vector": {
    "resnet293": [...],
    "eres2net": [...],
    "wavlm": [...]
  },

  "payload": {
    "episode_id": 1842,
    "episode_speaker_id": "ep1842_spk02",

    "start": 812.30,
    "end": 824.10,

    "duration": 11.80,

    "overlap": false,
    "quality_score": 0.94,

    "global_speaker_id": null
  }
}
```

Each named vector can have its own dimensionality. ([Qdrant][5])

Cosine similarity should initially be used unless the specific model documentation recommends another scoring representation. Qdrant normalizes vectors automatically when cosine distance is configured. ([Qdrant][6])

---

# 15. Store exemplars, not only centroids

The primary Qdrant point should represent a **clean voice exemplar** rather than an entire person.

For example:

```text
episode_1842:speaker_02

exemplar A
exemplar B
exemplar C
exemplar D
exemplar E
```

Each gets its own Qdrant point.

At expected archive scale:

```text
4,000 episodes
× ~4 speakers
× ~7 exemplars
≈ 112,000 points
```

This is a very modest vector workload for Qdrant.

Individual exemplars provide several important advantages.

### Diagnostic value

A suspicious match can be traced back to the exact audio responsible.

### Robustness

One corrupted or contaminated segment does not dominate the person's representation.

### Model experimentation

Embeddings can later be recomputed independently.

### Statistical verification

Multiple sample pairs can be compared instead of trusting one distance.

---

# 16. Episode-speaker centroids

An aggregate representation should additionally be calculated for each episode-speaker.

For each model:

```text
E = {e1, e2, e3, ..., en}
```

produce a robust normalized aggregate.

Initially this can simply be the normalized mean after rejecting obvious embedding outliers.

More sophisticated aggregation can be introduced later.

The centroid is useful for:

* initial candidate retrieval
* visualization
* coarse clustering

It should not replace the original exemplar embeddings.

---

# 17. Cross-episode speaker search

Given a speaker from an episode:

```text
episode 3318 / speaker 02
```

the system should search for historical candidate matches.

A two-stage retrieval architecture is proposed.

## Stage 1 — candidate retrieval

Use ResNet293 and/or ERes2Net centroids to retrieve approximately:

```text
top 20–50 episode-speaker candidates
```

from Qdrant.

This should be fast.

## Stage 2 — verification

For each candidate, compare multiple exemplars under all three embedding models.

Suppose speaker A has:

```text
a1 a2 a3 a4 a5
```

and candidate B has:

```text
b1 b2 b3 b4 b5
```

The system should not simply take:

```text
max cosine similarity
```

because a single accidental match is insufficient evidence.

Instead calculate a robust score such as:

```text
median of top mutually consistent sample matches
```

for each model.

This yields:

```text
ResNet293 score
ERes2Net score
WavLM score
```

These scores are then calibrated into an ensemble match score.

---

# 18. Ensemble scoring

The final speaker match score should not initially use arbitrary hard-coded weights such as:

```text
0.4 × A + 0.3 × B + 0.3 × C
```

Instead, the three scores should be calibrated using a small labelled dataset from the actual podcast.

The system will learn how each model behaves on:

```text
same speaker
different speaker
```

under the podcast's real recording conditions.

Possible final methods include:

* calibrated weighted average
* logistic regression
* shallow classifier
* probabilistic score fusion

The input remains simple:

```text
ResNet293 similarity statistics
ERes2Net similarity statistics
WavLM similarity statistics
duration/quality metadata
```

The system should remain interpretable.

A deep identity-classification model is unnecessary.

---

# 19. Conservative identity policy

Speaker identity should optimize for:

> **very low false-merge probability**

rather than maximum automatic recall.

There are two types of error.

## False split

```text
Mario
  ↓
global_017

Mario
  ↓
global_042
```

This is undesirable but recoverable.

## False merge

```text
Mario ─┐
       ├── global_017
Paolo ─┘
```

This contaminates the entire identity graph.

A false merge is therefore substantially more damaging.

The initial system should deliberately favor false splits.

Suggested states:

```text
HIGH CONFIDENCE
→ automatic link allowed

POSSIBLE MATCH
→ keep as candidate / human review

LOW CONFIDENCE
→ unrelated
```

The exact thresholds must come from the podcast-specific benchmark.

---

# 20. Global speaker graph

Global speaker identities should be represented conceptually as a graph.

Nodes:

```text
episode-speaker instances
```

Edges:

```text
high-confidence same-person evidence
```

Example:

```text
ep3821:s2 ───── ep3102:s1
     │              │
     │              │
ep2440:s4 ───── ep0811:s2
```

A connected, strongly supported group can become:

```text
global_speaker_0071
```

This is preferable to blindly applying generic unsupervised clustering to every embedding.

Candidate edges should be created only when multi-model verification is sufficiently strong.

---

# 21. Automatic clustering

Global clustering should be conservative.

The initial implementation should use:

1. nearest-neighbor candidate generation;
2. multi-model pairwise verification;
3. high-confidence graph edges;
4. conservative connected components or agglomerative merging.

HDBSCAN may still be useful for exploration, but should not initially define canonical identities.

The global identity layer needs provenance.

For every merge the system should be able to answer:

> Why do we believe these two episode-speakers are the same person?

---

# 22. Known-person enrollment

Once a global cluster is manually identified:

```text
global_speaker_0071 = "Mario Rossi"
```

the mapping should propagate automatically to all linked appearances.

Known identities should retain several high-quality references rather than one canonical recording.

Example:

```text
Mario Rossi
├── studio 2026
├── studio 2024
├── remote call 2022
└── old archive 2018
```

This allows the representation to account for:

* microphone changes
* aging
* different recording locations
* telephone codecs
* compression
* emotional state

---

# 23. Forgotten-guest discovery

The system should also support speaker discovery without known identities.

Example output:

```text
UNKNOWN_GLOBAL_0147

appears in:
episode 83
episode 417
episode 901
episode 2441
episode 3732
```

A user can listen to representative excerpts and determine:

```text
UNKNOWN_GLOBAL_0147 = Alessandro Bianchi
```

At that moment every associated historical appearance becomes labelled.

This is one of the principal benefits of processing the entire archive rather than only recent episodes.

---

# 24. Human-review interface

A minimal speaker-review interface should eventually show:

```text
Candidate identity: global_speaker_0147

Episode 83      [play]
Episode 417     [play]
Episode 901     [play]
Episode 2441    [play]

ResNet293     0.xx
ERes2Net      0.xx
WavLM         0.xx
ensemble      0.xx

[ same person ]
[ different person ]
[ uncertain ]
```

The selected audio should preferably be the highest-quality clean exemplar from each appearance.

Human decisions can then become labelled data for progressively improving score calibration.

---

# 25. Metadata database

Qdrant should contain vectors and retrieval-oriented payload.

Canonical relational relationships should live separately in PostgreSQL, SQLite, or another conventional database.

Suggested entities:

```text
episodes
speaker_turns
episode_speakers
speaker_exemplars
global_speakers
speaker_identity_labels
speaker_links
transcript_words
processing_runs
models
```

For example:

```text
episode_speakers
────────────────────────────
id
episode_id
local_speaker_label
speech_duration
overlap_duration
centroid_status
global_speaker_id
```

and:

```text
speaker_exemplars
────────────────────────────
id
episode_speaker_id
start
end
quality_score
qdrant_point_id
embedding_model_version
```

Model versions must always be recorded.

---

# 26. Transcript data model

The final transcript should not be stored only as human-readable prose.

A structured representation should preserve word timing and provenance.

Example:

```json
{
  "episode_id": 1842,
  "segments": [
    {
      "start": 812.30,
      "end": 824.10,
      "episode_speaker_id": "ep1842_spk02",
      "global_speaker_id": "global_0071",
      "speaker_name": "Mario Rossi",
      "text": "Secondo me il problema principale è...",
      "overlap": false
    }
  ]
}
```

Output renderers can then produce:

* TXT
* SRT
* VTT
* JSON
* searchable web views

without discarding underlying information.

---

# 27. Evaluation dataset

Before running the entire archive, build a small internal benchmark.

## Speaker-identity benchmark

Select approximately:

* 20 known recurring people
* 4–6 appearances each
* preferably spread over multiple years
* varied recording conditions

This gives roughly 100 labelled episode-speaker appearances.

Include deliberately difficult negative examples:

* speakers of the same gender
* similar voices
* similar accents
* recurring hosts
* family members if applicable
* remote-call audio
* old low-quality audio

## Metrics

Evaluate each embedding model independently and the ensemble.

Primary retrieval metrics:

```text
Recall@1
Recall@5
Recall@10
Recall@20
Mean Reciprocal Rank
```

Verification metrics:

```text
ROC
EER
false-positive rate
false-negative rate
```

Most importantly measure:

```text
false global-speaker merge rate
```

---

# 28. Same-person versus different-person distributions

For each model, plot similarity distributions:

```text
            different speakers
        ███████████████
     █████████████████████
────────────────────────────────────

                         same speaker
                      █████████████
                  ███████████████████
────────────────────────────────────
```

The important property is the separation on **this archive**, not the model's published VoxCeleb number.

This evaluation determines:

* model selection
* score fusion
* acceptance thresholds
* uncertainty region

---

# 29. Processing phases

The archive should not initially be processed as a single monolithic job.

## Phase 0 — benchmark

Approximately 10–20 representative episodes.

Goals:

* validate Opus decoding
* benchmark ASR
* inspect diarization
* compare the three embedding models
* establish initial similarity distributions
* measure GPU throughput

---

## Phase 1 — recent archive

Approximately:

```text
400 episodes
≈ 600 hours
```

Run the complete pipeline:

```text
ASR
diarization
speaker alignment
embedding ensemble
Qdrant ingestion
global speaker discovery
```

This produces immediate usable transcripts and enough cross-episode recurrence to calibrate identity matching.

---

## Phase 2 — entire archive identity pass

Approximately:

```text
4,000 episodes
≈ 6,000 hours
```

Priority:

```text
diarization
clean exemplars
three-model embeddings
Qdrant indexing
global identity discovery
```

Full ASR is optional at this stage.

The objective is to make the entire archive searchable **by voice** quickly.

An old episode discovered through voice matching can subsequently be prioritized for transcription.

---

## Phase 3 — historical ASR

Transcribe the remaining archive incrementally.

Potential priorities:

1. episodes containing newly discovered recurring guests;
2. episodes referenced by recent discussions;
3. historically significant episodes;
4. finally the remaining archive.

---

# 30. Job orchestration

Every processing stage should be idempotent and resumable.

A processing record should track:

```text
episode
stage
model
model_version
configuration_hash
started_at
completed_at
status
error
```

Example:

```text
episode 1842
  decode           COMPLETE
  asr-parakeet-v3  COMPLETE
  pyannote-c1      COMPLETE
  embeddings-v1    COMPLETE
  qdrant-index     COMPLETE
```

If the machine crashes at episode 2,718, processing should resume from the missing stage rather than restarting the archive.

---

# 31. Model versioning

Every artifact must retain the model that created it.

For example:

```json
{
  "embedding_model": "wespeaker-resnet293-LM",
  "embedding_model_version": "...",
  "pipeline_version": 1
}
```

This becomes important when a better embedding model appears.

A new model can be added as:

```text
embedding_v2
```

while keeping the original index until migration has been validated.

---

# 32. Qdrant collection strategy

Initial recommendation:

```text
collection:
podcast_speaker_exemplars
```

Named vectors:

```text
resnet293
eres2net
wavlm
```

Payload indexes should likely include:

```text
episode_id
episode_speaker_id
global_speaker_id
overlap
quality_score
recording_year
```

Qdrant supports multiple named vector spaces in a single point, each with independent dimensionality and distance configuration. ([Qdrant][5])

This is preferable to maintaining three unrelated collections because the three embeddings all describe the same physical exemplar.

---

# 33. Potential future use of Qdrant multivectors

Qdrant also supports multivectors and a `max_sim` comparator. ([Qdrant][5])

This could eventually allow one episode-speaker to be represented by several exemplar embeddings inside one logical vector object.

However, this should not be used in the first implementation.

Individual exemplar points provide better:

* diagnostics
* provenance
* threshold analysis
* audio traceability

Multivectors can be investigated after the matching methodology has been validated.

---

# 34. Model ensemble execution strategy

Two operating modes should be benchmarked.

## Full ensemble mode

Every clean exemplar is embedded by all three models:

```text
ResNet293
ERes2Net
WavLM
```

Advantages:

* simplest reasoning
* maximum information retained
* no need to recompute later

Given the relatively small amount of clean exemplar audio, this may be entirely practical on an RTX 5090.

## Cascade mode

Every exemplar receives:

```text
ResNet293
ERes2Net
```

WavLM is invoked only for:

* candidate links
* ambiguous comparisons
* high-value identity decisions

This reduces computation.

The benchmark should determine whether the saving is worthwhile.

Given the hardware available, **full three-model embedding is the preferred starting assumption**.

---

# 35. Expected archive scale

A plausible upper-bound estimate:

```text
4,000 episodes
× 5 episode-speakers
× 10 exemplars
= 200,000 exemplars
```

With three embeddings per exemplar:

```text
≈ 600,000 vectors
```

This remains a small vector-search workload by modern Qdrant standards.

The project should therefore optimize for **quality and traceability rather than aggressive vector compression**.

No vector quantization should initially be used.

---

# 36. Privacy and biometrics

Speaker embeddings should be treated as sensitive biometric-derived data even when the source podcast is public.

The system should therefore:

* remain local by default;
* avoid exposing the Qdrant collection publicly;
* protect the identity database;
* avoid repurposing embeddings outside the podcast archive;
* keep human identity labels separate from raw vectors where practical;
* log identity merges and manual corrections.

The system should be designed for archival/search purposes rather than inferring demographic or sensitive personal attributes from voices.

---

# 37. Initial model stack

The provisional stack is therefore:

### Audio

**FFmpeg**

Purpose:

```text
Opus decoding
resampling
segment extraction
```

### ASR

**NVIDIA Parakeet TDT 0.6B v3**

Purpose:

```text
Italian transcription
word timestamps
high-throughput local processing
```

### Diarization

**pyannote Community-1**

Purpose:

```text
speaker segmentation
speaker counting
overlap-aware diarization
exclusive diarization for ASR alignment
```

Community-1 is downloadable for local/offline use under CC-BY-4.0. ([Hugging Face][1])

### Embedding model 1

**WeSpeaker ResNet293-LM**

Purpose:

```text
primary speaker-verification representation
candidate retrieval
```

The released checkpoint provides 256-dimensional embeddings and CUDA-capable inference. ([Hugging Face][4])

### Embedding model 2

**ERes2Net-large / best validated ERes2Net checkpoint**

Purpose:

```text
architecture-diverse speaker verification
```

Exact checkpoint to be frozen after podcast-specific benchmarking.

### Embedding model 3

**WavLM Large + speaker-verification head**

Purpose:

```text
high-capacity acoustic identity representation
difficult cross-year / cross-channel verification
```

Exact checkpoint to be frozen after benchmarking.

### Vector database

**Qdrant**

Purpose:

```text
three named embedding vectors
nearest-neighbor candidate retrieval
metadata filtering
```

### Relational metadata

**PostgreSQL preferred if already available; SQLite acceptable initially**

Purpose:

```text
episodes
segments
speaker instances
global identities
processing state
model provenance
human corrections
```

---

# 38. Key open questions for the benchmark phase

The following should deliberately remain unresolved until measured.

### ASR

Does Parakeet provide sufficiently good Italian transcription relative to Scribe/Voxtral/MAI?

### Diarization

How frequently does Community-1:

* overestimate speaker count?
* split one speaker into several?
* merge similar speakers?
* miss short interruptions?

### Exemplars

What duration gives the best archive-specific speaker retrieval performance?

Likely candidates:

```text
5 s
10 s
15 s
20 s
```

### Embedding models

Which exact ERes2Net and WavLM checkpoints generalize best to this Italian podcast?

### Ensemble

How much independent information do the three embedding models provide?

### Identity threshold

What threshold produces an acceptably low false-merge probability?

These questions should be answered empirically rather than from published benchmark tables.

---

# 39. Desired end-state

After processing, it should be possible to query the archive in several ways.

## Transcript search

> Find every episode where someone mentions a particular person or topic.

## Speaker search

> Find every historical episode containing this voice.

## Identity history

> Show every episode in which Mario Rossi appears.

## Anonymous recurrence

> Show unidentified speakers who appear in more than one episode.

## Candidate identity

> Given this guest in episode 3,821, rank likely appearances in every earlier episode.

## Timeline

> When did this person first appear?

## Context retrieval

> Show what this speaker said around each appearance.

The archive therefore becomes both:

```text
a searchable text corpus
```

and:

```text
a searchable speaker corpus
```

---

# 40. Recommended first implementation milestone

Do **not** begin by processing 4,000 episodes.

The first complete milestone should consist of approximately **20 carefully chosen episodes**.

Those episodes should include:

* regular hosts
* known recurring guests
* one-off guests
* old and new recordings
* clean and degraded recordings
* heavy interruption/crosstalk
* several known instances of the same person

Run the complete pipeline:

```text
Opus
 ↓
FFmpeg
 ↓
Parakeet
 +
Community-1
 ↓
clean exemplar extraction
 ↓
ResNet293
ERes2Net
WavLM
 ↓
Qdrant
 ↓
cross-episode speaker matching
```

The milestone is successful when the system can take one known speaker appearance and reliably retrieve the same speaker from other episodes while keeping unrelated speakers below the automatic-link threshold.

Only after that behavior has been measured and calibrated should the pipeline be allowed to process the 400-episode recent archive and subsequently the full 4,000-episode collection.

One adjustment I deliberately made from our earlier discussion: I would **not yet lock the exact ERes2Net and WavLM checkpoints**. ResNet293-LM has a very clear production-ready checkpoint and documented 256-D output, while the other two families have multiple training recipes/checkpoints whose published benchmark ranking does not necessarily tell us which will work best on old Italian Opus podcast audio. The 20-episode benchmark should freeze those two exact choices. ([Hugging Face][4])

I’d also keep the three embeddings as **Qdrant named vectors on each exemplar**, rather than three collections. Qdrant explicitly supports independently sized named vector spaces on a single point, which maps almost perfectly to this design. ([Qdrant][5])

[1]: https://huggingface.co/pyannote/speaker-diarization-community-1?utm_source=chatgpt.com "pyannote/speaker-diarization-community-1 · Hugging Face"
[2]: https://www.pyannote.ai/blog/community-1?utm_source=chatgpt.com "Community-1: Unleashing open-source diarization - pyannoteAI Speaker Intelligence and Diarization"
[3]: https://huggingface.co/pyannote-community/speaker-diarization-community-1/blob/main/config.yaml?utm_source=chatgpt.com "config.yaml · pyannote-community/speaker-diarization-community-1 at main"
[4]: https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM?utm_source=chatgpt.com "Wespeaker/wespeaker-voxceleb-resnet293-LM · Hugging Face"
[5]: https://qdrant.tech/documentation/manage-data/vectors/?utm_source=chatgpt.com "Vectors - Qdrant"
[6]: https://qdrant.tech/documentation/manage-data/collections/?utm_source=chatgpt.com "Collections - Qdrant"

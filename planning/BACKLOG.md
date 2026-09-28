# Ordered backlog

Revised 2026-09-28. [The working plan](README.md) governs scope; numeric Order and
direct deliverable prerequisites govern selection. Closed issues retain historical
records. Deferred work is Backlog and never blocks core milestones. Retired work
is replaced explicitly. Phase labels are organizational.

| ID/body | Deliverable | Scope | Blocked by | Executor | Order |
|---|---|---|---|---|---|
| [P0](issues/P0.md) | Foundation and reproducible corpus | Historical/completed | P0-01, P0-02, P0-03, P0-04, P0-05, P0-06 | Human | 0 |
| [P0-01](issues/P0-01.md) | Package layout, uv configuration and CPU-only CI | Historical/completed | — | Luna | 10 |
| [P0-02](issues/P0-02.md) | Frozen 20-episode manifest and read-only source validation | Historical/completed | P0-01 | Luna | 20 |
| [P0-03](issues/P0-03.md) | Typed artifact, model, timestamp, job and API contracts | Historical/completed | P0-01 | Luna | 30 |
| [P0-04](issues/P0-04.md) | SQLite migrations, canonical entities and artifact publication | Historical/completed | P0-03 | Luna | 40 |
| [P0-05](issues/P0-05.md) | Resumable worker, stage fingerprints and crash recovery | Historical/completed | P0-04 | Luna | 50 |
| [P0-06](issues/P0-06.md) | Repository Sol phase-review skill and Luna handoff | Historical/completed | P0-01 | Luna | 60 |
| [P1](issues/P1.md) | Reusable transcription and diarization services | Active | P1-H01, P1-09, P1-01, P1-02, P1-03, P1-04, P1-05, P1-07, P1-10 | Human | 1000 |
| [P1-H01](issues/P1-H01.md) | Operator: accept model terms and provision download access | Historical/completed | P0 | Human | 1005 |
| [P1-09](issues/P1-09.md) | Shared GPU runtime base, Python 3.14 and model-image disk budget | Historical/completed | P0, P1-H01 | Luna | 1009 |
| [P1-01](issues/P1-01.md) | Model locks, CUDA compatibility and container smoke checks | Historical/completed | P0, P1-H01, P1-09 | Luna | 1010 |
| [P1-02](issues/P1-02.md) | Parakeet service and genuine word timestamps | Historical/completed | P0, P1-01 | Luna | 1020 |
| [P1-03](issues/P1-03.md) | Community-1 service with standard and exclusive diarization | Historical/completed | P0, P1-01 | Luna | 1030 |
| [P1-04](issues/P1-04.md) | Word attribution and structured transcript exports | Historical/completed | P0, P1-02, P1-03 | Luna | 1040 |
| [P1-05](issues/P1-05.md) | Local annotation UI and versioned reviewed exports | Historical/completed | P0, P1-04 | Luna | 1050 |
| [P1-06](issues/P1-06.md) | Human: review golden transcript and freeze evaluation split | Historical/completed | P0, P1-05 | Human | 1060 |
| [P1-07](issues/P1-07.md) | ASR, diarization, attribution and timing evaluation harness | Historical/completed | P0, P1-05 | Luna | 1070 |
| [P1-08](issues/P1-08.md) | Execute local baseline and publish measured evidence | Historical/completed | P0, P1-06, P1-07 | Luna | 1080 |
| [P1-10](issues/P1-10.md) | Prefer niquests and minimize HTTP client dependencies | Historical/completed | P0 | Luna | 1090 |
| [P1R](issues/P1R.md) | Small local ASR comparison | Active | P1R-01, P1R-02, P1R-03, P1R-03D, P1R-04, P1R-05, P1R-06, P1R-07, P1R-08, P1R-08B, P1R-08C, P1R-08A, P1R-09, P1R-10, P1R-11, P1R-H01, P1R-12, P1R-13, P1R-14, P1R-16, P1R-17 | Human | 1500 |
| [P1R-01](issues/P1R-01.md) | Supersede legacy evaluation design and migrate planning contracts | Historical/completed | P0 | Luna | 1510 |
| [P1R-02](issues/P1R-02.md) | Define chunk and hypothesis canonical schemas | Historical/completed | P0, P1R-01 | Luna | 1520 |
| [P1R-03](issues/P1R-03.md) | Implement deterministic adaptive chunk segmentation | Historical/completed | P0, P1R-02 | Luna | 1530 |
| [P1R-03A](issues/P1R-03A.md) | Community-1-aware adaptive chunk boundary selection | Historical/completed | P0, P1R-02, P1R-03, P1-03 | Luna | 1535 |
| [P1R-03B](issues/P1R-03B.md) | Use exclusive diarization turn boundaries to reduce hard-max chunking | Historical/completed | P0, P1R-02, P1R-03A, P1-03 | Luna | 1536 |
| [P1R-03D](issues/P1R-03D.md) | Native Community-1 speech-activity adaptive chunking | Historical/completed | P0, P1R-02, P1-03 | Luna | 1538 |
| [P1R-04](issues/P1R-04.md) | Adapt Parakeet to the chunk-first transcription contract | Historical/completed | P0, P1R-02 | Luna | 1540 |
| [P1R-05](issues/P1R-05.md) | Add Whisper Large v3 local ASR service | Historical/completed | P0, P1R-02 | Luna | 1550 |
| [P1R-06](issues/P1R-06.md) | Add Voxtral Mini 4B Realtime local ASR service | Historical/completed | P0, P1R-02 | Luna | 1560 |
| [P1R-07](issues/P1R-07.md) | Orchestrate multi-model chunk inference | Historical/completed | P0, P1R-03, P1R-04, P1R-05, P1R-06 | Luna | 1570 |
| [P1R-08](issues/P1R-08.md) | Derive speaker-count and overlap metadata from Community-1 | Historical/completed | P0, P1R-03 | Luna | 1580 |
| [P1R-08B](issues/P1R-08B.md) | Prepare dedicated music calibration review workflow | Historical/completed | P0, P1R-03 | Luna | 1584 |
| [P1R-08C](issues/P1R-08C.md) | Generate deterministic development calibration batches | Historical/completed | P0, P1R-03 | Luna | 1585 |
| [P1R-08A](issues/P1R-08A.md) | Classify acoustic conditions with AudioSet AST | Historical/completed | P0, P1R-03, P1R-08B, P1R-08C | Luna | 1586 |
| [P1R-09](issues/P1R-09.md) | Add local Qwen annotation-assistance service | Historical/completed | P0, P1R-07 | Luna | 1590 |
| [P1R-12](issues/P1R-12.md) | Implement chunk-level ASR scoring harness | Historical/completed | P0, P1R-02 | Luna | 1630 |
| [P1R-13](issues/P1R-13.md) | Rebuild diarization evaluation independently of ASR words | Historical/completed | P0, P1R-02, P1R-08 | Luna | 1640 |
| [P1R-14](issues/P1R-14.md) | Implement integrated who-said-what scoring | Historical/completed | P0, P1R-02, P1R-12, P1R-13 | Luna | 1650 |
| [P2](issues/P2.md) | Voice discovery and human identity review | Active | P2-01, P2-02, P2-05, P2-06, P2-07, P2-08, P2-09 | Human | 2000 |
| [P3](issues/P3.md) | Useful lexical transcript search | Active | P3-03, P3-04, P3-06, P3-07 | Human | 3000 |
| [P4](issues/P4.md) | Usable archive website on the LAN | Active | P4-01, P4-02, P4-04, P4-06 | Human | 4000 |
| [P5](issues/P5.md) | Useful and recoverable initial 20 | Active | P5-01, P5-03, P5-05, P5-06 | Human | 5000 |
| [P6](issues/P6.md) | 400 recent episodes | Active | P6-01, P6-02, P6-03, P6-04 | Human | 6000 |
| [P7](issues/P7.md) | Historical voice index | Active | P7-01, P7-02, P7-03, P7-04 | Human | 7000 |
| [P8](issues/P8.md) | Remaining historical transcription | Active | P8-01, P8-02, P8-03, P8-04 | Human | 8000 |
| [P3-03](issues/P3-03.md) | Speaker-aware transcript chunks and SQLite FTS5 | Active | P1-04 | Luna | 9000 |
| [P5-06](issues/P5-06.md) | Produce and index the initial 20 transcripts | Active | P3-03 | Luna | 9005 |
| [P3-04](issues/P3-04.md) | Filtered lexical transcript search | Active | P3-03 | Luna | 9010 |
| [P4-01](issues/P4-01.md) | Reuse the local web app for archive navigation | Active | P1-05 | Luna | 9020 |
| [P4-02](issues/P4-02.md) | Text search, transcripts and timestamped playback | Active | P3-04, P4-01 | Luna | 9030 |
| [P1R-10](issues/P1R-10.md) | Finish the chunk review interface | Active | P1R-07, P1R-08, P1R-03D | Luna | 9040 |
| [P1R-11](issues/P1R-11.md) | Freeze an 80-chunk ASR pilot | Active | P1R-03D, P1R-08 | Luna | 9050 |
| [P1R-H01](issues/P1R-H01.md) | Human: review the small ASR pilot | Active | P1R-10, P1R-11 | Human | 9060 |
| [P1R-16](issues/P1R-16.md) | Compare three local ASR models on the pilot | Active | P1R-07, P1R-11, P1R-H01, P1R-12, P1R-13, P1R-14 | Luna | 9070 |
| [P1R-17](issues/P1R-17.md) | Record the selected ASR operating settings | Active | P1R-16 | Luna | 9080 |
| [P2-01](issues/P2-01.md) | Extract clean speaker exemplars | Active | P1-03 | Luna | 9090 |
| [P2-02](issues/P2-02.md) | ResNet293 speaker embedding adapter | Active | P1-01 | Luna | 9100 |
| [P2-05](issues/P2-05.md) | Index ResNet exemplars and centroids in Qdrant | Active | P2-01, P2-02 | Luna | 9110 |
| [P2-06](issues/P2-06.md) | Retrieve and verify ResNet voice candidates | Active | P2-05 | Luna | 9120 |
| [P2-09](issues/P2-09.md) | Human identity links, contradiction checks, undo and split | Active | P2-06 | Luna | 9130 |
| [P4-04](issues/P4-04.md) | Voice comparison and anonymous appearance history | Active | P4-01, P2-06, P2-09 | Luna | 9140 |
| [P2-07](issues/P2-07.md) | Human: review recurring voices and difficult negatives | Active | P2-06 | Human | 9150 |
| [P2-08](issues/P2-08.md) | Evaluate the first voice retrieval baseline | Active | P2-06, P2-07 | Luna | 9160 |
| [P3-06](issues/P3-06.md) | Human: check ten useful Italian search queries | Active | P4-02 | Human | 9170 |
| [P3-07](issues/P3-07.md) | Summarize lexical search usefulness | Active | P3-06 | Luna | 9180 |
| [P4-06](issues/P4-06.md) | Check the core browser journeys | Active | P4-02, P4-04 | Luna | 9190 |
| [P5-01](issues/P5-01.md) | Run the workstation with serial GPU jobs | Active | P4-02, P2-06 | Luna | 9200 |
| [P5-03](issues/P5-03.md) | Demonstrate one backup and restore | Active | P5-01, P2-09 | Luna | 9210 |
| [P5-05](issues/P5-05.md) | Document the working archive and recovery commands | Active | P4-06, P5-03 | Luna | 9220 |
| [P6-01](issues/P6-01.md) | Freeze 400 recent episodes and incremental estimates | Active | P5-06 | Luna | 9240 |
| [P6-02](issues/P6-02.md) | Human: authorize 400-episode expansion from estimates | Active | P6-01 | Human | 9250 |
| [P6-03](issues/P6-03.md) | Process 40-total canary and check drift/regressions | Active | P6-02 | Luna | 9260 |
| [P6-04](issues/P6-04.md) | Process remaining 400 manifest and publish coverage | Active | P6-03 | Luna | 9270 |
| [P7-01](issues/P7-01.md) | Freeze historical manifest and execute stratified voice canary | Active | P6-04 | Luna | 9280 |
| [P7-02](issues/P7-02.md) | Human: cross-year labels and historical retrieval evaluation | Active | P7-01 | Human | 9290 |
| [P7-03](issues/P7-03.md) | Human: authorize historical voice processing | Active | P7-02 | Human | 9300 |
| [P7-04](issues/P7-04.md) | Resumable historical voice index and ASR-missing appearances | Active | P7-03 | Luna | 9310 |
| [P8-01](issues/P8-01.md) | Prioritized remaining transcription queue and estimates | Active | P7-04 | Luna | 9320 |
| [P8-02](issues/P8-02.md) | Human: authorize remaining transcription resources | Active | P8-01 | Human | 9330 |
| [P8-03](issues/P8-03.md) | Bounded historical transcription and searchable publication | Active | P8-02 | Luna | 9340 |
| [P8-04](issues/P8-04.md) | Final coverage, quality, recovery and documentation audit | Active | P8-03 | Luna | 9350 |
| [P1R-15](issues/P1R-15.md) | Evaluate annotation assistance when needed | Deferred | P1R-H01 | Luna | 20000 |
| [P2-03](issues/P2-03.md) | Compare ERes2Net after measured ResNet misses | Deferred | P2-08 | Luna | 20010 |
| [P2-04](issues/P2-04.md) | Compare WavLM after measured ResNet misses | Deferred | P2-08 | Luna | 20020 |
| [P3-01](issues/P3-01.md) | Extend shared-inference transcription when needed | Deferred | P1R-16 | Luna | 20030 |
| [P3-02](issues/P3-02.md) | Add dense retrieval for demonstrated lexical gaps | Deferred | P3-07 | Luna | 20040 |
| [P3-H01](issues/P3-H01.md) | Human: provision a capped key if cloud comparison is needed | Deferred | P1R-16 | Human | 20050 |
| [P3-05](issues/P3-05.md) | Run bounded provider comparison when justified | Deferred | P3-01, P3-H01 | Luna | 20060 |
| [P4-03](issues/P4-03.md) | Add uploaded voice queries when needed | Deferred | P4-04 | Luna | 20070 |
| [P4-05](issues/P4-05.md) | Add evaluation dashboard when reports become awkward | Deferred | P5-06 | Luna | 20080 |
| [P5-H01](issues/P5-H01.md) | Human: provision remote access when needed | Deferred | P5-06 | Human | 20090 |
| [P5-02](issues/P5-02.md) | Protect remote access before Internet exposure | Deferred | P5-01, P5-H01 | Luna | 20100 |
| [P5-04](issues/P5-04.md) | Tune GPU concurrency after observed contention | Deferred | P5-01 | Luna | 20110 |
| [P1R-03C](issues/P1R-03C.md) | Use exclusive diarization turn boundaries to reduce hard-max chunking | Retired | — | Luna | 21000 |

## Requirement coverage

| ID | Requirement | Issues |
|---|---|---|
| R01 | Python 3.14, uv/src package, Python 3.11 ML isolation and CPU CI | [P0-01](issues/P0-01.md), [P1-09](issues/P1-09.md), [P1-01](issues/P1-01.md) |
| R02 | Frozen exactly-20 corpus, golden episode, readonly NFS source and millisecond metadata | [P0-02](issues/P0-02.md), [P0-03](issues/P0-03.md) |
| R03 | Local SQLite WAL/FTS5, dedicated Qdrant, atomic artifacts and resumable worker | [P0-04](issues/P0-04.md), [P0-05](issues/P0-05.md), [P2-05](issues/P2-05.md), [P3-03](issues/P3-03.md) |
| R04 | Locked model revisions/licenses/head and measured RTX 5090 compatibility | [P1-H01](issues/P1-H01.md), [P1-09](issues/P1-09.md), [P1-01](issues/P1-01.md), [P2-02](issues/P2-02.md), [P2-03](issues/P2-03.md), [P2-04](issues/P2-04.md), [P3-02](issues/P3-02.md) |
| R05 | Legacy retained: optional word timing, episode-wide standard/exclusive diarization and production attribution | [P1-02](issues/P1-02.md), [P1-03](issues/P1-03.md), [P1-04](issues/P1-04.md), [P1R-04](issues/P1R-04.md) |
| R06 | Legacy golden artifacts retained as non-authoritative transcription evidence | [P1-05](issues/P1-05.md), [P1-06](issues/P1-06.md), [P1-07](issues/P1-07.md), [P1-08](issues/P1-08.md), [P1R-01](issues/P1R-01.md) |
| R29 | Chunk-first schemas, deterministic model-independent segmentation, multi-model immutable hypotheses and condition metadata | [P1R-02](issues/P1R-02.md), [P1R-03](issues/P1R-03.md), [P1R-03A](issues/P1R-03A.md), [P1R-04](issues/P1R-04.md), [P1R-05](issues/P1R-05.md), [P1R-06](issues/P1R-06.md), [P1R-07](issues/P1R-07.md), [P1R-08](issues/P1R-08.md), [P1R-08A](issues/P1R-08A.md) |
| R30 | Human-reviewed chunk truth, episode-independent representative/stress benchmark and separate ASR/diarization/attribution evidence | [P1R-09](issues/P1R-09.md), [P1R-10](issues/P1R-10.md), [P1R-11](issues/P1R-11.md), [P1R-H01](issues/P1R-H01.md), [P1R-12](issues/P1R-12.md), [P1R-13](issues/P1R-13.md), [P1R-14](issues/P1R-14.md), [P1R-16](issues/P1R-16.md), [P1R-17](issues/P1R-17.md) |
| R07 | Clean exemplars, ResNet voice index, centroids and deterministic retrieval; extra encoders optional | [P2-01](issues/P2-01.md), [P2-05](issues/P2-05.md), [P2-06](issues/P2-06.md) |
| R08 | Human voice truth, valid calibration, host/non-host evaluation and uncertainty | [P2-07](issues/P2-07.md), [P2-08](issues/P2-08.md) |
| R09 | Human-only global identities, manual names, contradictions, undo and split | [P2-09](issues/P2-09.md), [P4-04](issues/P4-04.md) |
| R10 | Deferred when justified: Shared-inference transcription upstream extension and merged pin | [P3-01](issues/P3-01.md) |
| R11 | Deferred when justified: Replaceable inference, capabilities and safe vector-generation swaps | [P3-02](issues/P3-02.md), [P3-05](issues/P3-05.md) |
| R12 | FTS5 transcript search, filters and playback; dense/hybrid on demonstrated need | [P3-03](issues/P3-03.md), [P3-04](issues/P3-04.md), [P4-02](issues/P4-02.md) |
| R13 | About ten reviewed Italian search examples; larger relevance comparison when needed | [P3-06](issues/P3-06.md), [P3-07](issues/P3-07.md) |
| R14 | Deferred when justified: Identical 20-minute local/cloud ASR subset and enforced US$10 total benchmark | [P3-H01](issues/P3-H01.md), [P3-05](issues/P3-05.md), [P3-07](issues/P3-07.md) |
| R15 | Private FastAPI/Jinja website, API v1, playback and bounded media access | [P4-01](issues/P4-01.md), [P4-02](issues/P4-02.md), [P4-06](issues/P4-06.md) |
| R16 | Deferred when justified: Archive-speaker/upload queries, formats/limits, selected speaker and one-hour expiry | [P4-03](issues/P4-03.md) |
| R17 | Candidate comparison, anonymous recurrence, appearances and chronology | [P4-04](issues/P4-04.md) |
| R18 | Evaluation/job dashboard, measured coverage/latency/cost and failure behavior | [P4-05](issues/P4-05.md), [P5](issues/P5.md) |
| R19 | Deferred when justified: Compose profiles, 4 GiB VRAM headroom and full ensemble contention benchmark | [P5-01](issues/P5-01.md), [P5-04](issues/P5-04.md) |
| R20 | Deferred when justified: Cloudflare Access on entire website, origin JWT and CSRF, operator deployment configuration | [P5-H01](issues/P5-H01.md), [P5-02](issues/P5-02.md) |
| R21 | Backups, Qdrant rebuild, interrupted job recovery and private evidence storage | [P5-03](issues/P5-03.md) |
| R22 | Accurate application/recovery runbook with private data excluded | [P5-05](issues/P5-05.md), [P8-04](issues/P8-04.md) |
| R23 | 20-episode acceptance and quality/regression gates, with P1R transcription evidence | [P1R-16](issues/P1R-16.md), [P1R-17](issues/P1R-17.md), [P2-08](issues/P2-08.md), [P3-07](issues/P3-07.md), [P5](issues/P5.md) |
| R24 | Gated 400 expansion, estimates and 40-total canary | [P6-01](issues/P6-01.md), [P6-02](issues/P6-02.md), [P6-03](issues/P6-03.md), [P6-04](issues/P6-04.md) |
| R25 | Gated historical voice-only pass, 20 canary and cross-year human labels | [P7-01](issues/P7-01.md), [P7-02](issues/P7-02.md), [P7-03](issues/P7-03.md), [P7-04](issues/P7-04.md) |
| R26 | Gated prioritized remaining ASR, bounded batches and final audit | [P8-01](issues/P8-01.md), [P8-02](issues/P8-02.md), [P8-03](issues/P8-03.md), [P8-04](issues/P8-04.md) |
| R27 | Focused issue implementation and optional integrated review; direct deliverable dependencies | [P0-06](issues/P0-06.md), [P0](issues/P0.md), [P1](issues/P1.md), [P1R](issues/P1R.md), [P2](issues/P2.md), [P3](issues/P3.md), [P4](issues/P4.md), [P5](issues/P5.md), [P6](issues/P6.md), [P7](issues/P7.md), [P8](issues/P8.md) |
| R28 | Maintain relevant GitHub issues/dependencies and Project status with targeted readback | [P0-06](issues/P0-06.md) |

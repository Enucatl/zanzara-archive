# Ordered backlog

[manifest.json](manifest.json) is the machine-readable authority for 68 issues (59 children and nine phase parents), fields, labels, views and requirement coverage. Each body is ready to publish using the link-rendering procedure in [GITHUB-SETUP](GITHUB-SETUP.md). P1-H01, P3-H01 and P5-H01 make required operator provisioning explicit; the numbered handoff IDs remain intact.

Every child depends on the preceding phase plus its listed local prerequisites. A parent is blocked by all children, including future review findings, and closes only after Sol review and user release. No child depends on its own parent. Kind/Executor fields distinguish operator work from implementation; a phase parent has Human executor. Select eligible Luna work by phase, priority, then numeric Order, not table position alone.

| ID/body | Deliverable | Parent | Blocked by | Executor | Kind | Priority | Order |
|---|---|---|---|---|---|---|---|
| [P0](issues/P0.md) | Foundation and reproducible corpus | — | P0-01, P0-02, P0-03, P0-04, P0-05, P0-06 | Human | Phase | P0 | 0 |
| [P0-01](issues/P0-01.md) | Package layout, uv configuration and CPU-only CI | P0 | — | Luna | Implementation | P1 | 10 |
| [P0-02](issues/P0-02.md) | Frozen 20-episode manifest and read-only source validation | P0 | P0-01 | Luna | Implementation | P1 | 20 |
| [P0-03](issues/P0-03.md) | Typed artifact, model, timestamp, job and API contracts | P0 | P0-01 | Luna | Implementation | P1 | 30 |
| [P0-04](issues/P0-04.md) | SQLite migrations, canonical entities and artifact publication | P0 | P0-03 | Luna | Implementation | P1 | 40 |
| [P0-05](issues/P0-05.md) | Resumable worker, stage fingerprints and crash recovery | P0 | P0-04 | Luna | Implementation | P1 | 50 |
| [P0-06](issues/P0-06.md) | Repository Sol phase-review skill and Luna handoff | P0 | P0-01 | Luna | Implementation | P1 | 60 |
| [P1](issues/P1.md) | Measured transcription and diarization baseline | — | P0, P1-H01, P1-09, P1-01, P1-02, P1-03, P1-04, P1-05, P1-06, P1-07, P1-08, P1-10 | Human | Phase | P0 | 1000 |
| [P1-H01](issues/P1-H01.md) | Operator: accept model terms and provision download access | P1 | P0 | Human | Operator | P0 | 1005 |
| [P1-09](issues/P1-09.md) | Shared GPU runtime base, Python 3.14 and model-image disk budget | P1 | P0, P1-H01 | Luna | Implementation | P1 | 1009 |
| [P1-01](issues/P1-01.md) | Model locks, CUDA compatibility and container smoke checks | P1 | P0, P1-H01, P1-09 | Luna | Implementation | P1 | 1010 |
| [P1-02](issues/P1-02.md) | Parakeet service and genuine word timestamps | P1 | P0, P1-01 | Luna | Implementation | P1 | 1020 |
| [P1-03](issues/P1-03.md) | Community-1 service with standard and exclusive diarization | P1 | P0, P1-01 | Luna | Implementation | P1 | 1030 |
| [P1-04](issues/P1-04.md) | Word attribution and structured transcript exports | P1 | P0, P1-02, P1-03 | Luna | Implementation | P1 | 1040 |
| [P1-05](issues/P1-05.md) | Local annotation UI and versioned reviewed exports | P1 | P0, P1-04 | Luna | Implementation | P1 | 1050 |
| [P1-06](issues/P1-06.md) | Human: review golden transcript and freeze evaluation split | P1 | P0, P1-05 | Human | Operator | P0 | 1060 |
| [P1-07](issues/P1-07.md) | ASR, diarization, attribution and timing evaluation harness | P1 | P0, P1-05 | Luna | Evaluation | P1 | 1070 |
| [P1-08](issues/P1-08.md) | Execute local baseline and publish measured evidence | P1 | P0, P1-06, P1-07 | Luna | Evaluation | P1 | 1080 |
| [P1-10](issues/P1-10.md) | Prefer niquests and minimize HTTP client dependencies | P1 | P0 | Luna | Implementation | P1 | 1090 |
| [P2](issues/P2.md) | Three-model voice retrieval and identity review | — | P1, P2-01, P2-02, P2-03, P2-04, P2-05, P2-06, P2-07, P2-08, P2-09 | Human | Phase | P0 | 2000 |
| [P2-01](issues/P2-01.md) | Deterministic clean-exemplar extraction | P2 | P1 | Luna | Implementation | P1 | 2010 |
| [P2-02](issues/P2-02.md) | ResNet293 service adapter and validated embedding fixtures | P2 | P1 | Luna | Implementation | P1 | 2020 |
| [P2-03](issues/P2-03.md) | ERes2Net service adapter and validated embedding fixtures | P2 | P1 | Luna | Implementation | P1 | 2030 |
| [P2-04](issues/P2-04.md) | WavLM Large verification service adapter and validated embedding fixtures | P2 | P1 | Luna | Implementation | P1 | 2040 |
| [P2-05](issues/P2-05.md) | Qdrant exemplar and centroid generations | P2 | P1, P2-01, P2-02, P2-03, P2-04 | Luna | Implementation | P1 | 2050 |
| [P2-06](issues/P2-06.md) | Candidate retrieval and robust exemplar verification | P2 | P1, P2-05 | Luna | Implementation | P1 | 2060 |
| [P2-07](issues/P2-07.md) | Human: cross-episode labels and frozen voice splits | P2 | P1, P2-06 | Human | Operator | P0 | 2070 |
| [P2-08](issues/P2-08.md) | Encoder and ensemble evaluation with conditional calibration | P2 | P1, P2-07 | Luna | Evaluation | P1 | 2080 |
| [P2-09](issues/P2-09.md) | Audited manual identity decisions, undo and split | P2 | P1, P2-06 | Luna | Implementation | P1 | 2090 |
| [P3](issues/P3.md) | Hybrid text retrieval and replaceable inference | — | P2, P3-01, P3-02, P3-03, P3-04, P3-H01, P3-05, P3-06, P3-07 | Human | Phase | P0 | 3000 |
| [P3-H01](issues/P3-H01.md) | Operator: provision capped OpenRouter benchmark key | P3 | P2 | Human | Operator | P0 | 3005 |
| [P3-01](issues/P3-01.md) | Upstream shared-inference transcription and pinned integration | P3 | P2 | Luna | Implementation | P1 | 3010 |
| [P3-02](issues/P3-02.md) | Local BGE-M3 and shared-inference embedding adapter | P3 | P2 | Luna | Implementation | P1 | 3020 |
| [P3-03](issues/P3-03.md) | Speaker-aware chunks, FTS5 and dense text indexing | P3 | P2, P3-02 | Luna | Implementation | P1 | 3030 |
| [P3-04](issues/P3-04.md) | Filtered lexical, dense and hybrid search | P3 | P2, P3-03 | Luna | Implementation | P1 | 3040 |
| [P3-05](issues/P3-05.md) | Provider probes, generation safety and cost controls | P3 | P2, P3-01, P3-02, P3-H01 | Luna | Implementation | P1 | 3050 |
| [P3-06](issues/P3-06.md) | Human: frozen Italian text-query relevance set | P3 | P2, P3-04 | Human | Operator | P0 | 3060 |
| [P3-07](issues/P3-07.md) | Cloud/local ASR and text-search evaluation reports | P3 | P2, P3-05, P3-06 | Luna | Evaluation | P1 | 3070 |
| [P4](issues/P4.md) | Complete private website | — | P3, P4-01, P4-02, P4-03, P4-04, P4-05, P4-06 | Human | Phase | P0 | 4000 |
| [P4-01](issues/P4-01.md) | Responsive application shell and API integration | P4 | P3 | Luna | Implementation | P1 | 4010 |
| [P4-02](issues/P4-02.md) | Text search, transcripts and timestamped audio playback | P4 | P3, P4-01 | Luna | Implementation | P1 | 4020 |
| [P4-03](issues/P4-03.md) | Archive-speaker and bounded upload voice-search jobs | P4 | P3, P4-01 | Luna | Implementation | P1 | 4030 |
| [P4-04](issues/P4-04.md) | Speaker comparison, identity review and chronology | P4 | P3, P4-01 | Luna | Implementation | P1 | 4040 |
| [P4-05](issues/P4-05.md) | Evaluation dashboard and job/error states | P4 | P3, P4-01 | Luna | Implementation | P1 | 4050 |
| [P4-06](issues/P4-06.md) | Browser tests for complete user journeys | P4 | P3, P4-02, P4-03, P4-04, P4-05 | Luna | Evaluation | P1 | 4060 |
| [P5](issues/P5.md) | Operational 20-episode release and public documentation | — | P4, P5-01, P5-H01, P5-02, P5-03, P5-04, P5-05, P5-06 | Human | Phase | P0 | 5000 |
| [P5-H01](issues/P5-H01.md) | Operator: provision Cloudflare Access and deployment configuration | P5 | P4 | Human | Operator | P0 | 5005 |
| [P5-01](issues/P5-01.md) | Production Compose profiles and configuration reference | P5 | P4 | Luna | Implementation | P1 | 5010 |
| [P5-02](issues/P5-02.md) | Cloudflare tunnel, origin JWT enforcement and provisioning runbook | P5 | P4, P5-01, P5-H01 | Luna | Implementation | P1 | 5020 |
| [P5-03](issues/P5-03.md) | Backup/restore, Qdrant rebuild and interrupted-job drills | P5 | P4, P5-01 | Luna | Evaluation | P1 | 5030 |
| [P5-04](issues/P5-04.md) | Concurrent GPU benchmark and measured serving defaults | P5 | P4, P5-01 | Luna | Evaluation | P1 | 5040 |
| [P5-05](issues/P5-05.md) | Public README and three architecture-generation prompts | P5 | P4, P5-02, P5-03, P5-04 | Luna | Implementation | P1 | 5050 |
| [P5-06](issues/P5-06.md) | Complete 20-episode acceptance and release evidence packet | P5 | P4, P5-05 | Luna | Evaluation | P1 | 5060 |
| [P6](issues/P6.md) | 400 recent episodes | — | P5, P6-01, P6-02, P6-03, P6-04 | Human | Phase | P0 | 6000 |
| [P6-01](issues/P6-01.md) | Freeze 400 recent episodes and incremental estimates | P6 | P5 | Luna | Implementation | P1 | 6010 |
| [P6-02](issues/P6-02.md) | Human: authorize 400-episode expansion from estimates | P6 | P5, P6-01 | Human | Operator | P0 | 6020 |
| [P6-03](issues/P6-03.md) | Process 40-total canary and check drift/regressions | P6 | P5, P6-02 | Luna | Evaluation | P1 | 6030 |
| [P6-04](issues/P6-04.md) | Process remaining 400 manifest and publish coverage | P6 | P5, P6-03 | Luna | Implementation | P1 | 6040 |
| [P7](issues/P7.md) | Historical voice index | — | P6, P7-01, P7-02, P7-03, P7-04 | Human | Phase | P0 | 7000 |
| [P7-01](issues/P7-01.md) | Freeze historical manifest and execute stratified voice canary | P7 | P6 | Luna | Implementation | P1 | 7010 |
| [P7-02](issues/P7-02.md) | Human: cross-year labels and historical retrieval evaluation | P7 | P6, P7-01 | Human | Operator | P0 | 7020 |
| [P7-03](issues/P7-03.md) | Human: authorize historical voice processing | P7 | P6, P7-02 | Human | Operator | P0 | 7030 |
| [P7-04](issues/P7-04.md) | Resumable historical voice index and ASR-missing appearances | P7 | P6, P7-03 | Luna | Implementation | P1 | 7040 |
| [P8](issues/P8.md) | Remaining historical transcription | — | P7, P8-01, P8-02, P8-03, P8-04 | Human | Phase | P0 | 8000 |
| [P8-01](issues/P8-01.md) | Prioritized remaining transcription queue and estimates | P8 | P7 | Luna | Implementation | P1 | 8010 |
| [P8-02](issues/P8-02.md) | Human: authorize remaining transcription resources | P8 | P7, P8-01 | Human | Operator | P0 | 8020 |
| [P8-03](issues/P8-03.md) | Bounded historical transcription and searchable publication | P8 | P7, P8-02 | Luna | Implementation | P1 | 8030 |
| [P8-04](issues/P8-04.md) | Final coverage, quality, recovery and documentation audit | P8 | P7, P8-03 | Luna | Evaluation | P1 | 8040 |

## Requirement coverage

R28's setup deliverable is completed by following the runbook before implementation starts; P0-06 preserves its relationship/selection rules in the repository skill. Credentials and view configuration needed before setup are explicit operator preflight actions in GITHUB-SETUP, not issues that require a nonexistent Project.

| ID | Requirement | Issues |
|---|---|---|
| R01 | Python 3.14, uv/src package, Python 3.11 ML isolation and CPU CI | [P0-01](issues/P0-01.md), [P1-01](issues/P1-01.md) |
| R02 | Frozen exactly-20 corpus, golden episode, readonly NFS source and millisecond metadata | [P0-02](issues/P0-02.md), [P0-03](issues/P0-03.md) |
| R03 | Local SQLite WAL/FTS5, dedicated Qdrant, atomic artifacts and resumable worker | [P0-04](issues/P0-04.md), [P0-05](issues/P0-05.md), [P2-05](issues/P2-05.md), [P3-03](issues/P3-03.md) |
| R04 | Locked model revisions/licenses/head and measured RTX 5090 compatibility | [P1-H01](issues/P1-H01.md), [P1-01](issues/P1-01.md), [P2-02](issues/P2-02.md), [P2-03](issues/P2-03.md), [P2-04](issues/P2-04.md), [P3-02](issues/P3-02.md) |
| R05 | Real ASR word timing, window boundaries, episode-wide standard/exclusive diarization and attribution | [P1-02](issues/P1-02.md), [P1-03](issues/P1-03.md), [P1-04](issues/P1-04.md) |
| R06 | Human golden annotations, 200 timing samples, frozen 80/20 splits and metrics | [P1-05](issues/P1-05.md), [P1-06](issues/P1-06.md), [P1-07](issues/P1-07.md), [P1-08](issues/P1-08.md) |
| R07 | Clean exemplars, three-vector indexes, centroids and deterministic retrieval | [P2-01](issues/P2-01.md), [P2-05](issues/P2-05.md), [P2-06](issues/P2-06.md) |
| R08 | Human voice truth, valid calibration, host/non-host evaluation and uncertainty | [P2-07](issues/P2-07.md), [P2-08](issues/P2-08.md) |
| R09 | Human-only global identities, manual names, contradictions, undo and split | [P2-09](issues/P2-09.md), [P4-04](issues/P4-04.md) |
| R10 | Shared-inference transcription upstream extension and merged pin | [P3-01](issues/P3-01.md) |
| R11 | Replaceable inference, capabilities and safe vector-generation swaps | [P3-02](issues/P3-02.md), [P3-05](issues/P3-05.md) |
| R12 | Hybrid text search, speaker-aware chunks, filters and lexical fallback | [P3-03](issues/P3-03.md), [P3-04](issues/P3-04.md), [P4-02](issues/P4-02.md) |
| R13 | 40 reviewed Italian queries and lexical/dense/hybrid quality comparison | [P3-06](issues/P3-06.md), [P3-07](issues/P3-07.md) |
| R14 | Identical 20-minute local/cloud ASR subset and enforced US$10 total benchmark | [P3-H01](issues/P3-H01.md), [P3-05](issues/P3-05.md), [P3-07](issues/P3-07.md) |
| R15 | Private FastAPI/Jinja website, API v1, playback and bounded media access | [P4-01](issues/P4-01.md), [P4-02](issues/P4-02.md), [P4-06](issues/P4-06.md) |
| R16 | Archive-speaker/upload queries, formats/limits, selected speaker and one-hour expiry | [P4-03](issues/P4-03.md) |
| R17 | Candidate comparison, anonymous recurrence, appearances and chronology | [P4-04](issues/P4-04.md) |
| R18 | Evaluation/job dashboard, measured coverage/latency/cost and failure behavior | [P4-05](issues/P4-05.md), [P5-06](issues/P5-06.md) |
| R19 | Compose profiles, 4 GiB VRAM headroom and full ensemble contention benchmark | [P5-01](issues/P5-01.md), [P5-04](issues/P5-04.md) |
| R20 | Cloudflare Access on entire website, origin JWT and CSRF, operator deployment configuration | [P5-H01](issues/P5-H01.md), [P5-02](issues/P5-02.md) |
| R21 | Backups, Qdrant rebuild, interrupted job recovery and private evidence storage | [P5-03](issues/P5-03.md) |
| R22 | Public README, screenshots and three implementation-grounded architecture prompts | [P5-05](issues/P5-05.md), [P8-04](issues/P8-04.md) |
| R23 | 20-episode acceptance and quality/regression gates | [P1-08](issues/P1-08.md), [P2-08](issues/P2-08.md), [P3-07](issues/P3-07.md), [P5-06](issues/P5-06.md) |
| R24 | Gated 400 expansion, estimates and 40-total canary | [P6-01](issues/P6-01.md), [P6-02](issues/P6-02.md), [P6-03](issues/P6-03.md), [P6-04](issues/P6-04.md) |
| R25 | Gated historical voice-only pass, 20 canary and cross-year human labels | [P7-01](issues/P7-01.md), [P7-02](issues/P7-02.md), [P7-03](issues/P7-03.md), [P7-04](issues/P7-04.md) |
| R26 | Gated prioritized remaining ASR, bounded batches and final audit | [P8-01](issues/P8-01.md), [P8-02](issues/P8-02.md), [P8-03](issues/P8-03.md), [P8-04](issues/P8-04.md) |
| R27 | Luna implementation, Sol review/remediation, user release of every phase | [P0-06](issues/P0-06.md), [P0](issues/P0.md), [P1](issues/P1.md), [P2](issues/P2.md), [P3](issues/P3.md), [P4](issues/P4.md), [P5](issues/P5.md), [P6](issues/P6.md), [P7](issues/P7.md), [P8](issues/P8.md) |
| R28 | Idempotent GitHub setup, native parents/blockers, fields/views and no milestones | [P0-06](issues/P0-06.md) |

## Review follow-ups

Sol adds follow-ups under the same parent with stable review markers, native blockers, Executor/Kind/Priority and Order `phase*1000 + 500 + sequence`. They need not be forced into the original fixed catalogue: live native children and the private setup mapping are authoritative for added findings. Never renumber original IDs. Validate the combined live graph and preserve the previous-phase prerequisite, while keeping the follow-up independent of its own parent's closure.

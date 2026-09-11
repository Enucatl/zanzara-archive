# Sources and dated observations

Verified on **2026-09-11 UTC** while preparing this folder. Observations are bounded to the revision/environment stated; recheck mutable APIs, model availability, pricing and credentials immediately before execution. No model weights were downloaded, no GPU inference or paid calls ran, and no GitHub objects were created during package preparation.

## Repository and environment evidence

| Item | Observed evidence | Implication |
|---|---|---|
| Archive repository | `Enucatl/zanzara-archive`, main commit `fbc329d703ec376fbf7b093511416f1a85aa6cba` (2026-09-08, Create plan.md); clean working tree before this task | Original README is only a title; original plan preserved byte-for-byte |
| Repository visibility | Authenticated `gh api repos/Enucatl/zanzara-archive`: `private=false`, repository admin/push permitted | Private Project does not protect public issue content |
| GitHub credential | `gh auth status`: account Enucatl; scopes `gist`, `read:org`, `repo` | Project write scope absent; preflight/remedy required |
| Tooling | gh 2.100.0 (2026-09-03), Node v26.8.2, ffprobe 6.1.1 | Local validator and metadata capture were available |
| Archive mount | `findmnt -T /export/scratch/archive/zanzara` reports NFS4 | Mount archive read-only in application containers; keep SQLite/job state local |
| GPU | `nvidia-smi`: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 590.48.01 | Hardware present; model/CUDA compatibility and >=4 GiB headroom are still unmeasured release requirements |
| Frozen corpus | Read and hashed exactly the explicit 20 handoff filenames, checked source size/mtime before/after, probed codec/duration/channels/rate | [corpus-20.json](corpus-20.json) is actual metadata, not a suggested filename list |
| Golden episode | `260910-lazanzara.opus`, 6,108,584 ms (101.8097 min), 32,215,323 bytes, Opus/mono/48 kHz; SHA-256 `06de18da0691a19738bcda30dace2d5be87c8be651e9bd9ed8e56b5828f64536` | Original-time millisecond reference for all stages |
| Initial corpus total | 20 episodes, approximately 33.6063 hours, July 1–September 10 2026 | 400/4,000-episode durations in the original plan remain estimates |

The corpus file records original source sample rates; 16 kHz mono is a **derived** model input. Host archive mount was observed writable, so the container's read-only bind and source-safety checks are requirements, not claims about the current host mount.

## Model and inference references

| Primary source | Verified support and limit |
|---|---|
| [Parakeet v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Italian support and timestamped NeMo inference documented. No inference quality/CUDA compatibility claim is made here. |
| [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1) | Standard and exclusive diarization and local use documented; access acceptance remains a human task. |
| [ResNet293-LM model card](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) | Dedicated speaker-verification candidate exists; P1-01 must lock actual weights/preprocessing/dimension/license. |
| [ERes2Net registry entry](https://www.modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k) and [official 3D-Speaker repository](https://github.com/modelscope/3D-Speaker) | Registry page is dynamically rendered and yielded no inspectable model-card text. The handoff fixes `v1.0.2`; revision/download/checksum/license must be verified during P1-01. This is an explicit acquisition risk, not a verified downloaded checkpoint. |
| [Microsoft UniSpeech speaker verification releases](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | Release table distinguishes fixed pretraining from fine-tuned WavLM Large (`Fix pre-train = No`). Use the released verification checkpoint/head from the latter row. Download availability, checksum and runtime compatibility remain P1-01 blockers until tested. |
| [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) | Multilingual dense 1,024-dimensional embeddings documented; sparse/multivector modes are outside initial scope. |
| [OpenRouter embedding catalogue](https://openrouter.ai/api/v1/embeddings/models) | Listed `baai/bge-m3` at inspection. Listing does not prove vector equivalence to a local deployment or future availability. |
| [OpenRouter transcription documentation](https://openrouter.ai/docs/guides/overview/multimodal/stt) | Documents JSON audio requests, verbose responses, timestamp requirements and provider options. Per-model/provider Italian/timing/pricing capabilities still need pre-execution probes. |
| [shared-inference inspected revision](https://github.com/Enucatl/shared-inference/tree/04c2eddcdf75e7893dc7648134ef02ebbf9dda9a) | Git tree plus `src/shared_inference/client.py` and `models.py` read through authenticated GitHub API. Client exposes async complete/embed/rerank, typed usage/results, errors and tracing; no transcribe method at this revision. |

In shared-inference, relevant files are `client.py`, `models.py`, `errors.py`, `tracing.py`, and `tests/test_client.py`/`test_live.py`. The inspected client uses `niquests.AsyncSession`, `llm_span`, `record_result`, `_post` and `InferenceError`/`InferenceHTTPError`/`InferenceTimeoutError`; its embedding result includes ordered embeddings, usage, request ID and raw response. Re-read current upstream before editing and pin the merged transcription extension, not this historical baseline as the final integration. Preserve tracing conventions while keeping full payload traces private.

## Storage, access and GitHub references

| Primary source | Verified support and limit |
|---|---|
| [SQLite FTS5](https://sqlite.org/fts5.html) | FTS5 lexical search/BM25 underpins the selected lexical component. Tokenization and fusion are project decisions. |
| [SQLite WAL](https://sqlite.org/wal.html) | WAL's shared-memory requirements rule out network-filesystem database placement. |
| [Qdrant named vectors](https://qdrant.tech/documentation/manage-data/vectors/) | Multiple independently named vector spaces supported. Collection generations and canonical SQLite ownership are project decisions. |
| [Cloudflare origin JWT validation](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/) | Origin validation uses signature keys, issuer/audience and token validity. Whole-site protection and CSRF remain implementation/test work. |
| [GitHub sub-issue REST API](https://docs.github.com/en/rest/issues/sub-issues) | Parent URL uses issue number; add-child payload uses numeric `sub_issue_id`; supports native parent/list read-back. |
| [GitHub dependency REST API](https://docs.github.com/en/rest/issues/issue-dependencies) | `POST /issues/{issue_number}/dependencies/blocked_by` takes prerequisite numeric `issue_id`; inverse blocking and blocker reads are documented. |
| [GitHub CLI Project manual](https://cli.github.com/manual/gh_project) | Installed `gh project --help` states `project` scope requirement and refresh remedy; field/item/link commands available. |
| [GitHub GraphQL reference](https://docs.github.com/en/graphql/reference) | Authenticated schema introspection, not guessed mutation signatures, confirms create/update Project views and update fields. |

Schema observation: `CreateProjectV2ViewInput` contains projectId/name/layout/configuration; `UpdateProjectV2ViewInput` also supports filter; `ProjectV2ViewConfigurationInput` exposes visibleFieldIds only. This supports API view creation/filter/columns, while saved sorting/grouping needs the explicit UI step in G3. `ProjectV2SingleSelectFieldOptionInput` includes optional existing id plus required name/color/description. Recheck live schema at execution; do not silently omit required view configuration if API support changes.

The GitHub REST documentation showed API version `2026-03-10` during inspection. No write endpoint was exercised. Source availability and schema inspection do not establish successful live Project creation. The final setup report must contain independent read-back evidence.

## Decisions versus measured findings

The user handoff supplies model candidates, workload limits, product targets and phases. D1–D12/E1–E7 supply deterministic implementation details, not results: ASR window defaults, matching/fusion algorithms, evidence minima, upload expiry, budget reservation, quality thresholds and release gates must be implemented and tested later. The observed GPU, source files and API shapes are the only environment measurements made for this package.

## Package verification

On 2026-09-11, `node planning/validate.mjs` passed for 66 issue bodies, 172 blocker edges, 28 requirement mappings, exactly 20 frozen episodes and all local Markdown links/anchors. Temporary-copy negative checks confirmed rejection of an unknown blocker, own-parent dependency, cross-child cycle, broken section link, changed corpus membership and a required Human task assigned to Luna. `node --check planning/validate.mjs` passed; `git diff --exit-code -- plan.md` confirmed the original plan was unchanged.

The proposed SKILL.md was extracted into a temporary directory and passed the bundled skill-creator `quick_validate.py` using `uv run --no-project --with pyyaml`. This validates skill structure, not live Sol review behavior or GitHub mutations; P0-06 retains those behavioral acceptance checks. Application, GPU, paid and browser tests are future issue requirements, not tests run on this documentation-only package.

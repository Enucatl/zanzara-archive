# Zanzara Archive planning package

This folder is the implementation specification and GitHub setup handoff. It contains the complete issue bodies for all phase parents and children, including explicit human tasks, and is the local source for publication and reconciliation. Preserve [the original plan](../plan.md) as background; the decisions here supersede it.

## Reading order

1. Read this file for scope, decisions and execution boundaries.
2. Read [SYSTEM-DESIGN](SYSTEM-DESIGN.md) (D1–D12) and [EVALUATION](EVALUATION.md) (E1–E7).
3. Read [GITHUB-SETUP](GITHUB-SETUP.md) for permissions, publication, native relationships, reconciliation and verification.
4. Read [BACKLOG](BACKLOG.md), [manifest.json](manifest.json) and the linked [issue bodies](issues/P0.md). The manifest defines titles, parents, blockers, executor, kind, priority, order, labels and Project configuration.
5. Read [REVIEW-SKILL](REVIEW-SKILL.md) for the complete proposed repository skill and [SOURCES](SOURCES.md) for dated observations and technical references.

[corpus-20.json](corpus-20.json) freezes the actual 20 episode filenames and their recorded metadata. [validate.mjs](validate.mjs) checks the package locally with Node >=22 and no dependencies. This is documentation tooling, not a JavaScript application dependency. P0-02 imports the corpus and establishes safe source paths; the persistent archive is not reverified during routine work. P0-06 will install the proposed review skill.

```bash
node planning/validate.mjs
```

The checks validate IDs, issue completeness/metadata, parent/blocker graph, release-gate structure, requirement mappings, local links/anchors and the exact frozen corpus. They cannot establish model quality, human review, hardware compatibility or live GitHub configuration. Test commands and CLI names in issue bodies are future contracts to implement in their owning issues; they do not claim those commands exist in the current repository.

## Agreed scope and superseding decisions

| Area | Binding decision | Supersedes original plan |
|---|---|---|
| Corpus | Exactly the supplied latest-20 snapshot, 2026-07-01–2026-09-10; golden `260910-lazanzara.opus` | Approximate/representative 10–20 initial selection, including historical episodes |
| Application | Python 3.14, uv/src package, FastAPI/Jinja/vanilla JS; separate locked Python 3.11 ML services | Unspecified application/runtime |
| Canonical state | SQLite WAL/FTS5 on local volumes, one durable worker | PostgreSQL preference and unspecified orchestration |
| Vectors | Dedicated Compose Qdrant; individual exemplars plus separate centroids, three named speaker spaces | Existing/shared Qdrant deployment |
| Models | Parakeet v3, Community-1, ResNet293-LM, fixed ERes2Net v1.0.3, released fine-tuned WavLM Large verification head; dense BGE-M3 | Open ERes2Net/WavLM checkpoint choice |
| Ensemble | Full three-model baseline; centroid union from all three, deterministic one-to-one exemplar matching, conditional logistic calibration | Cascade as initial alternative, vague matching/calibration |
| Identity | Human-confirmed global membership, manual names, reject/uncertain/undo/split with contradiction checks | Automatic high-confidence linking/clustering |
| Excerpts/time | >=3 seconds, prefer 8–15, max ten, exclude overlap/250 ms transitions; persisted integer milliseconds | Broader duration guidance and seconds-based examples |
| Evaluation | Human golden truth, fixed splits, reviewed voice/text labels, measurable release gates | Suggested benchmark without complete annotation/split procedure |
| Paid benchmark | Identical stratified 20-minute development subset, MAI and Voxtral, US$10 total including probes/retries | 3–5-hour multi-provider comparison including Scribe |
| Cloud | shared-inference embeddings plus bounded upstream transcription extension; validated timing and index equivalence | Unspecified cloud replacement interface |
| Website | Entire website behind Cloudflare Access with origin JWT/CSRF checks; bounded ephemeral uploads | Incomplete runtime/security/search contract |
| Expansion | Separate P6/P7/P8 approvals and canaries after released initial 20 | Earlier broader phase outline |
| Delivery | Private owner Project, native children/blockers, Luna implementation, Sol integrated review, explicit user release of every phase | Unspecified implementation workflow |

These are fixed initial candidates and acceptance targets, not claims that the models are optimal or will pass. Missing checkpoint access, GPU compatibility or quality evidence becomes a blocker. No silent substitutions, fabricated timing/confidence, vector-space mixing or automatic identity merges.

The specification resolves execution gaps without changing scope: P1-H01/P3-H01/P5-H01 make operator inputs explicit; P7-01 runs the bounded 20 historical canary before its human evaluation; P6-03 obtains human annotations for new canary material as needed. Five contiguous golden blocks fix 80/20 development/held-out membership. Expansion uses the original date cutoff for nested 20/40/400 cohorts. Deterministic defaults and evidentiary minima are documented in D6–D9 and E3; changing them requires a recorded design decision and review, not silent tuning on held-out data.

## Instructions for the GitHub setup agent

Under GPT 5.6-Luna, use GITHUB-SETUP exactly: validate all local inputs; preflight credentials before any creation; reconcile the private owner Project/fields/labels; publish parents then children; attach native sub-issues and blocking dependencies; add Project items/fields; configure views; read all objects back. Keep a private atomic mapping of stable IDs to issue numbers/database IDs/node IDs/Project item IDs. Reruns must create no duplicates and preserve human edits/comments.

Current credentials lack the Project scope; `gh auth refresh -s project` is the operator remedy during setup. Project view sorting/grouping also requires the explicit UI action documented in G3 with the inspected API schema. These are future setup requirements, not blockers to producing this folder. Planning files need a published commit before issue links can point to that commit. Do not claim setup complete while permissions, saved views, links or native relationships remain unverified.

Stop at the verified GitHub structure and creation report. Do not begin P0 implementation in the setup run. The private Project does not hide public repository issues: raw archive audio, annotations, identities, embeddings, keys and sensitive traces stay in private local storage.

## Luna implementation and phase review

Select only eligible open Luna children, including registered review follow-ups, using native blockers and completion evidence. Sort by phase, priority, then numeric Order. A child waits for the **preceding** phase's release, never its own phase's closure. Human tasks and phase parents are not unattended implementation work. A prerequisite closed as `not_planned` needs an explicit user-approved replacement.

Implement one bounded Luna issue, run its required checks, commit the result directly to `main`, push `origin/main`, and record the commit and evidence on the child issue. Mark that child Done only after its acceptance checklist passes; Operator and other Human issues remain explicit human actions. There is no issue-level PR review gate. At each phase boundary present `$review-phase <parent-issue-number>` for invocation using the model currently enabled in chat. The skill reviews integrated behavior, verifies current evidence and creates deduplicated native remediation children; it does not fix them or release the phase. Every phase stays open until a current PASS and the user's `Release Pn at <reviewed-commit>` record.

## Package completion checklist

- [x] Every handoff phase/child ID retained, with full bodies and explicit role/prerequisites.
- [x] Operator model access, benchmark credentials, annotations, deployment and release actions represented.
- [x] Requirement-to-issue mapping, acyclic dependencies and no own-parent closure requirement.
- [x] Actual frozen 20-file membership and recorded media metadata included.
- [x] Budget, timestamp capability, vector generation and human identity invariants specified.
- [x] Resumable GitHub publication and native read-back verification documented.
- [x] Sol follow-up selection and user release gates remain extensible.
- [x] Proposed review skill and all required future README/architecture prompt deliverables specified.
- [ ] Future setup run: permissions, published planning links, private Project, issues, relationships, views and ID map verified.
- [ ] Future implementation: P0–P8 executed and individually released by the user.

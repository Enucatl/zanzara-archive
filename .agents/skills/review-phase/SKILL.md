---
name: review-phase
description: Review integrated Zanzara Archive phase behavior and publish concrete findings when requested or warranted by material integration risk. Phase reviews are optional quality checks, not mandatory release ceremonies.
---

# Review a Zanzara Archive phase

Use the currently enabled model. Read `planning/README.md`, the requested parent issue, relevant open children and actual deliverable dependencies. Inspect the design, evaluation, and code sections needed to assess the phase. Reuse session context and inspect changed scope; do not run the GitHub setup preflight or reconcile the entire Project as a review prerequisite.

Record the code state and inspect integrated success and material failure/recovery behavior. Preserve local changes; an isolated checkout is useful only when those changes make the review ambiguous. Focus on observable defects, unmet current requirements, data loss, security, timing, model/vector compatibility, human decisions, and actual quality evidence. Phase labels, missing administrative hashes, and an unrelated new commit are not product defects.

Use existing checks and reports when their code behavior, model/configuration, and inputs remain applicable. Run a focused missing check once when it would resolve a real uncertainty. Do not rerun GPU jobs or full suites merely to obtain results at the latest commit. Do not rehash trusted archive sources, checkpoint caches, or evidence files. Paths and relevant run/model/configuration IDs suffice for routine evidence; retain existing hashes used as identity or cache keys. Request byte verification only for changed inputs, recovery, or suspected corruption. Human gold, held-out separation, real GPU measurements, and honest limits on evidence remain necessary where the deliverable depends on them.

Publish only actionable findings. Before creating a follow-up, check the relevant existing findings for the same root cause and reuse a matching open issue. A short issue needs the defect, reproduction/evidence, expected result, focused acceptance check, and real deliverable dependencies. Use a stable `<!-- zanzara-review:Pn:descriptive-slug -->` marker for deduplication, attach it to the phase, and add its Project item when available. Avoid generic template expansion, whole-project readback, and phase-parent blocker edges. A Project metadata failure need not prevent publishing a useful finding; report the limitation.

Publish one concise report with the reviewed scope, findings/links, relevant checks and results, evidence paths/run IDs, and remaining uncertainties. Update a prior report when this is a continuation of the same review, preserving discussion. State whether the deliverable is ready or which concrete defects block it. No exact-commit PASS, separate `Release Pn` comment, or named-model ceremony is required. Optional/deferred children do not block a completed milestone. A material subsequent change needs review of the affected behavior; unrelated changes do not invalidate the report.

Invocation authorizes review reports and bounded remediation issues. It does not by itself authorize implementing fixes, human annotation or identity decisions, accepting model terms, spending, public deployment, or bulk expansion. Keep detailed private evidence and sensitive data local. Read `planning/GITHUB-SETUP.md` only when a structural GitHub edit actually needs its API guidance.

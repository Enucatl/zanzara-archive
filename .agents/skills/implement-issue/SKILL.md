---
name: implement-issue
description: Implement an eligible Zanzara Archive issue or a tightly related authorized batch, verify the affected behavior, commit to main, push, and record completion. Use for implementation, evaluation, and review fixes; human actions still require the human.
---

# Implement Zanzara Archive work

Read `planning/README.md` for current scope and workflow policy, then the selected live issue and its direct deliverable dependencies. Read the relevant design, evaluation, and code sections as needed. Reuse context already established in the session; a fresh isolated agent and a full Project reconciliation are not required. Use the current model; `Executor: Luna` is a workflow label, not a model requirement.

Select open implementation, evaluation, or review work whose actual inputs are available. Exclude manifest entries marked `deferred: true` or `retired: true` from automatic selection. Without a requested issue, choose authorized ready work by global numeric Order, with priority as a tie-break, not phase-first order. Phase parents are milestones, not prerequisites or release ceremonies. Deferred or superseded tasks do not block their replacement. Respect the current scope in `planning/README.md`; removing a phase gate does not authorize deferred work, paid calls, public deployment, or bulk expansion.

Implement the requested outcome, inspecting callers and preserving unrelated changes. Batch tightly related authorized issues when this avoids repeated setup and checks. For work that benefits from a visible claim, set the selected Project item In progress; do not require a separate start comment or inspect every historical issue.

Run focused checks once for the affected behavior and format/lint changed code. Run broader tests only for changes affecting shared behavior or unresolved failures. Documentation changes need relevant link/structure checks, not CPU or GPU inference suites. Use existing results when the tested behavior, model, configuration, and inputs have not changed. A new unrelated commit does not invalidate evidence.

Trust the fixed workstation's recorded archive and model inputs during routine work. Do not rehash source audio, checkpoint caches, or evidence trees. Record the commit, relevant model/configuration or run IDs, commands/results, and output paths; preserve cheap identity/cache hashes already used by the application. Full byte verification is for import, changed inputs, recovery, or suspected corruption. Real GPU behavior and human listening/identity decisions still need real evidence.

Stage only the intended files, make a concise imperative commit, and push directly to `origin/main`. A successful push is sufficient; inspect the remote commit if the push result is ambiguous. Preserve user changes and never force-push. Record one concise completion note per issue or batch with the commit, checks, result, and any material limitation. Mark completed issues and Project items Done when their deliverables are satisfied; mark a concrete missing input or human action Blocked.

Continue authorized ready work without a separate phase review/release step. Use `$review-phase` when the user requests an integrated review or a material integration risk warrants it. Human annotation, identity approval, model access/terms, spending, public exposure, and newly scoped bulk work retain their actual authorization requirements. Keep audio, annotations, identities, embeddings, credentials, and detailed private evidence out of public issues.

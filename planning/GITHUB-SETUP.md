# GitHub setup runbook

Use this runbook for initial setup or an actual structural issue/Project change. Ordinary implementation and review read the selected issues and direct dependencies; they do not repeat setup or reconcile the whole Project. The current scope and delivery policy in [README](README.md) override historical phase-release boilerplate.

## G1 — Inputs and credential preflight

The repository is `Enucatl/zanzara-archive`; the private owner Project is **Zanzara Archive — Evaluation and Search**. Issues and comments are public. Keep raw audio, annotations, identities, embeddings, credentials, and detailed evidence in private local storage.

Read the relevant [manifest](manifest.json) entries and live issues. Run `node planning/validate.mjs` when modifying the planning structure. For first access or an authentication failure, inspect `gh auth status` and the permissions needed for the intended action. Request the missing scope only when needed (`gh auth refresh -s project` for an OAuth token lacking Project access); never log tokens. Successful existing access is sufficient for routine work.

Use a planning commit and stable issue IDs for provenance. Do not hash the manifest, issue bodies, mapping, or reports as an administrative requirement.

## G2 — Mapping, reconciliation and concurrency

Match managed issues using exact `<!-- zanzara-plan:ID -->` markers. Search relevant existing issues before creating new ones; include closed issues when recovering an uncertain create result. A duplicate marker is a conflict to resolve, not permission to create another issue.

A private ID map is a convenience for the current Project number/node ID, field option IDs, issue numbers, and Project item IDs. Reuse existing mappings and discover missing IDs from API responses. There is no required checksum ledger, exclusive setup lock, or per-mutation mapping snapshot for this single-operator project.

Before changing an issue body, read the current body and preserve human edits and comments. Compare actual body text to the intended edit; hashes do not improve that decision. After an ambiguous network result, read the affected object before retrying. Respect API rate limits. Preserve existing statuses unless the requested work changes them.

## G3 — Project, fields, labels and views

Reuse the existing private Project and its fields. Initial setup creates one owner Project, links this repository, and confirms privacy once before adding items. Do not create milestones in addition to phase-parent issues.

| Field | Type | Options |
|---|---|---|
| Status | Existing single select | Backlog, Blocked, Ready, In progress, In review, Done |
| Phase | Single select | P0, P1, P1R, P2, P3, P4, P5, P6, P7, P8 |
| Kind | Single select | Phase, Implementation, Evaluation, Operator, Review follow-up |
| Executor | Single select | Luna, Human |
| Priority | Single select | P0, P1, P2 |
| Order | Number | Current delivery order |

`Executor: Luna` means agent-owned work and does not select a runtime model. Phase is a grouping label, not a sequencing constraint. Retain existing labels and optional saved views when useful; cosmetic view settings do not block implementation.

Use `gh project field-list`, `gh project item-list`, and Project GraphQL only for fields/items affected by the requested change. Reuse option IDs. Pagination is needed when the relevant connection has more results, not as a reason to fetch every unrelated object. For an unavailable UI-only view setting, document the limitation and continue useful work.

## G4 — Publish parents, then children

Create a missing phase parent before its children. Use complete intended bodies with stable IDs and current scope. Submit JSON through `gh api --input` or use `gh issue create --body-file` / `gh issue edit --body-file` to preserve actual newlines.

Published links must resolve: use canonical issue URLs for known issues and GitHub repository URLs for planning files. Read live body edits before replacement. Preserve historical issues and comments; mark superseded or deferred work truthfully rather than deleting history. A managed relationship section can provide human-readable links, but it need not duplicate every API detail.

## G5 — Native relationships and project items

Native parent/child relationships group the work. Native blockers represent actual required deliverables only. Do not add previous-phase blockers or a child's own parent as a blocker. A phase parent may depend on its core deliverables; optional/deferred children remain grouped without blocking the milestone.

REST issue numbers identify URL paths. The numeric issue database ID is used in relationship payloads; GraphQL node IDs and Project item IDs are different identifiers. Example commands with populated values:

```bash
gh api --method POST "repos/Enucatl/zanzara-archive/issues/$ZANZARA_PARENT_NUMBER/sub_issues" -F sub_issue_id="$ZANZARA_CHILD_DATABASE_ID"
gh api --method POST "repos/Enucatl/zanzara-archive/issues/$ZANZARA_BLOCKED_NUMBER/dependencies/blocked_by" -F issue_id="$ZANZARA_PREREQUISITE_DATABASE_ID"
```

Read the affected relationships first, apply the intended changes, and avoid duplicate edges or cycles. Do not replace an unrelated parent. Add a missing Project item once and set the fields needed for selection and visibility. A platform limit or unavailable cosmetic field is a documented limitation, not a fabricated product dependency.

## G6 — Read back and completion evidence

Read back the affected issue body, relationship, or Project field once after structural changes. Compare it to the intended result. A whole-repository graph audit, inverse-edge inventory, body hashes, mapping checksums, and a second zero-mutation rehearsal are not routine requirements.

For a large initial import, one summary of created/reused issue IDs, Project URL, and unresolved discrepancies is enough. For ordinary edits, report what changed and any remaining limitation. Use a successful Git push as publication evidence; inspect the remote SHA if the push result is ambiguous.

## G7 — Luna implementation handoff and release rules

Use [implement-issue](../.agents/skills/implement-issue/SKILL.md) for authorized implementation and [review-phase](../.agents/skills/review-phase/SKILL.md) when an integrated review is requested or warranted. Read only the relevant task, direct deliverable dependencies, and affected contracts. Reuse session context; batch tightly related work when useful.

Exclude manifest entries marked `deferred: true` or `retired: true` from automatic selection. Among authorized ready work, use global numeric Order, with priority as a tie-break, not phase-first order. Respect actual dependencies even if a Project item says Ready. A superseded issue closed as not planned does not invalidate a documented replacement.

Run relevant checks once and reuse unchanged evidence. Routine evidence consists of a commit, affected checks/results, model/configuration or run IDs, and private artifact paths. Full checkpoint/archive/evidence rehashing is reserved for import, changed inputs, recovery, or suspected corruption. Real inference and human truth cannot be replaced by mocks.

Complete satisfied issues and milestones with concise results. There is no mandatory phase review, exact-commit PASS, model switch, or separate `Release Pn` comment. The removal of those ceremonies does not authorize human annotation/identity decisions, acceptance of access terms, spending, public deployment, or newly scoped bulk work. Keep those requirements attached to the actions that need them.

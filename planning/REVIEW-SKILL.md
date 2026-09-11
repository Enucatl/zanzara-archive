# Repository phase-review skill specification

P0-06 installs the proposed content below at `.agents/skills/review-phase/SKILL.md` and verifies repository-relative references. This planning package does not install or invoke it. The operator selects GPT 5.6-Sol before invoking `$review-phase <phase-issue-number>`; a skill cannot switch its own model. Model choice is a workflow requirement, not an API identifier claim.

The skill is limited to integrated phase review and GitHub remediation/report publication. Invocation authorizes those follow-up issues and review comments. It does not authorize fixes, merging PRs, closing the phase, relaxing targets, or starting the next phase. It should inspect current live state, never trust a static child list alone.

## Proposed SKILL.md

Copy the contents of this fence verbatim, replacing no placeholders:

````markdown
---
name: review-phase
description: Review an implemented Zanzara Archive phase under GPT 5.6-Sol, verify integrated code and evaluation evidence, and publish deduplicated remediation sub-issues and a release-readiness report. Use when asked to review a phase issue; do not use for routine implementation or isolated PR review.
---

# Review a Zanzara Archive phase

Run under GPT 5.6-Sol. If the active model is known to differ, stop before publishing and ask the operator to select Sol and invoke again. If the runtime does not expose model identity, record the operator's declared model and the verification limitation; never claim to have switched models. Input is the parent issue number in `Enucatl/zanzara-archive`.

Read `planning/README.md`, `planning/SYSTEM-DESIGN.md`, `planning/EVALUATION.md`, `planning/GITHUB-SETUP.md` and the matching manifest parent. Resolve paths from the Git repository root. Apply the setup runbook's credential preflight, stable-marker reconciliation and native relationship procedures. Read all pages of current children, dependencies, comments, linked PRs, review reports and project fields, including follow-ups absent from the original manifest. A permission failure must produce a local draft report and an explicit publication blocker, not a claimed successful review.

## Establish the reviewed state

1. Resolve the parent by number and stable marker; reject a child or unrelated issue as input. Read its release criteria and preceding-phase release evidence. Record repository HEAD, default-branch commit, working-tree state and reviewed commit. Review a concrete commit in a clean isolated checkout if local edits would make evidence ambiguous; preserve user changes.
2. Fetch all native children and blockers with pagination. Check close reason `completed`, merged PR commits and acceptance evidence. `not_planned` is not completion unless the user explicitly approves a replacement and its evidence. Include unresolved review follow-ups in readiness. Detect human tasks marked done without a human action record.
3. Read relevant current code and contracts across the whole phase, not just PR diffs. Trace at least one integrated success path and material failure/recovery path. Check source/model/generation isolation, real timing, human identity decisions, privacy, paid-call gating and phase-specific requirements. Do not run paid experiments or bulk processing as a side effect of review.
4. Verify evidence files exist, checksums match, and commands/results apply to the reviewed commit and frozen corpus/reference/splits. A newer implementation without current results is missing evidence. Run bounded relevant CPU checks where useful; inspect actual GPU/human/paid artifacts instead of substituting mocks for them. Respect the held-out protocol; do not tune after inspecting held-out metrics. Report missing resources as blockers.

## Publish concrete findings

A finding must describe an observed defect, unmet requirement or missing required evidence, with affected code/contract and reproduction. Assign priority P0 (gate/correctness/security/data integrity) or P1 (other required work). Do not file speculative cleanup.

Before creating anything, search all open and closed phase children and previous reports for the same root cause. Reuse an unresolved matching finding. Use a stable marker `<!-- zanzara-review:Pn:descriptive-slug -->`; keep it stable across reruns. If a completed finding has recurred, reopen it with evidence rather than duplicate it, unless it is a distinct cause requiring a new issue. Preserve comments, user edits and unrelated fields. Conflicting content requires a reported conflict, not overwrite.

Each follow-up body contains outcome/user requirement, parent, explicit prerequisites, exact design/evaluation sections, bounded steps, input/output and failure behavior, exclusions, acceptance checklist, commands, required evidence, stop conditions and completion rule. Use `planning/issues/` bodies as the format. Set Executor Luna for implementation/evaluation fixes or Human for required annotation/credentials/release actions. Never ask Luna to manufacture human truth.

Attach each follow-up with the native sub-issue API under the same parent; wire real blocker edges with the dependency API, then add its Project item and Phase, Kind=`Review follow-up`, Executor, Priority, Order and Status. Give it order `phase*1000 + 500 + next_followup_sequence` and add links in the parent/report. Include the previous phase's released parent as blocker, never its own parent. Make the phase parent blocked by this follow-up, not the reverse. Validate cycles before mutations. Update the private mapping/report with new IDs; new findings extend the phase without changing its original stable IDs.

## Report and stop

Publish one report comment per phase/reviewed commit using marker `<!-- zanzara-review-report:Pn:commit-sha -->` with the actual commit in the marker. On rerun reconcile that comment; preserve discussion and state which previous findings are fixed or still open. Include:

- reviewed commit, phase, date, declared/verified model and review scope;
- children/PRs reviewed and completion/close reasons;
- commands and evidence paths/checksums, real vs synthetic provenance;
- integrated contract/behavior findings and links to remediation issues;
- metric values, reference/split/model hashes, sample sizes and relevant thresholds;
- explicit **PASS — awaiting user release** or **BLOCKED**, with missing evidence/actions.

PASS requires every child/follow-up completed, required PRs merged, all phase acceptance criteria satisfied and evidence current. Set the parent's project Status to In review when awaiting release or Blocked when findings prevent it; keep the issue open in both cases. Do not implement findings, merge PRs, relax thresholds or start another phase. State the exact release action: the user posts `Release Pn at <reviewed-commit>` after this report, then the parent may be closed as completed and marked Done. Any subsequent relevant code, reference or child change makes the report stale until rerun.
````

## Validation required in P0-06

Validate frontmatter/name and all referenced repository paths. Exercise read-only fixtures for: a clean phase; an unmerged child PR; closed-not-planned prerequisite; missing GPU measurements; unmet WER; new merged code after a prior pass; an existing open finding; a completed recurring finding; and a human annotation blocker. Demonstrate that generated mutations create one native child and the correct blocker direction, never a cycle or duplicate. Test against fixture/API stubs before any real issue publication. Record model declaration accurately and demonstrate that a PASS leaves the parent open. No automated test should pretend a model self-switch occurred.

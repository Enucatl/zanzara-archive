---
name: implement-issue
description: In a fresh isolated sub-agent, implement one eligible Zanzara Archive Luna child issue, commit it directly to main, push it, and record acceptance evidence. Use for bounded implementation, evaluation, or review-follow-up work; do not use for Human tasks, phase release, or phase review.
---

# Implement one Zanzara Archive issue

## Execution boundary

Run this skill in a fresh sub-agent, with no inherited coordinator/main-thread context. The coordinator must pass only this skill prompt and the requested issue number or stable ID; use the equivalent of `fork_turns=none` when delegating. Reconstruct all repository, GitHub, dependency, and evidence context from the live workspace and the instructions below. Do not rely on summaries, claims, or decisions from the coordinator thread.

Run under GPT 5.6-Luna. This skill cannot switch its own model. Input may be an issue number or stable ID; without one, select the next eligible child from the live GitHub Project.

## Select exactly one issue

Read `planning/README.md`, `planning/SYSTEM-DESIGN.md`, `planning/EVALUATION.md`, `planning/GITHUB-SETUP.md`, the manifest, and the selected issue body. Read live issues, native blockers, comments, follow-ups, and Project fields before claiming work.

An eligible issue is open, has Executor `Luna`, and has Kind `Implementation`, `Evaluation`, or `Review follow-up`. Exclude phase parents, Operator/Human work, and any issue with an incomplete native blocker. A blocker is complete only with close reason `completed` and acceptance evidence. A preceding phase additionally needs a current Sol PASS and the user's release comment. A `not_planned` prerequisite never satisfies a dependency without an explicit user-approved replacement. The current phase parent stays open and does not block its own children.

Choose the lowest phase, then priority (`P0`, `P1`, `P2`), then numeric Order. If an issue was supplied, verify that it meets these rules rather than bypassing selection. Never work on more than one issue in a run.

Claim the issue with a comment containing the stable ID, agent/model declaration, and start commit, then set its Project Status to `In progress`. Preserve existing human comments, edits, and unrelated working-tree changes.

## Implement and verify

Inspect the current code and implement only the bounded outcome in the issue body. Preserve the interfaces, data ownership, provenance, timing, budget, privacy, and failure behavior in the planning package. Do not add speculative cleanup or replace a required model/provider with an unrecorded substitute.

Run every verification command in the issue body and the common checks established by P0 (`uv run ruff check .`, `uv run ruff format --check .`, and the relevant `uv run pytest` commands). Real GPU, paid, credential, deployment, and human-review requirements cannot be faked with CPU tests or mocks. If a required resource or human action is missing, set the issue `Blocked`, state the exact operator action, and do not claim completion.

## Commit directly to main

Before writing, confirm the tree is clean or that unrelated user changes can be preserved. Stage only files for this issue. Create one imperative commit with a concise subject, then push it directly with `git push origin main`. Never force-push, reset, rebase, or overwrite unrelated work. Verify `git rev-parse HEAD` equals `git ls-remote origin refs/heads/main` before reporting success. If a push outcome is uncertain, inspect the remote SHA before retrying so duplicate commits are not created.

Comment on the child issue with the commit SHA, commands and results, acceptance evidence paths/checksums, limitations, and any real-versus-synthetic provenance. Set the child Project Status to `Done` only when every acceptance item is satisfied and the commit is on `origin/main`; otherwise leave it `In progress` or `Blocked` with the next action.

## Stop at the phase gate

When all current-phase children and follow-ups are complete, stop and present `$review-phase <phase-parent-number>` for execution using the model currently enabled in the chat. Do not close or release the phase parent, start the next phase, perform excluded Human actions, or run unapproved paid/bulk work. The phase parent remains open until a current PASS is published and the user posts `Release Pn at <reviewed-commit>`.

If the tree conflicts, a dependency is ambiguous, acceptance evidence is missing, or the remote cannot be verified, stop with a precise blocker and leave the issue unclaimed or in the truthful status. A local or unpushed commit is never completion.

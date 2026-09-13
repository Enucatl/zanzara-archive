# GitHub setup runbook

The setup agent's deliverable is a verified GitHub structure, ID mapping and creation report. It must stop before implementation. The local planning package is authoritative input; do not redesign issues or infer permission to release a phase. This runbook describes a future setup run, not objects already created.

## G1 — Inputs and credential preflight

Use `Enucatl/zanzara-archive`, owner `Enucatl`, and the exact project title **Zanzara Archive — Evaluation and Search**. Create one private owner-level Project, link this repository, and create **no milestones**. Repository issues are public because the repository is public; their bodies contain specifications and synthetic examples only. A private Project does not make issues or comments private.

From the repository root, before any GitHub mutation:

```bash
node planning/validate.mjs
gh auth status
gh api repos/Enucatl/zanzara-archive --jq '{full_name,private,permissions}'
gh api graphql -f query='query { viewer { login } user(login:"Enucatl") { id projectsV2(first:100) { nodes { id number title public } pageInfo { hasNextPage endCursor } } } }'
gh api --paginate 'repos/Enucatl/zanzara-archive/issues?state=all&per_page=100'
```

Prerequisites: Node >=22 for the dependency-free local validator, `gh`, `jq`, repository Issues write permission and owner Project write permission. Authenticate as the intended owner or a principal with the equivalent explicit rights. Current inspected OAuth token has `repo`, `read:org`, `gist` but lacks `project`; the operator remedy is **`gh auth refresh -s project`**. Fine-grained tokens need repository Issues write plus appropriate owner Projects write permissions; OAuth scope alone is not the only possible credential scheme. Never log tokens. Check authenticated identity, repository permissions and successful Project reads; confirm Project write scope/permission before creating anything. A failed preflight stops with zero new objects and exact remedy. Do not refresh auth interactively on the user's behalf or downgrade to a repository-only issue dump.

Read [manifest.json](manifest.json), all its bodies, [BACKLOG](BACKLOG.md), and [REVIEW-SKILL](REVIEW-SKILL.md). Record the planning commit (if uncommitted, a sorted file-hash snapshot) and SHA-256 of the manifest/bodies. Run the local validator first; confirm that every issue and human action is represented and that graph edges point from blocked issue to prerequisite.

## G2 — Mapping, reconciliation and concurrency

Keep execution state outside the public repository, for example in a caller-selected local directory recorded as `ZANZARA_SETUP_STATE`. Use a single setup process and an exclusive lock there. Atomically save `github-map.json` after every successful mutation. Its shape is:

```json
{
  "schema_version": 1,
  "repository": "Enucatl/zanzara-archive",
  "planning_sha256": "actual-manifest-sha256",
  "project": {"number": 1, "node_id": "actual-project-node-id", "url": "actual-project-url"},
  "fields": {},
  "views": {},
  "issues": {
    "P0-01": {
      "number": 1,
      "database_id": 123,
      "node_id": "actual-issue-node-id",
      "project_item_id": "actual-project-item-id",
      "source_body_sha256": "actual-source-hash",
      "published_body_sha256": "actual-published-hash"
    }
  }
}
```

Values above demonstrate field types, not real IDs. REST issue **number** is used in URL paths; REST numeric **id** is the database ID used in relationship payloads; **node_id** is for GraphQL; the Project **item ID** is a fourth independent ID.

On first run and reruns, fetch **all pages of all open and closed repository issues**, exclude pull requests, and match exact `<!-- zanzara-plan:ID -->` markers. Titles are not identity. Zero matches permits creation; one reconciles; multiple are a conflict and block that operation. Recover unknown outcomes (network timeout after POST) by reading markers again before retrying. A conflict is not permission to delete or duplicate an issue. Respect secondary rate limits and Retry-After; do not create concurrently without a serialized mapping checkpoint.

Discover Projects by exact owner/title and then stored ID/description marker `zanzara-plan:project:v1`. Multiple matches or an unrelated existing same-title project are conflicts. Create only when absent; preserve unrelated owner projects, labels, comments and fields. For issue-body updates use three-way comparison: source body, last published hash, and current remote body. Update managed content only when the remote is unchanged from the last publication; otherwise report the conflicting edit. On first adoption verify the body rather than assuming the marker grants overwrite permission. Human comments are never replaced. Reruns must not reset completed/in-progress statuses or reopen closed issues automatically.

## G3 — Project, fields, labels and views

Create when absent, immediately enforce privacy, then link and read back:

```bash
gh project create --owner Enucatl --title 'Zanzara Archive — Evaluation and Search' --format json
gh project edit "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --visibility PRIVATE --description 'zanzara-plan:project:v1 — phased implementation with human release gates'
gh project link "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --repo Enucatl/zanzara-archive
```

Capture IDs from responses; never hardcode them. Stop immediately if privacy cannot be verified. Project creation is private by default but verify `public=false`. No issues are added before this check.

| Field | Type | Exact options/order |
|---|---|---|
| Status | Existing single select | Backlog, Blocked, Ready, In progress, In review, Done |
| Phase | Single select | P0, P1, P1R, P2, P3, P4, P5, P6, P7, P8 |
| Kind | Single select | Phase, Implementation, Evaluation, Operator, Review follow-up |
| Executor | Single select | Luna, Human |
| Priority | Single select | P0, P1, P2 |
| Order | Number | Manifest order |

Use `gh project field-list ... --format json` and paginate if necessary. Reconcile by exact name/type. Create custom fields with `gh project field-create`; example:

```bash
gh project field-create "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --name Phase --data-type SINGLE_SELECT --single-select-options P0,P1,P1R,P2,P3,P4,P5,P6,P7,P8 --format json
```

Do not create a second Status field. Update its existing options with GraphQL `updateProjectV2Field`. Input JSON has `fieldId` and `singleSelectOptions`, each with `name`, `color`, `description`, and existing option `id` where applicable. Use GRAY/RED/BLUE/YELLOW/PURPLE/GREEN respectively for the six statuses. Preserve IDs for already matching options. In a freshly created project replace default Todo/In Progress/Done with the exact options; for an adopted project, incompatible options or item assignments require a reported conflict rather than silent destructive replacement. Write JSON to a file and use `gh api graphql --input`; do not interpolate multiline JSON or descriptions into shell command text. Snapshot option IDs before setting item fields.

Create labels from `manifest.json.labels` with `gh label create NAME --repo Enucatl/zanzara-archive --color COLOR --description DESCRIPTION` only when absent. Labels are `phase:P0`…`phase:P8`, `kind:phase`, `kind:implementation`, `kind:evaluation`, `kind:operator`, `kind:review-follow-up`, `executor:luna`, `executor:human`, and `priority:P0`…`priority:P2`. Colors/descriptions are fixed in the manifest. Preserve unrelated labels. An existing matching name with conflicting metadata is reported for reconciliation, not overwritten with `--force`.

Create/reconcile the following views by name. Use GraphQL `createProjectV2View(input:{projectId,name,layout:TABLE_LAYOUT,configuration:{visibleFieldIds:[...]}})` and capture returned view ID. Set the saved filter through `updateProjectV2View(input:{viewId,filter})`. Include title, Status, Phase, Kind, Executor, Priority, Order fields in that order. Read current schema before mutations because the API evolves. The inspected schema supports creation, filters and visible columns, but **does not expose sort/group writes** in `ProjectV2ViewConfigurationInput` or `UpdateProjectV2ViewInput`; configure sorting/grouping in GitHub's Project UI as the explicit operator step below. Do not invent mutation fields or claim view completion merely because a named blank view exists.

| View | Saved filter | Sort/group |
|---|---|---|
| Implementation queue | `Executor:Luna -Status:Done` | Order ascending; group by Status |
| Phase progress | `Kind:Phase` | Order ascending; no group |
| Evaluation work | `Kind:Evaluation` | Order ascending; group by Phase |
| Human actions | `Executor:Human -Status:Done` | Order ascending; group by Phase |

Operator action: open each named view, verify the saved filter/columns, set sort/group, then **Save changes**. Provide project/view URLs and read-back confirmation. The implementation queue includes Blocked/In review so progress is visible; readiness is determined by G7, not by the view filter alone. If no interactive UI tool is available, produce a pending operator checklist and stop short of claiming setup is complete. The setup run may resume after this action. Read views, columns, groups and sort direction using the query below; advance the outer cursor and each nested connection if `hasNextPage` is true.

```graphql
query($project: ID!, $after: String) {
  node(id: $project) {
    ... on ProjectV2 {
      public
      views(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id name layout filter
          fields(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes {
              ... on ProjectV2Field { id name }
              ... on ProjectV2SingleSelectField { id name }
            }
          }
          groupByFields(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes {
              ... on ProjectV2Field { id name }
              ... on ProjectV2SingleSelectField { id name }
            }
          }
          sortByFields(first: 100) {
            pageInfo { hasNextPage endCursor }
            nodes { direction field { ... on ProjectV2Field { id name } } }
          }
        }
      }
    }
  }
}
```

## G4 — Publish parents, then children

Use the manifest titles, labels and **complete** Markdown bodies. Each body has a stable marker, executor, prerequisites and parent. First create/reconcile all phase parents, then children in numeric Order. Submit structured JSON using `gh api --method POST repos/Enucatl/zanzara-archive/issues --input BODY_FILE`, where BODY_FILE contains `{title,body,labels}`. This returns `number`, `id`, `node_id`, `html_url`; record them immediately. `gh issue create --body-file` is also valid if followed by a GET to capture all IDs. Do not assign the human username automatically to Luna tasks; Executor is a Project field/label, not a bot account.

Issue-local planning links such as `../SYSTEM-DESIGN.md#...` need to work after publication: render them to absolute GitHub blob URLs at the planning commit, resolving from `planning/issues/`. Rewrite links to `issues/ID.md` or sibling issue bodies into canonical issue URLs once IDs exist. For an uncommitted package, publish/commit the planning files through the authorized repository workflow before publishing links; otherwise stop with a concrete missing-publication blocker. Keep source bodies unchanged locally and record both source and rendered hashes. Complete prose and section IDs still make each body understandable before supplemental cross-links are reconciled.

After all numbers exist, append a managed relationship section with actual parent and prerequisite links. These links are supplemental; checkboxes or Markdown task lists do not replace native relationships. Issue titles and bodies must not use automatic-closing syntax for a phase parent.

## G5 — Native relationships and project items

For each child, check current parent, then attach to its manifest parent if absent. If it belongs to another parent, report conflict; never send `replace_parent=true` automatically. For each manifest `blocked_by` edge, create it on the **blocked issue** using the prerequisite's database ID. Parents are blocked by their release-blocking children; a child marked `release_blocker: false` remains a native sub-issue but does not prevent phase release. Example templates (variables are populated from the mapping):

```bash
gh api --method POST "repos/Enucatl/zanzara-archive/issues/$ZANZARA_PARENT_NUMBER/sub_issues" -F sub_issue_id="$ZANZARA_CHILD_DATABASE_ID"
gh api --method POST "repos/Enucatl/zanzara-archive/issues/$ZANZARA_BLOCKED_NUMBER/dependencies/blocked_by" -F issue_id="$ZANZARA_PREREQUISITE_DATABASE_ID"
```

Use `Accept: application/vnd.github+json` and `X-GitHub-Api-Version: 2026-03-10` headers (verified documentation version; recheck server support during preflight). Read existing relationships first and add only missing edges. Children depend on their preceding phase parent plus local prerequisites; parents depend on their children and the preceding phase. No child depends on its **own** parent. This prevents cycles while ensuring the next phase cannot start before user release. See [native sub-issues](https://docs.github.com/en/rest/issues/sub-issues) and [blocking dependencies](https://docs.github.com/en/rest/issues/issue-dependencies).

Add each issue once with `gh project item-add "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --url ISSUE_URL --format json`; first reconcile current items by issue node ID. Capture Project item ID. Set fields with `gh project item-edit --id ITEM_ID --project-id PROJECT_NODE_ID --field-id FIELD_ID --single-select-option-id OPTION_ID`; Order uses `--number ORDER`. Set initial Status to `Ready` for children with zero blockers (including Human actions), `Blocked` for other children and parents; use manifest `initial_status`. Preserve live status on reruns after implementation starts. Keep all phase parents open. No milestones.

## G6 — Read back and completion evidence

After writes, read every issue and all pages of relationships/items, not just successful HTTP responses:

```bash
gh api "repos/Enucatl/zanzara-archive/issues/$ZANZARA_ISSUE_NUMBER"
gh api --paginate "repos/Enucatl/zanzara-archive/issues/$ZANZARA_ISSUE_NUMBER/sub_issues?per_page=100"
gh api "repos/Enucatl/zanzara-archive/issues/$ZANZARA_CHILD_NUMBER/parent"
gh api --paginate "repos/Enucatl/zanzara-archive/issues/$ZANZARA_ISSUE_NUMBER/dependencies/blocked_by?per_page=100"
gh api --paginate "repos/Enucatl/zanzara-archive/issues/$ZANZARA_ISSUE_NUMBER/dependencies/blocking?per_page=100"
gh project view "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --format json
gh project field-list "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --format json --limit 100
gh project item-list "$ZANZARA_PROJECT_NUMBER" --owner Enucatl --format json --limit 1000
```

Use GraphQL cursor pagination beyond CLI limits, and verify result counts against totals. Confirm relationship endpoint pagination support/shape from current docs; follow provided next pages until exhausted. Compare managed native parents, blockers and inverse blocking edges exactly to manifest plus registered review follow-ups. Report unexpected edges instead of deleting them. Validate the resulting DAG too. Verify one issue per stable ID, no duplicate items/markers, correct titles/body hashes/labels/fields/order, all parents open at initial setup, no milestones created, Project private and repository linked, and all four views' saved settings. Preserve preexisting unrelated objects.

Save a private `creation-report.md` with planning hashes, identity and permission preflight (redacted), Project URL/ID, counts, IDs, read-back discrepancies, any conflicts/operator actions, and timestamps. Store a sanitized report in the repo only if authorized and free of private details. A successful report says `GitHub structure verified; implementation not started`. Missing scopes, unconfigured view filters, broken planning links or unwired native edges means incomplete setup. A second reconciliation run must make zero mutations when the expected state is unchanged. Do not claim this live verification was performed during planning-package creation.

## G7 — Luna implementation handoff and release rules

At implementation time, under GPT 5.6-Luna:

1. Read planning/README and live GitHub state, including follow-ups. Select an open child with Executor=Luna and Kind Implementation, Evaluation or Review follow-up. Human tasks and phase parents are never unattended selections.
2. Check every native blocker's close reason and completion evidence. A `completed` preceding phase also requires a current Sol PASS and user release comment. `not_planned` never satisfies a prerequisite without an explicit user-approved replacement. A Project Ready flag alone is insufficient.
3. Select lowest phase, then P0/P1/P2 priority, then numeric Order. The current phase parent remains open while its children execute; **only the preceding phase must be released**. Do not mistakenly exclude all children of an open parent. A registered current-phase follow-up is eligible by the same rules.
4. Claim one issue with an implementation comment and In progress status after checking it is unclaimed. Inspect current code, implement the bounded body, preserve unrelated edits, run specified checks and attach evidence. Do not create an additional implementation plan in place of the work.
5. Commit the bounded change directly to `main`, push `origin/main`, verify the remote commit, and comment on the child with the commit, acceptance checklist, commands/results, limitations and evidence. Set the child Project Status to Done only after acceptance is satisfied. Missing human actions/resources mean Blocked plus a precise operator request/remediation, not fabricated evidence.
6. At the phase boundary stop and present `$review-phase <parent-number>` for execution under GPT 5.6-Sol. The user releases via `Release Pn at <reviewed-commit>` after the current PASS; only then close the parent as completed and set Done. Any relevant subsequent commit or new finding requires another review before release. Setup/review agents never release phases on the user's behalf.

Local checks named in issue bodies are future implementation contracts: P0 establishes common commands, and each issue adds the referenced focused tests/CLI action where absent. GPU tests and paid/human runs are explicit; CPU success does not satisfy them. Human acceptance gates never become Luna work merely because there is no other eligible issue.

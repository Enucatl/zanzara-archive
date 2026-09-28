# Repository agent instructions

## Review web server

- Run the operator review web app on this host, bound to `0.0.0.0` for trusted LAN access.
- Give users links using `http://complex.home.arpa:<port>/...`, not the host's raw IP.
- The P1R-03D human review page is `/p1r-03d` (port `8000` by default).

## Solo workstation delivery

- Follow the current scope and workflow in `planning/README.md`; older issue boilerplate and release ceremonies do not override it.
- Read the selected work and actual dependencies. Reuse session context and batch tightly related authorized work instead of rebuilding context for every issue.
- Run relevant checks once. Reuse evidence after unrelated edits; reserve broad suites and GPU reruns for changes that affect their results.
- Trust recorded local archive/model inputs. Do not routinely rehash source audio, checkpoint caches, or evidence trees. Preserve existing identity/cache hashes; use full byte verification for import, changed inputs, recovery, or suspected corruption.
- Phase parents are milestones, not automatic blockers or separate release approvals. Human decisions, spending, public exposure, and newly scoped bulk processing retain their actual authorization requirements.
- Keep GitHub updates proportional: one concise completion record and targeted readback of structural edits, without full Project reconciliation for ordinary work.

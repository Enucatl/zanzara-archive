# P5-03 recovery drill — 2026-09-29

Private evidence: `.git/zanzara-evidence/p5-03-20260929/`. The one-time driver is
`.git/zanzara-state/p5-03-drill.py`; it uses `sqlite3.Connection.backup()` while
the live database remains available, copies immutable artifacts and the corpus
and model lock records, then restores into a separate local directory. It remaps
recorded artifact paths in the restored database, including relative paths. The
working database and Qdrant storage were not changed.

| Check | Result |
|---|---|
| Restored SQLite `PRAGMA integrity_check` | `ok` |
| Recorded artifacts in the restored tree | 115/115 resolve; sampled attribution manifest is complete |
| Human-reviewed transcript revisions | 33 retained |
| Restored FTS5 transcript search | Result returned with source timestamps |
| Isolated Qdrant rebuild from retained ResNet293 vectors | 2 episodes, 280 exemplars, 46 centroids; generation `54f295c140b8bd373eed80ab` |
| Restored voice search | 23 candidates with source timestamps |
| Interrupted synthetic job on restored SQLite | Expired lease recovered; second attempt succeeded; artifact count unchanged |
| Human identity decisions in the first snapshot | 0; repeated after human review below |

Commands used: `uv run python .git/zanzara-state/p5-03-drill.py`, an isolated
`qdrant/qdrant:v1.19.1` container on `127.0.0.1:6334`, and the existing
`publish_generation`, `retrieve_candidates`, `SQLiteRepository.search_text`,
`recover_expired_leases`, and `DurableWorker` entry points. The isolated Qdrant
container was removed after the check; its private rebuilt storage remains with
the drill evidence. No inference or archive rehash was needed.

After human review, `uv run python .git/zanzara-state/p5-03-identity-restore.py`
repeated the backup and isolated restore at
`.git/zanzara-evidence/p5-03-identity-20260929/`. The restored SQLite passed
`integrity_check` and `foreign_key_check`. Its identity revision (2), two active
decisions (one same-person, one different-person), two audit rows, two
memberships, and global-speaker record matched the backup row for row. The
same-person pair retained one membership; the different-person pair remained
separate. Both decision revisions had corresponding audit entries, their
evidence artifact IDs existed, and all 115 restored artifact paths resolved.
The earlier isolated Qdrant, query, and interrupted-job checks above remain
applicable; this repeat changed no recovery implementation or source inputs.

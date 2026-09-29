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
| Human identity decisions in the live snapshot | 0; real identity history cannot yet be inspected after restore |

Commands used: `uv run python .git/zanzara-state/p5-03-drill.py`, an isolated
`qdrant/qdrant:v1.19.1` container on `127.0.0.1:6334`, and the existing
`publish_generation`, `retrieve_candidates`, `SQLiteRepository.search_text`,
`recover_expired_leases`, and `DurableWorker` entry points. The isolated Qdrant
container was removed after the check; its private rebuilt storage remains with
the drill evidence. No inference or archive rehash was needed.

The remaining P5-03 check needs an actual human-confirmed identity link. Once
one exists, repeat the backup and isolated restore and compare the decision,
membership and history on both sides. A synthetic decision would not establish
that real review state survived recovery.

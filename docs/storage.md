# Canonical storage and artifact publication

`zanzara_archive.storage` owns mutable archive state in a local SQLite
database. `open_database()` rejects `:memory:`, UNC/network paths, and known
network filesystems before enabling foreign keys, WAL, a bounded busy timeout,
and FTS5. It applies the versioned migrations and refuses a database newer than
the application understands. SQLite state and worker job state therefore stay
on the local `state` volume; the canonical archive source remains read-only.

`zanzara_archive.artifacts.ArtifactPublisher` writes immutable stage output at
`<root>/<source_sha256>/<stage>/<stage_key>/`. It writes and fsyncs every file,
writes `manifest.json` last, fsyncs the temporary directory, and atomically
renames that directory into place. The manifest records source, upstream,
model, pipeline, schema, and file checksums. A complete directory is recorded
in SQLite only after the filesystem rename. If the database transaction fails,
the complete directory is a recoverable orphan; `reconcile()` validates its
manifest and checksums before inserting it. Directories without a completed
manifest, or with a bad checksum, are never treated as artifacts.

Index generation pointers, identity decisions and audit revisions, and cost
reservations are relational tables. Qdrant remains a rebuildable projection;
it is never the canonical identity store.

CPU checks:

```bash
uv run pytest tests/test_storage.py tests/test_artifacts.py
```

# Canonical SQLite storage

SQLite is the authoritative store for mutable archive state. Schema version 4
adds the D3 records that were previously only represented by in-memory
contracts: transcript words, standard and exclusive diarization turns,
overlap intervals, annotation revisions, review and split history, upload-job
metadata, and aggregate evaluation reports.

Every stage-derived record keeps its episode/source and artifact provenance;
evaluation reports additionally retain split, model, configuration, and
reviewed-commit hashes. Interval and lifecycle checks are enforced by SQLite,
and all references to episodes, artifacts, jobs, speakers, and annotation
revisions use foreign keys. Migrations are forward-only and applied one
version at a time inside the connection transaction. Existing version-3 data
is preserved when upgrading to version 4.

The repository keeps report persistence intentionally small until the owning
transcript, annotation, upload, and evaluation stages define their write
flows. Those stages must use the canonical tables rather than treating JSON
artifacts or Qdrant as authoritative state.

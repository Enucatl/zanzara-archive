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
version at a time inside an explicit transaction. Schema version 5 adds
insert/update guards for timed ASR and diarization records: millisecond values
must have SQLite integer storage, must fit the referenced episode duration, and
must agree with both the episode and artifact source/model provenance. Existing
version-3 and version-4 data is preserved when upgrading.

The repository keeps report persistence intentionally small until the owning
transcript, annotation, upload, and evaluation stages define their write
flows. Those stages must use the canonical tables rather than treating JSON
artifacts or Qdrant as authoritative state.

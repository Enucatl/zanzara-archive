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

Annotation revisions use the existing `annotation_revisions` table as an
append-only history and `review_records` as the per-revision audit record.
Their JSON payload retains source-artifact, source checksum, model/configuration
provenance, reviewer/review time, exact word/turn/overlap edits,
unintelligible spans, and the deterministic E1 split. Filesystem exports are
private immutable artifacts;
SQLite remains authoritative for revision and optimistic-conflict state.

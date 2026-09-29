# P6-01: frozen recent cohort and expansion estimate

The [400-episode manifest](corpus-400.json) freezes the 400 most recent
`*.opus` archive files dated no later than the original 2026-09-10 cutoff.
Its first 20 records exactly match [the initial manifest](corpus-20.json).
The remaining 380 source files were imported on 2026-09-29: each was read
once for SHA-256, probed for duration/codec, and checked for stable size and
modification time across the read. This does not authorize processing them.

| Scope | Episodes | Audio hours | Source bytes |
|---|---:|---:|---:|
| Initial, already processed | 20 | 33.606 | 0.634 GB |
| Added, awaiting authorization | 380 | 601.597 | 11.286 GB |
| Frozen total | 400 | 635.204 | 11.920 GB |

The added range is 2024-10-31 through 2026-06-30. Source bytes already exist
in the archive and do not consume new workstation storage. No member is missing
from the frozen selection. P5-06 accounted for all initial 20 with no
unresolved processing failures; transcript quality remains subject to human
review. The added 380 have no processing or quality result.

**Serial time estimate:** The measured warm P1-08 baseline on the 6,108.584 s
golden episode took 350.146 s ASR, 73.378 s diarization and 2.798 s
attribution (private run:
`.git/zanzara-evidence/P1-08-baseline-final/run.json`).
The run used Parakeet revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`
and Community-1 revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.
Its combined real-time factor is 0.06979. Applied to 601.597 audio hours,
that is **about 42 hours** of these three stages on the fixed workstation.
Allow additional time for service startup, retries, text indexing, canary
review and any voice stages. One P5-06 episode required 30-second ASR windows
after five-minute request timeouts, so this estimate is a planning baseline,
not a run deadline.

**New storage estimate:** The initial 20 used 201.219 MB for ASR,
diarization and attribution artifact directories and 21.086 MB for SQLite
text chunk/FTS tables (measured on 2026-09-29). Scaling by audio duration
gives **about 3.60 GB artifacts plus 0.38 GB index** for the added 380,
approximately **4.0 GB** total before backups, temporary files or voice
artifacts. The index includes content and FTS pages; actual usage will vary
with speech density and retained artifact generations.

Next, P6-02 needs the operator to authorize the 400-episode resource envelope
from these estimates. P6-03 then selects a nested 40-total canary for review
before P6-04 processes the remainder. No new audio was processed for P6-01.

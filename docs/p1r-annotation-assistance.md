# P1R local annotation assistance

`AnnotationAssistanceService` presents the current chunk's immutable model
hypotheses to a replaceable local helper. It includes at most two preceding
reference/draft context records, labels the candidate order, and persists the
prompt version, prompt hash, model fingerprint, inputs, and result in the
append-only `annotation_assistance_drafts` table.

The default adapter is `QwenAnnotationAdapter`, an OpenAI-compatible HTTP
adapter restricted to loopback endpoints. It expects exactly one JSON output
field, `current_chunk_text`. Qwen output is never written to
`transcript_references`, `reference_revisions`, or the episode annotation
tables. An unavailable or invalid helper is recorded as an `unavailable` draft
and the review API remains usable.

The API route is:

```text
POST /api/v1/chunks/{chunk_id}/annotation-assistance
{
  "hypothesis_ids": ["hypothesis-..."],
  "preceding_chunks": [
    {"rank": 1, "chunk_id": "chunk-...", "reference_text": "...", "draft_text": "..."}
  ]
}
```

This is assistance provenance, not human truth. The chunk-centric UI consumes
the draft state in P1R-10.

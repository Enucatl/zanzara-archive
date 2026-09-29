"""Deterministic clean excerpts from episode-wide diarization."""

from __future__ import annotations

from collections import defaultdict

from .contracts import DiarizationResult

MIN_MS = 3_000
PREFERRED_MIN_MS = 8_000
MAX_MS = 15_000
TRANSITION_MS = 250


def _subtract(interval: tuple[int, int], blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the pieces of one interval outside sorted blocked intervals."""

    pieces = [interval]
    for left, right in blocked:
        pieces = [
            piece
            for start, end in pieces
            for piece in ((start, min(end, left)), (max(start, right), end))
            if piece[1] > piece[0]
        ]
    return pieces


def select_exemplars(result: DiarizationResult) -> dict[str, dict[str, object]]:
    """Select at most ten clean source intervals for each diarized speaker."""

    turns: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for turn in result.standard_turns:
        turns[turn.speaker_id].append((turn.start_ms, turn.end_ms))
    for turn in result.exclusive_turns:
        turns.setdefault(turn.speaker_id, [])
    blocked = sorted(
        (
            max(0, overlap.start_ms - TRANSITION_MS),
            min(result.duration_ms, overlap.end_ms + TRANSITION_MS),
        )
        for overlap in result.overlaps
    )
    output: dict[str, dict[str, object]] = {}
    for speaker_id, intervals in sorted(turns.items()):
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        candidates: list[tuple[int, int]] = []
        exclusions: list[dict[str, object]] = []
        for start, end in merged:
            exclusions.append(
                {
                    "start_ms": start,
                    "end_ms": min(end, start + TRANSITION_MS),
                    "reason": "transition",
                }
            )
            exclusions.append(
                {"start_ms": max(start, end - TRANSITION_MS), "end_ms": end, "reason": "transition"}
            )
            exclusions.extend(
                {"start_ms": max(start, left), "end_ms": min(end, right), "reason": "overlap"}
                for left, right in blocked
                if max(start, left) < min(end, right)
            )
            # ponytail: scan overlaps per turn; use an interval index if long episodes make this slow.
            clean = _subtract((start + TRANSITION_MS, end - TRANSITION_MS), blocked)
            for left, right in clean:
                if right - left < MIN_MS:
                    exclusions.append(
                        {"start_ms": left, "end_ms": right, "reason": "under_3_seconds"}
                    )
                    continue
                length = right - left
                count = (length + MAX_MS - 1) // MAX_MS
                if length >= count * PREFERRED_MIN_MS:
                    boundaries = [left + length * index // count for index in range(count + 1)]
                else:
                    boundaries = list(range(left, right, MAX_MS)) + [right]
                for offset, stop in zip(boundaries, boundaries[1:], strict=False):
                    if stop - offset >= MIN_MS:
                        candidates.append((offset, stop))
                    else:
                        exclusions.append(
                            {"start_ms": offset, "end_ms": stop, "reason": "under_3_seconds"}
                        )
            if not clean:
                exclusions.append(
                    {"start_ms": start, "end_ms": end, "reason": "overlap_or_transition"}
                )
        candidates.sort(
            key=lambda item: (item[1] - item[0] < PREFERRED_MIN_MS, -(item[1] - item[0]), item[0])
        )
        chosen = candidates[:10]
        exclusions.extend(
            {"start_ms": start, "end_ms": end, "reason": "ten_exemplar_limit"}
            for start, end in candidates[10:]
        )
        output[speaker_id] = {
            "status": "voice_searchable" if chosen else "not_voice_searchable",
            "excerpts": [{"start_ms": start, "end_ms": end} for start, end in chosen],
            "exclusions": exclusions,
        }
    return output

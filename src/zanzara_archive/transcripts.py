"""Word attribution and deterministic transcript export views.

Attribution is deliberately a pure, CPU-only operation.  It consumes the
timed ASR and episode-wide diarization contracts, then publishes only derived
views; the input artifacts are never edited.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    ApiEnvelope,
    ContractValidationError,
    DiarizationResult,
    ModelFingerprint,
    Overlap,
    TimedWord,
    TranscriptResult,
    Turn,
)
from .inference import derive_overlap_intervals

EXPORT_SCHEMA_VERSION = 1
UNASSIGNED_LABEL = "UNASSIGNED"


def _intersection_ms(word: TimedWord, turn: Turn) -> int:
    """Return the positive temporal intersection of a word and a turn."""

    return max(0, min(word.end_ms, turn.end_ms) - max(word.start_ms, turn.start_ms))


def _contains_midpoint(word: TimedWord, turn: Turn) -> bool:
    """Test half-open midpoint containment without converting to floating point."""

    midpoint_numerator = word.start_ms + word.end_ms
    return 2 * turn.start_ms <= midpoint_numerator and midpoint_numerator < 2 * turn.end_ms


def _overlap_for_word(word: TimedWord, overlaps: Sequence[Overlap]) -> bool:
    """Mark a word when it touches any standard-turn overlap interval."""

    return any(
        overlap.start_ms < word.end_ms and overlap.end_ms > word.start_ms for overlap in overlaps
    )


def _validated_overlaps(
    standard_turns: Sequence[Turn], overlaps: Sequence[Overlap] | None
) -> tuple[Overlap, ...]:
    """Derive overlap annotations and reject a contradictory supplied view."""

    derived = derive_overlap_intervals(standard_turns)
    if overlaps is None:
        return derived
    supplied = tuple(
        sorted(overlaps, key=lambda item: (item.start_ms, item.end_ms, item.speaker_ids))
    )
    if supplied != derived:
        raise ContractValidationError("diarization overlap intervals do not match standard turns")
    return supplied


def attribute_word(
    word: TimedWord,
    exclusive_turns: Sequence[Turn],
    *,
    standard_overlaps: Sequence[Overlap] = (),
) -> TimedWord:
    """Attribute one word by greatest exclusive-turn intersection.

    Ties first prefer a turn containing the word midpoint, then the
    lexicographically smallest stable local speaker ID.  A word with no
    positive intersection remains explicitly unassigned.
    """

    intersections = [
        (turn, _intersection_ms(word, turn))
        for turn in exclusive_turns
        if _intersection_ms(word, turn) > 0
    ]
    speaker_id: str | None = None
    if intersections:
        greatest = max(intersection for _, intersection in intersections)
        tied = [turn for turn, intersection in intersections if intersection == greatest]
        midpoint_tied = [turn for turn in tied if _contains_midpoint(word, turn)]
        candidates = midpoint_tied or tied
        speaker_id = min(turn.speaker_id for turn in candidates)
    return TimedWord(
        word_id=word.word_id,
        text=word.text,
        start_ms=word.start_ms,
        end_ms=word.end_ms,
        confidence=word.confidence,
        speaker_id=speaker_id,
        overlap=_overlap_for_word(word, standard_overlaps),
    )


def attribute_words(
    words: Sequence[TimedWord],
    exclusive_turns: Sequence[Turn],
    standard_turns: Sequence[Turn],
    *,
    overlaps: Sequence[Overlap] | None = None,
) -> tuple[TimedWord, ...]:
    """Attribute ASR words and independently annotate standard-turn overlap."""

    validated_overlaps = _validated_overlaps(standard_turns, overlaps)
    return tuple(
        attribute_word(word, exclusive_turns, standard_overlaps=validated_overlaps)
        for word in words
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AttributedTranscript:
    """Structured transcript payload with both diarization representations."""

    artifact_id: str
    source_sha256: str
    duration_ms: int
    transcript_artifact_id: str
    diarization_artifact_id: str
    transcript_model: ModelFingerprint
    diarization_model: ModelFingerprint
    transcript_text: str
    transcript_status: str
    timestamp_granularities: tuple[str, ...]
    words: tuple[TimedWord, ...]
    standard_turns: tuple[Turn, ...]
    exclusive_turns: tuple[Turn, ...]
    overlaps: tuple[Overlap, ...]
    request_id: str | None = None
    episode_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id or not isinstance(self.artifact_id, str):
            raise ContractValidationError("attribution artifact_id must be non-empty text")
        if not self.source_sha256 or len(self.source_sha256) != 64:
            raise ContractValidationError("attribution source_sha256 must be a SHA-256")
        if self.duration_ms <= 0:
            raise ContractValidationError("attribution duration_ms must be positive")
        if self.transcript_status != "timed":
            raise ContractValidationError("attribution requires a timed transcript")
        if "word" not in self.timestamp_granularities:
            raise ContractValidationError("attribution requires word timestamp granularity")
        _validated_overlaps(self.standard_turns, self.overlaps)
        for word in self.words:
            if word.end_ms > self.duration_ms:
                raise ContractValidationError("attributed word exceeds episode duration")
        if self.request_id is not None and not self.request_id:
            raise ContractValidationError("request_id must be non-empty when provided")

    @property
    def unassigned_word_count(self) -> int:
        """Return the number of words that have no exclusive speaker."""

        return sum(word.speaker_id is None for word in self.words)

    @property
    def overlap_word_count(self) -> int:
        """Return the number of words touching standard diarization overlap."""

        return sum(word.overlap for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the API data while retaining exact IDs, offsets and provenance."""

        return {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "episode_id": self.episode_id,
            "source_sha256": self.source_sha256,
            "duration_ms": self.duration_ms,
            "time_origin_ms": 0,
            "transcript": {
                "artifact_id": self.transcript_artifact_id,
                "model": self.transcript_model.to_dict(),
                "text": self.transcript_text,
                "status": self.transcript_status,
                "timestamp_granularities": list(self.timestamp_granularities),
            },
            "diarization": {
                "artifact_id": self.diarization_artifact_id,
                "model": self.diarization_model.to_dict(),
                "standard_turns": [turn.to_dict() for turn in self.standard_turns],
                "exclusive_turns": [turn.to_dict() for turn in self.exclusive_turns],
                "overlaps": [overlap.to_dict() for overlap in self.overlaps],
            },
            "words": [word.to_dict() for word in self.words],
            "provenance": dict(self.provenance),
            "request_id": self.request_id,
        }

    def to_api_payload(self) -> dict[str, Any]:
        """Wrap the structured data in the stable D10 success envelope."""

        request_id = self.request_id or f"attribution-{self.artifact_id}"
        return ApiEnvelope(request_id, "ok", data=self.to_dict()).to_dict()


def build_attributed_transcript(
    transcript: TranscriptResult,
    diarization: DiarizationResult,
    *,
    artifact_id: str | None = None,
    episode_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> AttributedTranscript:
    """Validate compatible inputs and build one immutable attributed payload."""

    transcript.require_production()
    if transcript.source_sha256 != diarization.source_sha256:
        raise ContractValidationError(
            "transcript and diarization artifacts have different source provenance"
        )
    if transcript.duration_ms is not None and transcript.duration_ms != diarization.duration_ms:
        raise ContractValidationError(
            "transcript and diarization artifacts have different duration provenance"
        )
    attributed_words = attribute_words(
        transcript.words,
        diarization.exclusive_turns,
        diarization.standard_turns,
        overlaps=diarization.overlaps,
    )
    artifact_id = artifact_id or (
        "attribution-"
        + _canonical_hash(
            {
                "transcript_artifact_id": transcript.artifact_id,
                "diarization_artifact_id": diarization.artifact_id,
                "source_sha256": transcript.source_sha256,
            }
        )[:24]
    )
    return AttributedTranscript(
        artifact_id=artifact_id,
        source_sha256=transcript.source_sha256,
        duration_ms=diarization.duration_ms,
        transcript_artifact_id=transcript.artifact_id,
        diarization_artifact_id=diarization.artifact_id,
        transcript_model=transcript.model,
        diarization_model=diarization.model,
        transcript_text=transcript.text,
        transcript_status=transcript.status,
        timestamp_granularities=tuple(transcript.timestamp_granularities),
        words=attributed_words,
        standard_turns=tuple(diarization.standard_turns),
        exclusive_turns=tuple(diarization.exclusive_turns),
        overlaps=tuple(diarization.overlaps),
        request_id=transcript.request_id,
        episode_id=episode_id,
        provenance=provenance or {},
    )


def _format_timestamp(milliseconds: int, *, separator: str) -> str:
    """Format a non-negative source offset for TXT/SRT/VTT."""

    if milliseconds < 0:
        raise ContractValidationError("export timestamps must be non-negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _export_words(value: AttributedTranscript | Sequence[TimedWord]) -> tuple[TimedWord, ...]:
    if isinstance(value, AttributedTranscript):
        return value.words
    return tuple(value)


def _caption(word: TimedWord) -> str:
    label = f"[{word.speaker_id or UNASSIGNED_LABEL}]"
    if word.overlap:
        label += " [OVERLAP]"
    return f"{label} {word.text}"


def render_txt(value: AttributedTranscript | Sequence[TimedWord]) -> str:
    """Render one timestamped, labelled line per word without shifting offsets."""

    return "".join(
        f"[{_format_timestamp(word.start_ms, separator='.')} --> "
        f"{_format_timestamp(word.end_ms, separator='.')}] {_caption(word)}\n"
        for word in _export_words(value)
    )


def render_srt(value: AttributedTranscript | Sequence[TimedWord]) -> str:
    """Render one SRT cue per word, preserving each word's exact interval."""

    blocks = []
    for index, word in enumerate(_export_words(value), start=1):
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(word.start_ms, separator=',')} --> "
            f"{_format_timestamp(word.end_ms, separator=',')}\n"
            f"{_caption(word)}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(value: AttributedTranscript | Sequence[TimedWord]) -> str:
    """Render one WebVTT cue per word, preserving each word's exact interval."""

    blocks = [
        f"{_format_timestamp(word.start_ms, separator='.')} --> "
        f"{_format_timestamp(word.end_ms, separator='.')}\n"
        f"{_caption(word)}\n"
        for word in _export_words(value)
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


render_transcript_txt = render_txt
render_transcript_srt = render_srt
render_transcript_vtt = render_vtt


def render_exports(value: AttributedTranscript | Sequence[TimedWord]) -> dict[str, str]:
    """Return all named derived transcript views."""

    return {
        "transcript.txt": render_txt(value),
        "transcript.srt": render_srt(value),
        "transcript.vtt": render_vtt(value),
    }


__all__ = [
    "AttributedTranscript",
    "EXPORT_SCHEMA_VERSION",
    "UNASSIGNED_LABEL",
    "attribute_word",
    "attribute_words",
    "build_attributed_transcript",
    "render_exports",
    "render_srt",
    "render_transcript_srt",
    "render_transcript_txt",
    "render_transcript_vtt",
    "render_txt",
    "render_vtt",
]

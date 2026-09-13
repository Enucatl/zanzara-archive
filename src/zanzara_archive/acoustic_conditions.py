"""Deterministic AudioSet AST condition metadata for P1R chunks.

The AST service owns AudioSet event probabilities.  This module owns the
versioned, model-independent windowing and mapping contract, deterministic
signal measurements, and the separate human-correction record.  It never
infers speaker count or overlap and never consumes ASR output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from .contracts import (
    AcousticConditionCorrection,
    AcousticConditionMetadata,
    AudioChunk,
    AudioSetScore,
    AudioSetWindowScores,
    ChunkCondition,
    ContractValidationError,
    SignalMeasurements,
)

AUDIOSET_MODEL_ID = "audioset_ast"
AUDIOSET_MODEL_REPOSITORY = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIOSET_MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
AUDIOSET_SAMPLE_RATE_HZ = 16_000
AUDIOSET_WINDOW_MS = 10_000
AUDIOSET_MAX_LENGTH = 1_024
ACOUSTIC_CONDITION_VERSION = "p1r-acoustic-condition-v1"
ACOUSTIC_THRESHOLD_VERSION = "p1r-acoustic-thresholds-v1"
SIGNAL_MEASUREMENT_VERSION = "p1r-signal-measurements-v1"

MUSIC_LABEL_KEYWORDS = frozenset({"music", "musical instrument", "singing", "song"})
SPEECH_LABEL_KEYWORDS = frozenset(
    {"speech", "conversation", "narration", "monologue", "babbling", "speech synthesizer"}
)
NON_SPEECH_LABEL_KEYWORDS = frozenset(
    {
        "applause",
        "breathing",
        "cough",
        "crowd",
        "crying",
        "engine",
        "laughter",
        "noise",
        "rain",
        "singing",
        "vehicle",
        "wind",
    }
)


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _label_matches(label: str, keywords: frozenset[str]) -> bool:
    lowered = label.casefold()
    return any(keyword in lowered for keyword in keywords)


@dataclass(frozen=True, slots=True)
class ASTWindow:
    """One source-relative interval sent to the fixed ten-second AST input."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int):
            raise ContractValidationError("AST window start_ms must be an integer")
        if isinstance(self.end_ms, bool) or not isinstance(self.end_ms, int):
            raise ContractValidationError("AST window end_ms must be an integer")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ContractValidationError("AST windows must be non-empty half-open intervals")

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, int]:
        return {"start_ms": self.start_ms, "end_ms": self.end_ms}


@dataclass(frozen=True, slots=True)
class AcousticThresholds:
    """Frozen development-only mapping thresholds."""

    version: str = ACOUSTIC_THRESHOLD_VERSION
    music_presence: float = 0.20
    music_dominant: float = 0.65
    speech_presence: float = 0.25
    activity_presence: float = 0.25
    clipping_fraction: float = 0.01
    low_rms_dbfs: float = -45.0
    high_silence_fraction: float = 0.85

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ContractValidationError("acoustic threshold version must be non-empty")
        for name in (
            "music_presence",
            "music_dominant",
            "speech_presence",
            "activity_presence",
            "clipping_fraction",
            "high_silence_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractValidationError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ContractValidationError(f"{name} must be between 0 and 1")
        if self.music_presence >= self.music_dominant:
            raise ContractValidationError("music_presence must be below music_dominant")
        if not math.isfinite(float(self.low_rms_dbfs)):
            raise ContractValidationError("low_rms_dbfs must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "music_presence": float(self.music_presence),
            "music_dominant": float(self.music_dominant),
            "speech_presence": float(self.speech_presence),
            "activity_presence": float(self.activity_presence),
            "clipping_fraction": float(self.clipping_fraction),
            "low_rms_dbfs": float(self.low_rms_dbfs),
            "high_silence_fraction": float(self.high_silence_fraction),
        }


def ast_preprocessing(*, sample_rate_hz: int = AUDIOSET_SAMPLE_RATE_HZ) -> dict[str, object]:
    """Return the canonical transform shared by chunks and AST windows."""

    if sample_rate_hz != AUDIOSET_SAMPLE_RATE_HZ:
        raise ContractValidationError("AudioSet AST requires 16 kHz input")
    return {
        "version": "p1r-audioset-preprocessing-v1",
        "sample_rate_hz": AUDIOSET_SAMPLE_RATE_HZ,
        "channels": 1,
        "window_ms": AUDIOSET_WINDOW_MS,
        "padding": "zero-pad-final-window-to-10s",
        "feature_extractor": "ASTFeatureExtractor",
        "feature_extractor_max_length": AUDIOSET_MAX_LENGTH,
        "time_origin": "source chunk start",
    }


def preprocessing_fingerprint(preprocessing: Mapping[str, object] | None = None) -> str:
    """Hash only deterministic AST preprocessing configuration."""

    return _hash_json(dict(preprocessing or ast_preprocessing()))


def build_ast_windows(
    start_ms: int,
    end_ms: int,
    *,
    window_ms: int = AUDIOSET_WINDOW_MS,
) -> tuple[ASTWindow, ...]:
    """Partition a chunk into contiguous, non-overlapping AST windows."""

    if isinstance(start_ms, bool) or isinstance(end_ms, bool) or start_ms < 0 or end_ms <= start_ms:
        raise ContractValidationError("AST chunk bounds must be a valid half-open interval")
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms <= 0:
        raise ContractValidationError("AST window_ms must be positive")
    result: list[ASTWindow] = []
    cursor = start_ms
    while cursor < end_ms:
        boundary = min(cursor + window_ms, end_ms)
        result.append(ASTWindow(cursor, boundary))
        cursor = boundary
    return tuple(result)


def relevant_class_ids(label_map: Mapping[int | str, str]) -> tuple[int, ...]:
    """Return the locked AudioSet IDs retained for condition evidence."""

    result = []
    for raw_id, label in label_map.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("AudioSet label-map IDs must be integers") from exc
        if not isinstance(label, str) or not label.strip():
            raise ContractValidationError("AudioSet label-map labels must be non-empty text")
        if (
            _label_matches(label, MUSIC_LABEL_KEYWORDS)
            or _label_matches(label, SPEECH_LABEL_KEYWORDS)
            or _label_matches(label, NON_SPEECH_LABEL_KEYWORDS)
        ):
            result.append(class_id)
    return tuple(sorted(set(result)))


def measure_signal(samples: Sequence[float]) -> SignalMeasurements:
    """Compute deterministic clipping, peak, RMS and silence measurements."""

    values = tuple(float(value) for value in samples)
    if not values:
        return SignalMeasurements(0.0, -120.0, -120.0, 1.0)
    if any(not math.isfinite(value) for value in values):
        raise ContractValidationError("AST signal samples must be finite")
    peak = max(abs(value) for value in values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))

    def to_dbfs(value: float) -> float:
        return 20.0 * math.log10(max(value, 1e-12))

    return SignalMeasurements(
        clipping_fraction=sum(abs(value) >= 0.999 for value in values) / len(values),
        peak_dbfs=to_dbfs(peak),
        rms_dbfs=to_dbfs(rms),
        silence_fraction=sum(abs(value) < 1e-4 for value in values) / len(values),
    )


def aggregate_audio_set_scores(
    windows: Sequence[AudioSetWindowScores],
) -> tuple[AudioSetScore, ...]:
    """Mean per-class probabilities while retaining each window unchanged."""

    if not windows:
        return ()
    labels: dict[int, str] = {}
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for window in windows:
        for score in window.scores:
            previous = labels.setdefault(score.class_id, score.label)
            if previous != score.label:
                raise ContractValidationError("AudioSet class labels drifted between windows")
            totals[score.class_id] = totals.get(score.class_id, 0.0) + score.probability
            counts[score.class_id] = counts.get(score.class_id, 0) + 1
    return tuple(
        AudioSetScore(class_id, labels[class_id], totals[class_id] / counts[class_id])
        for class_id in sorted(totals)
    )


def _score(scores: Sequence[AudioSetScore], keywords: frozenset[str]) -> float:
    return max(
        (item.probability for item in scores if _label_matches(item.label, keywords)),
        default=0.0,
    )


def _music_level(music_score: float, speech_score: float, thresholds: AcousticThresholds) -> str:
    if music_score < thresholds.music_presence:
        return "none"
    if music_score >= thresholds.music_dominant:
        return "dominant"
    if speech_score >= thresholds.speech_presence:
        return "background"
    return "uncertain"


def _audio_quality(signal: SignalMeasurements, thresholds: AcousticThresholds) -> str:
    if (
        signal.clipping_fraction > thresholds.clipping_fraction
        or signal.rms_dbfs < thresholds.low_rms_dbfs
        or signal.silence_fraction > thresholds.high_silence_fraction
    ):
        return "degraded"
    return "clean"


def _unknown_metadata(
    *,
    signal: SignalMeasurements,
    model_fingerprint: str | None,
    label_map_sha256: str | None,
    preprocessing_sha256: str | None,
) -> AcousticConditionMetadata:
    return AcousticConditionMetadata(
        version=ACOUSTIC_CONDITION_VERSION,
        threshold_version=ACOUSTIC_THRESHOLD_VERSION,
        has_music=False,
        music_level="uncertain",
        has_speech=False,
        non_speech_activity=(),
        audio_quality="uncertain",
        signal_measurements=signal,
        status="unknown",
        origin="machine_seed",
        model_fingerprint=model_fingerprint,
        label_map_sha256=label_map_sha256,
        preprocessing_sha256=preprocessing_sha256,
    )


def derive_acoustic_condition(
    chunk: AudioChunk,
    *,
    window_scores: Sequence[AudioSetWindowScores] = (),
    signal_measurements: SignalMeasurements | None = None,
    model_fingerprint: str | None = None,
    label_map_sha256: str | None = None,
    preprocessing_sha256: str | None = None,
    thresholds: AcousticThresholds | None = None,
) -> AudioChunk:
    """Attach AST and signal metadata without touching speaker-owned fields."""

    policy = thresholds or AcousticThresholds()
    windows = tuple(window_scores)
    for window in windows:
        if window.start_ms < chunk.start_ms or window.end_ms > chunk.end_ms:
            raise ContractValidationError("AudioSet window exceeds chunk interval")
    signal = signal_measurements or SignalMeasurements(0.0, -120.0, -120.0, 1.0)
    if not windows:
        metadata = _unknown_metadata(
            signal=signal,
            model_fingerprint=model_fingerprint,
            label_map_sha256=label_map_sha256,
            preprocessing_sha256=preprocessing_sha256,
        )
    else:
        raw_scores = aggregate_audio_set_scores(windows)
        music_score = _score(raw_scores, MUSIC_LABEL_KEYWORDS)
        speech_score = _score(raw_scores, SPEECH_LABEL_KEYWORDS)
        activity = tuple(
            sorted(
                score.label
                for score in raw_scores
                if score.probability >= policy.activity_presence
                and _label_matches(score.label, NON_SPEECH_LABEL_KEYWORDS)
            )
        )
        level = _music_level(music_score, speech_score, policy)
        metadata = AcousticConditionMetadata(
            version=ACOUSTIC_CONDITION_VERSION,
            threshold_version=policy.version,
            has_music=level in {"background", "dominant"},
            music_level=level,  # type: ignore[arg-type]
            has_speech=speech_score >= policy.speech_presence,
            non_speech_activity=activity,
            audio_quality=_audio_quality(signal, policy),
            signal_measurements=signal,
            raw_scores=raw_scores,
            window_scores=windows,
            music_score=music_score,
            speech_score=speech_score,
            model_fingerprint=model_fingerprint,
            label_map_sha256=label_map_sha256,
            preprocessing_sha256=preprocessing_sha256,
        )
    current = chunk.condition or ChunkCondition()
    condition = replace(
        current,
        acoustic_labels=metadata.non_speech_activity,
        music=metadata.has_music,
        degraded=metadata.audio_quality == "degraded",
        acoustic_metadata=metadata,
        acoustic_correction=None,
    )
    return replace(chunk, condition=condition)


def derive_acoustic_conditions(
    chunks: Sequence[AudioChunk],
    *,
    window_scores: Mapping[str, Sequence[AudioSetWindowScores]] | None = None,
    signal_measurements: Mapping[str, SignalMeasurements] | None = None,
    model_fingerprint: str | None = None,
    label_map_sha256: str | None = None,
    preprocessing_sha256: str | None = None,
    thresholds: AcousticThresholds | None = None,
) -> tuple[AudioChunk, ...]:
    """Apply the same deterministic mapping independently to frozen chunks."""

    scores = window_scores or {}
    signals = signal_measurements or {}
    return tuple(
        derive_acoustic_condition(
            chunk,
            window_scores=scores.get(chunk.chunk_id, ()),
            signal_measurements=signals.get(chunk.chunk_id),
            model_fingerprint=model_fingerprint,
            label_map_sha256=label_map_sha256,
            preprocessing_sha256=preprocessing_sha256,
            thresholds=thresholds,
        )
        for chunk in chunks
    )


def apply_acoustic_condition_correction(
    chunk: AudioChunk, correction: AcousticConditionCorrection
) -> AudioChunk:
    """Return a view with a human correction while retaining the machine seed."""

    if chunk.condition is None or chunk.condition.acoustic_metadata is None:
        raise ContractValidationError("acoustic correction requires an acoustic machine seed")
    if correction.seed_version != chunk.condition.acoustic_metadata.version:
        raise ContractValidationError("acoustic correction seed version does not match chunk seed")
    return replace(chunk, condition=replace(chunk.condition, acoustic_correction=correction))


def development_confusion(
    examples: Sequence[Mapping[str, object]],
    *,
    thresholds: AcousticThresholds | None = None,
) -> dict[str, object]:
    """Report deterministic music-level confusion for development examples only."""

    policy = thresholds or AcousticThresholds()
    levels = ("none", "background", "dominant", "uncertain")
    confusion = {actual: {predicted: 0 for predicted in levels} for actual in levels}
    for index, example in enumerate(examples):
        if example.get("partition", "development") != "development":
            raise ContractValidationError(f"calibration example {index} is not development data")
        actual = example.get("music_level")
        if actual not in levels:
            raise ContractValidationError(f"calibration example {index} has an invalid label")
        predicted = _music_level(
            float(example["music_score"]), float(example["speech_score"]), policy
        )
        confusion[actual][predicted] += 1
    return {
        "version": policy.version,
        "partition": "development",
        "thresholds": policy.to_dict(),
        "confusion": confusion,
        "count": len(examples),
    }


# Descriptive aliases used by stage callers and focused tests.
attach_acoustic_condition = derive_acoustic_condition
classify_acoustic_conditions = derive_acoustic_condition
build_development_confusion = development_confusion


__all__ = [
    "ACOUSTIC_CONDITION_VERSION",
    "ACOUSTIC_THRESHOLD_VERSION",
    "AUDIOSET_MAX_LENGTH",
    "AUDIOSET_MODEL_ID",
    "AUDIOSET_MODEL_REPOSITORY",
    "AUDIOSET_MODEL_REVISION",
    "AUDIOSET_SAMPLE_RATE_HZ",
    "AUDIOSET_WINDOW_MS",
    "AcousticThresholds",
    "ASTWindow",
    "SIGNAL_MEASUREMENT_VERSION",
    "aggregate_audio_set_scores",
    "apply_acoustic_condition_correction",
    "ast_preprocessing",
    "attach_acoustic_condition",
    "build_ast_windows",
    "build_development_confusion",
    "classify_acoustic_conditions",
    "derive_acoustic_condition",
    "derive_acoustic_conditions",
    "development_confusion",
    "measure_signal",
    "preprocessing_fingerprint",
    "relevant_class_ids",
]

"""Frozen corpus loading and read-only source verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Final


class CorpusValidationError(ValueError):
    """Raised when a manifest or archive source cannot be verified."""


_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EPISODE_KEYS: Final = frozenset(
    {
        "relative_filename",
        "episode_date",
        "sha256",
        "size_bytes",
        "duration_ms",
        "codec",
        "channels",
        "sample_rate_hz",
    }
)


def _safe_relative_path(filename: str, *, label: str) -> Path:
    if not filename or "\x00" in filename:
        raise CorpusValidationError(f"{label} must be non-empty text")
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise CorpusValidationError(f"{label} must stay beneath the archive root")
    if "\\" in filename:
        raise CorpusValidationError(f"{label} must use POSIX separators")
    return path


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    """Expected immutable metadata for one source episode."""

    relative_filename: str
    episode_date: date
    sha256: str
    size_bytes: int
    duration_ms: int
    codec: str
    channels: int
    sample_rate_hz: int

    @classmethod
    def from_mapping(cls, value: object, index: int) -> EpisodeManifest:
        if not isinstance(value, dict):
            raise CorpusValidationError(f"episodes[{index}] must be an object")
        missing = _REQUIRED_EPISODE_KEYS - value.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise CorpusValidationError(f"episodes[{index}] missing keys: {names}")

        filename = value["relative_filename"]
        if not isinstance(filename, str):
            raise CorpusValidationError(
                f"episodes[{index}].relative_filename must be non-empty text"
            )
        _safe_relative_path(filename, label=f"episodes[{index}].relative_filename")

        raw_date = value["episode_date"]
        try:
            episode_date = date.fromisoformat(raw_date) if isinstance(raw_date, str) else None
        except ValueError as exc:
            raise CorpusValidationError(f"episodes[{index}].episode_date is not ISO-8601") from exc
        if episode_date is None:
            raise CorpusValidationError(f"episodes[{index}].episode_date is not ISO-8601")

        sha256 = value["sha256"]
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise CorpusValidationError(f"episodes[{index}].sha256 must be a lowercase SHA-256")

        numbers: dict[str, int] = {}
        for key in ("size_bytes", "duration_ms", "channels", "sample_rate_hz"):
            number = value[key]
            if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                raise CorpusValidationError(f"episodes[{index}].{key} must be a positive integer")
            numbers[key] = number

        codec = value["codec"]
        if not isinstance(codec, str) or not codec:
            raise CorpusValidationError(f"episodes[{index}].codec must be non-empty text")
        return cls(
            relative_filename=filename,
            episode_date=episode_date,
            sha256=sha256,
            codec=codec,
            **numbers,
        )

    def as_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["episode_date"] = self.episode_date.isoformat()
        return result


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """The frozen 20-episode manifest and its source-byte checksum."""

    schema_version: int
    observed_at: date
    selection: str
    golden_episode: str
    episodes: tuple[EpisodeManifest, ...]
    sha256: str

    @classmethod
    def from_bytes(cls, payload: bytes) -> CorpusManifest:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusValidationError("manifest must be valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise CorpusValidationError("manifest must be a JSON object")
        if value.get("schema_version") != 1:
            raise CorpusValidationError("manifest schema_version must be 1")
        observed_at_raw = value.get("observed_at")
        try:
            observed_at = (
                date.fromisoformat(observed_at_raw) if isinstance(observed_at_raw, str) else None
            )
        except ValueError as exc:
            raise CorpusValidationError("manifest observed_at is not ISO-8601") from exc
        if observed_at is None:
            raise CorpusValidationError("manifest observed_at is not ISO-8601")
        selection = value.get("selection")
        if not isinstance(selection, str) or not selection:
            raise CorpusValidationError("manifest selection must be non-empty text")
        golden_episode = value.get("golden_episode")
        if not isinstance(golden_episode, str) or not golden_episode:
            raise CorpusValidationError("manifest golden_episode must be non-empty text")
        raw_episodes = value.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) != 20:
            count = len(raw_episodes) if isinstance(raw_episodes, list) else "not a list"
            raise CorpusValidationError(f"manifest must contain exactly 20 episodes (got {count})")
        episodes = tuple(
            EpisodeManifest.from_mapping(item, index) for index, item in enumerate(raw_episodes)
        )
        filenames = [episode.relative_filename for episode in episodes]
        if len(set(filenames)) != len(filenames):
            raise CorpusValidationError("manifest episode filenames must be unique")
        if golden_episode not in filenames:
            raise CorpusValidationError("manifest golden_episode must name an episode")
        dates = [episode.episode_date for episode in episodes]
        if min(dates) != date(2026, 7, 1) or max(dates) != date(2026, 9, 10):
            raise CorpusValidationError("manifest date range must be 2026-07-01 through 2026-09-10")
        return cls(
            schema_version=1,
            observed_at=observed_at,
            selection=selection,
            golden_episode=golden_episode,
            episodes=episodes,
            sha256=hashlib.sha256(payload).hexdigest(),
        )


def load_manifest(path: str | os.PathLike[str]) -> CorpusManifest:
    """Load and validate the supplied frozen manifest without touching SQLite."""

    manifest_path = Path(path)
    try:
        payload = manifest_path.read_bytes()
    except OSError as exc:
        raise CorpusValidationError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return CorpusManifest.from_bytes(payload)


def _root_and_source(root: Path, relative_filename: str) -> tuple[Path, Path]:
    _safe_relative_path(relative_filename, label="relative_filename")
    try:
        resolved_root = root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise CorpusValidationError(f"archive root is not readable: {root}: {exc}") from exc
    if not resolved_root.is_dir():
        raise CorpusValidationError(f"archive root is not a directory: {root}")
    source = root / relative_filename
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as exc:
        raise CorpusValidationError(f"source is missing: {relative_filename}: {exc}") from exc
    try:
        resolved_source.relative_to(resolved_root)
    except ValueError as exc:
        raise CorpusValidationError(
            f"source escapes archive root through traversal or symlink: {relative_filename}"
        ) from exc
    if not resolved_source.is_file():
        raise CorpusValidationError(f"source is not a regular file: {relative_filename}")
    return resolved_root, resolved_source


def resolve_source(root: str | os.PathLike[str], relative_filename: str) -> Path:
    """Resolve one manifest path and reject traversal and escaping symlinks."""

    return _root_and_source(Path(root), relative_filename)[1]


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _hash_stable(path: Path) -> tuple[str, int, tuple[int, int, int, int]]:
    before = _stat_signature(path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise CorpusValidationError(f"cannot read source {path.name}: {exc}") from exc
    after = _stat_signature(path)
    if before != after or byte_count != before[2]:
        raise CorpusValidationError(f"source changed while reading: {path.name}")
    return digest.hexdigest(), byte_count, after


def _ffprobe(path: Path, executable: str) -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,channels,sample_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        value = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise CorpusValidationError(f"ffprobe failed for {path.name}: {exc}") from exc
    streams = value.get("streams")
    format_value = value.get("format")
    if not isinstance(streams, list) or not streams or not isinstance(format_value, dict):
        raise CorpusValidationError(f"ffprobe returned incomplete metadata for {path.name}")
    stream = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"),
        None,
    )
    if stream is None:
        raise CorpusValidationError(f"ffprobe returned no stream metadata for {path.name}")
    try:
        duration_ms = int(
            (Decimal(str(format_value["duration"])) * 1000).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
        channels = int(stream["channels"])
        sample_rate_hz = int(stream["sample_rate"])
        codec = str(stream["codec_name"])
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        raise CorpusValidationError(f"ffprobe returned invalid metadata for {path.name}") from exc
    return {
        "duration_ms": duration_ms,
        "codec": codec,
        "channels": channels,
        "sample_rate_hz": sample_rate_hz,
    }


def _empty_result(episode: EpisodeManifest) -> dict[str, Any]:
    return {
        "relative_filename": episode.relative_filename,
        "episode_date": episode.episode_date.isoformat(),
        "expected": episode.as_json(),
        "observed": None,
        "sha256": None,
        "size_bytes": None,
        "status": "error",
        "error": None,
    }


def verify_corpus(
    manifest: CorpusManifest,
    archive_root: str | os.PathLike[str],
    *,
    ffprobe_binary: str = "ffprobe",
) -> dict[str, Any]:
    """Verify all frozen sources and return a report without writing to the archive."""

    root = Path(archive_root)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for episode in manifest.episodes:
        result = _empty_result(episode)
        try:
            source = resolve_source(root, episode.relative_filename)
            digest, size_bytes, _ = _hash_stable(source)
            probe_before = _stat_signature(source)
            observed = _ffprobe(source, ffprobe_binary)
            if probe_before != _stat_signature(source):
                raise CorpusValidationError(
                    f"source changed while probing: {episode.relative_filename}"
                )
            result["sha256"] = digest
            result["size_bytes"] = size_bytes
            result["observed"] = observed
            mismatches: list[str] = []
            if digest != episode.sha256:
                mismatches.append(f"sha256 expected {episode.sha256}, got {digest}")
            if size_bytes != episode.size_bytes:
                mismatches.append(f"size_bytes expected {episode.size_bytes}, got {size_bytes}")
            for key in ("duration_ms", "codec", "channels", "sample_rate_hz"):
                expected = getattr(episode, key)
                actual = observed[key]
                if actual != expected:
                    mismatches.append(f"{key} expected {expected}, got {actual}")
            if mismatches:
                raise CorpusValidationError("; ".join(mismatches))
            result["status"] = "ok"
        except CorpusValidationError as exc:
            result["error"] = str(exc)
            errors.append(f"{episode.relative_filename}: {exc}")
        results.append(result)

    report: dict[str, Any] = {
        "report_schema_version": 1,
        "valid": not errors,
        "verified_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "stage": "corpus-validation",
            "manifest_sha256": manifest.sha256,
            "hash_algorithm": "sha256",
            "ffprobe": ffprobe_binary,
            "duration_rounding": "decimal half-up to integer milliseconds",
            "model": None,
            "configuration": {
                "read_only_archive": True,
                "time_origin": "original episode start; intervals are [start_ms,end_ms)",
            },
        },
        "manifest": {
            "schema_version": manifest.schema_version,
            "observed_at": manifest.observed_at.isoformat(),
            "selection": manifest.selection,
            "golden_episode": manifest.golden_episode,
            "episode_count": len(manifest.episodes),
            "date_min": min(item.episode_date for item in manifest.episodes).isoformat(),
            "date_max": max(item.episode_date for item in manifest.episodes).isoformat(),
        },
        "archive": {
            "root": str(root),
            "mount_must_be_read_only": True,
            "writes_performed": False,
        },
        "files": results,
        "errors": errors,
    }
    return report


def write_report(report: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Atomically publish a successful report outside the source archive."""

    if not report.get("valid"):
        raise CorpusValidationError("refusing to publish an invalid corpus report")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise CorpusValidationError(f"cannot publish report {destination}: {exc}") from exc

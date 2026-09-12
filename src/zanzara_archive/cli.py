"""Command-line entry point for the Zanzara Archive application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from typing import Any

from zanzara_archive import __version__
from zanzara_archive.artifacts import ArtifactPublisher
from zanzara_archive.asr import transcribe_windowed
from zanzara_archive.contracts import AdapterFailure, AudioArtifact
from zanzara_archive.corpus import (
    CorpusValidationError,
    load_manifest,
    resolve_source,
    verify_corpus,
    write_report,
)
from zanzara_archive.jobs import DurableWorker, synthetic_runner
from zanzara_archive.model_locks import (
    ModelLockError,
    model_fingerprint_from_lock,
    validate_model_lock,
)
from zanzara_archive.storage import SQLiteRepository, StorageError


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser without starting services or reading archive data."""
    parser = argparse.ArgumentParser(
        prog="zanzara",
        description="Local-first tools for the Zanzara audio archive.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    corpus = commands.add_parser("corpus", help="validate frozen archive sources")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    verify = corpus_commands.add_parser("verify", help="verify manifest hashes and media metadata")
    verify.add_argument("--manifest", required=True, help="path to the frozen corpus manifest")
    verify.add_argument("--archive-root", required=True, help="read-only archive root")
    verify.add_argument("--output", help="write the successful JSON report outside the archive")
    verify.add_argument(
        "--ffprobe", default="ffprobe", help="ffprobe executable (default: ffprobe)"
    )
    models = commands.add_parser("models", help="validate the immutable model lock")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_verify = model_commands.add_parser("verify", help="verify model artifacts and smoke lock")
    model_verify.add_argument("--lock", required=True, help="path to models.lock.json")
    process = commands.add_parser("process", help="run one local processing stage")
    process.add_argument("--manifest", required=True, help="path to the frozen corpus manifest")
    process.add_argument("--episode", required=True, help="manifest relative filename")
    process.add_argument("--stage", required=True, choices=("asr", "diarization"))
    process.add_argument(
        "--archive-root",
        default=os.environ.get("ZANZARA_ARCHIVE_ROOT", "/export/scratch/archive/zanzara"),
        help="read-only archive root",
    )
    process.add_argument("--endpoint", help="override the local inference endpoint")
    process.add_argument(
        "--diarization-endpoint",
        default=os.environ.get("DIARIZATION_ENDPOINT", "http://127.0.0.1:18081"),
        help="local Community-1 endpoint",
    )
    process.add_argument(
        "--artifact-root",
        default=os.environ.get("ZANZARA_ARTIFACT_ROOT", ".git/zanzara-artifacts"),
        help="private immutable artifact root",
    )
    process.add_argument("--model-lock", default="models.lock.json")
    process.add_argument("--ffmpeg", default="ffmpeg")
    process.add_argument("--language")
    jobs = commands.add_parser("jobs", help="manage durable worker jobs")
    job_commands = jobs.add_subparsers(dest="jobs_command", required=True)
    enqueue = job_commands.add_parser("enqueue", help="enqueue one stage job")
    enqueue.add_argument("--database", required=True)
    enqueue.add_argument("--job-id", required=True)
    enqueue.add_argument("--stage", required=True)
    enqueue.add_argument("--source-sha256")
    enqueue.add_argument("--request-id")
    enqueue.add_argument("--paid", action="store_true")
    status = job_commands.add_parser("status", help="show one job")
    status.add_argument("--database", required=True)
    status.add_argument("job_id")
    retry = job_commands.add_parser("retry", help="retry an exhausted failed job")
    retry.add_argument("--database", required=True)
    retry.add_argument("job_id")
    cancel = job_commands.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("--database", required=True)
    cancel.add_argument("job_id")
    recover = job_commands.add_parser("recover", help="requeue expired leases")
    recover.add_argument("--database", required=True)
    worker = commands.add_parser("worker", help="run the single durable synthetic worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    run = worker_commands.add_parser("run", help="claim and complete one queued job")
    run.add_argument("--database", required=True)
    run.add_argument("--owner", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the currently available application commands."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "models":
        try:
            result = validate_model_lock(arguments.lock)
        except ModelLockError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "process":
        try:
            result = _process_stage(arguments)
        except (AdapterFailure, CorpusValidationError, OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if arguments.command not in {"corpus", "jobs", "worker"}:
        parser.print_help()
        return 0
    if arguments.command == "jobs":
        try:
            repository = SQLiteRepository.open(arguments.database)
            try:
                if arguments.jobs_command == "enqueue":
                    result = repository.enqueue_job(
                        job_id=arguments.job_id,
                        stage=arguments.stage,
                        source_sha256=arguments.source_sha256,
                        request_id=arguments.request_id,
                        paid=arguments.paid,
                    )
                elif arguments.jobs_command == "status":
                    result = repository.fetch_job(arguments.job_id)
                    if result is None:
                        parser.error(f"unknown job {arguments.job_id}")
                elif arguments.jobs_command == "retry":
                    result = repository.retry_job(arguments.job_id)
                elif arguments.jobs_command == "cancel":
                    result = repository.cancel_job(arguments.job_id)
                else:
                    result = {"recovered": repository.recover_expired_leases()}
            finally:
                repository.close()
        except (StorageError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if arguments.command == "worker":
        try:
            repository = SQLiteRepository.open(arguments.database)
            try:
                result = DurableWorker(
                    repository,
                    arguments.owner,
                    synthetic_runner(),
                ).run_once()
            finally:
                repository.close()
        except (StorageError, KeyError, ValueError) as exc:
            parser.error(str(exc))
        payload = {
            "job": result.job.to_dict() if result.job else None,
            "completed": result.completed,
            "error": result.error.to_dict() if result.error else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    try:
        manifest = load_manifest(arguments.manifest)
        report = verify_corpus(
            manifest,
            arguments.archive_root,
            ffprobe_binary=arguments.ffprobe,
        )
        if arguments.output and report["valid"]:
            write_report(report, arguments.output)
    except CorpusValidationError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["valid"] else 1


def _process_stage(arguments: argparse.Namespace) -> dict[str, Any]:
    """Process one episode through a bounded local stage and publish atomically."""

    if arguments.stage == "diarization":
        return _process_diarization_stage(arguments)
    return _process_asr_stage(arguments)


def _process_asr_stage(arguments: argparse.Namespace) -> dict[str, Any]:
    """Process one episode through the bounded Parakeet stage."""

    manifest = load_manifest(arguments.manifest)
    episode = next(
        (item for item in manifest.episodes if item.relative_filename == arguments.episode), None
    )
    if episode is None:
        raise CorpusValidationError(f"episode is not present in manifest: {arguments.episode}")
    source = resolve_source(arguments.archive_root, episode.relative_filename)
    source_audio = AudioArtifact(
        artifact_id=f"source-{episode.sha256[:16]}",
        source_sha256=episode.sha256,
        format=episode.codec,
        duration_ms=episode.duration_ms,
        sample_rate_hz=episode.sample_rate_hz,
        channels=episode.channels,
        preprocessing={"manifest": manifest.sha256, "relative_filename": episode.relative_filename},
    )
    model = model_fingerprint_from_lock(arguments.model_lock, "parakeet")
    adapter_endpoint = arguments.endpoint or os.environ.get(
        "PARAKEET_ENDPOINT", "http://127.0.0.1:18080"
    )
    request_id = f"asr-{episode.sha256[:16]}"

    def load_window(window: Any) -> bytes:
        completed = subprocess.run(
            [
                arguments.ffmpeg,
                "-v",
                "error",
                "-i",
                str(source),
                "-ss",
                f"{window.decode_start_ms / 1000:.3f}",
                "-t",
                f"{window.decode_duration_ms / 1000:.3f}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "wav",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise OSError(f"ffmpeg could not decode {episode.relative_filename}: {detail}")
        return completed.stdout

    from zanzara_archive.inference import ParakeetAdapter

    transcript, windows, raw_responses = transcribe_windowed(
        ParakeetAdapter(adapter_endpoint, model),
        source_audio,
        load_window,
        request_id=request_id,
        language=arguments.language,
    )
    transcript.require_production()
    configuration = {
        "stage": "asr",
        "window_ms": 300_000,
        "context_ms": 5_000,
        "language": arguments.language,
        "decoder": "ffmpeg",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "manifest_sha256": manifest.sha256,
        "source_sha256": episode.sha256,
        "model_fingerprint_sha256": model.fingerprint_sha256,
    }
    stage_key = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    publisher = ArtifactPublisher(arguments.artifact_root)
    artifact = publisher.publish(
        source_sha256=episode.sha256,
        stage="asr",
        stage_key=stage_key,
        files={
            "transcript.json": json.dumps(
                transcript.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "windows.json": json.dumps(
                [window.to_dict() for window in windows], sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "raw_responses.json": json.dumps(
                list(raw_responses), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        },
        pipeline_version="p1-02",
        model_fingerprint_sha256=model.fingerprint_sha256,
        artifact_id=f"asr-{stage_key[:24]}",
        provenance={
            "configuration": configuration,
            "source": source_audio.to_dict(),
            "model": model.to_dict(),
            "endpoint": adapter_endpoint,
            "window_count": len(windows),
        },
    )
    return {
        "stage": "asr",
        "artifact": artifact.to_dict(),
        "artifact_path": str(publisher.artifact_path(episode.sha256, "asr", stage_key)),
        "word_count": len(transcript.words),
        "duration_ms": transcript.duration_ms,
    }


def _process_diarization_stage(arguments: argparse.Namespace) -> dict[str, Any]:
    """Diarize one complete episode once and publish both views."""

    manifest = load_manifest(arguments.manifest)
    episode = next(
        (item for item in manifest.episodes if item.relative_filename == arguments.episode), None
    )
    if episode is None:
        raise CorpusValidationError(f"episode is not present in manifest: {arguments.episode}")
    source = resolve_source(arguments.archive_root, episode.relative_filename)
    source_audio = AudioArtifact(
        artifact_id=f"source-{episode.sha256[:16]}",
        source_sha256=episode.sha256,
        format=episode.codec,
        duration_ms=episode.duration_ms,
        sample_rate_hz=episode.sample_rate_hz,
        channels=episode.channels,
        preprocessing={"manifest": manifest.sha256, "relative_filename": episode.relative_filename},
    )
    model = model_fingerprint_from_lock(arguments.model_lock, "diarization")
    from zanzara_archive.inference import DiarizerAdapter, render_rttm

    request_id = f"diarization-{episode.sha256[:16]}"
    endpoint = arguments.endpoint or arguments.diarization_endpoint
    source_bytes = source.read_bytes()
    parsed = DiarizerAdapter(endpoint, model).diarize_bytes(
        source_audio,
        source_bytes,
        request_id=request_id,
    )
    result = parsed.result
    configuration = {
        "stage": "diarization",
        "clustering": "episode-wide",
        "decoder": "ffmpeg in Community-1 service",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "manifest_sha256": manifest.sha256,
        "source_sha256": episode.sha256,
        "model_fingerprint_sha256": model.fingerprint_sha256,
        "rttm_boundary_precision_ms": 1,
    }
    stage_key = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    publisher = ArtifactPublisher(arguments.artifact_root)
    artifact = publisher.publish(
        source_sha256=episode.sha256,
        stage="diarization",
        stage_key=stage_key,
        files={
            "diarization.json": json.dumps(
                result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "standard.rttm": render_rttm(
                result.standard_turns, file_id=episode.relative_filename
            ).encode("utf-8"),
            "exclusive.rttm": render_rttm(
                result.exclusive_turns, file_id=episode.relative_filename
            ).encode("utf-8"),
            "raw_response.json": json.dumps(
                parsed.raw_response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
        },
        pipeline_version="p1-03",
        model_fingerprint_sha256=model.fingerprint_sha256,
        artifact_id=f"diarization-{stage_key[:24]}",
        provenance={
            "configuration": configuration,
            "source": source_audio.to_dict(),
            "decoded_input": {
                "decoder": "ffmpeg in Community-1 service",
                "target_sample_rate_hz": 16_000,
                "target_channels": 1,
                "time_origin": "original episode",
            },
            "model": model.to_dict(),
            "endpoint": endpoint,
            "episode_wide": True,
            "standard_turn_count": len(result.standard_turns),
            "exclusive_turn_count": len(result.exclusive_turns),
            "overlap_count": len(result.overlaps),
        },
    )
    return {
        "stage": "diarization",
        "artifact": artifact.to_dict(),
        "artifact_path": str(publisher.artifact_path(episode.sha256, "diarization", stage_key)),
        "standard_turn_count": len(result.standard_turns),
        "exclusive_turn_count": len(result.exclusive_turns),
        "overlap_count": len(result.overlaps),
        "speaker_ids": sorted(
            {turn.speaker_id for turn in result.standard_turns}
            | {turn.speaker_id for turn in result.exclusive_turns}
        ),
        "duration_ms": result.duration_ms,
    }

"""Command-line entry point for the Zanzara Archive application."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zanzara_archive import __version__
from zanzara_archive.artifacts import ArtifactPublicationError, ArtifactPublisher
from zanzara_archive.asr import transcribe_windowed
from zanzara_archive.calibration import build_batch, validate_batch
from zanzara_archive.chunking import (
    Community1AdaptiveConfig,
    Community1NativeAdaptiveConfig,
    segment_chunks_with_metadata,
    validate_chunk_coverage,
)
from zanzara_archive.contracts import (
    AdapterFailure,
    AudioArtifact,
    AudioChunk,
    DiarizationResult,
    TranscriptResult,
)
from zanzara_archive.corpus import (
    CorpusValidationError,
    load_manifest,
    resolve_source,
    verify_corpus,
    write_report,
)
from zanzara_archive.evaluation import (
    EvaluationValidationError,
    current_git_commit,
    evaluate_files,
    validate_reference,
    write_evaluation_artifacts,
)
from zanzara_archive.jobs import DurableWorker, synthetic_runner
from zanzara_archive.model_locks import (
    ModelLockError,
    model_fingerprint_from_lock,
    validate_model_lock,
)
from zanzara_archive.stages import stage_fingerprint
from zanzara_archive.storage import SQLiteRepository, StorageError
from zanzara_archive.transcripts import build_attributed_transcript, render_exports

LOOPBACK_WEB_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


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
    evaluation = commands.add_parser("evaluation", help="validate private evaluation inputs")
    evaluation_commands = evaluation.add_subparsers(dest="evaluation_command", required=True)
    reference = evaluation_commands.add_parser(
        "validate-reference", help="validate the reviewed golden reference contract"
    )
    reference.add_argument("--corpus", required=True, help="path to the frozen corpus manifest")
    reference.add_argument(
        "--reference", required=True, help="reference.json or its private artifact directory"
    )
    score = evaluation_commands.add_parser(
        "score", help="score a reference and hypothesis and write a private evaluation run"
    )
    score.add_argument(
        "--reference", required=True, help="reference.json or private artifact directory"
    )
    score.add_argument(
        "--hypothesis", required=True, help="hypothesis.json or private artifact directory"
    )
    score.add_argument("--output", required=True, help="new private E6 evaluation run directory")
    score.add_argument(
        "--reviewed-commit",
        help="code commit to record; defaults to the current Git HEAD when available",
    )
    run = evaluation_commands.add_parser(
        "run", help="execute the real local golden baseline and write a private E6 run"
    )
    run.add_argument("--suite", required=True, choices=("golden",))
    run.add_argument(
        "--reference", required=True, help="reviewed reference JSON or artifact directory"
    )
    run.add_argument("--split", required=True, help="frozen E1 split JSON")
    run.add_argument("--output", required=True, help="new private E6 evaluation run directory")
    run.add_argument("--corpus", default="planning/corpus-20.json")
    run.add_argument(
        "--archive-root",
        default=os.environ.get("ZANZARA_ARCHIVE_ROOT", "/export/scratch/archive/zanzara"),
    )
    run.add_argument(
        "--artifact-root",
        default=os.environ.get("ZANZARA_ARTIFACT_ROOT", ".git/zanzara-artifacts"),
    )
    run.add_argument(
        "--database", default=os.environ.get("ZANZARA_DATABASE", ".git/zanzara-state/state.db")
    )
    run.add_argument("--model-lock", default="models.lock.json")
    run.add_argument("--endpoint", help="override the local Parakeet endpoint")
    run.add_argument(
        "--diarization-endpoint",
        default=os.environ.get("DIARIZATION_ENDPOINT", "http://127.0.0.1:18081"),
    )
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--language")
    web = commands.add_parser("web", help="run the annotation UI")
    web.add_argument("--database", required=True, help="local SQLite state path")
    web.add_argument(
        "--artifact-root",
        default=os.environ.get("ZANZARA_ARTIFACT_ROOT", ".git/zanzara-artifacts"),
    )
    web.add_argument(
        "--archive-root",
        default=os.environ.get("ZANZARA_ARCHIVE_ROOT", "/export/scratch/archive/zanzara"),
    )
    web.add_argument("--manifest", default="planning/corpus-20.json")
    web.add_argument(
        "--host",
        default=os.environ.get("ZANZARA_WEB_HOST", "127.0.0.1"),
        help="bind address; non-loopback addresses require --allow-network",
    )
    web.add_argument("--port", type=int, default=8000)
    web.add_argument(
        "--allow-network",
        action="store_true",
        help="explicitly allow non-loopback access to the unauthenticated local editor",
    )
    calibration = commands.add_parser(
        "calibration", help="prepare and export development music review batches"
    )
    calibration_commands = calibration.add_subparsers(dest="calibration_command", required=True)
    generate = calibration_commands.add_parser(
        "generate", help="generate a deterministic development review batch"
    )
    generate.add_argument("--split", required=True, help="private episode split JSON")
    generate.add_argument("--manifest", default="planning/corpus-20.json")
    generate.add_argument("--batch-id", required=True)
    generate.add_argument("--clips-per-episode", type=int, default=8)
    generate.add_argument("--output", required=True, help="private batch JSON output")
    prepare = calibration_commands.add_parser(
        "prepare", help="validate and persist a calibration batch"
    )
    prepare.add_argument("--batch", required=True, help="private batch JSON")
    prepare.add_argument("--manifest", default="planning/corpus-20.json")
    prepare.add_argument(
        "--database", default=os.environ.get("ZANZARA_DATABASE", ".git/zanzara-state/state.db")
    )
    export = calibration_commands.add_parser(
        "export", help="export append-only calibration decisions"
    )
    export.add_argument("--batch-id", required=True)
    export.add_argument(
        "--database", default=os.environ.get("ZANZARA_DATABASE", ".git/zanzara-state/state.db")
    )
    export.add_argument("--output", required=True)
    chunks = commands.add_parser("chunks", help="build and validate benchmark chunk manifests")
    chunk_commands = chunks.add_subparsers(dest="chunks_command", required=True)
    chunk_build = chunk_commands.add_parser(
        "build", help="build Community-1 adaptive chunks from known artifacts"
    )
    chunk_build.add_argument(
        "--algorithm",
        required=True,
        choices=("community1-adaptive-v1", "community1-native-adaptive-v2"),
    )
    chunk_build.add_argument("--manifest", required=True)
    chunk_build.add_argument("--output", required=True)
    chunk_build.add_argument(
        "--artifact-root",
        default=os.environ.get("ZANZARA_ARTIFACT_ROOT", ".git/zanzara-artifacts"),
    )
    chunk_build.add_argument("--model-lock", default="models.lock.json")
    chunk_build.add_argument("--episode", action="append")
    chunk_validate = chunk_commands.add_parser(
        "validate", help="validate a generated benchmark chunk manifest"
    )
    chunk_validate.add_argument("--manifest", required=True)
    chunk_validate.add_argument("--output", help="optional normalized validation output")
    chunk_validate.add_argument(
        "--corpus", default="planning/corpus-20.json", help="frozen corpus manifest"
    )
    chunk_report = chunk_commands.add_parser(
        "report", help="compare chunk duration and boundary distributions"
    )
    chunk_report.add_argument("--manifest", required=True)
    chunk_report.add_argument("--compare", required=True)
    chunk_report.add_argument("--output", required=True)
    diarization = commands.add_parser(
        "diarization", help="validate and export Community-1 diarization artifacts"
    )
    diarization_commands = diarization.add_subparsers(dest="diarization_command", required=True)
    native_activity = diarization_commands.add_parser(
        "native-activity", help="export captured native speaker-count activity"
    )
    native_activity.add_argument("--manifest", required=True)
    native_activity.add_argument("--output", required=True)
    native_activity.add_argument(
        "--artifact-root",
        default=os.environ.get("ZANZARA_ARTIFACT_ROOT", ".git/zanzara-artifacts"),
    )
    native_activity.add_argument("--model-lock", default="models.lock.json")
    native_activity.add_argument("--episode", action="append")
    process = commands.add_parser("process", help="run one local processing stage")
    process.add_argument("--manifest", required=True, help="path to the frozen corpus manifest")
    process.add_argument("--episode", required=True, help="manifest relative filename")
    process.add_argument("--stage", required=True, choices=("asr", "diarization", "attribution"))
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
    process.add_argument(
        "--asr-artifact",
        help=(
            "explicit P1-02 artifact directory for attribution (defaults to its locked generation)"
        ),
    )
    process.add_argument(
        "--diarization-artifact",
        help=(
            "explicit P1-03 artifact directory for attribution (defaults to its locked generation)"
        ),
    )
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
    if arguments.command == "evaluation":
        if arguments.evaluation_command == "validate-reference":
            report = validate_reference(arguments.corpus, arguments.reference)
            print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
            return 0 if report["valid"] else 1
        if arguments.evaluation_command == "score":
            try:
                report = evaluate_files(
                    arguments.reference,
                    arguments.hypothesis,
                    reviewed_commit=arguments.reviewed_commit or current_git_commit(),
                )
                artifacts = write_evaluation_artifacts(report, arguments.output)
            except (EvaluationValidationError, OSError) as exc:
                parser.error(str(exc))
            print(
                json.dumps(
                    {**artifacts, "pass_block": report["pass_block"]}, indent=2, sort_keys=True
                )
            )
            return 0 if report["verdict"] == "pass" else 1
        if arguments.evaluation_command == "run":
            try:
                from zanzara_archive.baseline import BaselineRunError, run_baseline

                artifacts = run_baseline(arguments)
            except (
                AdapterFailure,
                ArtifactPublicationError,
                BaselineRunError,
                CorpusValidationError,
                EvaluationValidationError,
                ModelLockError,
                OSError,
                StorageError,
                ValueError,
            ) as exc:
                parser.error(str(exc))
            print(json.dumps(artifacts, indent=2, sort_keys=True))
            return 0 if artifacts["verdict"] == "pass" else 1
    if arguments.command == "calibration":
        try:
            if arguments.calibration_command == "generate":
                manifest = load_manifest(arguments.manifest)
                split = json.loads(Path(arguments.split).read_text(encoding="utf-8"))
                result = build_batch(
                    manifest,
                    split,
                    batch_id=arguments.batch_id,
                    clips_per_episode=arguments.clips_per_episode,
                )
                output = Path(arguments.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                result = {
                    "batch_id": result["batch_id"],
                    "content_sha256": result["content_sha256"],
                    "path": str(output),
                    "total_count": len(result["chunks"]),
                }
            elif arguments.calibration_command == "prepare":
                manifest = load_manifest(arguments.manifest)
                payload = json.loads(Path(arguments.batch).read_text(encoding="utf-8"))
                normalized = validate_batch(payload, manifest)
                repository = SQLiteRepository.open(arguments.database)
                try:
                    repository.record_calibration_batch(normalized)
                finally:
                    repository.close()
                result = {
                    "batch_id": normalized["batch_id"],
                    "content_sha256": normalized["content_sha256"],
                    "total_count": len(normalized["chunks"]),
                }
            else:
                repository = SQLiteRepository.open(arguments.database)
                try:
                    batch = repository.fetch_calibration_batch(arguments.batch_id)
                finally:
                    repository.close()
                if batch is None:
                    parser.error("calibration batch was not found")
                output = Path(arguments.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                payload = {"artifact_type": "p1r-development-music-review", "calibration": batch}
                output.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                result = {
                    "path": str(output),
                    "reviewed_count": len(batch.get("decisions", {})),
                    "total_count": len(batch["chunks"]),
                }
        except (
            CorpusValidationError,
            StorageError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "chunks":
        try:
            if arguments.chunks_command == "build":
                result = _build_chunk_manifest(arguments)
            elif arguments.chunks_command == "validate":
                result = _validate_chunk_manifest(arguments)
            else:
                result = _report_chunk_manifests(arguments)
        except (
            ArtifactPublicationError,
            CorpusValidationError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if arguments.command == "diarization":
        try:
            result = _export_native_activity(arguments)
        except (CorpusValidationError, OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if arguments.command == "web":
        if arguments.host not in LOOPBACK_WEB_HOSTS and not arguments.allow_network:
            parser.error(
                "the annotation UI is loopback-only by default; pass --allow-network "
                "to explicitly enable a non-loopback bind"
            )
        try:
            import uvicorn

            from zanzara_archive.web import create_app
        except ImportError as exc:
            parser.error(f"the web extras are unavailable: {exc}")
        uvicorn.run(
            create_app(
                arguments.database,
                artifact_root=arguments.artifact_root,
                archive_root=arguments.archive_root,
                manifest_path=arguments.manifest,
                network_access=arguments.host not in LOOPBACK_WEB_HOSTS,
            ),
            host=arguments.host,
            port=arguments.port,
        )
        return 0
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
        except (
            AdapterFailure,
            ArtifactPublicationError,
            CorpusValidationError,
            OSError,
            ValueError,
        ) as exc:
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
    if arguments.stage == "attribution":
        return _process_attribution_stage(arguments)
    return _process_asr_stage(arguments)


def _configuration_key(configuration: dict[str, Any]) -> str:
    """Match the immutable keys used by the existing ASR/diarization commands."""

    return hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_digest(manifest: Any) -> str:
    """Hash the canonical manifest representation used as an upstream input."""

    payload = json.dumps(
        manifest.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _chunk_artifact_path(arguments: argparse.Namespace, episode: Any, corpus: Any) -> Path:
    """Resolve one deterministic Community-1 artifact without guessing between runs."""

    expected = _default_attribution_input_path(arguments, episode, corpus, stage="diarization")
    if expected.is_dir() and (expected / "diarization.json").is_file():
        return expected
    root = Path(arguments.artifact_root).expanduser().resolve()
    candidates = sorted(
        path.parent
        for path in (root / episode.sha256 / "diarization").glob("*/diarization.json")
        if path.is_file()
    )
    if not candidates:
        raise ValueError(f"no complete Community-1 artifact found for {episode.relative_filename}")
    if len(candidates) > 1:
        raise ValueError(
            f"multiple Community-1 artifacts found for {episode.relative_filename}; pass one run"
        )
    return candidates[0]


def _build_chunk_manifest(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build a public-safe chunk manifest from private Community-1 artifacts."""

    corpus = load_manifest(arguments.manifest)
    selected = set(arguments.episode or (item.relative_filename for item in corpus.episodes))
    known = {item.relative_filename: item for item in corpus.episodes}
    unknown = selected - known.keys()
    if unknown:
        raise CorpusValidationError(f"episode is not present in manifest: {sorted(unknown)[0]}")
    config = (
        Community1NativeAdaptiveConfig()
        if arguments.algorithm == "community1-native-adaptive-v2"
        else Community1AdaptiveConfig()
    )
    episode_payloads: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for episode in corpus.episodes:
        if episode.relative_filename not in selected:
            continue
        artifact_path = _chunk_artifact_path(arguments, episode, corpus)
        artifact_manifest, diarization_payload, _, _ = _load_stage_payload(
            arguments.artifact_root,
            artifact_path,
            expected_stage="diarization",
            payload_name="diarization.json",
        )
        try:
            diarization = DiarizationResult.from_dict(diarization_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Community-1 artifact for {episode.relative_filename}: {exc}"
            ) from exc
        _validate_stage_provenance(
            artifact_manifest,
            diarization,
            expected_source_sha256=episode.sha256,
            stage="diarization",
        )
        result = segment_chunks_with_metadata(
            episode.relative_filename,
            episode.sha256,
            episode.duration_ms,
            algorithm=arguments.algorithm,
            config=config,
            diarization=diarization,
            partition="development",
        )
        episode_chunks = [chunk.to_dict() for chunk in result.chunks]
        episode_diagnostics = [dict(item) for item in result.boundary_diagnostics]
        chunks.extend(episode_chunks)
        diagnostics.extend(
            {"episode_id": episode.relative_filename, **item} for item in episode_diagnostics
        )
        episode_payloads.append(
            {
                "episode_id": episode.relative_filename,
                "source_sha256": episode.sha256,
                "duration_ms": episode.duration_ms,
                "diarization_artifact_id": diarization.artifact_id,
                "native_activity_artifact_id": (
                    diarization.native_activity.artifact_id
                    if diarization.native_activity is not None
                    else None
                ),
                "diarization_model_fingerprint": diarization.model.fingerprint_sha256,
                "diarization_manifest_sha256": _manifest_digest(artifact_manifest),
                "chunk_count": len(episode_chunks),
            }
        )
    if not chunks:
        raise ValueError("chunk manifest requires at least one selected episode")
    chunks.sort(key=lambda item: (item["episode_id"], item["start_ms"], item["end_ms"]))
    reason_counts: dict[str, int] = {}
    hard_maximum_count = 0
    overlap_conflicted_count = 0
    short_terminal_count = 0
    selection_phase_counts: dict[str, int] = {}
    durations: list[int] = []
    for raw in chunks:
        durations.append(raw["end_ms"] - raw["start_ms"])
        reason = raw["boundary_end_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        hard_maximum_count += reason == "hard_maximum"
        overlap_conflicted_count += bool(raw["boundary_overlap_conflict"])
        short_terminal_count += reason == "episode_end" and durations[-1] < config.minimum_chunk_ms
        phase = raw.get("selection_phase")
        if phase is not None:
            selection_phase_counts[phase] = selection_phase_counts.get(phase, 0) + 1
    normalized: dict[str, Any] = {
        "algorithm": arguments.algorithm,
        "segmentation_version": config.version,
        "segmentation_configuration": config.to_dict(),
        "segmentation_configuration_hash": config.configuration_sha256,
        "corpus_manifest_sha256": corpus.sha256,
        "episodes": episode_payloads,
        "chunks": chunks,
        "boundary_diagnostics": diagnostics,
        "statistics": {
            "episode_count": len(episode_payloads),
            "chunk_count": len(chunks),
            "mean_duration_ms": sum(durations) / len(durations),
            "median_duration_ms": sorted(durations)[len(durations) // 2],
            "p90_duration_ms": sorted(durations)[
                min(len(durations) - 1, (len(durations) * 90 + 99) // 100 - 1)
            ],
            "p95_duration_ms": sorted(durations)[
                min(len(durations) - 1, (len(durations) * 95 + 99) // 100 - 1)
            ],
            "duration_bins": {
                "8-10s": sum(8_000 <= value < 10_000 for value in durations),
                "10-12s": sum(10_000 <= value < 12_000 for value in durations),
                "12-14s": sum(12_000 <= value < 14_000 for value in durations),
                "14-16s": sum(14_000 <= value < 16_000 for value in durations),
                "16-18s": sum(16_000 <= value < 18_000 for value in durations),
                "18-20s": sum(18_000 <= value < 20_000 for value in durations),
                "20-25s": sum(20_000 <= value < 25_000 for value in durations),
                "25-<30s": sum(25_000 <= value < 30_000 for value in durations),
                "exactly-30s": sum(value == 30_000 for value in durations),
            },
            "boundary_reason_counts": reason_counts,
            "selection_phase_counts": selection_phase_counts,
            "hard_maximum_count": hard_maximum_count,
            "acceptance_percentages": {
                "8-16s": 100
                * sum(8_000 <= value <= 16_000 for value in durations)
                / len(durations),
                ">16s": 100 * sum(value > 16_000 for value in durations) / len(durations),
                "exact_hard_maximum": 100
                * sum(value == config.hard_max_ms for value in durations)
                / len(durations),
            },
            "overlap_conflicted_boundary_count": overlap_conflicted_count,
            "short_terminal_chunk_count": short_terminal_count,
        },
    }
    normalized["content_sha256"] = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {
        "path": str(output),
        "content_sha256": normalized["content_sha256"],
        "episode_count": len(episode_payloads),
        "chunk_count": len(chunks),
    }


def _validate_chunk_manifest(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate chunk intervals, provenance, ordering and configuration identity."""

    corpus = load_manifest(arguments.corpus)
    payload = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("chunk manifest must be a JSON object")
    if payload.get("algorithm") not in {
        "community1-adaptive-v1",
        "community1-native-adaptive-v2",
    }:
        raise ValueError("chunk manifest algorithm is not a supported Community-1 algorithm")
    if payload.get("corpus_manifest_sha256") != corpus.sha256:
        raise ValueError("chunk manifest corpus hash does not match the frozen corpus")
    config = (
        Community1NativeAdaptiveConfig.from_dict(payload.get("segmentation_configuration", {}))
        if payload.get("algorithm") == "community1-native-adaptive-v2"
        else Community1AdaptiveConfig.from_dict(payload.get("segmentation_configuration", {}))
    )
    if payload.get("segmentation_version") != config.version:
        raise ValueError("chunk manifest segmentation version does not match configuration")
    if payload.get("segmentation_configuration_hash") != config.configuration_sha256:
        raise ValueError("chunk manifest configuration hash does not match configuration")
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunk manifest requires a non-empty chunks array")
    episodes = {episode.relative_filename: episode for episode in corpus.episodes}
    parsed = []
    for index, raw in enumerate(raw_chunks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"chunks[{index}] must be an object")
        chunk = AudioChunk.from_dict(raw)
        episode = episodes.get(chunk.episode_id)
        if episode is None or chunk.source_sha256 != episode.sha256:
            raise ValueError(f"chunks[{index}] has invalid episode/source provenance")
        if chunk.partition != "development":
            raise ValueError(f"chunks[{index}] is not a development chunk")
        if chunk.segmentation_version != config.version:
            raise ValueError(f"chunks[{index}] has a mismatched segmentation version")
        if chunk.segmentation_configuration_hash != config.configuration_sha256:
            raise ValueError(f"chunks[{index}] has a mismatched configuration hash")
        if chunk.diarization_artifact_id is None:
            raise ValueError(f"chunks[{index}] is missing Community-1 artifact provenance")
        if payload.get("algorithm") == "community1-native-adaptive-v2":
            if chunk.community1_artifact_id is None or chunk.native_activity_artifact_id is None:
                raise ValueError(f"chunks[{index}] is missing native Community-1 provenance")
        parsed.append(chunk)
    sorted_chunks = sorted(parsed, key=lambda item: (item.episode_id, item.start_ms, item.end_ms))
    if list(raw_chunks) != [chunk.to_dict() for chunk in sorted_chunks]:
        raise ValueError("chunk manifest chunks are not stably sorted")
    for episode_id in sorted({chunk.episode_id for chunk in parsed}):
        episode = episodes[episode_id]
        validate_chunk_coverage(
            [chunk for chunk in parsed if chunk.episode_id == episode_id],
            ((0, episode.duration_ms),),
            duration_ms=episode.duration_ms,
            hard_max_ms=config.hard_max_ms,
        )
    if payload.get("algorithm") == "community1-native-adaptive-v2":
        diagnostics = payload.get("boundary_diagnostics")
        if not isinstance(diagnostics, list):
            raise ValueError("native chunk manifest requires boundary diagnostics")
        by_boundary = {
            (str(item.get("episode_id")), item.get("start_ms")): item
            for item in diagnostics
            if isinstance(item, Mapping)
        }
        for chunk in parsed:
            diagnostic = by_boundary.get((chunk.episode_id, chunk.start_ms))
            if not isinstance(diagnostic, Mapping):
                raise ValueError("native chunk is missing its boundary diagnostic")
            if (
                diagnostic.get("end_ms") != chunk.end_ms
                or diagnostic.get("reason") != chunk.boundary_end_reason
            ):
                raise ValueError("native boundary diagnostic disagrees with chunk selection")
            if diagnostic.get("selected_candidate_timestamp_ms") != chunk.end_ms:
                raise ValueError("selected candidate timestamp must equal chunk end")
            if chunk.boundary_end_reason == "hard_maximum" and (
                diagnostic.get("exclusive_transition_count_18_30") != 0
                or diagnostic.get("native_pause_count_18_30") != 0
            ):
                raise ValueError("hard maximum has a natural candidate in [18,30]")
    expected_content = dict(payload)
    content_sha256 = expected_content.pop("content_sha256", None)
    actual_content = hashlib.sha256(
        json.dumps(
            expected_content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if content_sha256 != actual_content:
        raise ValueError("chunk manifest content hash does not match payload")
    result = {
        "valid": True,
        "content_sha256": actual_content,
        "episode_count": len({chunk.episode_id for chunk in parsed}),
        "chunk_count": len(parsed),
        "algorithm": config.version,
    }
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _export_native_activity(arguments: argparse.Namespace) -> dict[str, Any]:
    """Export captured native activity without running Community-1 again."""

    corpus = load_manifest(arguments.manifest)
    selected = set(arguments.episode or (item.relative_filename for item in corpus.episodes))
    known = {item.relative_filename: item for item in corpus.episodes}
    unknown = selected - known.keys()
    if unknown:
        raise CorpusValidationError(f"episode is not present in manifest: {sorted(unknown)[0]}")
    episodes: list[dict[str, Any]] = []
    for episode in corpus.episodes:
        if episode.relative_filename not in selected:
            continue
        artifact_path = _chunk_artifact_path(arguments, episode, corpus)
        _, payload, _, _ = _load_stage_payload(
            arguments.artifact_root,
            artifact_path,
            expected_stage="diarization",
            payload_name="diarization.json",
        )
        try:
            diarization = DiarizationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Community-1 artifact for {episode.relative_filename}: {exc}"
            ) from exc
        if diarization.native_activity is None:
            raise ValueError(
                f"Community-1 artifact for {episode.relative_filename} has no native "
                "speaker-count activity"
            )
        if diarization.native_activity.capture_version != "community1-speaker-count-snapshot-v1":
            raise ValueError(
                f"Community-1 artifact for {episode.relative_filename} lacks current native "
                "speaker-count capture provenance"
            )
        episodes.append(
            {
                "episode_id": episode.relative_filename,
                "source_sha256": episode.sha256,
                "native_activity": diarization.native_activity.to_dict(),
                "community1_artifact_id": diarization.artifact_id,
            }
        )
    if not episodes:
        raise ValueError("native activity export requires at least one selected episode")
    normalized: dict[str, Any] = {
        "artifact_type": "community1-native-activity-v1",
        "corpus_manifest_sha256": corpus.sha256,
        "episodes": episodes,
    }
    normalized["content_sha256"] = hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {
        "path": str(output),
        "content_sha256": normalized["content_sha256"],
        "episode_count": len(episodes),
    }


def _chunk_distribution(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("chunk manifest requires a non-empty chunks array")
    durations = sorted(
        int(item["end_ms"]) - int(item["start_ms"])
        for item in raw_chunks
        if isinstance(item, Mapping)
    )
    if len(durations) != len(raw_chunks):
        raise ValueError("chunk manifest contains a non-object chunk")
    reasons: dict[str, int] = {}
    candidate_types: dict[str, int] = {}
    selection_phases: dict[str, int] = {}
    for item in raw_chunks:
        reason = str(item.get("boundary_end_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
        candidate = str(item.get("selected_candidate_type") or reason)
        candidate_types[candidate] = candidate_types.get(candidate, 0) + 1
        phase = item.get("selection_phase")
        if phase is not None:
            selection_phases[str(phase)] = selection_phases.get(str(phase), 0) + 1

    def percentile(fraction: float) -> int:
        index = min(len(durations) - 1, max(0, int(len(durations) * fraction + 0.999999) - 1))
        return durations[index]

    bins = {
        "8-10s": sum(8_000 <= value < 10_000 for value in durations),
        "10-12s": sum(10_000 <= value < 12_000 for value in durations),
        "12-14s": sum(12_000 <= value < 14_000 for value in durations),
        "14-16s": sum(14_000 <= value < 16_000 for value in durations),
        "16-18s": sum(16_000 <= value < 18_000 for value in durations),
        "18-20s": sum(18_000 <= value < 20_000 for value in durations),
        "20-25s": sum(20_000 <= value < 25_000 for value in durations),
        "25-<30s": sum(25_000 <= value < 30_000 for value in durations),
        "exactly-30s": sum(value == 30_000 for value in durations),
    }
    return {
        "chunk_count": len(durations),
        "mean_duration_ms": sum(durations) / len(durations),
        "median_duration_ms": durations[len(durations) // 2],
        "p90_duration_ms": percentile(0.90),
        "p95_duration_ms": percentile(0.95),
        "duration_bins": bins,
        "boundary_reason_counts": reasons,
        "boundary_type_counts": candidate_types,
        "selection_phase_counts": selection_phases,
        "hard_maximum_count": reasons.get("hard_maximum", 0),
        "acceptance_percentages": {
            "8-16s": 100 * sum(8_000 <= value <= 16_000 for value in durations) / len(durations),
            ">16s": 100 * sum(value > 16_000 for value in durations) / len(durations),
            "exact_hard_maximum": 100
            * sum(value == 30_000 for value in durations)
            / len(durations),
        },
    }


def _report_chunk_manifests(arguments: argparse.Namespace) -> dict[str, Any]:
    """Write a sanitized comparison report for two private chunk manifests."""

    current = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    previous = json.loads(Path(arguments.compare).read_text(encoding="utf-8"))
    if not isinstance(current, Mapping) or not isinstance(previous, Mapping):
        raise ValueError("chunk reports require JSON object manifests")
    report = {
        "algorithm": current.get("algorithm"),
        "current_content_sha256": current.get("content_sha256"),
        "previous_content_sha256": previous.get("content_sha256"),
        "current": _chunk_distribution(current),
        "previous": _chunk_distribution(previous),
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    report["path"] = str(output)
    return report


def _default_attribution_input_path(
    arguments: argparse.Namespace,
    episode: Any,
    manifest: Any,
    *,
    stage: str,
) -> Path:
    """Resolve a prerequisite's deterministic generation without scanning directories."""

    if stage == "asr":
        model = model_fingerprint_from_lock(arguments.model_lock, "parakeet")
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
    else:
        model = model_fingerprint_from_lock(arguments.model_lock, "diarization")
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
    return (
        Path(arguments.artifact_root).expanduser()
        / episode.sha256
        / stage
        / _configuration_key(configuration)
    )


def _load_stage_payload(
    artifact_root: str | os.PathLike[str],
    artifact_path: str | os.PathLike[str] | None,
    *,
    expected_stage: str,
    payload_name: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path]:
    """Read and checksum one complete prerequisite artifact."""

    publisher = ArtifactPublisher(artifact_root)
    root = publisher.root
    candidate = Path(artifact_path).expanduser() if artifact_path is not None else None
    if candidate is None:
        raise ValueError(f"no {expected_stage} artifact was resolved")
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink():
        raise ValueError(f"{expected_stage} artifact must not be a symlink")
    candidate = candidate.resolve()
    if root not in candidate.parents:
        raise ValueError(f"{expected_stage} artifact must be inside the artifact root")
    manifest = publisher._read_complete(candidate)
    if manifest.stage != expected_stage:
        raise ValueError(f"expected a {expected_stage} artifact, found stage {manifest.stage!r}")
    payload_path = candidate / payload_name
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        provenance_path = candidate / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {}
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactPublicationError(
            f"cannot read complete {expected_stage} artifact {candidate}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(provenance, dict):
        raise ArtifactPublicationError(f"{expected_stage} artifact payloads must be JSON objects")
    return manifest, payload, provenance, candidate


def _validate_stage_provenance(
    manifest: Any,
    result: TranscriptResult | DiarizationResult,
    *,
    expected_source_sha256: str,
    stage: str,
) -> None:
    """Reject an input whose payload disagrees with its immutable manifest."""

    if (
        manifest.source_sha256 != expected_source_sha256
        or result.source_sha256 != expected_source_sha256
    ):
        raise ValueError(f"{stage} artifact source provenance does not match the episode")
    if manifest.model_fingerprint_sha256 != result.model.fingerprint_sha256:
        raise ValueError(f"{stage} artifact model provenance does not match its payload")


def _process_attribution_stage(arguments: argparse.Namespace) -> dict[str, Any]:
    """Attribute validated prerequisite artifacts and publish derived exports."""

    manifest = load_manifest(arguments.manifest)
    episode = next(
        (item for item in manifest.episodes if item.relative_filename == arguments.episode), None
    )
    if episode is None:
        raise CorpusValidationError(f"episode is not present in manifest: {arguments.episode}")

    asr_path = arguments.asr_artifact or _default_attribution_input_path(
        arguments, episode, manifest, stage="asr"
    )
    diarization_path = arguments.diarization_artifact or _default_attribution_input_path(
        arguments, episode, manifest, stage="diarization"
    )
    asr_manifest, asr_payload, asr_provenance, asr_directory = _load_stage_payload(
        arguments.artifact_root,
        asr_path,
        expected_stage="asr",
        payload_name="transcript.json",
    )
    diarization_manifest, diarization_payload, diarization_provenance, diarization_directory = (
        _load_stage_payload(
            arguments.artifact_root,
            diarization_path,
            expected_stage="diarization",
            payload_name="diarization.json",
        )
    )
    try:
        transcript = TranscriptResult.from_dict(asr_payload)
        diarization = DiarizationResult.from_dict(diarization_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactPublicationError(f"invalid attribution input payload: {exc}") from exc
    _validate_stage_provenance(
        asr_manifest, transcript, expected_source_sha256=episode.sha256, stage="ASR"
    )
    _validate_stage_provenance(
        diarization_manifest,
        diarization,
        expected_source_sha256=episode.sha256,
        stage="diarization",
    )
    if transcript.duration_ms != diarization.duration_ms:
        raise ValueError("ASR and diarization duration provenance does not match")

    asr_digest = _manifest_digest(asr_manifest)
    diarization_digest = _manifest_digest(diarization_manifest)
    configuration = {
        "stage": "attribution",
        "tie_policy": "greatest_exclusive_intersection_then_midpoint_then_stable_speaker_id",
        "no_intersection_policy": "speaker_id_null",
        "overlap_policy": "standard_turns_intersection",
        "export_cue_policy": "one_word_per_cue",
        "manifest_sha256": manifest.sha256,
        "source_sha256": episode.sha256,
        "asr_artifact_manifest_sha256": asr_digest,
        "diarization_artifact_manifest_sha256": diarization_digest,
        "transcript_model_fingerprint_sha256": transcript.model.fingerprint_sha256,
        "diarization_model_fingerprint_sha256": diarization.model.fingerprint_sha256,
    }
    upstream = (asr_digest, diarization_digest)
    stage_key = stage_fingerprint(
        "attribution",
        source_sha256=episode.sha256,
        upstream_artifact_hashes=upstream,
        configuration=configuration,
        pipeline_version="p1-04",
    )
    artifact_id = f"attribution-{stage_key[:24]}"
    provenance = {
        "configuration": configuration,
        "source": {
            "episode_id": episode.relative_filename,
            "source_sha256": episode.sha256,
            "duration_ms": episode.duration_ms,
            "time_origin_ms": 0,
        },
        "inputs": {
            "asr": {
                "artifact_id": asr_manifest.artifact_id,
                "stage_key": asr_manifest.stage_key,
                "manifest_sha256": asr_digest,
                "model_fingerprint_sha256": asr_manifest.model_fingerprint_sha256,
                "directory": str(asr_directory),
                "upstream_provenance": asr_provenance.get("configuration", {}),
            },
            "diarization": {
                "artifact_id": diarization_manifest.artifact_id,
                "stage_key": diarization_manifest.stage_key,
                "manifest_sha256": diarization_digest,
                "model_fingerprint_sha256": diarization_manifest.model_fingerprint_sha256,
                "directory": str(diarization_directory),
                "upstream_provenance": diarization_provenance.get("configuration", {}),
            },
        },
        "models": {
            "transcript": transcript.model.to_dict(),
            "diarization": diarization.model.to_dict(),
        },
    }
    attributed = build_attributed_transcript(
        transcript,
        diarization,
        artifact_id=artifact_id,
        episode_id=episode.relative_filename,
        provenance=provenance,
    )
    files = {
        "attributed.json": json.dumps(
            attributed.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
    }
    files.update(
        {name: content.encode("utf-8") for name, content in render_exports(attributed).items()}
    )
    publisher = ArtifactPublisher(
        arguments.artifact_root, repository=getattr(arguments, "repository", None)
    )
    artifact = publisher.publish(
        source_sha256=episode.sha256,
        stage="attribution",
        stage_key=stage_key,
        files=files,
        upstream_artifact_hashes=upstream,
        pipeline_version="p1-04",
        artifact_id=artifact_id,
        provenance=provenance,
        job=getattr(arguments, "job", None),
    )
    return {
        "stage": "attribution",
        "artifact": artifact.to_dict(),
        "artifact_path": str(
            ArtifactPublisher(arguments.artifact_root).artifact_path(
                episode.sha256, "attribution", stage_key
            )
        ),
        "word_count": len(attributed.words),
        "unassigned_word_count": attributed.unassigned_word_count,
        "overlap_word_count": attributed.overlap_word_count,
        "duration_ms": attributed.duration_ms,
        "source_sha256": attributed.source_sha256,
    }


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
    publisher = ArtifactPublisher(
        arguments.artifact_root, repository=getattr(arguments, "repository", None)
    )
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
        job=getattr(arguments, "job", None),
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
    publisher = ArtifactPublisher(
        arguments.artifact_root, repository=getattr(arguments, "repository", None)
    )
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
        job=getattr(arguments, "job", None),
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

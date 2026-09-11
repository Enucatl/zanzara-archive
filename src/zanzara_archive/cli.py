"""Command-line entry point for the Zanzara Archive application."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from zanzara_archive import __version__
from zanzara_archive.corpus import CorpusValidationError, load_manifest, verify_corpus, write_report
from zanzara_archive.jobs import DurableWorker, synthetic_runner
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

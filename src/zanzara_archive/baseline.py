"""Run the reviewed golden baseline through the durable local worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .annotations import canonical_hash, split_manifest
from .corpus import CorpusValidationError, load_manifest, resolve_source
from .evaluation import (
    EvaluationValidationError,
    evaluate_documents,
    validate_reference,
    write_evaluation_artifacts,
)
from .jobs import DurableWorker
from .model_locks import model_fingerprint_from_lock, validate_model_lock
from .storage import SQLiteRepository


class BaselineRunError(EvaluationValidationError):
    """Raised when a real baseline cannot be completed safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineRunError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineRunError(f"{label} JSON must be an object: {path}")
    return value


def _reference_file(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        return candidate
    for name in ("reference.json", "annotation.json", "reviewed-reference.json"):
        selected = candidate / name
        if selected.is_file():
            return selected
    raise BaselineRunError(f"reference directory has no reference JSON: {candidate}")


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _gpu_snapshot() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
        fields = [item.strip() for item in completed.stdout.splitlines()[0].split(",")]
        if len(fields) != 5:
            return None
        return {
            "gpu": fields[0],
            "driver": fields[1],
            "vram_total_mib": int(fields[2]),
            "vram_used_mib": int(fields[3]),
            "gpu_utilization_percent": int(fields[4]),
        }
    except (
        OSError,
        IndexError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None


def _docker_workloads() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    workloads: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        name, separator, rest = line.partition("\t")
        image, separator, status = rest.partition("\t")
        if separator and name and image and status:
            workloads.append({"name": name, "image": image, "status": status})
    return workloads


class _ResourceSampler:
    """Sample host process RSS and GPU memory while one worker job runs."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def _sample(self) -> None:
        gpu = _gpu_snapshot()
        self._samples.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "gpu": gpu,
            }
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="zanzara-baseline-sampler")
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        gpu_samples = [item["gpu"] for item in self._samples if item["gpu"] is not None]
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self._samples),
            "initial": self._samples[0],
            "final": self._samples[-1],
            "peak_process_rss_mib": max(item["process_rss_mib"] for item in self._samples),
            "peak_vram_used_mib": max(
                (item["vram_used_mib"] for item in gpu_samples), default=None
            ),
            "peak_gpu_utilization_percent": max(
                (item["gpu_utilization_percent"] for item in gpu_samples), default=None
            ),
        }


def _validate_inputs(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.suite != "golden":
        raise BaselineRunError(f"unsupported evaluation suite: {arguments.suite}")
    try:
        corpus = load_manifest(arguments.corpus)
    except CorpusValidationError as exc:
        raise BaselineRunError(str(exc)) from exc
    episode = next(
        (item for item in corpus.episodes if item.relative_filename == corpus.golden_episode), None
    )
    if episode is None:
        raise BaselineRunError("frozen corpus golden episode is missing")
    source = resolve_source(arguments.archive_root, episode.relative_filename)
    if source.stat().st_size != episode.size_bytes:
        raise BaselineRunError(
            f"golden source size differs from the frozen manifest: {source.stat().st_size}"
        )

    reference_path = _reference_file(arguments.reference)
    validation = validate_reference(arguments.corpus, reference_path)
    if not validation["valid"]:
        raise BaselineRunError(
            "golden reference validation failed: " + "; ".join(validation["errors"])
        )
    reference = _read_json(reference_path, label="reference")
    split_path = Path(arguments.split).expanduser()
    split = _read_json(split_path, label="split")
    expected_split = split_manifest(episode.duration_ms, episode.sha256)
    if split != expected_split:
        raise BaselineRunError("split does not match the frozen E1 golden split")
    if reference.get("split_sha256") != canonical_hash(split):
        raise BaselineRunError("reference split hash does not match the supplied split")

    try:
        model_validation = validate_model_lock(arguments.model_lock)
    except ValueError as exc:
        raise BaselineRunError(str(exc)) from exc
    lock = _read_json(Path(arguments.model_lock), label="model lock")
    asr_model = model_fingerprint_from_lock(arguments.model_lock, "parakeet")
    diarization_model = model_fingerprint_from_lock(arguments.model_lock, "diarization")
    asr_endpoint = arguments.endpoint or os.environ.get(
        "PARAKEET_ENDPOINT", "http://127.0.0.1:18080"
    )
    diarization_endpoint = arguments.diarization_endpoint
    reference_file_sha256 = _sha256_file(reference_path)
    split_file_sha256 = _sha256_file(split_path)
    frozen_configuration = {
        "suite": arguments.suite,
        "corpus_manifest_sha256": corpus.sha256,
        "episode": episode.relative_filename,
        "source_sha256": episode.sha256,
        "duration_ms": episode.duration_ms,
        "reference_file_sha256": reference_file_sha256,
        "reference_payload_sha256": _canonical_sha256(reference),
        "split_file_sha256": split_file_sha256,
        "split_sha256": canonical_hash(split),
        "model_lock_sha256": _sha256_file(Path(arguments.model_lock)),
        "model_validation": model_validation,
        "models": {
            "parakeet": asr_model.fingerprint_sha256,
            "diarization": diarization_model.fingerprint_sha256,
        },
        "endpoints": {"parakeet": asr_endpoint, "diarization": diarization_endpoint},
        "decoder": {"binary": arguments.ffmpeg, "sample_rate_hz": 16_000, "channels": 1},
        "asr": {"window_ms": 300_000, "context_ms": 5_000, "language": arguments.language},
        "diarization": {"episode_wide": True},
        "worker": {"owner": "p1-08-baseline", "lease_seconds": 120, "heartbeat_seconds": 30},
    }
    return {
        "corpus": corpus,
        "episode": episode,
        "source": source,
        "reference_path": reference_path,
        "split_path": split_path,
        "reference": reference,
        "lock": lock,
        "asr_model": asr_model,
        "diarization_model": diarization_model,
        "asr_endpoint": asr_endpoint,
        "diarization_endpoint": diarization_endpoint,
        "frozen_configuration": frozen_configuration,
        "configuration_sha256": _canonical_sha256(frozen_configuration),
    }


def _run_stage(
    repository: SQLiteRepository,
    arguments: argparse.Namespace,
    context: Mapping[str, Any],
    stage: str,
    *,
    upstream_artifact_hashes: tuple[str, ...] = (),
) -> dict[str, Any]:
    episode = context["episode"]
    frozen_configuration = dict(context["frozen_configuration"])
    stage_configuration = {**frozen_configuration, "stage": stage}
    if stage == "attribution":
        model_fingerprint = None
    elif stage == "asr":
        model_fingerprint = context["asr_model"].fingerprint_sha256
    else:
        model_fingerprint = context["diarization_model"].fingerprint_sha256
    job_id = f"p1-08-{stage}-{context['run_nonce']}"
    repository.enqueue_job(
        job_id=job_id,
        stage=stage,
        source_sha256=episode.sha256,
        upstream_artifact_hashes=upstream_artifact_hashes,
        model_fingerprint_sha256=model_fingerprint,
        configuration=stage_configuration,
        pipeline_version="p1-08",
        stage_key=f"p1-08-{stage}-{context['run_nonce']}",
        request_id=f"p1-08-{stage}-{episode.sha256[:16]}",
    )
    stage_result: dict[str, Any] = {}

    def runner(job: Any) -> Mapping[str, Any]:
        from .cli import _process_stage

        process_arguments = argparse.Namespace(
            manifest=arguments.corpus,
            episode=episode.relative_filename,
            stage=stage,
            archive_root=arguments.archive_root,
            endpoint=context["asr_endpoint"] if stage == "asr" else None,
            diarization_endpoint=context["diarization_endpoint"],
            artifact_root=arguments.artifact_root,
            model_lock=arguments.model_lock,
            asr_artifact=None,
            diarization_artifact=None,
            ffmpeg=arguments.ffmpeg,
            language=arguments.language,
            repository=repository,
            job=job,
        )
        stage_result.update(_process_stage(process_arguments))
        return stage_result

    sampler = _ResourceSampler()
    started = time.perf_counter()
    sampler.start()
    try:
        worker_result = DurableWorker(repository, "p1-08-baseline", runner).run_once()
    finally:
        resources = sampler.stop()
    elapsed = time.perf_counter() - started
    final = worker_result.job if "worker_result" in locals() else repository.fetch_job(job_id)
    if not worker_result.completed or final is None:
        detail = worker_result.error.to_dict() if worker_result.error else None
        raise BaselineRunError(f"{stage} worker failed: {json.dumps(detail, sort_keys=True)}")
    artifact_path = Path(stage_result["artifact_path"]).expanduser().resolve()
    artifact_manifest = artifact_path / "manifest.json"
    artifact_sha256 = _sha256_file(artifact_manifest)
    source_bytes = context["source"].stat().st_size
    if stage == "attribution":
        input_bytes = sum(
            _artifact_bytes(Path(item["artifact_path"]))
            for item in context["stage_results"]
            if item["stage"] in {"asr", "diarization"}
        )
    else:
        input_bytes = source_bytes
    return {
        "stage": stage,
        "job": final.to_dict(),
        "artifact_path": str(artifact_path),
        "artifact_manifest_sha256": artifact_sha256,
        "artifact_file_bytes": _artifact_bytes(artifact_path),
        "input_bytes": input_bytes,
        "output_bytes": _artifact_bytes(artifact_path),
        "audio_duration_seconds": episode.duration_ms / 1000,
        "worker_end_to_end_seconds": elapsed,
        "real_time_factor": elapsed / (episode.duration_ms / 1000),
        "warm_latency_seconds": elapsed,
        "cold_latency_seconds": None,
        "model_only_seconds": None,
        "decode_seconds": None,
        "queue_wait_seconds": None,
        "network_seconds": None,
        "failures": 0,
        "retries": max(0, final.attempts - 1),
        "cost_usd": 0,
        "paid": False,
        "resources": resources,
    }


def run_baseline(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate inputs, execute real local stages, score, and publish atomically."""

    context = _validate_inputs(arguments)
    started_at = datetime.now(UTC).isoformat()
    context = dict(context)
    context["run_nonce"] = hashlib.sha256(
        f"{context['configuration_sha256']}:{started_at}".encode()
    ).hexdigest()[:16]
    repository = SQLiteRepository.open(arguments.database)
    try:
        context["stage_results"] = []
        asr = _run_stage(repository, arguments, context, "asr")
        context["stage_results"].append(asr)
        diarization = _run_stage(repository, arguments, context, "diarization")
        context["stage_results"].append(diarization)
        attribution = _run_stage(
            repository,
            arguments,
            context,
            "attribution",
            upstream_artifact_hashes=(
                asr["artifact_manifest_sha256"],
                diarization["artifact_manifest_sha256"],
            ),
        )
        context["stage_results"].append(attribution)
    finally:
        repository.close()

    from .cli import _load_stage_payload

    _, hypothesis, _, _ = _load_stage_payload(
        arguments.artifact_root,
        attribution["artifact_path"],
        expected_stage="attribution",
        payload_name="attributed.json",
    )
    reviewed_commit = _current_commit()
    report = evaluate_documents(context["reference"], hypothesis, reviewed_commit=reviewed_commit)
    hardware_lock = context["lock"].get("hardware", {})
    execution = {
        "suite": arguments.suite,
        "started_at": started_at,
        "reviewed_commit": reviewed_commit,
        "pipeline": "asr-diarization-attribution",
        "execution_engine": "DurableWorker",
        "real_model_inference": True,
        "synthetic_provenance": [],
        "command_contract": (
            'uv run zanzara evaluation run --suite golden --reference "$ZANZARA_REFERENCE" '
            '--split "$ZANZARA_GOLDEN_SPLIT" --output "$ZANZARA_RESULTS"'
        ),
        "resolved_inputs": {
            "corpus": str(arguments.corpus),
            "reference": str(context["reference_path"]),
            "split": str(context["split_path"]),
            "output": str(arguments.output),
            "archive_root": str(arguments.archive_root),
            "artifact_root": str(arguments.artifact_root),
            "database": str(arguments.database),
            "model_lock": str(arguments.model_lock),
        },
        "frozen_configuration": context["frozen_configuration"],
        "configuration_sha256": context["configuration_sha256"],
        "hardware": {"lock": hardware_lock, "live": _gpu_snapshot()},
        "concurrent_workloads": _docker_workloads(),
        "stages": context["stage_results"],
        "scope": {
            "full_episode": "slices.all",
            "development": "slices.development and slices.block-0..block-3",
            "held_out": "slices.held_out and slices.block-4",
        },
        "limitations": [
            "Cold-start latency was not measured because the locked services were already running.",
            (
                "The service contract does not expose model-only, decode-only, queue, or network "
                "timing; null fields are not inferred from worker wall time."
            ),
            (
                "Peak RAM is the baseline process RSS; service-container RAM is not included in "
                "that process metric."
            ),
            "No paid request was made, so no cost-ledger.json is emitted.",
        ],
    }
    report["execution"] = execution
    report["provenance"].update(
        {
            "corpus_manifest_sha256": context["corpus"].sha256,
            "reference_file_sha256": context["frozen_configuration"]["reference_file_sha256"],
            "split_file_sha256": context["frozen_configuration"]["split_file_sha256"],
            "model_lock_sha256": context["frozen_configuration"]["model_lock_sha256"],
            "execution_configuration_sha256": context["configuration_sha256"],
            "hardware_context": execution["hardware"],
        }
    )
    report["limitations"].extend(execution["limitations"])
    return write_evaluation_artifacts(report, arguments.output, run_metadata=execution)


def _current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


__all__ = ["BaselineRunError", "run_baseline"]

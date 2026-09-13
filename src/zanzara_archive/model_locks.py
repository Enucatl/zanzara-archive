"""Validation for the immutable model and service lock."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from .contracts import ModelFingerprint


class ModelLockError(ValueError):
    """Raised when a model lock is incomplete or its local artifacts drift."""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelLockError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: object, name: str) -> str:
    value = _require_text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ModelLockError(f"{name} must be a lowercase SHA-256")
    return value


def _artifact_path(artifact: dict[str, Any]) -> Path:
    kind = _require_text(artifact.get("cache"), "artifact.cache")
    relative = Path(_require_text(artifact.get("path"), "artifact.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelLockError("artifact.path must be relative to its cache")
    if kind == "huggingface":
        root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        return root / "hub" / relative
    if kind == "zanzara":
        root = Path(
            os.environ.get(
                "ZANZARA_MODEL_CACHE", str(Path.home() / ".cache" / "zanzara-archive" / "models")
            )
        )
        return root / relative
    raise ModelLockError(f"unsupported artifact cache {kind!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_lock(
    path: str | os.PathLike[str],
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate a lock and, by default, every locally materialized locked artifact."""

    lock_path = Path(path)
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLockError(f"cannot read model lock {lock_path}: {exc}") from exc
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise ModelLockError("model lock schema_version must be 1")

    hardware = lock.get("hardware")
    if not isinstance(hardware, dict):
        raise ModelLockError("model lock hardware must be an object")
    for key in ("gpu", "driver", "cuda", "compute_capability"):
        _require_text(hardware.get(key), f"hardware.{key}")
    if hardware.get("gpu") != "NVIDIA GeForce RTX 5090":
        raise ModelLockError("hardware.gpu must identify the RTX 5090")
    if hardware.get("vram_total_mib") != 32607:
        raise ModelLockError("hardware.vram_total_mib must be the observed RTX 5090 capacity")

    models = lock.get("models")
    if not isinstance(models, list) or len(models) != 7:
        raise ModelLockError("model lock must contain exactly seven model records")
    ids: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ModelLockError(f"models[{index}] must be an object")
        model_id = _require_text(model.get("id"), f"models[{index}].id")
        if model_id in ids:
            raise ModelLockError(f"duplicate model id {model_id}")
        ids.add(model_id)
        for key in (
            "role",
            "repository",
            "revision",
            "license",
            "preprocessing",
            "dimensions",
            "precision",
            "runtime",
            "service",
            "smoke",
        ):
            if key not in model:
                raise ModelLockError(f"models[{index}] missing {key}")
        revision = model["revision"]
        if not isinstance(revision, str) or not revision.strip():
            raise ModelLockError(f"models[{index}].revision must be immutable text")
        files = model.get("artifacts")
        if not isinstance(files, list) or not files:
            raise ModelLockError(f"models[{index}].artifacts must be non-empty")
        for file_index, artifact in enumerate(files):
            if not isinstance(artifact, dict):
                raise ModelLockError(f"models[{index}].artifacts[{file_index}] must be an object")
            artifact_path = _artifact_path(artifact)
            _require_sha256(
                artifact.get("sha256"),
                f"models[{index}].artifacts[{file_index}].sha256",
            )
            size = artifact.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ModelLockError(
                    f"models[{index}].artifacts[{file_index}].size_bytes must be positive"
                )
            if verify_artifacts:
                if not artifact_path.is_file():
                    raise ModelLockError(f"locked artifact is missing: {artifact_path}")
                if artifact_path.stat().st_size != size:
                    raise ModelLockError(f"locked artifact size changed: {artifact_path}")
                if _sha256(artifact_path) != artifact["sha256"]:
                    raise ModelLockError(f"locked artifact checksum changed: {artifact_path}")
        smoke = model["smoke"]
        if not isinstance(smoke, dict) or smoke.get("status") != "passed":
            raise ModelLockError(f"models[{index}].smoke.status must be passed")
        for key in ("shape", "elapsed_ms", "peak_vram_mib", "evidence_sha256"):
            if key not in smoke:
                raise ModelLockError(f"models[{index}].smoke missing {key}")
        _require_sha256(smoke["evidence_sha256"], f"models[{index}].smoke.evidence_sha256")

    services = lock.get("services")
    if not isinstance(services, list) or len(services) != 7:
        raise ModelLockError("model lock must contain seven service records")
    service_ids = {service.get("id") for service in services if isinstance(service, dict)}
    if service_ids != ids:
        raise ModelLockError("service IDs must match model IDs")
    for index, service in enumerate(services):
        if not isinstance(service, dict):
            raise ModelLockError(f"services[{index}] must be an object")
        for key in ("id", "directory", "image", "image_digest", "python", "uv_lock_sha256"):
            _require_text(service.get(key), f"services[{index}].{key}")
        _require_sha256(service["image_digest"], f"services[{index}].image_digest")
        _require_sha256(service["uv_lock_sha256"], f"services[{index}].uv_lock_sha256")
        lock_file = lock_path.parent / service["directory"] / "uv.lock"
        if not lock_file.is_file():
            raise ModelLockError(f"service lock is missing: {lock_file}")
        if _sha256(lock_file) != service["uv_lock_sha256"]:
            raise ModelLockError(f"service lock checksum changed: {lock_file}")

    return {
        "schema_version": 1,
        "models": len(models),
        "services": len(services),
        "hardware": hardware,
        "python": platform.python_version(),
    }


def model_fingerprint_from_lock(
    path: str | os.PathLike[str], model_id: str = "parakeet"
) -> ModelFingerprint:
    """Build a typed fingerprint from one model record in the immutable lock."""

    lock_path = Path(path)
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLockError(f"cannot read model lock {lock_path}: {exc}") from exc
    models = lock.get("models") if isinstance(lock, dict) else None
    record = next(
        (
            model
            for model in models or ()
            if isinstance(model, dict) and model.get("id") == model_id
        ),
        None,
    )
    if record is None:
        raise ModelLockError(f"model lock does not contain {model_id!r}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelLockError(f"model lock record {model_id!r} has no artifacts")
    checkpoint_sha256 = tuple(
        artifact.get("sha256")
        for artifact in artifacts
        if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str)
    )
    if len(checkpoint_sha256) != len(artifacts):
        raise ModelLockError(f"model lock record {model_id!r} has an invalid artifact hash")
    dimensions = record.get("dimensions")
    output_dimension = None
    if isinstance(dimensions, dict):
        for key in ("embedding", "output"):
            candidate = dimensions.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                output_dimension = candidate
                break
    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        raise ModelLockError(f"model lock record {model_id!r} has invalid runtime metadata")
    return ModelFingerprint(
        name=str(record.get("id")),
        repository=_require_text(record.get("repository"), f"models.{model_id}.repository"),
        revision=_require_text(record.get("revision"), f"models.{model_id}.revision"),
        checkpoint_sha256=checkpoint_sha256,
        dimensions=output_dimension,
        preprocessing={
            "description": _require_text(
                record.get("preprocessing"), f"models.{model_id}.preprocessing"
            )
        },
        precision=_require_text(record.get("precision"), f"models.{model_id}.precision"),
        runtime={
            str(key): _require_text(value, f"models.{model_id}.runtime.{key}")
            for key, value in runtime.items()
        },
        terms_evidence=_require_text(record.get("license"), f"models.{model_id}.license"),
    )

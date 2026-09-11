"""Crash-safe immutable artifact publication and reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from .contracts import ArtifactManifest, ContractValidationError
from .storage import SQLiteRepository


class ArtifactPublicationError(RuntimeError):
    """Raised when an artifact cannot be published as a complete generation."""


def _safe_component(value: str, field_name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ArtifactPublicationError(f"{field_name} must be one path component")
    return value


def _safe_source_hash(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ArtifactPublicationError("source_sha256 must be a lowercase SHA-256")
    return value


def _safe_file_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactPublicationError("artifact file paths must stay inside the artifact")
    return path


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_source(
    destination: Path, source: bytes | bytearray | memoryview | os.PathLike[str]
) -> str:
    digest = hashlib.sha256()
    if isinstance(source, (bytes, bytearray, memoryview)):
        with destination.open("wb") as target:
            target.write(source)
            digest.update(source)
    else:
        source_path = Path(source)
        try:
            with source_path.open("rb") as source_file, destination.open("wb") as target:
                while chunk := source_file.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise ArtifactPublicationError(
                f"cannot copy artifact source {source_path}: {exc}"
            ) from exc
    target_fileno = destination.open("rb")
    try:
        os.fsync(target_fileno.fileno())
    finally:
        target_fileno.close()
    return digest.hexdigest()


class ArtifactPublisher:
    """Publish immutable artifacts under the D3 hashed path layout."""

    def __init__(self, root: str | os.PathLike[str], repository: SQLiteRepository | None = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.repository = repository

    def artifact_path(self, source_sha256: str, stage: str, stage_key: str) -> Path:
        _safe_source_hash(source_sha256)
        _safe_component(stage, "stage")
        _safe_component(stage_key, "stage_key")
        return self.root / source_sha256 / stage / stage_key

    def publish(
        self,
        *,
        source_sha256: str,
        stage: str,
        stage_key: str,
        files: Mapping[str, bytes | bytearray | memoryview | os.PathLike[str]],
        upstream_artifact_hashes: tuple[str, ...] = (),
        pipeline_version: str = "dev",
        schema_version: int = 1,
        model_fingerprint_sha256: str | None = None,
        artifact_id: str | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> ArtifactManifest:
        """Write files and a completed manifest, then atomically publish the directory."""

        if not files:
            raise ArtifactPublicationError("an artifact must contain at least one file")
        if provenance is not None:
            if "provenance.json" in files:
                raise ArtifactPublicationError(
                    "provenance.json is reserved for publication metadata"
                )
            files = dict(files)
            files["provenance.json"] = json.dumps(
                provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        destination = self.artifact_path(source_sha256, stage, stage_key)
        artifact_id = artifact_id or f"{stage}:{stage_key}"
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            manifest = self._read_complete(destination)
            if self.repository is not None:
                self.repository.reconcile_artifact(manifest, destination)
            return manifest
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
        checksums: dict[str, str] = {}
        try:
            for name, source in sorted(files.items()):
                relative = _safe_file_name(name)
                output = temporary.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                checksums[str(relative)] = _write_source(output, source)
            manifest = ArtifactManifest(
                artifact_id=artifact_id,
                source_sha256=source_sha256,
                stage=stage,
                stage_key=stage_key,
                upstream_artifact_hashes=tuple(upstream_artifact_hashes),
                file_checksums=checksums,
                pipeline_version=pipeline_version,
                schema_version=schema_version,
                model_fingerprint_sha256=model_fingerprint_sha256,
            )
            payload = json.dumps(
                manifest.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(parent)
        except (OSError, ContractValidationError) as exc:
            raise ArtifactPublicationError(f"artifact publication failed: {exc}") from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
        if self.repository is not None:
            # A database failure leaves a valid, discoverable orphan for reconciliation.
            self.repository.record_artifact(manifest, destination)
        return manifest

    def _read_complete(self, directory: Path) -> ArtifactManifest:
        manifest_path = directory / "manifest.json"
        try:
            if manifest_path.is_symlink():
                raise ArtifactPublicationError("manifest must not be a symlink")
            manifest = ArtifactManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            for name, expected in manifest.file_checksums.items():
                file_path = directory.joinpath(*PurePosixPath(name).parts)
                if file_path.is_symlink() or directory.resolve() not in file_path.resolve().parents:
                    raise ArtifactPublicationError(f"artifact file escapes its directory: {name}")
                if (
                    not file_path.is_file()
                    or hashlib.sha256(file_path.read_bytes()).hexdigest() != expected
                ):
                    raise ArtifactPublicationError(f"incomplete or corrupt artifact file: {name}")
        except (OSError, ValueError, KeyError, TypeError, ContractValidationError) as exc:
            raise ArtifactPublicationError(
                f"invalid completed artifact {directory}: {exc}"
            ) from exc
        return manifest

    def reconcile(self) -> tuple[tuple[ArtifactManifest, ...], tuple[str, ...]]:
        """Recover complete orphan directories and ignore incomplete directories."""

        recovered: list[ArtifactManifest] = []
        errors: list[str] = []
        if not self.root.exists():
            return (), ()
        for source_dir in sorted(self.root.iterdir()):
            if not source_dir.is_dir() or source_dir.name.startswith("."):
                continue
            for stage_dir in sorted(source_dir.iterdir()):
                if not stage_dir.is_dir():
                    continue
                for artifact_dir in sorted(stage_dir.iterdir()):
                    if (
                        not artifact_dir.is_dir()
                        or artifact_dir.is_symlink()
                        or artifact_dir.name.startswith(".")
                    ):
                        continue
                    if not (artifact_dir / "manifest.json").exists():
                        continue
                    try:
                        manifest = self._read_complete(artifact_dir)
                        if (
                            self.artifact_path(
                                manifest.source_sha256, manifest.stage, manifest.stage_key
                            )
                            != artifact_dir
                        ):
                            raise ArtifactPublicationError(
                                f"artifact manifest is in the wrong hashed path: {artifact_dir}"
                            )
                        if self.repository is not None and self.repository.reconcile_artifact(
                            manifest, artifact_dir
                        ):
                            recovered.append(manifest)
                    except ArtifactPublicationError as exc:
                        errors.append(str(exc))
        return tuple(recovered), tuple(errors)


ImmutableArtifactPublisher = ArtifactPublisher

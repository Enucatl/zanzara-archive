"""CPU checks for the model-lock schema."""

import hashlib
import json
from pathlib import Path

import pytest

from zanzara_archive import cli, model_locks
from zanzara_archive.model_locks import ModelLockError, validate_model_lock


def test_model_lock_requires_nine_passed_smokes(tmp_path) -> None:
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ModelLockError, match="hardware"):
        validate_model_lock(path, verify_artifacts=False)


def test_model_lock_validation_does_not_hash_checkpoints(monkeypatch) -> None:
    original_sha256 = model_locks._sha256

    def service_lock_only(path: Path) -> str:
        assert path.name == "uv.lock", "routine validation must not read model bytes"
        return original_sha256(path)

    monkeypatch.setattr(model_locks, "_sha256", service_lock_only)
    result = validate_model_lock("models.lock.json")
    assert result["models"] == 9
    assert result["services"] == 9
    assert result["artifacts_verified"] is False


def test_explicit_model_audit_detects_same_size_corruption(tmp_path, monkeypatch) -> None:
    lock = json.loads(Path("models.lock.json").read_text())
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("ZANZARA_MODEL_CACHE", str(tmp_path))
    for model in lock["models"]:
        model["artifacts"] = [
            {
                "cache": "zanzara",
                "path": checkpoint.name,
                "sha256": hashlib.sha256(b"weights").hexdigest(),
                "size_bytes": 7,
            }
        ]
    for service in lock["services"]:
        service_lock = tmp_path / service["directory"] / "uv.lock"
        service_lock.parent.mkdir(parents=True, exist_ok=True)
        service_lock.write_bytes(b"lock")
        service["uv_lock_sha256"] = hashlib.sha256(b"lock").hexdigest()
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps(lock))

    assert validate_model_lock(path, verify_artifacts=True)["artifacts_verified"] is True
    checkpoint.write_bytes(b"damaged")
    assert validate_model_lock(path)["artifacts_verified"] is False
    with pytest.raises(ModelLockError, match="checksum changed"):
        validate_model_lock(path, verify_artifacts=True)


@pytest.mark.parametrize("audit", [False, True])
def test_models_verify_cli_requires_explicit_audit(audit, monkeypatch, capsys) -> None:
    def validate(path: str, *, verify_artifacts: bool) -> dict[str, bool]:
        assert path == "models.lock.json"
        assert verify_artifacts is audit
        return {"artifacts_verified": verify_artifacts}

    monkeypatch.setattr(cli, "validate_model_lock", validate)
    arguments = ["models", "verify", "--lock", "models.lock.json"]
    if audit:
        arguments.append("--verify-artifacts")
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["artifacts_verified"] is audit

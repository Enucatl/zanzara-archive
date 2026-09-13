"""CPU checks for the model-lock schema."""

import json

import pytest

from zanzara_archive.model_locks import ModelLockError, validate_model_lock


def test_model_lock_requires_seven_passed_smokes(tmp_path) -> None:
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ModelLockError, match="hardware"):
        validate_model_lock(path, verify_artifacts=False)


def test_model_lock_is_available_and_fully_materialized() -> None:
    result = validate_model_lock("models.lock.json")
    assert result["models"] == 7
    assert result["services"] == 7

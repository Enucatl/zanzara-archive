"""P1-08 evaluator scalability and real-run contract checks."""

from __future__ import annotations

import json

from zanzara_archive.cli import build_parser
from zanzara_archive.evaluation_metrics import (
    _edit_alignment,
    _speaker_mapping,
    write_evaluation_artifacts,
)


def test_large_common_regions_do_not_require_a_quadratic_alignment_matrix() -> None:
    reference = tuple(f"token-{index}" for index in range(10_000))
    hypothesis = (*reference[:5_000], "replacement", *reference[5_001:])

    alignment = _edit_alignment(reference, hypothesis)

    assert alignment.substitutions == 1
    assert alignment.deletions == 0
    assert alignment.insertions == 0
    assert len(alignment.operations) == len(reference)


def test_speaker_mapping_scales_to_the_real_golden_speaker_count() -> None:
    events = tuple(
        (
            index * 1_000,
            (index + 1) * 1_000,
            frozenset({f"REFERENCE_{index:02d}"}),
            frozenset({f"HYPOTHESIS_{index:02d}"}),
        )
        for index in range(23)
    )

    assert _speaker_mapping(events) == {
        f"HYPOTHESIS_{index:02d}": f"REFERENCE_{index:02d}" for index in range(23)
    }


def test_real_baseline_command_is_registered() -> None:
    arguments = build_parser().parse_args(
        [
            "evaluation",
            "run",
            "--suite",
            "golden",
            "--reference",
            "reference.json",
            "--split",
            "split.json",
            "--output",
            "run",
        ]
    )

    assert arguments.evaluation_command == "run"
    assert arguments.suite == "golden"
    assert arguments.corpus == "planning/corpus-20.json"


def test_real_run_metadata_marks_model_execution_and_preserves_private_artifact_contract(
    tmp_path,
) -> None:
    report = {
        "schema_version": 1,
        "provenance": {"source_sha256": "a" * 64},
        "slices": {},
        "limitations": [],
        "pass_block": {},
        "verdict": "pass",
        "private_errors": {},
    }
    metadata = {"real_model_inference": True, "stages": [{"stage": "asr"}]}

    write_evaluation_artifacts(report, tmp_path / "run", run_metadata=metadata)

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["commands_require_no_model_or_paid_call"] is False
    assert run["execution"] == metadata

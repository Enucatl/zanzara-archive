"""Contract checks for the repository phase-review skill.

The review skill is an operational workflow, so these fixtures exercise its
decision boundaries without making live GitHub mutations or pretending to
have GPU, human, or paid evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
SKILL = ROOT / ".agents" / "skills" / "review-phase" / "SKILL.md"
PROPOSAL = ROOT / "planning" / "REVIEW-SKILL.md"
FIXTURES = ROOT / "tests" / "fixtures" / "phase_review_cases.json"


def _skill() -> str:
    return SKILL.read_text(encoding="utf-8")


def _cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _expected_decision(facts: dict[str, Any]) -> str:
    if facts.get("matching_open_finding"):
        return "reuse-existing-finding"
    if facts.get("matching_completed_finding_recurred"):
        return "reopen-existing-finding"
    if facts.get("not_planned_prerequisite"):
        return "BLOCKED"
    if not facts.get("commits_pushed") or not facts.get("evidence_current"):
        return "BLOCKED"
    if facts.get("missing_gpu") or facts.get("unmet_wer") or facts.get("newer_code_after_pass"):
        return "BLOCKED"
    if facts.get("missing_human_annotation"):
        return "BLOCKED"
    if facts.get("children_complete"):
        return "PASS — awaiting user release"
    return "BLOCKED"


def test_installed_skill_matches_the_planned_proposal() -> None:
    proposal = PROPOSAL.read_text(encoding="utf-8")
    start = proposal.index("````markdown\n", proposal.index("## Proposed SKILL.md")) + len(
        "````markdown\n"
    )
    end = proposal.index("\n````", start)
    assert _skill() == proposal[start:end] + "\n"


def test_frontmatter_and_repository_references_are_valid() -> None:
    skill = _skill()
    match = re.match(r"^---\nname: ([a-z0-9-]+)\ndescription: .+\n---\n", skill)
    assert match is not None
    assert match.group(1) == "review-phase"

    required_paths = (
        "planning/README.md",
        "planning/SYSTEM-DESIGN.md",
        "planning/EVALUATION.md",
        "planning/GITHUB-SETUP.md",
        "planning/issues/",
    )
    for relative_path in required_paths:
        assert (ROOT / relative_path).exists()


def test_fixture_outcomes_cover_review_gates_without_live_mutations() -> None:
    cases = _cases()
    assert {case["id"] for case in cases} == {
        "clean-phase",
        "unpushed-child-commit",
        "closed-not-planned-prerequisite",
        "missing-gpu-measurement",
        "unmet-wer",
        "code-after-prior-pass",
        "existing-open-finding",
        "completed-recurring-finding",
        "human-annotation-blocker",
    }
    for case in cases:
        assert _expected_decision(case["facts"]) == case["expected"]


def test_skill_preserves_the_fixture_decision_boundaries() -> None:
    skill = _skill()
    required_clauses = (
        "Use whatever model is currently enabled in the chat",
        "do not gate, switch, or request a model change",
        "Check close reason `completed`, pushed implementation commits and acceptance evidence",
        "`not_planned` is not completion unless the user explicitly approves a replacement",
        "Report missing resources as blockers",
        "A newer implementation without current results is missing evidence",
        "inspect actual GPU/human/paid artifacts instead of substituting mocks",
        "Reuse an unresolved matching finding",
        "If a completed finding has recurred, reopen it with evidence rather than duplicate it",
        "Attach each follow-up with the native sub-issue API under the same parent",
        "Make the phase parent blocked by this follow-up, not the reverse",
        "PASS requires every child/follow-up completed",
        "keep the issue open in both cases",
        "Do not implement findings",
        "the user posts `Release Pn at <reviewed-commit>`",
    )
    for clause in required_clauses:
        assert clause in skill


def test_current_chat_model_is_not_a_review_gate() -> None:
    skill = _skill()
    assert "select Sol" not in skill
    assert "GPT-5.6-Sol" not in skill
    facts = {
        "model": "arbitrary-current-chat-model",
        "children_complete": True,
        "commits_pushed": True,
        "evidence_current": True,
    }
    assert _expected_decision(facts) == "PASS — awaiting user release"


def test_clean_pass_cannot_close_or_release_the_phase() -> None:
    skill = _skill()
    report_section = skill[skill.index("## Report and stop") :]
    assert "PASS — awaiting user release" in report_section
    assert "keep the issue open in both cases" in report_section
    assert "then the parent may be closed as completed and marked Done" in report_section
    assert "Do not implement findings" in report_section

"""Checks for clean, stable diarization excerpt selection."""

from zanzara_archive.contracts import DiarizationResult, ModelFingerprint, Overlap, Turn
from zanzara_archive.exemplars import select_exemplars


def test_excerpts_exclude_overlap_and_transitions_deterministically() -> None:
    """Keep only valid source intervals and explain speakers with no excerpt."""

    result = DiarizationResult(
        artifact_id="diarization-test",
        source_sha256="a" * 64,
        duration_ms=30_000,
        model=ModelFingerprint(
            name="test",
            repository="test/model",
            revision="rev",
            checkpoint_sha256=("b" * 64,),
        ),
        standard_turns=(
            Turn("A", 0, 20_000),
            Turn("B", 9_000, 11_000),
            Turn("C", 22_000, 24_000),
        ),
        exclusive_turns=(),
        overlaps=(Overlap(("A", "B"), 9_000, 11_000),),
    )
    selected = select_exemplars(result)
    assert selected == select_exemplars(result)
    assert selected["B"]["status"] == "not_voice_searchable"
    assert selected["C"]["status"] == "not_voice_searchable"
    assert selected["A"]["status"] == "voice_searchable"
    assert selected["A"]["excerpts"] == [
        {"start_ms": 250, "end_ms": 8_750},
        {"start_ms": 11_250, "end_ms": 19_750},
    ]
    assert {reason["reason"] for reason in selected["A"]["exclusions"]} >= {"transition", "overlap"}
    assert all(
        0 <= item["start_ms"] < item["end_ms"] <= 30_000 for item in selected["A"]["excerpts"]
    )

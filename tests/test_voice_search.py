"""Checks for deterministic cross-episode ResNet candidate retrieval."""

from __future__ import annotations

from typing import Any

import pytest

from zanzara_archive.contracts import ModelFingerprint
from zanzara_archive.voice_index import ALIASES, build_points
from zanzara_archive.voice_search import match_exemplars, retrieve_candidates

MODEL = ModelFingerprint(
    name="resnet293",
    repository="test/resnet293",
    revision="rev",
    checkpoint_sha256=("a" * 64,),
    dimensions=2,
)


def _record(episode: str, vectors: list[list[float]]) -> dict[str, Any]:
    """Make a retained embedding artifact with one searchable and one empty speaker."""

    return {
        "episode_id": episode,
        "source_sha256": "b" * 64,
        "embedding_artifact_id": f"artifact-{episode}",
        "embedding_manifest_sha256": "c" * 64,
        "model": MODEL.to_dict(),
        "speakers": {
            "A": {
                "status": "voice_searchable",
                "excerpts": [
                    {
                        "exemplar_id": f"{episode}-{index}",
                        "start_ms": 1000 + 5000 * index,
                        "end_ms": 4000 + 5000 * index,
                        "vector": vector,
                    }
                    for index, vector in enumerate(vectors)
                ],
            },
            "B": {"status": "not_voice_searchable", "excerpts": []},
        },
    }


def test_greedy_matching_uses_unique_excerpts_and_stable_ties() -> None:
    """Tied pairs choose stable IDs and cannot reuse either endpoint."""

    records = [_record("query.opus", [[1, 0], [1, 0]]), _record("other.opus", [[1, 0], [0, 1]])]
    _, points, _ = build_points(records)
    score, matches = match_exemplars(points[:2], points[2:])
    assert score == pytest.approx(0.5)
    assert [
        (m["query_excerpt"]["exemplar_id"], m["candidate_excerpt"]["exemplar_id"]) for m in matches
    ] == [
        ("query.opus-0", "other.opus-0"),
        ("query.opus-1", "other.opus-1"),
    ]


def test_retrieval_excludes_query_episode_and_returns_playable_evidence() -> None:
    """Use the filtered centroid shortlist and rerank visible source excerpts."""

    query = _record("query.opus", [[1, 0], [0, 1]])
    other = _record("other.opus", [[1, 0], [0, 1]])
    weaker = _record("weaker.opus", [[-1, 0], [0, -1]])
    _, exemplars, centroids = build_points([query, other, weaker])

    def request(method: str, path: str, payload: object | None) -> Any:
        """Return the Qdrant fields needed by the search contract."""

        if path == "/aliases":
            return {
                "aliases": [
                    {"alias_name": alias, "collection_name": f"zanzara_resnet293_{kind}_gen"}
                    for kind, alias in ALIASES.items()
                ]
            }
        if path.endswith("/points/query"):
            assert payload["limit"] == 50
            assert payload["filter"]["must_not"][0]["match"]["value"] == "query.opus"
            return {
                "points": [
                    {"payload": centroids[2]["payload"], "score": 0.9},
                    {"payload": centroids[1]["payload"], "score": 0.8},
                ]
            }
        if path.endswith("/points/scroll"):
            speaker = payload["filter"]["must"][0]["match"]["value"]
            points = exemplars[2:4] if speaker == "artifact-other.opus:A" else exemplars[4:]
            return {"points": points, "next_page_offset": None}
        if path.endswith("/points"):
            assert payload["ids"] == [centroids[0]["id"]]
            return [{"payload": centroids[0]["payload"]}]
        raise AssertionError((method, path))

    result = retrieve_candidates(query, "A", request)
    assert result["generation"] == "gen"
    assert len(result["candidates"]) == 2
    candidate = result["candidates"][0]
    assert candidate["episode_id"] == "other.opus"
    assert candidate["score"] == pytest.approx(1)
    assert candidate["representative_match"]["candidate_excerpt"]["listen_url"] == (
        "http://complex.home.arpa:8000/episodes/other.opus?t=1"
    )
    assert candidate["matches"][1]["candidate_excerpt"]["start_ms"] == 6000
    assert retrieve_candidates(query, "B", request)["status"] == "not_voice_searchable"
    empty = _record("empty.opus", [])
    empty["speakers"] = {"B": empty["speakers"]["B"]}
    assert retrieve_candidates(empty, "B", request)["candidates"] == []

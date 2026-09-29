"""Checks for rebuildable, coherently activated ResNet voice generations."""

from __future__ import annotations

from typing import Any

import pytest

from zanzara_archive.contracts import ModelFingerprint
from zanzara_archive.voice_index import ALIASES, build_points, publish_generation

MODEL = ModelFingerprint(
    name="resnet293",
    repository="test/resnet293",
    revision="rev",
    checkpoint_sha256=("a" * 64,),
    dimensions=2,
)


def _record() -> dict[str, Any]:
    """Make one synthetic episode with two selected excerpts and one empty speaker."""

    return {
        "episode_id": "episode.opus",
        "source_sha256": "b" * 64,
        "embedding_artifact_id": "embedding-1",
        "embedding_manifest_sha256": "c" * 64,
        "model": MODEL.to_dict(),
        "speakers": {
            "A": {
                "status": "voice_searchable",
                "excerpts": [
                    {"exemplar_id": "first", "start_ms": 1000, "end_ms": 5000, "vector": [2, 0]},
                    {"exemplar_id": "second", "start_ms": 6000, "end_ms": 9000, "vector": [0, 3]},
                ],
            },
            "B": {"status": "not_voice_searchable", "excerpts": []},
        },
    }


def test_points_are_normalized_and_counts_follow_selected_excerpts() -> None:
    """Keep source offsets and derive one centroid per indexable speaker."""

    model, exemplars, centroids = build_points([_record()])
    assert model == MODEL
    assert len(exemplars) == 2
    assert len(centroids) == 1
    assert exemplars[0]["vector"] == [1.0, 0.0]
    assert exemplars[1]["vector"] == [0.0, 1.0]
    assert centroids[0]["vector"] == pytest.approx([2**-0.5, 2**-0.5])
    assert exemplars[0]["payload"]["start_ms"] == 1000
    assert exemplars[0]["id"] == build_points([_record()])[1][0]["id"]


def test_rejects_mixed_models_and_bad_vectors() -> None:
    """Never combine incompatible model spaces or publish invalid dimensions."""

    other = _record()
    other["episode_id"] = "other.opus"
    other["model"] = {**MODEL.to_dict(), "revision": "other"}
    with pytest.raises(ValueError, match="different model"):
        build_points([_record(), other])
    bad = _record()
    bad["speakers"]["A"]["excerpts"][0]["vector"] = [1, 2, 3]
    with pytest.raises(ValueError, match="dimension"):
        build_points([bad])


class FakeQdrant:
    """Record collection writes and keep alias updates atomic for this check."""

    def __init__(self, *, fail_count: bool = False) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.aliases: dict[str, str] = {alias: "old" for alias in ALIASES.values()}
        self.fail_count = fail_count
        self.alias_updates = 0

    def request(self, method: str, path: str, payload: object | None) -> Any:
        """Return the Qdrant response fields used by the publisher."""

        if path == "/collections":
            return {"collections": [{"name": name} for name in self.collections]}
        if path == "/aliases":
            return {
                "aliases": [
                    {"alias_name": alias, "collection_name": collection}
                    for alias, collection in self.aliases.items()
                ]
            }
        if path == "/collections/aliases":
            self.alias_updates += 1
            for action in payload["actions"]:
                if "delete_alias" in action:
                    self.aliases.pop(action["delete_alias"]["alias_name"])
                else:
                    item = action["create_alias"]
                    self.aliases[item["alias_name"]] = item["collection_name"]
            return True
        collection = path.split("/")[2]
        if method == "PUT" and path == f"/collections/{collection}":
            self.collections.setdefault(collection, [])
            return True
        if method == "GET":
            return {"config": {"params": {"vectors": {"size": 2, "distance": "Cosine"}}}}
        if path.endswith("/points/count"):
            return {"count": len(self.collections[collection]) + int(self.fail_count)}
        if path.endswith("/points?wait=true"):
            existing = {point["id"]: point for point in self.collections[collection]}
            existing.update({point["id"]: point for point in payload["points"]})
            self.collections[collection] = list(existing.values())
            return {"status": "completed"}
        raise AssertionError((method, path))


def test_failed_build_does_not_switch_either_active_alias() -> None:
    """A count failure leaves both active collections on the old generation."""

    qdrant = FakeQdrant(fail_count=True)
    with pytest.raises(ValueError, match="count"):
        publish_generation([_record()], qdrant.request)
    assert qdrant.alias_updates == 0
    assert set(qdrant.aliases.values()) == {"old"}

    qdrant = FakeQdrant()
    result = publish_generation([_record()], qdrant.request)
    assert result["exemplar_count"] == 2
    assert result["centroid_count"] == 1
    assert qdrant.alias_updates == 1
    assert set(qdrant.aliases.values()) == set(result["collections"].values())
    assert publish_generation([_record()], qdrant.request) == result
    assert qdrant.alias_updates == 1

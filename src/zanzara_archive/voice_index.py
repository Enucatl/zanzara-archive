"""Build complete ResNet293 voice generations from retained embedding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import niquests

from .artifacts import ArtifactPublisher
from .contracts import EmbeddingBatch, ModelFingerprint

ALIASES = {
    "exemplars": "zanzara_resnet293_exemplars_active",
    "centroids": "zanzara_resnet293_centroids_active",
}


def _unit(vector: Sequence[float]) -> list[float]:
    """Return a unit vector, rejecting a zero or invalid norm."""

    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("voice vector has an invalid norm")
    return [value / norm for value in vector]


def _point_id(kind: str, item_id: str) -> str:
    """Make a Qdrant UUID from stable canonical identity."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"zanzara:{kind}:{item_id}"))


def build_points(
    records: Sequence[Mapping[str, Any]],
) -> tuple[ModelFingerprint, list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate original vectors and derive exemplar and speaker points."""

    if not records:
        raise ValueError("at least one embedding artifact is required")
    model = ModelFingerprint.from_dict(records[0]["model"])
    if model.name != "resnet293" or model.dimensions is None:
        raise ValueError("embedding artifact must declare a ResNet293 vector space")
    exemplars: list[dict[str, Any]] = []
    centroids: list[dict[str, Any]] = []
    episodes: set[str] = set()
    item_ids: set[str] = set()
    for record in records:
        if ModelFingerprint.from_dict(record["model"]) != model:
            raise ValueError("embedding artifacts use different model/config generations")
        episode_id = record["episode_id"]
        if episode_id in episodes:
            raise ValueError(f"duplicate episode in voice index: {episode_id}")
        episodes.add(episode_id)
        source = record["source_sha256"]
        artifact_id = record["embedding_artifact_id"]
        speakers = record["speakers"]
        if not isinstance(speakers, Mapping):
            raise ValueError("embedding speakers must be an object")
        for local_id, speaker in sorted(speakers.items()):
            excerpts = speaker["excerpts"]
            status = speaker["status"]
            if status not in {"voice_searchable", "not_voice_searchable"}:
                raise ValueError("invalid voice searchable status")
            if bool(excerpts) != (status == "voice_searchable"):
                raise ValueError("speaker status disagrees with selected excerpts")
            ids = [item["exemplar_id"] for item in excerpts]
            vectors = [item["vector"] for item in excerpts]
            EmbeddingBatch(model, tuple(ids), tuple(tuple(vector) for vector in vectors))
            speaker_id = f"{artifact_id}:{local_id}"
            normalized: list[list[float]] = []
            for item, vector in zip(excerpts, vectors, strict=True):
                item_id = item["exemplar_id"]
                if item_id in item_ids:
                    raise ValueError(f"duplicate exemplar ID: {item_id}")
                item_ids.add(item_id)
                start, end = item["start_ms"], item["end_ms"]
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                ):
                    raise ValueError("invalid exemplar source interval")
                unit = _unit(vector)
                normalized.append(unit)
                exemplars.append(
                    {
                        "id": _point_id("exemplar", item_id),
                        "vector": unit,
                        "payload": {
                            "exemplar_id": item_id,
                            "episode_speaker_id": speaker_id,
                            "episode_id": episode_id,
                            "source_sha256": source,
                            "embedding_artifact_id": artifact_id,
                            "start_ms": start,
                            "end_ms": end,
                            "model_fingerprint_sha256": model.fingerprint_sha256,
                        },
                    }
                )
            if normalized:
                mean = [sum(values) / len(values) for values in zip(*normalized, strict=True)]
                centroids.append(
                    {
                        "id": _point_id("centroid", speaker_id),
                        "vector": _unit(mean),
                        "payload": {
                            "episode_speaker_id": speaker_id,
                            "episode_id": episode_id,
                            "source_sha256": source,
                            "embedding_artifact_id": artifact_id,
                            "exemplar_count": len(normalized),
                            "model_fingerprint_sha256": model.fingerprint_sha256,
                        },
                    }
                )
    if not exemplars:
        raise ValueError("voice index has no selected excerpts")
    return model, exemplars, centroids


def load_embedding_artifacts(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read complete immutable embedding artifacts for a rebuild."""

    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        manifest = ArtifactPublisher(path.parent.parent.parent)._read_complete(path)
        if manifest.stage != "speaker_embeddings_resnet293":
            raise ValueError(f"not a ResNet293 embedding artifact: {path}")
        if "embeddings.json" not in manifest.file_checksums:
            raise ValueError(f"embedding artifact does not checksum its vectors: {path}")
        record = json.loads((path / "embeddings.json").read_text(encoding="utf-8"))
        model = ModelFingerprint.from_dict(record["model"])
        if (
            record["source_sha256"] != manifest.source_sha256
            or manifest.model_fingerprint_sha256 != model.fingerprint_sha256
            or record["exemplars_manifest_sha256"] not in manifest.upstream_artifact_hashes
        ):
            raise ValueError(f"embedding artifact provenance disagrees with manifest: {path}")
        record["embedding_artifact_id"] = manifest.artifact_id
        record["embedding_manifest_sha256"] = hashlib.sha256(
            (path / "manifest.json").read_bytes()
        ).hexdigest()
        records.append(record)
    return records


class Qdrant:
    """Small REST client for collection generations and atomic alias switches."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def request(self, method: str, path: str, payload: object | None = None) -> dict[str, Any]:
        """Call a local Qdrant endpoint and require an OK JSON response."""

        with niquests.request(
            method,
            f"{self.endpoint}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            timeout=60,
            retries=0,
        ) as response:
            response.raise_for_status()
            body = response.json()
        if body.get("status") != "ok":
            raise ValueError(f"Qdrant {method} {path} failed: {body}")
        return body["result"]


def publish_generation(
    records: Sequence[Mapping[str, Any]], request: Callable[[str, str, object | None], Any]
) -> dict[str, Any]:
    """Build both complete collections before atomically switching aliases."""

    model, exemplars, centroids = build_points(records)
    key = hashlib.sha256(
        json.dumps(
            {
                "model": model.fingerprint_sha256,
                "artifacts": sorted(record["embedding_manifest_sha256"] for record in records),
                "index_version": "p2-05",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]
    collections = {kind: f"zanzara_resnet293_{kind}_{key}" for kind in ALIASES}
    existing_collections = {
        item["name"] for item in request("GET", "/collections", None)["collections"]
    }
    for kind, points in (("exemplars", exemplars), ("centroids", centroids)):
        name = collections[kind]
        if name not in existing_collections:
            request(
                "PUT",
                f"/collections/{name}",
                {"vectors": {"size": model.dimensions, "distance": "Cosine"}},
            )
        collection = request("GET", f"/collections/{name}", None)
        vector_config = collection["config"]["params"]["vectors"]
        if (
            vector_config.get("size") != model.dimensions
            or vector_config.get("distance") != "Cosine"
        ):
            raise ValueError(f"{kind} collection has a mismatched vector space")
        for offset in range(0, len(points), 128):
            result = request(
                "PUT",
                f"/collections/{name}/points?wait=true",
                {"points": points[offset : offset + 128]},
            )
            if result.get("status") != "completed":
                raise ValueError(f"Qdrant did not complete {kind} point upsert")
        count = request("POST", f"/collections/{name}/points/count", {"exact": True})["count"]
        if count != len(points):
            raise ValueError(f"{kind} count is {count}, expected {len(points)}")
    existing = {
        entry["alias_name"]: entry["collection_name"]
        for entry in request("GET", "/aliases", None)["aliases"]
    }
    actions = []
    for kind, alias in ALIASES.items():
        if existing.get(alias) != collections[kind]:
            if alias in existing:
                actions.append({"delete_alias": {"alias_name": alias}})
            actions.append(
                {"create_alias": {"alias_name": alias, "collection_name": collections[kind]}}
            )
    if actions:
        request("POST", "/collections/aliases", {"actions": actions})
    return {
        "generation": key,
        "model_fingerprint_sha256": model.fingerprint_sha256,
        "collections": collections,
        "exemplar_count": len(exemplars),
        "centroid_count": len(centroids),
        "episode_count": len(records),
    }


def main() -> None:
    """Build a voice generation from private retained embedding artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-artifact", action="append", required=True)
    parser.add_argument(
        "--qdrant-endpoint", default=os.environ.get("QDRANT_ENDPOINT", "http://127.0.0.1:6333")
    )
    args = parser.parse_args()
    result = publish_generation(
        load_embedding_artifacts(args.embedding_artifact), Qdrant(args.qdrant_endpoint).request
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

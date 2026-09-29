"""Retrieve cross-episode ResNet voice candidates from one active generation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import Any
from urllib.parse import quote

from .contracts import ModelFingerprint
from .voice_index import ALIASES, Qdrant, build_points, load_embedding_artifacts


def _excerpt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expose source offsets and a playable link without copying private audio."""

    episode_id = payload["episode_id"]
    start_ms = payload["start_ms"]
    episode_path = quote(episode_id, safe="")
    return {
        "exemplar_id": payload["exemplar_id"],
        "episode_id": episode_id,
        "source_sha256": payload["source_sha256"],
        "embedding_artifact_id": payload["embedding_artifact_id"],
        "start_ms": start_ms,
        "end_ms": payload["end_ms"],
        "listen_url": (
            f"http://complex.home.arpa:8000/episodes/{episode_path}?t={start_ms / 1000:g}"
        ),
    }


def match_exemplars(
    query: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> tuple[float, list[dict[str, Any]]]:
    """Greedily match distinct excerpts by cosine and return their median score."""

    pairs = sorted(
        (
            -sum(a * b for a, b in zip(left["vector"], right["vector"], strict=True)),
            left["payload"]["exemplar_id"],
            right["payload"]["exemplar_id"],
            left,
            right,
        )
        for left in query
        for right in candidate
    )
    used_query: set[str] = set()
    used_candidate: set[str] = set()
    matches: list[dict[str, Any]] = []
    for negative_score, query_id, candidate_id, left, right in pairs:
        if query_id in used_query or candidate_id in used_candidate:
            continue
        used_query.add(query_id)
        used_candidate.add(candidate_id)
        matches.append(
            {
                "score": -negative_score,
                "query_excerpt": _excerpt(left["payload"]),
                "candidate_excerpt": _excerpt(right["payload"]),
            }
        )
    if not matches:
        raise ValueError("candidate has no eligible exemplars")
    return median(match["score"] for match in matches), matches


def retrieve_candidates(
    record: Mapping[str, Any],
    local_speaker_id: str,
    request: Callable[[str, str, object | None], Any],
) -> dict[str, Any]:
    """Search one coherent active generation for an indexed episode speaker."""

    speakers = record["speakers"]
    if local_speaker_id not in speakers:
        raise ValueError(f"unknown episode speaker: {local_speaker_id}")
    model = ModelFingerprint.from_dict(record["model"])
    if model.name != "resnet293":
        raise ValueError("query artifact is not a ResNet293 vector space")
    aliases = {
        item["alias_name"]: item["collection_name"]
        for item in request("GET", "/aliases", None)["aliases"]
    }
    names = {kind: aliases.get(alias, "") for kind, alias in ALIASES.items()}
    prefixes = {kind: f"zanzara_resnet293_{kind}_" for kind in names}
    generations = {
        name.removeprefix(prefixes[kind])
        for kind, name in names.items()
        if name.startswith(prefixes[kind])
    }
    if any(not names[kind].startswith(prefixes[kind]) for kind in names) or len(generations) != 1:
        raise ValueError("active ResNet voice aliases do not share a generation")
    generation = generations.pop()
    speaker_id = f"{record['embedding_artifact_id']}:{local_speaker_id}"
    result: dict[str, Any] = {
        "episode_speaker_id": speaker_id,
        "episode_id": record["episode_id"],
        "status": speakers[local_speaker_id]["status"],
        "generation": generation,
        "model_fingerprint_sha256": model.fingerprint_sha256,
        "score_kind": "uncalibrated_cosine_median",
        "candidates": [],
    }
    if result["status"] == "not_voice_searchable":
        return result
    model, exemplars, centroids = build_points([record])
    centroid = next(
        point for point in centroids if point["payload"]["episode_speaker_id"] == speaker_id
    )
    stored = request(
        "POST",
        f"/collections/{names['centroids']}/points",
        {"ids": [centroid["id"]], "with_payload": True, "with_vector": False},
    )
    if len(stored) != 1 or stored[0]["payload"] != centroid["payload"]:
        raise ValueError("query speaker is absent from the active ResNet generation")
    query_exemplars = [
        point for point in exemplars if point["payload"]["episode_speaker_id"] == speaker_id
    ]
    hits = request(
        "POST",
        f"/collections/{names['centroids']}/points/query",
        {
            "query": centroid["vector"],
            "filter": {
                "must_not": [{"key": "episode_id", "match": {"value": record["episode_id"]}}]
            },
            "limit": 50,
            "with_payload": True,
        },
    )["points"]
    for hit in hits:
        payload = hit["payload"]
        if (
            payload["episode_id"] == record["episode_id"]
            or payload["model_fingerprint_sha256"] != model.fingerprint_sha256
        ):
            raise ValueError("candidate has incompatible episode or model provenance")
        page = request(
            "POST",
            f"/collections/{names['exemplars']}/points/scroll",
            {
                "filter": {
                    "must": [
                        {
                            "key": "episode_speaker_id",
                            "match": {"value": payload["episode_speaker_id"]},
                        }
                    ]
                },
                "limit": 11,
                "with_payload": True,
                "with_vector": True,
            },
        )
        points = page["points"]
        if (
            len(points) != payload["exemplar_count"]
            or page["next_page_offset"] is not None
            or any(
                point["payload"]["model_fingerprint_sha256"] != model.fingerprint_sha256
                or point["payload"]["embedding_artifact_id"] != payload["embedding_artifact_id"]
                for point in points
            )
        ):
            raise ValueError("candidate exemplars disagree with active centroid")
        score, matches = match_exemplars(query_exemplars, points)
        result["candidates"].append(
            {
                "episode_speaker_id": payload["episode_speaker_id"],
                "episode_id": payload["episode_id"],
                "source_sha256": payload["source_sha256"],
                "embedding_artifact_id": payload["embedding_artifact_id"],
                "centroid_score": hit["score"],
                "score": score,
                "representative_match": matches[0],
                "matches": matches,
            }
        )
    result["candidates"].sort(key=lambda item: (-item["score"], item["episode_speaker_id"]))
    return result


def main() -> None:
    """Print private cross-episode voice candidates as JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-artifact", required=True)
    parser.add_argument("--speaker", required=True, help="episode-local speaker ID")
    parser.add_argument(
        "--qdrant-endpoint", default=os.environ.get("QDRANT_ENDPOINT", "http://127.0.0.1:6333")
    )
    args = parser.parse_args()
    record = load_embedding_artifacts([args.embedding_artifact])[0]
    print(
        json.dumps(
            retrieve_candidates(record, args.speaker, Qdrant(args.qdrant_endpoint).request),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

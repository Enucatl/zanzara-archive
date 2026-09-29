"""Focused checks for the frozen 80-chunk pilot."""

from zanzara_archive.contracts import AudioChunk, ChunkBenchmarkManifest
from zanzara_archive.pilot import freeze_pilot


def test_freeze_pilot_preserves_episode_split_and_ignores_model_scores():
    """Keep 40/40 disjoint and choose only from source and review evidence."""

    episodes = [f"episode-{index:02d}" for index in range(20)]
    chunks = [
        AudioChunk(
            chunk_id=f"chunk-{episode}-{index}",
            episode_id=episode,
            source_sha256=f"{episode_number + 1:064x}",
            start_ms=index * 12_000,
            end_ms=(index + 1) * 12_000,
            duration_ms=240_000,
            segmentation_fingerprint="a" * 64,
            boundary_end_reason="hard_maximum" if index == 19 else "speaker_change",
        ).to_dict()
        for episode_number, episode in enumerate(episodes)
        for index in range(20)
    ]
    source = {
        "algorithm": "community1-native-adaptive-v2",
        "content_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "segmentation_version": "community1-native-adaptive-v2",
        "segmentation_configuration_hash": "d" * 64,
        "segmentation_configuration": {"target_ms": 12_000},
        "chunks": chunks,
    }
    split = {"development": episodes[:16], "held_out": episodes[16:]}
    queue = {
        "rows": [
            {"chunk_id": f"chunk-{episode}-1", "categories": ["overlap_containing"]}
            for episode in episodes
        ]
    }
    result = freeze_pilot(source, split, queue)
    again = freeze_pilot({**source, "model_scores": {"all": -999}}, split, queue)
    assert result == again
    manifest = ChunkBenchmarkManifest.from_dict(result["manifest"])
    assert len(manifest.chunks) == 80
    assert {name: len(ids) for name, ids in manifest.partitions.items()} == {
        "development": 40,
        "held_out": 40,
    }
    assert {name: result["selection"]["coverage"][name]["difficult"] for name in split} == {
        "development": 10,
        "held_out": 10,
    }
    for name, episode_set in split.items():
        selected = [chunk for chunk in manifest.chunks if chunk.partition == name]
        assert {chunk.episode_id for chunk in selected} == set(episode_set)
        assert all(0 <= chunk.start_ms < chunk.end_ms <= 240_000 for chunk in selected)
    assert result["selection"]["coverage"]["held_out"]["untagged_difficult"] == 2

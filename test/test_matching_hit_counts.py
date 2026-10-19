"""Reused hit counts preserve geometric validation, ordering and RNG consumption."""

from types import SimpleNamespace

import pytest
import torch
import numpy as np

from utils import geometry_3d
from utils.matching_algorithms import rank_projected_bbox_hits


def test_inclusive_counts_preserve_edge_points_nonfinite_domain_and_stable_ties():
    points = torch.tensor([[0., 0.], [1., 1.], [2., 1.], [float("nan"), 1.], [float("inf"), 1.]])
    boxes = [[0., 0., 1., 1.], [1., 1., 2., 1.], [0., 0., 1., 1.], [3., 3., 4., 4.]]
    counts, ranked = rank_projected_bbox_hits(points, boxes)
    assert counts == [2, 2, 2, 0]
    assert ranked == [0, 1, 2, 3]


def test_reused_topk_counts_preserve_matches_sample_order_and_rng(monkeypatch):
    points = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    boxes = [
        {"object_id": 0, "bbox": [0., 0., 1., 1.]},
        {"object_id": 1, "bbox": [1., 1., 4., 4.]},
        {"object_id": 2, "bbox": [1., 1., 4., 4.]},
        {"object_id": 3, "bbox": [8., 8., 9., 9.]},
    ]
    counts, ranked = rank_projected_bbox_hits(points, [box["bbox"] for box in boxes])
    top_indices = ranked[:3]
    candidates = [boxes[index] for index in top_indices]
    config = SimpleNamespace(min_hit_ratio=0.5, min_3d_sample_points=3, max_3d_distance=0.4)
    reference = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
    sampled_boxes = []

    def sample(_scene, _image, bbox, *_args):
        sampled_boxes.append(list(bbox))
        # A real RNG draw detects any changed number/order of sampling calls.
        return reference + torch.rand(1).item() * 0.01

    monkeypatch.setattr(geometry_3d, "sample_3d_points_from_non_overlap_regions", sample)
    transform = SimpleNamespace(map_bbox_to_original=lambda box: box)
    args = (points, candidates, config, {}, 0, transform, reference.mean(0), 1.)
    torch.manual_seed(17)
    recomputed = geometry_3d.find_best_matching_bbox_with_3d_validation(*args, ref_points_3d=reference)
    recomputed_samples, final_rng = sampled_boxes[:], torch.random.get_rng_state().clone()
    sampled_boxes.clear()
    torch.manual_seed(17)
    monkeypatch.setattr(geometry_3d, "projected_bbox_hit_counts", lambda *_a: pytest.fail("recomputed supplied counts"))
    reused = geometry_3d.find_best_matching_bbox_with_3d_validation(
        *args, ref_points_3d=reference, hit_counts=[counts[index] for index in top_indices],
    )
    # The existing best candidate also appears inside its own validated list.
    # Compare the published fields without recursively comparing that cycle.
    def fields(candidate):
        return {key: value for key, value in candidate.items() if key != "validated_candidates"}

    assert fields(reused) == fields(recomputed)
    assert [fields(row) for row in reused["validated_candidates"]] == [fields(row) for row in recomputed["validated_candidates"]]
    assert sampled_boxes == recomputed_samples == [boxes[1]["bbox"], boxes[2]["bbox"]]
    assert torch.equal(final_rng, torch.random.get_rng_state())
    assert [entry["target_bbox_info"]["object_id"] for entry in reused["validated_candidates"]] == [1, 2]


def test_precomputed_counts_require_matching_candidate_length():
    with pytest.raises(ValueError, match="hit_counts"):
        geometry_3d.find_best_matching_bbox_with_3d_validation(
            torch.ones(5, 2), [{"bbox": [0., 0., 1., 1.]}], None, {}, 0, None, None, 1., hit_counts=[],
        )


def test_matching_caller_aligns_counts_with_stable_topk_order(tmp_path, monkeypatch):
    from utils import matching_algorithms as matching
    from utils.config import SKUMatchingConfig

    config = SKUMatchingConfig.for_3d_mapping(backend="pi3", device="cpu")
    config.enable_sam3_mask_sampling = False
    config.output_dir = str(tmp_path / "output_3dmapping_pi3/0")
    config.min_bbox_area = 0
    config.min_3d_sample_points = 1
    config.max_3d_validation_candidates = 2
    path = tmp_path / "pi3_cache/predictions.npz"
    path.parent.mkdir()
    np.savez(path, image_ids=[0, 1], depth=np.ones((2, 4, 4, 1), np.float32),
             depth_conf=np.ones((2, 4, 4), np.float32), world_points=np.ones((2, 4, 4, 3), np.float32),
             world_points_conf=np.ones((2, 4, 4), np.float32),
             extrinsic=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
             intrinsic=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)))
    monkeypatch.setattr(matching, "PI3_SCENE_CACHE", {})
    monkeypatch.setattr(matching, "sample_3d_points_from_non_overlap_regions", lambda *_a: torch.ones(5, 3))
    projected = torch.tensor([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    monkeypatch.setattr(matching, "project_3d_to_2d", lambda *_a: projected)
    seen = []

    def validate(points, candidates, *_args, hit_counts, **_kwargs):
        assert points is projected
        seen.append(([box["object_id"] for box in candidates], hit_counts))
        return None

    monkeypatch.setattr(matching, "find_best_matching_bbox_with_3d_validation", validate)
    make_box = lambda position: {"position": position, "confidences": {"det": 1.0}}
    targets = [make_box([0., 0., 1., 1.]), make_box([1., 1., 4., 4.]), make_box([1., 1., 4., 4.])]
    transforms = [SimpleNamespace(image_id=index, map_bbox_to_final=lambda box: list(box)) for index in range(2)]
    matching.find_correspondences_3d_mapping(None, [{"objects": [targets[0]]}, {"objects": targets}],
                                           torch.zeros(2, 3, 4, 4), config, transforms_info=transforms)
    assert seen == [([1, 2], [4, 4])]

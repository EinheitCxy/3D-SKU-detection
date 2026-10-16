"""PVS adapter: shared encoding, box order, empty masks and device binding."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from utils import sam31_pvs, sam3_utils


def test_native_pvs_encodes_once_preserves_tail_and_round_clip(monkeypatch):
    boxes = np.asarray([
        [0.6, 0.6, 3.4, 4.4], [4.2, 1.2, 7.8, 5.8],
        [1.0, 2.0, 3.0, 6.0], [2.0, 0.0, 6.0, 7.0],
        [0.0, 1.0, 5.0, 8.0], [5.0, 3.0, 9.0, 7.0],
    ], dtype=np.float32)
    features = [torch.zeros(1, 256, 2, 2) for _ in range(3)]
    seen = []

    class Model:
        interactivity_no_mem_embed = torch.ones(1, 1, 256)

        def forward_image(self, image, *, need_interactive_out):
            seen.append("encode")
            assert image.shape == (1, 3, 1008, 1008)
            assert need_interactive_out
            return {"interactive": {"backbone_fpn": [
                SimpleNamespace(tensors=f) for f in features
            ]}}

        def _forward_sam_heads(self, embedding, **kwargs):
            n = len(kwargs["objects_to_interact"])
            offset = sum(value for value in seen if isinstance(value, int))
            seen.append(n)
            assert torch.equal(embedding, torch.ones_like(embedding))
            assert kwargs["interactive_high_res_features"][0] is features[0]
            assert kwargs["multimask_output"] is False
            assert kwargs["multiplex_state"] is None
            points = kwargs["point_inputs"]
            expected = boxes[offset:offset+n] * np.asarray([100.8, 126.0] * 2)
            np.testing.assert_allclose(points["point_coords"].reshape(-1, 4), expected, rtol=1e-6)
            assert points["point_labels"].tolist() == [[2, 3]] * n
            logits = torch.ones(n, 1, 8, 10)
            if offset == 0:
                logits[2] = -1  # Native absence stays empty; no center-point retry.
            return {"high_res_masks": logits, "ious": torch.ones(n, 1),
                    "object_score_logits": torch.ones(n, 1)}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch, "autocast", lambda *_a, **_k: nullcontext())
    masks, *_ = sam31_pvs.predict(Model(), Image.new("RGB", (10, 8)), boxes, 4, "cpu")
    assert seen == ["encode", 4, 2]
    assert masks.shape == (6, 8, 10) and masks.dtype == np.bool_
    for index, box in enumerate(boxes):
        expected = np.zeros((8, 10), dtype=bool)
        if index != 2:
            x1, y1, x2, y2 = np.round(box).astype(int)
            expected[y1:y2, x1:x2] = True
        np.testing.assert_array_equal(masks[index], expected)


def test_empty_and_invalid_boxes_never_load_model(monkeypatch):
    monkeypatch.setattr(sam31_pvs, "load_model", lambda *_a, **_k: pytest.fail("loaded model"))
    kwargs = dict(image_path=Image.new("RGB", (8, 6)), checkpoint_path="unused", device="cpu")
    assert sam3_utils.sam31_masks_from_bboxes(bboxes_xyxy=[], **kwargs) == []
    for boxes in [[[1, 2, 1, 3]], [[0, 0, float("nan"), 4]]]:
        with pytest.raises(ValueError, match="PVS boxes"):
            sam3_utils.sam31_masks_from_bboxes(bboxes_xyxy=boxes, **kwargs)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        sam3_utils.sam31_masks_from_bboxes(bboxes_xyxy=[[0, 0, 4, 4]], **kwargs)


def test_warmup_and_matching_share_pvs_arguments(monkeypatch):
    from utils.config import SKUMatchingConfig
    from utils.sam3_mask_cache import PVS_INFERENCE_CONTRACT

    calls = []
    monkeypatch.setattr(sam3_utils, "_SAM3_CODE_ROOT_OVERRIDE", None)
    monkeypatch.setattr(sam3_utils, "sam31_masks_from_bboxes", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sam3_utils, "sam3_masks_self_exemplar", lambda **_kw: pytest.fail("detector loaded"))
    config = SKUMatchingConfig.for_3d_mapping(backend="da3", device="cpu", sam3_checkpoint_path="unused")
    sam3_utils.warmup_sam3_pvs(config)
    assert len(calls) == 1
    assert calls[0]["max_batch_size"] == PVS_INFERENCE_CONTRACT["max_batch_size"] == 4

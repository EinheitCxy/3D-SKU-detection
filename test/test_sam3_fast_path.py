"""SAM3.1 self-exemplar fast-path equivalence and checkpoint-coverage tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from utils import sam3_utils

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sam31_tree(monkeypatch: pytest.MonkeyPatch):
    """Import the real sam31 tree, then restore the prior `sam3` modules."""
    if not (ROOT / "sam31" / "sam3" / "__init__.py").exists():
        pytest.skip("sam31 code tree is not available")
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "sam3"}
    for name in saved:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(ROOT / "sam31"))
    monkeypatch.setattr(sam3_utils, "_SAM3_CODE_ROOT_OVERRIDE", ROOT / "sam31")
    try:
        yield
    finally:
        for name in [k for k in sys.modules if k.split(".")[0] == "sam3"]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_sam31_top1_masks_match_postprocessor_extraction(sam31_tree) -> None:
    """Top-1-before-upsample must reproduce process_results + top-1 extraction."""
    from sam3.eval.postprocessors import PostProcessImage  # type: ignore
    from sam3.model.model_misc import SAM3Output  # type: ignore

    generator = torch.Generator().manual_seed(0)
    num_queries, num_dets, h, w = 6, 20, 37, 53
    pred_logits = torch.randn(num_queries, num_dets, 1, generator=generator) * 3
    pred_logits[2] = -10.0  # no detection above the threshold
    pred_logits[4] = -10.0
    pred_logits[4, 7] = 10.0  # exactly one kept detection
    outputs = {
        "pred_logits": pred_logits,
        "presence_logit_dec": torch.randn(num_queries, 1, generator=generator) + 2,
        "pred_boxes": torch.rand(num_queries, num_dets, 4, generator=generator),
        "pred_masks": torch.randn(num_queries, num_dets, 16, 16, generator=generator),
    }
    output = SAM3Output([[outputs]], iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
    row_query_ids = [105, 101, 103, 100, 104, 102]
    meta = SimpleNamespace(
        coco_image_id=torch.tensor(row_query_ids),
        original_size=torch.tensor([[h, w]] * num_queries),
    )
    batch = SimpleNamespace(find_metadatas=[meta])
    query_id_to_local_idx = {100: 3, 101: 0, 102: 5, 103: 1, 104: 2, 105: 4}
    bboxes = [[2.0, 3.0, 30.0, 20.0], [0.0, 0.0, 53.0, 37.0], [10.4, 5.6, 40.2, 33.1],
              [-5.0, -5.0, 20.0, 20.0], [30.0, 10.0, 60.0, 40.0], [5.0, 5.0, 6.0, 6.0]]
    postprocessor = PostProcessImage(
        max_dets_per_img=1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=True,
        always_interpolate_masks_on_gpu=False,
    )

    expected = sam3_utils._self_exemplar_masks_from_processed_results(
        postprocessor.process_results(output, batch.find_metadatas),
        query_id_to_local_idx, bboxes, w, h,
    )
    actual = sam3_utils._sam31_self_exemplar_top1_masks(
        output, batch, postprocessor, query_id_to_local_idx, bboxes, w, h
    )

    assert len(actual) == len(expected) == num_queries
    assert not expected[query_id_to_local_idx[103]].any()
    assert any(mask.any() for mask in expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype == np.bool_
        assert np.array_equal(got, want)


class _Attn(nn.Module):
    def __init__(self, freqs_cis: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("freqs_cis", freqs_cis)
        self.register_buffer("freqs_cis_real", torch.zeros_like(freqs_cis.real))
        self.register_buffer("freqs_cis_imag", torch.zeros_like(freqs_cis.imag))


class _TinyDetector(nn.Module):
    def __init__(self, freqs_cis: torch.Tensor) -> None:
        super().__init__()
        self.blk = nn.Module()
        self.blk.attn = _Attn(freqs_cis)
        self.proj = nn.Linear(2, 2)


def _freqs_cis() -> torch.Tensor:
    return torch.polar(torch.ones(3, 2), torch.arange(6.0).reshape(3, 2))


def _save_ckpt(path: Path, tensors: dict[str, torch.Tensor]) -> str:
    torch.save({"model": {f"detector.{k}": v for k, v in tensors.items()}}, path)
    return str(path)


def test_load_sam31_ckpt_loads_full_coverage_and_constructed_rope(tmp_path: Path) -> None:
    freqs = _freqs_cis()
    weight, bias = torch.randn(2, 2), torch.randn(2)
    ckpt = _save_ckpt(
        tmp_path / "full.pt",
        {"blk.attn.freqs_cis": freqs, "proj.weight": weight, "proj.bias": bias},
    )

    model = sam3_utils._load_sam31_ckpt(_TinyDetector(freqs), ckpt)

    assert torch.equal(model.proj.weight, weight)
    assert torch.equal(model.proj.bias, bias)
    assert torch.equal(model.blk.attn.freqs_cis_real, freqs.real)
    assert torch.equal(model.blk.attn.freqs_cis_imag, freqs.imag)


def test_load_sam31_ckpt_rejects_uncovered_parameter(tmp_path: Path) -> None:
    freqs = _freqs_cis()
    ckpt = _save_ckpt(
        tmp_path / "partial.pt",
        {"blk.attn.freqs_cis": freqs, "proj.weight": torch.randn(2, 2)},
    )

    with pytest.raises(RuntimeError, match="uninitialized: \\['proj.bias'\\]"):
        sam3_utils._load_sam31_ckpt(_TinyDetector(freqs), ckpt)


def test_load_sam31_ckpt_rejects_wrong_constructed_rope(tmp_path: Path) -> None:
    freqs = _freqs_cis()
    ckpt = _save_ckpt(
        tmp_path / "rope.pt",
        {"blk.attn.freqs_cis": freqs * 2, "proj.weight": torch.randn(2, 2),
         "proj.bias": torch.randn(2)},
    )

    with pytest.raises(RuntimeError, match="differs from checkpoint"):
        sam3_utils._load_sam31_ckpt(_TinyDetector(freqs), ckpt)


def test_load_sam31_ckpt_rejects_unverified_nonpersistent_buffer(tmp_path: Path) -> None:
    freqs = _freqs_cis()
    model = _TinyDetector(freqs)
    model.register_buffer("scratch", torch.empty(2), persistent=False)
    ckpt = _save_ckpt(
        tmp_path / "np.pt",
        {"blk.attn.freqs_cis": freqs, "proj.weight": torch.randn(2, 2),
         "proj.bias": torch.randn(2)},
    )

    with pytest.raises(RuntimeError, match="unverified non-persistent buffers"):
        sam3_utils._load_sam31_ckpt(model, ckpt)

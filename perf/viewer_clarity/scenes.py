"""Locked source inputs for the two Viewer clarity scenes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES: dict[str, dict] = {
    "video3": {
        "cache": ROOT / "runtime/video3-resolution-review/504/outputs/dataset/da3_cache/predictions.npz",
        "mapping": ROOT / "runtime/video3-resolution-review/504/outputs/dataset/dedup_detections/global_mapping.json",
        "images": ROOT / "runtime/video3-resolution-review/dataset/images",
        "masks": ROOT / "runtime/video3-resolution-review/504/outputs/dataset/sam3_mask_cache/v2",
        "eval_frames": (4, 14, 24),
    },
    "fd6": {
        "cache": ROOT / "Output/floor_display6/da3_cache/predictions.npz",
        "mapping": ROOT / "Output/floor_display6/dedup_detections/global_mapping.json",
        "images": ROOT / "imdata/floor_display6/images",
        "masks": ROOT / "Output/floor_display6/sam3_mask_cache_sam31/v2",
        "eval_frames": (1, 5, 9),
    },
}

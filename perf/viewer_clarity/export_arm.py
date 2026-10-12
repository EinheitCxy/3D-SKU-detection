"""Run one registered export arm and save its immutable bundle receipt."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from .pack import pack_bundle
from .scenes import ROOT, SCENES


def _tuning_for_arm(tuning_type, arm: str, owner_patch_cells: int | None = None):
    if arm == "XQ":
        return tuning_type(quality_filter=True, product_spill_trim=True, background_edge_weights=(2, 1, 1))
    if arm == "XS4":
        return tuning_type(single_owner=True, owner_patch_cells=4)
    if arm == "XS8":
        return tuning_type(single_owner=True, owner_patch_cells=8)
    if arm == "XQS":
        if owner_patch_cells not in (4, 8):
            raise ValueError("XQS requires --owner-patch-cells 4 or 8, selected from the XS arms")
        return tuning_type(quality_filter=True, product_spill_trim=True, background_edge_weights=(2, 1, 1), single_owner=True, owner_patch_cells=owner_patch_cells)
    raise ValueError(f"no tuning is defined for {arm}")


def current_generation(output_dir: Path) -> Path:
    pointer = json.loads((output_dir / "CURRENT").read_text(encoding="utf-8"))
    generation = output_dir / "runs" / pointer["run_id"]
    if not generation.is_dir():
        raise FileNotFoundError(generation)
    return generation


def run(scene: str, arm: str, owner_patch_cells: int | None = None) -> dict:
    if scene not in SCENES:
        raise ValueError(f"unknown scene: {scene}")
    if arm not in {"X0", "XQ", "XS4", "XS8", "XQS"}:
        raise ValueError(f"unknown export arm: {arm}")
    if arm == "XQS" and owner_patch_cells not in (4, 8):
        raise ValueError("XQS requires --owner-patch-cells 4 or 8, selected from the XS arms")
    if arm != "XQS" and owner_patch_cells is not None:
        raise ValueError("--owner-patch-cells is only valid for XQS")
    if arm != "X0":
        try:
            from src.viewer_export_tuning import ViewerExportTuning
            from src.web_viewer_export import export_web_viewer_bundle
        except ImportError as error:
            raise RuntimeError(f"{arm} requires the approved WP-X3 tuning interface") from error
        tuning = _tuning_for_arm(ViewerExportTuning, arm, owner_patch_cells)
        scene_inputs = SCENES[scene]
        out = ROOT / "runtime/viewer-clarity-20261012" / scene / arm
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        exported = export_web_viewer_bundle(
            dataset_name=f"{scene}-{arm}", da3_cache_path=scene_inputs["cache"],
            global_mapping_path=scene_inputs["mapping"], output_dir=out / "viewer",
            source_images_dir=scene_inputs["images"], sam3_mask_cache_root=scene_inputs["masks"],
            surfel_texture_edge=1920, tuning=tuning,
        )
        seconds = time.monotonic() - started
        generation = Path(exported["manifest_path"]).parent
        tuning = {key: getattr(tuning, key) for key in tuning.__dataclass_fields__}
    else:
        out = ROOT / "runtime/viewer-clarity-20261012" / scene / arm
        generation = current_generation(out / "viewer")
        original = json.loads((out / "export.json").read_text(encoding="utf-8"))
        seconds = float(original["seconds"])
        tuning = None
        exported = original
    archive_size = pack_bundle(generation, out / "viewer_bundle.zip")
    metrics = _point_counts(generation)
    receipt = {
        "scene": scene, "arm": arm, "generation_dir": str(generation),
        "seconds": seconds, "point_count": metrics["point_count"],
        "product_points": metrics["product_points"], "background_points": metrics["background_points"],
        "zip_bytes": archive_size, "tuning": tuning,
        "git_sha": exported.get("git_sha") if arm == "X0" else subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "manifest_path": str(generation / "manifest.json"),
        "export": exported,
    }
    (out / "export.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def _point_counts(generation: Path) -> dict:
    surfel = json.loads((generation / "surfel.json").read_text(encoding="utf-8"))
    objects = json.loads((generation / "objects.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    product = sum(int(end) - int(start) for entry in objects.values() for start, end in entry.get("point_ranges", []))
    return {"point_count": count, "product_points": product, "background_points": count - product}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SCENES))
    parser.add_argument("--arm", required=True, choices=("X0", "XQ", "XS4", "XS8", "XQS"))
    parser.add_argument("--owner-patch-cells", type=int, choices=(4, 8))
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.arm, args.owner_patch_cells), ensure_ascii=False))


if __name__ == "__main__":
    main()

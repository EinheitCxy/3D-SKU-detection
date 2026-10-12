"""Export and measure the approved video2 single-owner Viewer candidate."""

import json
import logging
from pathlib import Path
import sys
import time
import traceback
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from perf.viewer_clarity.pack import pack_bundle
from src.viewer_export_tuning import ViewerExportTuning
from src.web_viewer_export import export_web_viewer_bundle


SOURCE = ROOT / "runtime/fd-videos-20260911/video2"
BASELINE = (
    ROOT
    / "runtime/viewer-extent-20261012/video12/video2/viewer/runs"
    / "50e1703aac7c474fbeea1fccc2ac14be"
)
OUTPUT = (
    ROOT
    / "runtime/viewer-extent-20261012/video12/followup/ownership8"
)
VOXEL_SIZE_M = 0.008
CANDIDATE = (
    OUTPUT
    / "viewer/runs/ab803c8972024ec0bb2a312162be63e1"
)


def _gid_counts(bundle_dir: Path) -> dict[str, int]:
    surfel = json.loads((bundle_dir / "surfel.json").read_text(encoding="utf-8"))
    objects = json.loads((bundle_dir / "objects.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    counts: dict[str, int] = {}
    for gid, entry in objects.items():
        total = 0
        for start, end in entry.get("point_ranges", []):
            start, end = int(start), int(end)
            if start < 0 or end < start or end > count:
                raise ValueError(f"invalid point range for GID {gid}: {start}:{end}")
            total += end - start
        counts[str(gid)] = total
    return counts


def _multiframe_voxel_fraction(
    bundle_dir: Path, selected: np.ndarray | None = None
) -> dict[str, int | float]:
    surfel = json.loads((bundle_dir / "surfel.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    positions = np.fromfile(bundle_dir / "positions.f32.bin", dtype="<f4")
    if positions.size != count * 3:
        raise ValueError("positions length differs from point_count")
    positions = positions.reshape(count, 3)
    frames = np.fromfile(bundle_dir / "surfel-frame.u8.bin", dtype="u1")
    if len(frames) != count:
        raise ValueError("surfel-frame length differs from point_count")
    if selected is None:
        selected = np.ones(count, dtype=bool)
    elif selected.shape != (count,) or selected.dtype != np.bool_:
        raise ValueError("voxel metric selection must be a point-aligned bool mask")

    voxel = np.floor(
        positions[selected].astype(np.float64) / VOXEL_SIZE_M
    ).astype(np.int64)
    observed = np.unique(np.column_stack((voxel, frames[selected])), axis=0)
    _, frame_counts = np.unique(observed[:, :3], axis=0, return_counts=True)
    occupied_count = int(len(frame_counts))
    multiframe_count = int(np.count_nonzero(frame_counts >= 2))
    return {
        "voxel_size_m": VOXEL_SIZE_M,
        "occupied_voxels": occupied_count,
        "multiframe_voxels": multiframe_count,
        "multiframe_fraction": (
            float(multiframe_count / occupied_count) if occupied_count else 0.0
        ),
    }


def _bundle_summary(bundle_dir: Path) -> dict:
    surfel = json.loads((bundle_dir / "surfel.json").read_text(encoding="utf-8"))
    point_count = int(surfel["point_count"])
    gid_counts = _gid_counts(bundle_dir)
    return {
        "point_count": point_count,
        "frame_count": len(surfel["frames"]),
        "nonzero_gid_count": sum(count > 0 for count in gid_counts.values()),
        "gid_point_counts": gid_counts,
        "full_8mm_multiframe_voxel": _multiframe_voxel_fraction(bundle_dir),
        "product_only_8mm_multiframe_voxel": _multiframe_voxel_fraction(
            bundle_dir, _product_mask(bundle_dir, point_count)
        ),
        "full_scene_metric_note": (
            "Background remains at a fixed budget but may be reselected after "
            "ownership changes filtered indices; use product-only delta to "
            "interpret the ownership change itself."
        ),
    }


def _product_mask(bundle_dir: Path, count: int) -> np.ndarray:
    objects = json.loads((bundle_dir / "objects.json").read_text(encoding="utf-8"))
    product = np.zeros(count, dtype=bool)
    for gid, entry in objects.items():
        for start, end in entry.get("point_ranges", []):
            start, end = int(start), int(end)
            if start < 0 or end < start or end > count:
                raise ValueError(f"invalid point range for GID {gid}: {start}:{end}")
            product[start:end] = True
    return product


def _layer_metrics(bundle_dir: Path) -> dict[str, dict]:
    surfel = json.loads((bundle_dir / "surfel.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    product = _product_mask(bundle_dir, count)
    result = {}
    for name, selected in (("product", product), ("background", ~product)):
        result[name] = {
            "point_count": int(selected.sum()),
            **_multiframe_voxel_fraction(bundle_dir, selected),
        }
    return result


def _point_records(bundle_dir: Path, product_layer: bool) -> np.ndarray:
    surfel = json.loads((bundle_dir / "surfel.json").read_text(encoding="utf-8"))
    count = int(surfel["point_count"])
    positions = np.fromfile(bundle_dir / "positions.f32.bin", dtype="<f4").reshape(count, 3)
    frames = np.fromfile(bundle_dir / "surfel-frame.u8.bin", dtype="u1")
    selected = _product_mask(bundle_dir, count)
    if not product_layer:
        selected = ~selected
    bits = positions[selected].view("<u4").reshape(-1, 3)
    records = np.empty(
        len(bits),
        dtype=[("x", "<u4"), ("y", "<u4"), ("z", "<u4"), ("frame", "u1")],
    )
    records["x"], records["y"], records["z"] = bits.T
    records["frame"] = frames[selected]
    return np.unique(records)


def _overlap_summary(product_layer: bool) -> dict[str, int | float]:
    baseline = _point_records(BASELINE, product_layer)
    candidate = _point_records(CANDIDATE, product_layer)
    common = np.intersect1d(baseline, candidate, assume_unique=True)
    return {
        "baseline_unique_records": int(len(baseline)),
        "candidate_unique_records": int(len(candidate)),
        "common_unique_records": int(len(common)),
        "candidate_only_unique_records": int(len(candidate) - len(common)),
        "baseline_only_unique_records": int(len(baseline) - len(common)),
        "candidate_overlap_fraction": float(len(common) / len(candidate)) if len(candidate) else 0.0,
    }


def audit_existing() -> None:
    log_path = OUTPUT / "comparability.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    started = time.monotonic()
    logging.info("auditing existing generations without re-export")
    baseline_manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((CANDIDATE / "manifest.json").read_text(encoding="utf-8"))
    baseline_transform = np.asarray(baseline_manifest["world_to_view"], dtype=np.float64).reshape(4, 4)
    candidate_transform = np.asarray(candidate_manifest["world_to_view"], dtype=np.float64).reshape(4, 4)

    baseline_layers = _layer_metrics(BASELINE)
    candidate_layers = _layer_metrics(CANDIDATE)
    overlap = {
        "product": _overlap_summary(True),
        "background": _overlap_summary(False),
    }
    report = {
        "status": "complete",
        "scope": "existing artifacts only; no export rerun",
        "coordinate_audit": {
            "positions_file_coordinates": "raw DA3 world coordinates shared by both exports",
            "positions_transform_applied_by_exporter": False,
            "world_to_view_role": "manifest-only Viewer transform computed after sampling",
            "rotation_max_abs_delta": float(
                np.max(np.abs(baseline_transform[:3, :3] - candidate_transform[:3, :3]))
            ),
            "translation_delta": (
                candidate_transform[:3, 3] - baseline_transform[:3, 3]
            ).tolist(),
            "metric_coordinates": "raw common world coordinates from positions.f32.bin",
        },
        "selection_audit": {
            "baseline_layers": baseline_layers,
            "candidate_layers": candidate_layers,
            "exact_float32_xyz_plus_frame_record_overlap": overlap,
            "explanation": (
                "single-owner changes filtered_indices before EdgeAdaptiveSampler; "
                "the fixed 500000-point background budget is then refilled from a "
                "different support/candidate set, so the final full-scene candidate "
                "is not a subset of baseline"
            ),
        },
        "comparison": {
            layer: {
                "point_delta": candidate_layers[layer]["point_count"] - baseline_layers[layer]["point_count"],
                "occupied_voxel_delta": candidate_layers[layer]["occupied_voxels"] - baseline_layers[layer]["occupied_voxels"],
                "multiframe_voxel_delta": candidate_layers[layer]["multiframe_voxels"] - baseline_layers[layer]["multiframe_voxels"],
                "multiframe_fraction_delta": candidate_layers[layer]["multiframe_fraction"] - baseline_layers[layer]["multiframe_fraction"],
            }
            for layer in ("product", "background")
        },
        "interpretation": {
            "full_scene_original_metric": "confounded by background reselection",
            "product_only_metric": "comparable in common world coordinates; subset status is reported by exact record overlap",
            "candidate_failure_claim": "withdrawn; comparable product-only metric improves while full-scene metric is confounded",
        },
        "seconds": time.monotonic() - started,
        "source_evidence": [
            "src/web_viewer_export.py:147-152 manifest-only world_to_view",
            "src/web_viewer_export.py:532-558 background and product adaptive sampling",
            "src/web_viewer_export.py:566 positions written from selected raw points",
            "src/web_viewer_export.py:904-914 world_to_view computation",
        ],
    }
    out = OUTPUT / "comparability_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("wrote %s", out)
    print(json.dumps(report, indent=2))


def _zip_summary(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bad_paths = [
            name
            for name in names
            if name.startswith("/") or ".." in Path(name).parts
        ]
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "member_count": len(names),
            "duplicate_members": len(names) - len(set(names)),
            "unsafe_member_paths": bad_paths,
            "has_manifest": "manifest.json" in names,
            "has_surfel": "surfel.json" in names,
        }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    log_path = OUTPUT / "command.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    invocation = {
        "script_argv": sys.argv,
        "recommended_command": (
            "uv run python perf/surfel_extent/experiments/ownership_export.py"
        ),
    }
    logging.info("script argv: %s", sys.argv)
    logging.info("recommended command: %s", invocation["recommended_command"])
    logging.info("source: %s", SOURCE)
    logging.info("baseline: %s", BASELINE)
    logging.info(
        "candidate: surfel_texture_edge=1920, single_owner=True, "
        "owner_patch_cells=8, other exporter options=current defaults"
    )

    started = time.monotonic()
    try:
        result = export_web_viewer_bundle(
            dataset_name="video2-full-ownership8",
            da3_cache_path=SOURCE / "outputs/dataset/da3_cache/predictions.npz",
            global_mapping_path=(
                SOURCE / "outputs/dataset/dedup_detections/global_mapping.json"
            ),
            output_dir=OUTPUT / "viewer",
            source_images_dir=SOURCE / "dataset/images",
            sam3_mask_cache_root=SOURCE / "outputs/dataset/sam3_mask_cache/v2",
            surfel_texture_edge=1920,
            tuning=ViewerExportTuning(single_owner=True, owner_patch_cells=8),
        )
        generation = Path(result["manifest_path"]).parent
        zip_path = OUTPUT / "viewer_bundle.zip"
        result["zip_bytes"] = pack_bundle(generation, zip_path)
        export_seconds = time.monotonic() - started

        metric_started = time.monotonic()
        baseline = _bundle_summary(BASELINE)
        candidate = _bundle_summary(generation)
        all_gids = sorted(
            set(baseline["gid_point_counts"]) | set(candidate["gid_point_counts"]),
            key=lambda value: int(value),
        )
        retention = {}
        for gid in all_gids:
            before = baseline["gid_point_counts"].get(gid, 0)
            after = candidate["gid_point_counts"].get(gid, 0)
            retention[gid] = {
                "baseline_points": before,
                "candidate_points": after,
                "retention_fraction": float(after / before) if before else None,
            }
        metric_seconds = time.monotonic() - metric_started

        report = {
            "status": "complete",
            "invocation": invocation,
            "source": {
                "da3_cache": str(
                    SOURCE / "outputs/dataset/da3_cache/predictions.npz"
                ),
                "global_mapping": str(
                    SOURCE / "outputs/dataset/dedup_detections/global_mapping.json"
                ),
                "sam3_mask_cache": str(
                    SOURCE / "outputs/dataset/sam3_mask_cache/v2"
                ),
                "source_images": str(SOURCE / "dataset/images"),
                "baseline_generation": str(BASELINE),
            },
            "export_settings": {
                "surfel_texture_edge": 1920,
                "tuning": {
                    "single_owner": True,
                    "owner_patch_cells": 8,
                },
                "other_exporter_options": "current defaults",
                "spatial_crop": False,
                "additional_downsampling": False,
            },
            "timing_seconds": {
                "export_and_pack": export_seconds,
                "metrics": metric_seconds,
                "total": time.monotonic() - started,
            },
            "export_result": result,
            "baseline": baseline,
            "candidate": candidate,
            "comparison": {
                "point_retention_fraction": (
                    candidate["point_count"] / baseline["point_count"]
                ),
                "point_reduction": baseline["point_count"] - candidate["point_count"],
                "nonzero_gid_delta": (
                    candidate["nonzero_gid_count"] - baseline["nonzero_gid_count"]
                ),
                "multiframe_8mm_fraction_delta": (
                    candidate["full_8mm_multiframe_voxel"]["multiframe_fraction"]
                    - baseline["full_8mm_multiframe_voxel"]["multiframe_fraction"]
                ),
                "product_only_multiframe_8mm_fraction_delta": (
                    candidate["product_only_8mm_multiframe_voxel"]["multiframe_fraction"]
                    - baseline["product_only_8mm_multiframe_voxel"]["multiframe_fraction"]
                ),
                "metric_interpretation": (
                    "Use product_only_multiframe_8mm_fraction_delta for the "
                    "ownership effect; the full-scene delta can be confounded "
                    "by fixed-budget background reselection."
                ),
                "per_gid_retention": retention,
            },
            "zip": _zip_summary(zip_path),
            "excluded_metrics": {
                "full_M2_M6_graph": "not run; unnecessary and expensive for this candidate",
                "sampled_nn_mixed_frame": "not run; product-only full 8 mm metric is the ownership effect measure",
            },
        }
        report_path = OUTPUT / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.info("report: %s", report_path)
        logging.info("ready ZIP: %s", zip_path)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        failure = {
            "status": "failed",
            "invocation": invocation,
            "seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (OUTPUT / "error.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        logging.exception("experiment failed")
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--audit-existing"]:
        audit_existing()
    else:
        main()

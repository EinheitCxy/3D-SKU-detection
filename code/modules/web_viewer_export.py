"""Strict schema-v2 DA3 to bundle-v1 static web-viewer exporter."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from modules.da3_footprint_stage import resolve_current_footprint_artifacts
from utils.global_id_mapper import GlobalIDMapper
from viewer.indexer import build_global_object_index


_BUNDLE_FILES = (
    "manifest.json",
    "positions.f32.bin",
    "colors.u8.bin",
    "confidences.f32.bin",
    "frame_ids.i32.bin",
    "objects.json",
    "footprints.json",
)
_REQUIRED_CACHE_FIELDS = frozenset(
    {
        "cache_schema_version",
        "source_model",
        "image_ids",
        "world_points",
        "world_points_conf",
        "images",
        "source_image_sizes",
        "source_to_processed_affine",
        "source_image_sha256",
        "affine_convention",
        "preprocess_resolution",
        "preprocess_method",
    }
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9._/-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WebViewerExportError(ValueError):
    """Raised when an input cannot satisfy the strict bundle-v1 contract."""


def export_web_viewer_bundle(
    *,
    da3_cache_path: Path,
    global_mapping_path: Path,
    footprint_root: Path,
    output_dir: Path,
    voxel_size_m: float = 0.01,
    max_points: int = 500_000,
) -> dict[str, object]:
    """Export verified formal artifacts and sampled DA3 points as bundle-v1."""
    voxel_size = _validate_export_options(voxel_size_m, max_points)
    cache = _load_da3_cache(Path(da3_cache_path))
    sampled = _sample_points(cache, voxel_size=voxel_size, max_points=max_points)
    artifacts = resolve_current_footprint_artifacts(Path(footprint_root))
    footprint, report_cache = _load_footprint(artifacts)
    if report_cache != cache["provenance"]:
        raise WebViewerExportError(
            "DA3 cache provenance does not match formal footprint report"
        )
    objects = build_global_object_index(GlobalIDMapper(str(global_mapping_path)))
    output_path = Path(output_dir)

    manifest = {
        "schema_version": "1.0.0",
        "coordinate_space": "da3_world_meters",
        "point_count": int(len(sampled["positions"])),
        "arrays": {
            "positions": {
                "path": "positions.f32.bin",
                "dtype": "float32",
                "components": 3,
                "byte_length": int(sampled["positions"].nbytes),
            },
            "colors": {
                "path": "colors.u8.bin",
                "dtype": "uint8",
                "components": 3,
                "byte_length": int(sampled["colors"].nbytes),
            },
            "confidences": {
                "path": "confidences.f32.bin",
                "dtype": "float32",
                "components": 1,
                "byte_length": int(sampled["confidences"].nbytes),
            },
            "frame_ids": {
                "path": "frame_ids.i32.bin",
                "dtype": "int32",
                "components": 1,
                "byte_length": int(sampled["frame_ids"].nbytes),
            },
        },
        "objects_path": "objects.json",
        "footprints_path": "footprints.json",
        "source": {
            "da3_cache": {
                **cache["provenance"],
            },
            "footprint": {
                "run_id": footprint["run_id"],
                "status": footprint["status"],
            },
            "export": {"voxel_size_m": voxel_size, "max_points": max_points},
        },
        "capabilities": {
            "point_picking": False,
            "footprint_picking": True,
            "formal_ground_footprint": True,
        },
    }
    generation = _publish_bundle(output_path, manifest, objects, footprint, sampled)
    return {
        "output_dir": str(output_path),
        "manifest_path": str(generation / "manifest.json"),
        "point_count": int(len(sampled["positions"])),
        "footprint_status": footprint["status"],
    }


def _validate_export_options(voxel_size_m: float, max_points: int) -> float:
    if isinstance(voxel_size_m, bool):
        raise WebViewerExportError("voxel_size_m must be a positive finite number")
    try:
        voxel_size = float(voxel_size_m)
    except (TypeError, ValueError) as error:
        raise WebViewerExportError("voxel_size_m must be a positive finite number") from error
    if not math.isfinite(voxel_size) or voxel_size <= 0:
        raise WebViewerExportError("voxel_size_m must be a positive finite number")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points <= 0:
        raise WebViewerExportError("max_points must be a positive integer")
    return voxel_size


def _load_da3_cache(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(_REQUIRED_CACHE_FIELDS - set(loaded.files))
            if missing:
                raise WebViewerExportError(
                    "DA3 cache missing required schema-v2 fields: " + ", ".join(missing)
                )
            cache = {field: loaded[field].copy() for field in _REQUIRED_CACHE_FIELDS}
    except WebViewerExportError:
        raise
    except (OSError, ValueError) as error:
        raise WebViewerExportError(f"cannot load DA3 cache: {error}") from error

    schema = cache["cache_schema_version"]
    if schema.shape != () or schema.dtype.kind not in "iu" or int(schema.item()) != 2:
        raise WebViewerExportError("DA3 cache schema version must be exactly 2")
    source_model = _unicode_scalar(cache["source_model"], "source_model")
    if not _MODEL_ID.fullmatch(source_model):
        raise WebViewerExportError("DA3 cache source_model is unsafe")
    points = cache["world_points"]
    confidence = cache["world_points_conf"]
    images = cache["images"]
    image_ids = cache["image_ids"]
    if points.dtype != np.dtype(np.float32) or points.ndim != 4 or points.shape[-1] != 3:
        raise WebViewerExportError("DA3 cache world_points must be float32 with shape (N, H, W, 3)")
    frame_count, height, width, _ = points.shape
    if frame_count < 1 or height < 1 or width < 1:
        raise WebViewerExportError("DA3 cache world_points grid must be nonempty")
    if confidence.dtype != np.dtype(np.float32) or confidence.shape != points.shape[:3]:
        raise WebViewerExportError("DA3 cache world_points_conf must be float32 and align with world_points")
    if images.dtype != np.dtype(np.uint8) or images.shape != (frame_count, height, width, 3):
        raise WebViewerExportError("DA3 cache images must be uint8 and align with world_points")
    if image_ids.dtype.kind not in "iu" or image_ids.shape != (frame_count,):
        raise WebViewerExportError("DA3 cache image_ids must be an integer vector aligned with frames")
    if len({int(value) for value in image_ids}) != frame_count:
        raise WebViewerExportError("DA3 cache image_ids must be unique")
    integer_info = np.iinfo(np.int32)
    if (
        (image_ids.dtype.kind == "i" and np.any(image_ids < integer_info.min))
        or np.any(image_ids > integer_info.max)
    ):
        raise WebViewerExportError("DA3 cache image_ids cannot be represented as int32")
    sizes = cache["source_image_sizes"]
    affine = cache["source_to_processed_affine"]
    hashes = cache["source_image_sha256"]
    convention = _unicode_scalar(cache["affine_convention"], "affine_convention")
    resolution = _integer_scalar(cache["preprocess_resolution"], "preprocess_resolution")
    method = _unicode_scalar(cache["preprocess_method"], "preprocess_method")
    if sizes.dtype.kind not in "iu" or sizes.shape != (frame_count, 2) or np.any(sizes <= 0):
        raise WebViewerExportError("DA3 cache source_image_sizes must be positive integer pairs")
    if affine.dtype.kind not in "fiu" or affine.shape != (frame_count, 2, 3) or not np.isfinite(affine).all():
        raise WebViewerExportError("DA3 cache source_to_processed_affine is invalid")
    _validate_affine_linear_parts(affine)
    if hashes.dtype.kind != "U" or hashes.shape != (frame_count,) or any(
        _SHA256.fullmatch(str(value)) is None for value in hashes
    ):
        raise WebViewerExportError("DA3 cache source_image_sha256 is invalid")
    if convention != "pixel_center_v1" or resolution <= 0 or method != "upper_bound_resize":
        raise WebViewerExportError("DA3 cache formal preprocessing metadata is invalid")
    provenance = {
        "schema_version": 2,
        "source_model": source_model,
        "affine_convention": convention,
        "preprocess_resolution": resolution,
        "preprocess_method": method,
        "frame_count": int(frame_count),
        "processed_size": [int(width), int(height)],
        "image_ids": [int(value) for value in image_ids],
        "source_image_sha256": [str(value) for value in hashes],
    }
    return {
        "image_ids": image_ids.astype(np.int32, copy=False),
        "points": points,
        "confidence": confidence,
        "images": images,
        "provenance": provenance,
    }


def _unicode_scalar(value: np.ndarray, field: str) -> str:
    if value.shape != () or value.dtype.kind != "U" or not value.item():
        raise WebViewerExportError(f"DA3 cache {field} must be a nonempty unicode scalar")
    return str(value.item())


def _integer_scalar(value: np.ndarray, field: str) -> int:
    if value.shape != () or value.dtype.kind not in "iu":
        raise WebViewerExportError(f"DA3 cache {field} must be an integer scalar")
    return int(value.item())


def _validate_affine_linear_parts(affine: np.ndarray) -> None:
    linear = affine[:, :, :2]
    if not np.allclose(linear[:, 0, 1], 0.0, rtol=0.0, atol=1e-8) or not np.allclose(
        linear[:, 1, 0], 0.0, rtol=0.0, atol=1e-8
    ):
        raise WebViewerExportError("DA3 cache affine linear part must be axis-aligned")
    if np.any(linear[:, 0, 0] <= 0.0) or np.any(linear[:, 1, 1] <= 0.0):
        raise WebViewerExportError("DA3 cache affine linear scales must be positive")
    if np.any(np.linalg.det(linear) <= 0.0):
        raise WebViewerExportError("DA3 cache affine determinant must be positive")
    if any(np.linalg.matrix_rank(matrix) != 2 for matrix in linear):
        raise WebViewerExportError("DA3 cache affine linear part must have rank two")


def _sample_points(cache: dict[str, Any], *, voxel_size: float, max_points: int) -> dict[str, np.ndarray]:
    points = cache["points"].reshape(-1, 3)
    confidence = cache["confidence"].reshape(-1)
    colors = cache["images"].reshape(-1, 3)
    frame_count, height, width, _ = cache["points"].shape
    frame_ids = np.repeat(cache["image_ids"], height * width)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.any(points != 0, axis=1)
        & np.isfinite(confidence)
        & (confidence >= 1.0)
    )
    points = points[valid]
    confidence = confidence[valid]
    colors = colors[valid]
    frame_ids = frame_ids[valid]
    if len(points):
        scaled = np.floor(points.astype(np.float64, copy=False) / voxel_size)
        integer_info = np.iinfo(np.int64)
        if not np.isfinite(scaled).all() or np.any(scaled < integer_info.min) or np.any(
            scaled > integer_info.max
        ):
            raise WebViewerExportError("DA3 voxel keys exceed int64 representation")
        selected: dict[tuple[int, int, int], int] = {}
        for index, key_values in enumerate(scaled.astype(np.int64)):
            key = tuple(int(value) for value in key_values)
            previous = selected.get(key)
            if previous is None or confidence[index] > confidence[previous]:
                selected[key] = index
        keep = np.fromiter(selected.values(), dtype=np.int64, count=len(selected))[:max_points]
        points = points[keep]
        confidence = confidence[keep]
        colors = colors[keep]
        frame_ids = frame_ids[keep]
    positions = np.ascontiguousarray(points, dtype="<f4")
    confidences = np.ascontiguousarray(confidence, dtype="<f4")
    if not np.isfinite(positions).all() or not np.isfinite(confidences).all():
        raise WebViewerExportError("valid DA3 points must be representable as finite float32")
    return {
        "positions": positions,
        "colors": np.ascontiguousarray(colors, dtype=np.uint8),
        "confidences": confidences,
        "frame_ids": np.ascontiguousarray(frame_ids, dtype="<i4"),
    }


def _load_footprint(artifacts: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report_path = Path(artifacts["measurement_report"])
        report = _read_json(report_path)
        geojson = _read_json(Path(artifacts["footprints_geojson"]))
        generation_manifest = _read_json(report_path.with_name("manifest.json"))
    except KeyError as error:
        raise WebViewerExportError("footprint resolver omitted a required artifact") from error
    if not isinstance(report, dict) or not isinstance(geojson, dict) or not isinstance(generation_manifest, dict):
        raise WebViewerExportError("formal footprint artifacts must contain JSON objects")
    report_cache = report.get("cache")
    if not isinstance(report_cache, dict):
        raise WebViewerExportError("formal footprint report cache provenance is invalid")
    run_id = generation_manifest.get("run_id")
    if not isinstance(run_id, str):
        raise WebViewerExportError("formal footprint manifest run_id is invalid")
    status = report.get("status")
    if report.get("metric") != "da3_ground_footprint_union" or report.get("unit") != "m2":
        raise WebViewerExportError("formal footprint report metric or unit is invalid")
    if status not in {"accepted", "rejected"}:
        raise WebViewerExportError("formal footprint report status is invalid")
    accepted = status == "accepted"
    value = report.get("value_m2")
    if accepted:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise WebViewerExportError("accepted formal footprint value_m2 must be finite")
    elif value is not None:
        raise WebViewerExportError("rejected formal footprint value_m2 must be null")
    if (
        geojson.get("type") != "FeatureCollection"
        or geojson.get("coordinate_space") != "local_support_plane_meters"
        or geojson.get("status") != status
        or geojson.get("measurement_complete") is not accepted
        or not isinstance(geojson.get("features"), list)
    ):
        raise WebViewerExportError("formal footprint GeoJSON contract is invalid")
    if not accepted:
        if geojson["features"]:
            raise WebViewerExportError("rejected formal footprint must not contain geometry")
        return {
            "metric": report["metric"],
            "unit": report["unit"],
            "status": status,
            "value_m2": None,
            "rejection_reason": report.get("rejection_reason"),
            "run_id": run_id,
            "support_plane": None,
            "per_global_id": {},
            "union": None,
        }, report_cache
    plane = _support_plane(report)
    per_global_id: dict[str, Any] = {}
    union: dict[str, Any] | None = None
    for feature in geojson["features"]:
        global_id, geometry = _parse_feature(feature)
        if global_id == "union":
            if union is not None:
                raise WebViewerExportError("formal footprint GeoJSON has multiple union features")
            union = geometry
        elif global_id in per_global_id:
            raise WebViewerExportError("formal footprint GeoJSON has duplicate global_id features")
        else:
            per_global_id[global_id] = geometry
    if union is None:
        raise WebViewerExportError("accepted formal footprint GeoJSON is missing union geometry")
    return {
        "metric": report["metric"],
        "unit": report["unit"],
        "status": status,
        "value_m2": float(value),
        "rejection_reason": report.get("rejection_reason"),
        "run_id": run_id,
        "support_plane": plane,
        "per_global_id": per_global_id,
        "union": union,
    }, report_cache


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise WebViewerExportError(f"cannot read JSON artifact {path}: {error}") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _support_plane(report: dict[str, Any]) -> dict[str, list[float]]:
    plane = report.get("plane")
    selected = plane.get("selected") if isinstance(plane, dict) else None
    if not isinstance(selected, dict):
        raise WebViewerExportError("accepted formal footprint report is missing support plane")
    result: dict[str, list[float]] = {}
    for field in ("point", "u_axis", "v_axis", "normal"):
        value = selected.get(field)
        if not isinstance(value, list) or len(value) != 3:
            raise WebViewerExportError(f"formal footprint support plane {field} is invalid")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
            raise WebViewerExportError(f"formal footprint support plane {field} is invalid")
        result[field] = [float(item) for item in value]
    return result


def _parse_feature(feature: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        raise WebViewerExportError("formal footprint GeoJSON feature is invalid")
    properties = feature.get("properties")
    if not isinstance(properties, dict) or not isinstance(properties.get("global_id"), str):
        raise WebViewerExportError("formal footprint GeoJSON global_id is invalid")
    global_id = properties["global_id"]
    if not global_id:
        raise WebViewerExportError("formal footprint GeoJSON global_id is invalid")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise WebViewerExportError("formal footprint GeoJSON geometry is invalid")
    polygons = _parse_polygons(geometry["type"], geometry.get("coordinates"))
    return global_id, {"rings": polygons, "properties": properties}


def _parse_polygons(geometry_type: str, coordinates: Any) -> list[list[list[list[float]]]]:
    raw_polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if not isinstance(raw_polygons, list) or not raw_polygons:
        raise WebViewerExportError("formal footprint GeoJSON coordinates are invalid")
    polygons: list[list[list[list[float]]]] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, list) or not raw_polygon:
            raise WebViewerExportError("formal footprint GeoJSON polygon rings are invalid")
        rings: list[list[list[float]]] = []
        for raw_ring in raw_polygon:
            if not isinstance(raw_ring, list) or len(raw_ring) < 4:
                raise WebViewerExportError("formal footprint GeoJSON ring is invalid")
            ring: list[list[float]] = []
            for coordinate in raw_ring:
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 2
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not math.isfinite(item)
                        for item in coordinate
                    )
                ):
                    raise WebViewerExportError("formal footprint GeoJSON coordinate is invalid")
                ring.append([float(coordinate[0]), float(coordinate[1])])
            if ring[0] != ring[-1]:
                raise WebViewerExportError("formal footprint GeoJSON ring must be closed")
            rings.append(ring)
        polygons.append(rings)
    return polygons


def _publish_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    objects: dict[str, Any],
    footprint: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_root = output_dir / "runs"
    runs_root.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    generation = runs_root / run_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    try:
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "positions.f32.bin").write_bytes(arrays["positions"].tobytes(order="C"))
        (temporary / "colors.u8.bin").write_bytes(arrays["colors"].tobytes(order="C"))
        (temporary / "confidences.f32.bin").write_bytes(arrays["confidences"].tobytes(order="C"))
        (temporary / "frame_ids.i32.bin").write_bytes(arrays["frame_ids"].tobytes(order="C"))
        _write_json(temporary / "objects.json", objects)
        _write_json(temporary / "footprints.json", footprint)
        os.rename(temporary, generation)
        _atomic_replace_current(
            output_dir / "CURRENT",
            {"complete": True, "run_id": run_id, "schema_version": "1.0.0"},
        )
        return generation
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _atomic_replace_current(current_path: Path, pointer: dict[str, object]) -> None:
    payload = (json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".CURRENT.", dir=current_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, current_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

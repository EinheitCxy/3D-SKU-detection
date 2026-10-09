"""Preserve valid primary textures and fill only coherent residual patches.

This isolated candidate changes source selection only. Projection, depth, mask,
and front-facing gates come unchanged from the original ROI experiment.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from perf.roi_fusion.texture import _patches, face_view_candidates


_FACE_BATCH = 32768
_MIN_PATCH_FACES = 4
_ANCHOR_NEIGHBORS = 4


def _candidates_in_batches(vertices, faces, data):
    """Bound projection temporaries while keeping the exact original gates."""
    count, views = len(faces), len(data["E"])
    visibility = np.empty((views, count), dtype=bool)
    scores = np.empty((views, count), dtype=np.float32)
    twice_area = np.empty(count, dtype=np.float64)
    centers = np.empty((count, 3), dtype=np.float64)
    normals = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, _FACE_BATCH):
        stop = min(start + _FACE_BATCH, count)
        visible, quality, area, center, normal = face_view_candidates(
            vertices, faces[start:stop], data
        )
        visibility[:, start:stop] = visible
        scores[:, start:stop] = quality
        twice_area[start:stop] = area
        centers[start:stop] = center
        normals[start:stop] = normal
    return visibility, scores, twice_area, centers, normals


def _boundary_anchors(centers, normals, sources, residual, spacing):
    """Nearest compatible frozen source, using at most four nearby faces.

    Anchors express a continuity preference only. They never make a source
    eligible, and residual assignments never become anchors for another patch.
    """
    anchors = np.full(len(residual), -1, dtype=np.int32)
    frozen = np.flatnonzero(sources >= 0)
    if not len(frozen) or not len(residual):
        return anchors
    tree = cKDTree(centers[frozen])
    cosine = np.cos(np.deg2rad(25))
    for start in range(0, len(residual), _FACE_BATCH):
        stop = min(start + _FACE_BATCH, len(residual))
        group = residual[start:stop]
        _, nearest = tree.query(
            centers[group], k=min(_ANCHOR_NEIGHBORS, len(frozen)),
            distance_upper_bound=2.5 * spacing,
        )
        if nearest.ndim == 1:
            nearest = nearest[:, None]
        found = nearest < len(frozen)
        indices = frozen[np.minimum(nearest, len(frozen) - 1)]
        found &= np.einsum("ijk,ik->ij", normals[indices], normals[group]) >= cosine
        first = found.argmax(axis=1)
        has_anchor = found.any(axis=1)
        selected = indices[np.arange(len(group)), first]
        anchors[start:stop][has_anchor] = sources[selected[has_anchor]]
    return anchors


def choose_sources(vertices, faces, data, *, preferred_sources=None):
    """Return int32 frame indices, with -1 for deliberately untextured faces.

    A visible original/primary frame is immutable. The remaining strictly
    observable faces form bounded spatial/normal patches using the old helper.
    A patch needs at least four faces and one view visible on *every* face.
    Choose one view for the complete patch, preferring a matching frozen source
    at its boundary and then the old summed projected-area quality. Never grow
    patches from newly assigned labels or chase individual remaining faces.
    """
    started = time.monotonic()
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    spacing = float(data["spacing"])
    if not len(faces) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Nonempty mesh and finite positive spacing are required")
    if preferred_sources is None:
        raise ValueError("preferred_sources is required for primary-source anchoring")
    preferred = np.asarray(preferred_sources)
    views, count = len(data["E"]), len(faces)
    if preferred.shape != (count,) or not np.issubdtype(preferred.dtype, np.integer):
        raise ValueError("preferred_sources must contain one integer frame index per face")
    if np.any((preferred < 0) | (preferred >= views)):
        raise ValueError("preferred_sources contains an invalid frame index")

    visibility, scores, twice_area, centers, normals = _candidates_in_batches(
        vertices, faces, data
    )
    candidate_seconds = time.monotonic() - started
    face_indices = np.arange(count)
    original_valid = visibility[preferred, face_indices]
    sources = np.full(count, -1, dtype=np.int32)
    sources[original_valid] = preferred[original_valid]
    observable = visibility.any(axis=0)
    residual = np.flatnonzero(~original_valid & observable)
    anchors = _boundary_anchors(centers, normals, sources, residual, spacing)
    patches = _patches(centers[residual], normals[residual], spacing) if len(residual) else []
    patch_source_counts = {}
    accepted_sizes = []
    deferred_small = deferred_conflict = 0
    deferred_small_area = deferred_conflict_area = 0.0
    anchored_patches = 0
    anchor_boundary_faces = anchor_agreement_faces = 0
    for patch in patches:
        selected = residual[patch]
        if len(patch) < _MIN_PATCH_FACES:
            deferred_small += 1
            deferred_small_area += float(twice_area[selected].sum() * .5)
            continue
        candidates = np.flatnonzero(visibility[:, selected].all(axis=1))
        if not len(candidates):
            deferred_conflict += 1
            deferred_conflict_area += float(twice_area[selected].sum() * .5)
            continue

        boundary = anchors[patch]
        has_anchor = boundary >= 0
        support = np.bincount(
            boundary[has_anchor], weights=twice_area[selected[has_anchor]],
            minlength=views,
        )[candidates]
        if support.max() > 0:
            candidates = candidates[support == support.max()]
            anchored_patches += 1
        # Visibility remains bool; only quality storage is float32. Accumulate
        # in float64 so large patches do not add float32 summation error.
        quality = scores[np.ix_(candidates, selected)].sum(axis=1, dtype=np.float64)
        source = int(candidates[np.argmax(quality)])
        sources[selected] = source
        patch_source_counts[str(source)] = patch_source_counts.get(str(source), 0) + 1
        accepted_sizes.append(len(patch))
        anchor_boundary_faces += int(has_anchor.sum())
        anchor_agreement_faces += int(np.count_nonzero(boundary == source))

    textured = sources >= 0
    recovered = textured & ~original_valid
    still_observable = ~textured & observable
    chosen_valid = visibility[sources[textured], face_indices[textured]]
    changed_original = original_valid & (sources != preferred)
    if np.any(changed_original) or not chosen_valid.all():
        raise RuntimeError("Source selection violated an immutable primary or visibility gate")

    def area(mask):
        return float(twice_area[mask].sum() * .5)

    total_area = float(twice_area.sum() * .5)
    denominator = max(total_area, 1e-20)
    before_area = area(~original_valid)
    recovered_area = area(recovered)
    untextured_area = area(~textured)
    pair_codes = preferred[recovered].astype(np.int64) * views + sources[recovered]
    pairs, pair_counts = np.unique(pair_codes, return_counts=True)
    source_changes = {
        f"{int(pair // views)}->{int(pair % views)}": int(number)
        for pair, number in zip(pairs, pair_counts)
    }
    stats = {
        "selection_mode": "frozen-primary-and-common-view-residual-patches",
        "faces": count,
        "textured_faces": int(textured.sum()),
        "untextured_faces": int((~textured).sum()),
        "total_triangle_area_m2": total_area,
        "original_valid_faces": int(original_valid.sum()),
        "preserved_original_faces": int(original_valid.sum()),
        "restored_original_source_area_m2": area(original_valid),
        "restored_original_source_area_fraction": area(original_valid) / denominator,
        "restored_area_definition": "valid primary area retained; not a measured diff from old spatial-patch labels",
        "initial_untextured_area_m2": before_area,
        "initial_untextured_area_fraction": before_area / denominator,
        "recovered_faces": int(recovered.sum()),
        "recovered_area_m2": recovered_area,
        "recovered_area_fraction": recovered_area / denominator,
        "recovered_fraction_of_initial_untextured_area": recovered_area / max(before_area, 1e-20),
        "untextured_area_m2": untextured_area,
        "untextured_area_fraction": untextured_area / denominator,
        "no_strict_view_faces": int((~observable).sum()),
        "no_strict_view_area_fraction": area(~observable) / denominator,
        "observable_but_unassigned_faces": int(still_observable.sum()),
        "observable_but_unassigned_area_fraction": area(still_observable) / denominator,
        "chosen_sources_valid": bool(chosen_valid.all()),
        "invalid_chosen_source_faces": int((~chosen_valid).sum()),
        "switched_valid_original_faces": int(changed_original.sum()),
        "alternative_source_transitions": source_changes,
        "patches": len(patches),
        "assigned_patches": len(accepted_sizes),
        "anchored_assigned_patches": anchored_patches,
        "unanchored_assigned_patches": len(accepted_sizes) - anchored_patches,
        "deferred_small_patches": deferred_small,
        "deferred_small_area_m2": deferred_small_area,
        "deferred_no_common_view_patches": deferred_conflict,
        "deferred_no_common_view_area_m2": deferred_conflict_area,
        "assigned_patch_faces_min": min(accepted_sizes, default=0),
        "assigned_patch_faces_max": max(accepted_sizes, default=0),
        "assigned_patch_source_counts": patch_source_counts,
        "assigned_boundary_anchor_faces": anchor_boundary_faces,
        "assigned_boundary_matching_source_faces": anchor_agreement_faces,
        "thresholds": {
            "patch_neighbor_radius_spacing": 2.5,
            "patch_neighbor_limit": 24,
            "patch_seed_radius_spacing": 12,
            "normal_degrees": 25,
            "minimum_residual_patch_faces": _MIN_PATCH_FACES,
            "required_common_view_face_fraction": 1.0,
            "anchor_neighbor_limit": _ANCHOR_NEIGHBORS,
            "anchor_radius_spacing": 2.5,
            "front_cosine": .15,
            "depth_absolute_tolerance_spacing": 2,
            "depth_relative_tolerance": .005,
            "visibility_samples": "3 vertices + centroid; all samples must lie in input valid-support mask (whole-shelf depth/confidence support, not instance segmentation)",
            "selection_priority": "common strict visibility; matching frozen boundary area; summed original quality; lower frame index",
        },
        "projection_batch_faces": _FACE_BATCH,
        "quality_storage_dtype": "float32; float64 patch sums; bool visibility unchanged",
        "candidate_arrays_bytes": int(sum(array.nbytes for array in (
            visibility, scores, twice_area, centers, normals,
        ))),
        "candidate_seconds": candidate_seconds,
        "selection_seconds": time.monotonic() - started,
        "full_scene_benefit_proven": False,
    }
    return sources, stats

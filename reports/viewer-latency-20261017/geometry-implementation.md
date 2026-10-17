# Geometry implementation handoff — 2026-10-17

Status: source implementation and focused engineering tests complete: **66 geometry/export tests passed**, see [command/resources](geometry-tests.json). Full-scene timing, RSS and render comparisons remain **PENDING**. No default approximation enabled; no formal experiment, GPU or publication performed.

## APIs and integration

- `utils.pointcloud_filter.prepare_scene_filter(points, config=None) -> PreparedSceneFilter` snapshots the float64 input in original order and the filter configuration, builds the original finite/nonzero base domain, computes original median NN, and stores the **raw** SOR mask. It does not OR product protection, apply the 20% guard, select the cloud, run DBSCAN, or remove ground.
- `PreparedSceneFilter` fields: `points`, `config`, `base_mask`, `median_nn`, `sor_mask`, `pcd`. `sor_mask=None` means disabled/small/degenerate/no SOR due to neighbor count; a computed all-false mask remains an array and must still pass through protection/guard. The retained `pcd` is the original base cloud. Final selections create new clouds.
- `filter_scene_points(points, config=None, protect_mask=None, *, prepared=None)` checks exact configuration and float64 input values/order before reuse. It then applies the original SOR protection OR, per-stage remaining-ratio guard and selection, followed by DBSCAN and ground. Passing no prepared state executes the same preparation synchronously. Returned masks remain writable, including disabled/small/degenerate branches.
- `prepare_web_viewer_geometry(..., filter_config=None, prepare_filter=True)` now prepares full original positive-depth/finite-confidence valid indices and optional raw filter state. It still keeps full native U/V/depth/textures and existing texture-edge behavior.
- New `PreparedViewerGeometry` fields: `filter_config`, `valid_indices`, `scene_filter`. Configuration is snapshotted. Export requires the same filter configuration; `_sample_points` verifies original valid indices/order and the prepared filter verifies original point values. Existing task-local cache/path/tuning checks remain in place; no hashes were added.
- The caller preparing geometry must pass the same `filter_config` as final export, including `enabled`, DBSCAN-off and proxy options. Main integration belongs to the coordinator/matching owner.
- `PointCloudFilterConfig.cluster_proxy_voxel_scale: float = 0.0`. Zero disables proxy mode. A positive finite scale uses `voxel_edge = original_median_nn * scale`. DBSCAN-off remains `keep_main_clusters=False`.

## Proxy semantics

Only DBSCAN is approximated, on the current **post-SOR/protection/guard** domain. The first original point in each voxel is its representative; representatives retain first-occurrence order. `eps` still uses original median NN and `cluster_eps_scale`. `cluster_min_points` counts **proxy neighbors**, not original members, and is deliberately not adjusted or weighted.

Proxy labels map back to every original member before cluster sizes are counted. Thus the default 1% cluster-size threshold, largest-cluster rule, protection OR and 20% guard all operate on original current-domain members. Ground removal still operates on selected original points. Proxy DBSCAN is not equivalent to dense DBSCAN. Unrepresentable proxy voxel coordinates raise an error rather than choosing a different algorithm.

## Early-background handoff

The `early_background_impl` worker has completed the opt-in early candidate implementation; see [early-background-implementation.md](early-background-implementation.md) for the current API, domain and support semantics, underfill reporting and pending validation. The `prepare_filter=False` deferral seam remains available; early tuning automatically defers full-domain NN/SOR and filters the reduced candidate domain without reusing full-domain raw state. Native grids/depth remain available. This report does not claim runtime or quality acceptance for that separate implementation.

## Validation and limitations

AST parsing passed for the four edited Python files with:

```sh
UV_CACHE_DIR=/tmp/uv-viewer-latency-geometry uv run --no-project --python /usr/bin/python3.11 python -c 'import ast; from pathlib import Path; paths = ["utils/pointcloud_filter.py", "src/web_viewer_export.py", "test/test_pointcloud_filter.py", "test/test_web_viewer_export.py"]; [ast.parse(Path(p).read_text(), filename=p) for p in paths]; print("AST parse passed: 4 files")'
```

The initial identical command without `UV_CACHE_DIR` failed before Python because `/home/xingyu/.cache/uv` was read-only in the sandbox. The retry uses only `/tmp`; no dependencies were installed.

Added focused tests: raw SOR against direct Open3D on original float64/base order; instrumented no-recompute reuse; protection before the 20% guard including all-rejected raw SOR; disabled/small/zero-median/insufficient-neighbor cases; exact input/order/config rejection; weighted proxy cluster-size retention and unchanged proxy-neighbor `min_points`; invalid proxy scales; prepared Viewer configuration/domain rejection; explicit preparation deferral. Existing serial/prepared Surfel comparison now passes its nondefault filter configuration into preparation.

Independent review follow-up restored validation order: after the existing float64 conversion, an invalid `protect_mask` shape raises before preparing NN/SOR or importing Open3D. A focused test replaces preparation with a failing sentinel to verify this rejection path. Source syntax and the subsequent focused test run passed.

The focused run passed 66 tests in 11.08 s without failures or reruns; outer wall was 15.696 s, child CPU 18.249 s and maximum RSS about 0.80 GiB. See [raw output](geometry-tests.log) and [receipt](geometry-tests.json). These fixture checks do not establish full-scene output equality, rendered quality or speedup. The point snapshot plus retained Open3D cloud increases preparation residency; full-scene RSS and copy/validation overhead belong in the pending latency comparison. Ground/orientation randomness must be fixed for final-mask experiments.

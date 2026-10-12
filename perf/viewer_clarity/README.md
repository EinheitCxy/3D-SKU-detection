# Viewer clarity evaluation tools

These scripts implement the locked WP-T protocol in `docs/superpowers/plans/2026-10-12-viewer-clarity.md`. They evaluate only the selected `video3` and `fd6` inputs, use CPU export, and render through the preinstalled Playwright/Chromium setup. No model inference or GPU selection is performed here.

## Bundle tools

`pack_bundle(generation_dir, out_zip)` writes the flat `ZIP_STORED` Viewer archive used by the mapping service. It supports Surfel schema 2 and 3; schema 3 includes `surfel-scale.f16.bin`.

`crop_bundle(generation_dir, out_dir, center_fraction=0.4)` uses the product points' 1–99% bounds after `manifest.world_to_view`, takes the centered fraction in view X/Y, and retains all view depth. It applies the same point mask to point-aligned binary arrays, remaps object ranges, removes empty GIDs, prunes unused thumbnails, and writes `crop.json` with removed GIDs and counts. The output directory must not already exist.

Examples, from the repository root:

```bash
UV_CACHE_DIR=/tmp/3d-clarity-uv-cache uv run --no-sync python -m perf.viewer_clarity.export_arm --scene video3 --arm X0
UV_CACHE_DIR=/tmp/3d-clarity-uv-cache uv run --no-sync python -m perf.viewer_clarity.crop --bundle-dir runtime/viewer-clarity-20261012/video3/X0/viewer/runs/<run-id> --out-dir runtime/viewer-clarity-20261012/video3/X0/crop
UV_CACHE_DIR=/tmp/3d-clarity-uv-cache uv run --no-sync python -m perf.viewer_clarity.metrics --bundle-dir runtime/viewer-clarity-20261012/video3/X0/viewer/runs/<run-id> --out runtime/viewer-clarity-20261012/video3/X0/metrics.json
```

Export arms XQ, XS4, XS8 and XQS require the `ViewerExportTuning` integration from WP-X3. XQS requires `--owner-patch-cells 4` or `8`, selected from the XS results. It does not infer a winning setting. Export time is measured around export only; packing is recorded separately in `zip_bytes`. For X0, the existing locked export is reused and its original `seconds` and `git_sha` receipt are preserved.

Metrics implement M1, M2, M3, M5, M6, M7 and M8. M4 is undefined by the registered protocol and is intentionally absent. M2 checks up to eight nearest other points after removing the queried point's own index, including duplicate-position points. M3 reports the fraction of occupied 8 mm world-space voxels containing multiple source frames. M5 tests whether a background Surfel has at least one scale axis at 0.5. M7 uses the frame's recorded world-to-camera matrix. Pass the X0 metrics JSON as `--baseline` to calculate M8 over every GID that had points in X0.

## Source-camera refview

Start Vite with the selected arm's own dependencies and source tree. V0 does not import modules added by later viewer arms:

```bash
VIEWER_SRC="$PWD/runtime/viewer-clarity-20261012/arms/V0/viewer/src" \
  runtime/viewer-clarity-20261012/arms/V0/viewer/node_modules/.bin/vite \
  --config perf/viewer_clarity/harness/vite.config.ts --host 127.0.0.1
```

Then render a frame (frames are cache indices; registered frames are `video3: 4,14,24` and `fd6: 1,5,9`):

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/viewer_clarity/harness/render.mjs --base-url http://127.0.0.1:5173 \
  --arm V0 --scene video3 --frame 4 --generation-dir runtime/viewer-clarity-20261012/video3/X0/crop \
  --zip runtime/viewer-clarity-20261012/video3/X0/crop/viewer_bundle.zip --radius-size 0.07
```

For a Surfel arm, use `--radius-size 0.004` for 1.05 px or `0.004952380952380953` for 1.3 px. For GLB source-camera refview, add `--mode glb --glb <scene.glb>`. GLB overview and close views are not implemented. The camera uses the stored source intrinsic/extrinsic matrices and the GLB path additionally applies the inverse row-major `world_to_view`. Each browser page is allowed up to 900 seconds. Chromium is launched with SwiftShader flags and the executable is supplied by `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`.

Score renders with:

```bash
UV_CACHE_DIR=/tmp/3d-clarity-uv-cache uv run --no-sync python -m perf.viewer_clarity.score \
  --generation-dir runtime/viewer-clarity-20261012/video3/X0/crop --renders-dir runtime/viewer-clarity-20261012/video3/X0/refview/V0 \
  --frames 4 --images-dir runtime/video3-resolution-review/dataset/images \
  --out runtime/viewer-clarity-20261012/video3/X0/refview/V0/score.json --mode surfel
```

Reference pixels are bilinearly sampled from the image through `processed_to_source`. `refwhite` means OpenCV RGB-to-gray brightness ≥245. Coverage excludes white reference pixels; holes are exact white `(255,255,255)` for Surfels and magenta `(255,0,255)` for GLB. PSNR uses 8-bit sRGB values over valid, non-white, non-hole pixels; SSIM uses the same mask over the full-image SSIM map. This is a source-camera, in-sample comparison, not held-out view or ground-truth geometry evaluation.

## Production Viewer capture

Build the chosen arm with its own package scripts, start its `vite preview`, then run the interaction capture. The script rewrites only the task ZIP fetch to the local ZIP, waits for the Viewer to draw, records `overview`, `close`, `side`, and `selected` page screenshots, and computes product-height fraction from WebGL canvas pixels. One complete V0 smoke passed with HTTP 200, no browser errors, and product-height fraction 0.1222:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/viewer_clarity/capture.mjs --base-url http://127.0.0.1:4173 \
  --task-id viewer-clarity-smoke --zip runtime/viewer-clarity-20261012/video3/X0/viewer_bundle.zip \
  --out-dir runtime/viewer-clarity-20261012/video3/X0/capture
```

`sheet.py --inputs runtime/viewer-clarity-20261012 --out-dir runtime/viewer-clarity-20261012/sheets` combines matching captures by scene/view and labels each arm with its refview PSNR and coverage when available. Screenshots are visual evidence; this tool does not make A/B adoption decisions.

## Limits

The render and score depend on source-camera pose/depth estimates from the reconstruction. White packaging pixels are excluded, only two scenes are registered, and side-view quality is judged from captured images. SwiftShader output is not GPU performance evidence. Export and ZIP timing are recorded separately.

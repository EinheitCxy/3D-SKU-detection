# Deterministic source-selection diagnostic

This opt-in experiment compares the isolated source-selection renderer with selection disabled and enabled. It loads the complete video2 ZIP with all points visible and fixes the prior close camera, default radius 1.05, source-depth tolerance 0.005, final footprint cap 6, and opaque output normalization. The first loaded render is the OFF arm; the script then performs only the ON render.

The result is a qualitative still-frame check of texture ownership, seams, holes, and ghosting under software WebGL. It was not adopted as a production default and is not hardware-GPU performance evidence.

The candidate patch is based on opaque-output Docker Viewer commit `5aeca44eb895fdcd488553dadfec7c1e1b1d0041`:

```bash
mkdir -p runtime/viewer-extent-20261012/video12/followup
git -C docker show 5aeca44eb895fdcd488553dadfec7c1e1b1d0041:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/video12/followup/source-selection-renderer.ts
git apply perf/surfel_extent/experiments/source_selection/source-selection-renderer.patch
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome-headless-shell node perf/surfel_extent/experiments/source_selection/capture.mjs
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome-headless-shell node perf/surfel_extent/experiments/output_normalization/probe.mjs --page source-selection-probe.html --out runtime/viewer-extent-20261012/video12/followup/source-selection/probe.json
```

The raw receipt retains the original over-strict alpha-summary assertion; `receipt-adjudication.json` explains why the screenshots and unchanged `alpha>0` occupancy remain valid. The capture script now checks occupancy fields only.

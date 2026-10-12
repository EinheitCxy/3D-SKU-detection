# Source-depth tolerance diagnostic

This opt-in experiment uses the complete video2 ZIP and prior frozen close camera to compare source-depth relative tolerance `0.005` with historical `0.015`. It keeps all points, source selection OFF, opaque output normalization, default radius 1.05, and final footprint cap 6. The experiment changes an isolated runtime renderer uniform and does not alter the production constant.

The alpha-occupancy readback tests whether the stricter tolerance removed rendered coverage. The screenshots remain qualitative software-WebGL evidence; added coverage can also cross depth boundaries, so a larger occupied area alone is not sufficient to promote `0.015`.

Prepare the runtime renderer using [the source-selection instructions](../source_selection/README.md), keep source selection disabled, then run:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome-headless-shell node perf/surfel_extent/experiments/source_depth_tolerance/capture.mjs
```

Runtime evidence is in `runtime/viewer-extent-20261012/video12/followup/source-depth-tolerance/`.

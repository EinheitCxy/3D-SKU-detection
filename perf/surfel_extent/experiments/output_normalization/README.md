# Docker output-normalization diagnostic

This opt-in experiment freezes the Docker candidate renderer and changes only its final full-screen output. `old` mixes normalized accumulated color with white using `smoothstep(0.03, 0.3, alpha)`. `opaque` displays normalized accumulated color wherever alpha is positive and white where alpha is zero, matching the proposed one-line Docker output change.

The full video2 ZIP, all point visibility, default radius 1.05, and the prior candidate-close camera receipt are fixed. Both arms read the same accumulation alpha and must report identical coverage. This experiment does not modify the production renderer and does not make the diagnostic URL mode a default.

The old-output snapshot is Docker Viewer commit `1b917cddab56817d58da360b58ddd74e7ad39c9f`. Recreate and run it from the repository root:

```bash
mkdir -p runtime/viewer-extent-20261012/video12/followup
git -C docker show 1b917cddab56817d58da360b58ddd74e7ad39c9f:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/video12/followup/diagnostic-renderer.ts
git apply perf/surfel_extent/experiments/output_normalization/diagnostic-renderer.patch
docker/viewer/node_modules/.bin/vite --config perf/surfel_extent/harness/vite.config.ts --host 127.0.0.1
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/path/to/chrome-headless-shell node perf/surfel_extent/experiments/output_normalization/capture.mjs
```

`capture-production.mjs --scene video1` and `--scene video2` capture production commit `5aeca44eb895fdcd488553dadfec7c1e1b1d0041`. Runtime evidence is in `runtime/viewer-extent-20261012/video12/followup/output-normalization/`.

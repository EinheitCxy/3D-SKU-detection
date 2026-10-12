# Surfel full-scene WebGL comparison

This harness compares the unchanged Docker Viewer renderer snapshot with the working-tree candidate by dynamically importing each real `createSurfelRenderer`. The default inputs are the complete video1 and video2 scenes; no spatial crop or point subsampling is performed by the harness. It renders only on load, camera/control changes, and A/B switches.

Run from the repository root with the English `docker` checkout, its Viewer dependencies, and `perf` Playwright dependencies installed. These ZIPs must already exist; this tool does not run reconstruction:

- `runtime/viewer-extent-20261012/video12/video1/viewer_bundle.zip`
- `runtime/viewer-extent-20261012/video12/video2/viewer_bundle.zip`

Input sources, point counts, and export settings are recorded in [the evaluation report](../../docs/surfel_extent_evaluation.md#video1--video2-完整场景复查). The ZIPs and screenshots remain local runtime assets, not Git-tracked files.

The renderer snapshots are local runtime inputs. Rebuild them with these exact refs (do not infer them from the current working-tree `HEAD`):

```bash
mkdir -p runtime/viewer-extent-20261012/visual
git -C docker show 8e53633:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/baseline-renderer.ts
git show 69420d3:modules/viewer_web/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/root-baseline-renderer.ts
git -C docker show 50df94d:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/uniform-renderer.ts
git -C docker show 1b917cd:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/major-renderer.ts
```

Start the local Viewer on its reserved port:

```bash
docker/viewer/node_modules/.bin/vite --config perf/surfel_extent/harness/vite.config.ts --host 127.0.0.1
```

Prepared LAN entries, currently disabled pending authorization (the scene selector also switches between them):

- `http://192.168.2.6:53110/?scene=video1` (default, full)
- `http://192.168.2.6:53110/?scene=video2` (full)
- `http://192.168.2.6:53110/?scene=video3` (legacy crop fixture)
- `http://192.168.2.6:53110/?scene=fd6` (legacy crop fixture)

The verified service for this run is local-only at `http://127.0.0.1:53110/`. Binding the new video1/video2 scene data to `0.0.0.0:53110` for the listed LAN URLs requires explicit Rick authorization; automatic approval review rejected that LAN exposure, so no tunnel, alternate port, or forwarding was used.

The initial camera uses source frame 0 orientation (`extrinsic` inverse followed by `world_to_view`) and fits the robust product bounds from `productViewBox`. `Full overview`, `Close`, and the exact `Source camera` pose remain selectable, as does every source frame. Before and After therefore use the same point set and camera. `candidate` now means the production Axis-cap renderer; retained older files named `candidate-*.png` came from the Uniform-cap stage and must not be read as current candidate evidence. Renderer controllers are initialized lazily and retained until page disposal because they share the loaded Surfel textures.

Reproduce the six fixed-view captures on the local service:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/surfel_extent/capture.mjs --base-url http://127.0.0.1:53110 \
  --out-dir runtime/viewer-extent-20261012/video12/visual --scenes video1,video2 \
  --arm candidate --states overview
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/surfel_extent/capture.mjs --base-url http://127.0.0.1:53110 \
  --out-dir runtime/viewer-extent-20261012/video12/visual --scenes video2 \
  --states default-close,radius2-close
```

The independent synthetic silhouette/readback tool remains `probe.mjs --base-url http://127.0.0.1:53110`; it was not rerun for this preview-only change.

The capture has a fixed 960×600 viewport and 960×543 canvas. By default it writes twelve images: Before/After at default-radius overview, default-radius close view, and radius-2 close view for video1 and video2. `capture-before-after.json` records the exact dimensions, source frame, camera, FOV, and browser errors. Use `--scenes video1`, `--arm candidate`, or `--states overview` for a smaller diagnostic run. The old `--arm major` path was retired; current production Axis cap is `--arm candidate`. The legacy crop entries remain interactive and can be explicitly captured with `--scenes video3,fd6`.

The frozen evidence actually captured in this run is six images, rather than the twelve-image CLI default: After overview for video1 and video2, plus video2 Before/After at default close and radius-2 close. `capture-candidate.json` records the two overviews; `capture-before-after.json` records the four close comparisons. Video1 was not image-compared Before/After because its v2 bundle has no adaptive scale and the default footprint cap is not expected to activate. The excluded first video1 attempt is retained as `candidate-overview-invalid-world-axis-fit.png` and is not comparison evidence.

Automated evidence uses Chrome headless shell with SwiftShader software WebGL. These screenshots and probes validate this final-footprint cap in the tested inputs; they are neither an old-D1 result nor hardware-GPU performance or real-world accuracy evidence.

## Follow-up: whitening and texture ghosting

The next experiments use the same full video1/video2 bundles and retain the final diameter cap. See [the follow-up report](../../docs/surfel_clarity_followup.md) for the output-normalization correction, source-selection candidate, and the ownership8 background-resampling confound. The current English renderer includes opaque color normalization; older captures above predate that correction.

`experiments/output_normalization/probe.mjs` loads `normalization-probe.html` and tests the actual English renderer with a single opaque black Surfel. It checks low-weight interior color, empty/hidden pixels, and picking. Use `--base-url http://127.0.0.1:53110` and the same `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` as above. The probe reports six checks and WebGL/browser errors; it does not infer hole coverage from final RGB color.

The local preview was stopped after the follow-up captures. Prepare both experiment renderer patches before restarting Vite; their READMEs document the Git revisions and commands. No LAN service was opened.

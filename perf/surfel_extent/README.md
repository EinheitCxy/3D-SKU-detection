# Surfel final-extent WebGL verification

This harness compares the unchanged Docker Viewer renderer snapshot with the working-tree candidate by dynamically importing each real `createSurfelRenderer`. It renders only on load, camera/control changes, and A/B switches.

Run from the repository root with the English `docker` checkout, its Viewer dependencies, and `perf` Playwright dependencies installed. The two crop ZIPs below must already exist; this tool does not run reconstruction. The four renderer snapshots are local runtime inputs. Rebuild them with these exact refs (do not infer them from the current working-tree `HEAD`):

```bash
mkdir -p runtime/viewer-extent-20261012/visual
git -C docker show 8e53633:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/baseline-renderer.ts
git show 69420d3:modules/viewer_web/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/root-baseline-renderer.ts
git -C docker show 50df94d:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/uniform-renderer.ts
git -C docker show 1b917cd:viewer/src/surfel-renderer.ts > runtime/viewer-extent-20261012/visual/major-renderer.ts
```

Start the LAN Viewer on its reserved port:

```bash
docker/viewer/node_modules/.bin/vite --config perf/surfel_extent/harness/vite.config.ts
```

Direct entries (the scene selector also switches between them):

- `http://192.168.2.6:53110/?scene=video3`
- `http://192.168.2.6:53110/?scene=fd6`

Both inputs are retained crop fixtures, not the full scenes. The interactive default is the production `Axis cap`; `Before` and `Uniform` remain available for comparison. Existing screenshot names containing `candidate-*.png` are the earlier uniform-cap arm. New captures name that frozen arm `uniform-*.png` so subsequent production changes cannot silently change its meaning.

Run the real WebGL silhouette/readback probe and fixed-view capture:

```bash
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/surfel_extent/probe.mjs --base-url http://192.168.2.6:53110
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/surfel_extent/capture.mjs --base-url http://192.168.2.6:53110 \
  --out-dir runtime/viewer-extent-20261012/visual
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/tmp/surfel-glb-browser/extracted/chrome-headless-shell-linux64/chrome-headless-shell \
  node perf/surfel_extent/capture.mjs --base-url http://192.168.2.6:53110 \
  --out-dir runtime/viewer-extent-20261012/visual --arm major
```

The Viewer serves only the locally retained `video3/X0/crop` and `fd6/X0/crop` ZIPs. `capture-uniform.json` records the original Before/Uniform run; `capture-major.json` records the corrected 960×543 Axis-cap run and exact camera state. `capture-major-invalid-layout.json` records and excludes an earlier toolbar-wrap attempt. `probe.json` records actual Docker and root production renderers, their own pre-cap baselines, the frozen Axis snapshot, raster silhouettes, and pick results.

Automated evidence uses Chrome headless shell with SwiftShader software WebGL. These screenshots and probes validate this final-footprint cap in the tested inputs; they are neither an old-D1 result nor hardware-GPU performance or real-world accuracy evidence.

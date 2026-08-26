# Self-contained Global-ID Mapping Docker Design

## Objective

Create one synchronous CUDA mapping service under `docker/`. It reuses the
current DA3/SAM3/global-ID/Viewer implementation, accepts external detector and
classifier results, returns the historical `global_skus` field plus a Viewer
bundle ZIP, and runs without network access or model volumes.

The work lives on branch `docker`. Historical `Global-ID-Mapping` code is not
restored or copied; only its BSON `POST /api` envelope and container port are
retained.

## Scope

Included:

- root mapping core (`main.py`, `src/`, `utils/`, configuration);
- DA3 and SAM3 source;
- baked DA3 and SAM3 weights;
- BSON/FastAPI service wrapper;
- minimal Viewer bundle export;
- one unified Python environment for host and one equivalent unified
  environment inside Docker.

Excluded:

- SKU detector implementation, environment and weights;
- personalcare classifier implementation, environment and weights;
- Viewer static web server;
- vendored Pi3/VGGT source trees and model weights; the existing lightweight
  wrapper modules may remain importable but the container request runner fixes
  both reconstruction and matching to DA3;
- cross-request caches and persistent runtime output.

## API Contract

`POST /api`, request and response encoded as BSON, container port `80`.

Request:

```jsonc
{
  "images": [<bytes>, ...],
  "skus": ["<sku-classifier frame JSON>", ...],
  "project_id": 51
}
```

For every frame, `skus[i]` contains classifier-produced `classes.cls`, object
`classes.cls` indices and `confidences.cls`, while retaining detector bbox,
class, confidence and object ordering. The adapter resolves
`classes.cls[object.classes.cls]` as `sku_id^sku_name` and writes the current
canonical object-level `classification`. `features` are not accepted or saved.

Success response:

```jsonc
{
  "global_skus": ["<per-frame JSON string>", ...],
  "viewer_bundle": <ZIP_STORED bytes>
}
```

The ZIP root contains only:

```text
CURRENT
runs/<viewer-run-id>/
  manifest.json
  positions.f32.bin
  colors.u8.bin
  normals.i8.bin
  objects.json
  thumbs/*.jpg
```

## Request Execution

The service is synchronous and uses one Uvicorn worker plus one process-wide
execution lock. Each request represents one floor-display dataset and receives
an isolated temporary work root.

```text
BSON decode and validation
  -> write numeric images and enriched detections
  -> child request runner
     -> DA3 reconstruction
     -> complete batch-all-refs SAM3 matching
     -> analysis and dedup
     -> Viewer export
  -> read global_skus.json
  -> ZIP Viewer CURRENT + run
  -> BSON encode
  -> remove request work root
```

The child process releases CUDA and module-level caches between requests. A
failed stage returns a BSON error containing `stage` and `message`; it never
returns an older request's artifacts.

## Classifier Switch

`main.py` gains a BooleanOptionalAction:

```text
--classifier      default; run the existing local classifier
--no-classifier   consume already enriched detections
```

External mode validates canonical classification records and passes the same
detection directory explicitly to dedup. Docker always uses
`--no-classifier`.

## Unified Environment

The root project becomes the single host environment. Target versions:

```text
Python 3.11
NumPy 1.26.4
Torch 2.7.1+cu126
TorchVision 0.22.1
xFormers 0.0.31
OpenCV 4.11
Open3D 0.19
```

Root dependency metadata also declares DA3 runtime packages, including
OmegaConf, e3nn, evo, moviepy, plyfile, pillow-heif, pycolmap and the missing
runtime dependency `addict`. DA3 remains a child process, but
`DA3_VENV_PYTHON` points to the root `.venv/bin/python`.

Migration is candidate-first: build a candidate, run dependency/tests/GPU
checks, atomically replace the root `.venv`, then remove the regenerable
`Depth-Anything-3/.venv` to recover disk. The old environment is not removed
before the candidate passes.

Docker builds the same locked dependency set inside its Ubuntu/Python 3.11
base; it does not copy the RHEL host venv.

## Offline Self-contained Image

The local-only base image is:

```text
harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4
```

Its small template `/app` content is removed. It contributes Python 3.11,
CUDA runtime and BSON/FastAPI libraries, not detector/classifier business code
or weights.

Build uses `--pull=false` and `--network=none`. Named local contexts provide:

- a container-built unified venv generated from the root lock and local uv
  cache;
- the complete DA3 Hugging Face model cache at snapshot
  `b2359bdf726fb44ef62acca04d629dcf158053e7`;
- `sam3/checkpoints/sam3.pt`.

The final image contains the unified venv once, both model weights and only the
mapping source required at runtime. Default tag is
`global-id-mapping:da3-self-contained`, overridable by `IMAGE_TAG`.

## Repository Layout

```text
docker/
  Dockerfile
  Dockerfile.dockerignore
  api.py
  processor.py
  request_runner.py
  build.sh
  test_api.py
  README.md
```

Only these deployment files, required core changes, focused tests and README
updates are staged. Existing research files and unrelated deletions remain
outside the branch commits.

## Acceptance

- existing owned Python suite and Viewer tests/build pass;
- unified environment passes `uv pip check`;
- classifier on/off paths have focused tests;
- external classifier adapter preserves frame/object order and rejects malformed
  input;
- API test confirms legacy `global_skus` string array and a readable Viewer ZIP;
- offline Docker build completes with no package/model download;
- container import smoke passes for core, DA3 and SAM3;
- one GPU dataset request completes through BSON response and Viewer ZIP;
- no detector/classifier business files or weights exist in the final image;
- no vendored `Pi3/` or `vggt-main/` tree exists in the final image.

# Self-contained Global-ID Mapping Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and GPU-validate one offline, self-contained BSON global-ID mapping Docker service that consumes external classifier results and returns legacy `global_skus` plus a zipped Viewer bundle.

**Architecture:** A single-worker FastAPI wrapper validates one floor-display request and launches a child request runner so CUDA/module caches die after every request. The runner uses the current DA3/SAM3 pipeline with local classification disabled, then exports the minimal Viewer bundle. Host and container each use one environment built from the same unified root lock.

**Tech Stack:** Python 3.11, uv, FastAPI, BSON, NumPy, PyTorch 2.7.1+cu126, xFormers 0.0.31, DA3, SAM3, Docker BuildKit, ZIP_STORED.

**Spec:** `docs/superpowers/specs/2026-08-26-global-id-mapping-docker-design.md`

## Global Constraints

- Work only on branch `docker` and preserve all unrelated dirty files.
- Use explicit staging allowlists; never use `git add -A`.
- Do not restore or copy historical `Global-ID-Mapping` business code.
- Do not include detector/classifier code, environments or weights in the image.
- Success BSON has exactly `global_skus` and `viewer_bundle`.
- Requests are synchronous and serialized; Uvicorn uses one worker.
- No cross-request output/model cache is allowed.
- Docker build uses `--pull=false`, `--network=none` and local contexts only.
- Host and Docker target Python 3.11, NumPy 1.26.4, Torch 2.7.1,
  TorchVision 0.22.1 and xFormers 0.0.31.
- Preserve DA3 as a child process while pointing `DA3_VENV_PYTHON` to the
  unified environment.
- Build and smoke tests use DA3 snapshot
  `b2359bdf726fb44ef62acca04d629dcf158053e7` and the local SAM3 checkpoint.

---

### Task 1: Add the external-classification pipeline switch

**Owner:** Terra worker; no other worker edits `main.py` or
`tests/test_main_pipeline.py` concurrently.

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `SKUDetectionMain.classifier_enabled: bool`
- Produces: CLI flags `--classifier` and `--no-classifier`
- Produces: `validate_external_classification_directory(dataset: Path) -> Path`
- External mode uses `<dataset>/detections_results` as the explicit dedup input.

- [ ] **Step 1: Write failing external-mode tests**

Add focused tests proving that `--no-classifier` never calls
`run_personalcare_classification`, validates every object classification, and
passes the input detection directory to dedup:

```python
def test_pipeline_external_classifier_uses_enriched_input(monkeypatch, tmp_path):
    app, dataset, calls = prepared_pipeline(tmp_path)
    app.classifier_enabled = False
    write_enriched_detections(dataset)
    monkeypatch.setattr(
        app,
        "run_personalcare_classification",
        lambda *_: pytest.fail("local classifier must not run"),
    )
    monkeypatch.setattr(
        app,
        "run_dedup_sequence",
        lambda *_args, **kwargs: calls.append(kwargs["detection_dir"])
        or {"success": True},
    )

    summary = app.run_complete_pipeline(str(dataset), algorithm="3d")

    assert summary["classification"] is True
    assert calls == [str(dataset / "detections_results")]
```

Also add CLI coverage asserting default classifier enabled and
`--no-classifier` disabled.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest \
  tests/test_main_pipeline.py::test_pipeline_external_classifier_uses_enriched_input \
  tests/test_main_pipeline.py::test_pipeline_cli_can_disable_classifier -q
```

Expected: failures because the switch and external validation do not exist.

- [ ] **Step 3: Implement the minimal switch**

Use `argparse.BooleanOptionalAction`:

```python
parser.add_argument(
    "--classifier",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run local classification; --no-classifier consumes enriched detections",
)
```

Set `app.classifier_enabled = args.classifier`. In
`run_complete_pipeline()`, create the executor only when enabled. External mode
must synchronously validate numeric detection files and every object via the
existing `validate_classification()`, then use:

```python
{"success": True, "detection_dir": str(dataset / "detections_results")}
```

Do not create publication pointers or compatibility copies in external mode.

- [ ] **Step 4: Run pipeline tests GREEN**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_main_pipeline.py \
  tests/test_classification_aggregation.py -q
```

- [ ] **Step 5: Update usage and commit**

```bash
git add main.py tests/test_main_pipeline.py README.md
git commit -m "feat: consume external classifier detections"
```

---

### Task 2: Merge core and DA3 dependencies into one host environment

**Owner:** Terra worker owns dependency metadata and the candidate builder. The
coordinator executes the environment swap after review.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/da3_3d_reconstructor.py`
- Modify: `tests/test_da3_3d_reconstructor.py`
- Modify: `tests/test_da3_import_isolation.py`
- Create: `scripts/3d/ops/build_unified_env.sh`
- Modify: `README.md`
- Modify: `docs/3d_core.md`

**Interfaces:**
- Produces: one root lock containing core, SAM3, DA3 and BSON API deps.
- Produces: default DA3 interpreter `<repo>/.venv/bin/python`.
- Produces: `build_unified_env.sh OUTPUT_DIR`.

- [ ] **Step 1: Write failing interpreter/dependency tests**

```python
def test_da3_default_interpreter_is_root_environment():
    expected = PROJECT_ROOT / ".venv" / "bin" / "python"
    assert DA33DReconstructor.DEFAULT_DA3_VENV_PYTHON == expected
```

Add a dependency-contract test requiring exact NumPy/Torch/TorchVision/xFormers
pins and the named DA3 runtime packages.

- [ ] **Step 2: Run tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_da3_3d_reconstructor.py \
  tests/test_da3_import_isolation.py -q
```

- [ ] **Step 3: Lock the unified environment**

Set:

```toml
requires-python = ">=3.11,<3.12"
"numpy==1.26.4"
"torch==2.7.1"
"torchvision==0.22.1"
"xformers==0.0.31"
```

Declare DA3 runtime packages, explicitly including `addict`. Declare
`fastapi`, `uvicorn` and `bson`. Regenerate with `uv lock`; never edit
lock entries manually.

- [ ] **Step 4: Implement candidate builder**

The script rejects an existing output, creates the candidate using the frozen
root lock, then runs:

```bash
uv pip check --python "$OUTPUT_DIR/bin/python"
"$OUTPUT_DIR/bin/python" -c \
  'import torch, xformers, depth_anything_3, omegaconf, e3nn, evo, sam3'
```

It must not modify either current venv.

- [ ] **Step 5: Build and validate candidate**

```bash
VIRTUAL_ENV=<candidate> UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest tests perf/tests -q
```

- [ ] **Step 6: Perform candidate-first host swap**

Stop active project Python processes, rename the current root `.venv` to a
bounded backup, rename the candidate to `.venv`, and immediately re-run import
and focused tests. Do not delete either old environment yet.

- [ ] **Step 7: Run GPU equivalence smoke**

Run fresh DA3 reconstruction plus complete matching in an isolated output root.
Verify DA3 cache schema/shape/model source, complete SAM3 v2 entries, assignment
counts, xFormers operators and absence of CUDA OOM.

- [ ] **Step 8: Remove superseded environments after acceptance**

After coordinator verification, delete only the regenerable root backup and
`Depth-Anything-3/.venv`, report disk recovered, and retain root `.venv`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/da3_3d_reconstructor.py \
  tests/test_da3_3d_reconstructor.py tests/test_da3_import_isolation.py \
  scripts/3d/ops/build_unified_env.sh README.md docs/3d_core.md
git commit -m "build: unify core and DA3 environment"
```

---

### Task 3: Implement the external classifier BSON adapter

**Owner:** Luna worker owns only `docker/processor.py` and its focused tests.

**Files:**
- Create: `docker/__init__.py`
- Create: `docker/processor.py`
- Create: `tests/test_docker_mapping_processor.py`

**Interfaces:**
- Produces: `PreparedRequest(dataset_dir: Path, project_id: int)`
- Produces: `prepare_request(inputs, work_root) -> PreparedRequest`
- Produces: `build_success_response(global_skus_path, viewer_root) -> dict`
- Produces: `pack_viewer_bundle(viewer_root) -> bytes`

- [ ] **Step 1: Write failing adapter tests**

```python
prepared = prepare_request(
    {"images": [image0, image1], "skus": [sku0, sku1], "project_id": 51},
    tmp_path,
)
frame = json.loads(
    (prepared.dataset_dir / "detections_results" / "0.json").read_text()
)
obj = frame["skus"][0]["objects"][0]
assert obj["classes"]["det"] == 0
assert obj["classification"]["sku_id"] == "430085"
assert obj["classification"]["confidence"] == 0.87
```

Add rejection cases for count mismatch, invalid images, missing/out-of-range cls
index, malformed label and non-finite confidence.

- [ ] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_docker_mapping_processor.py -q
```

- [ ] **Step 3: Implement strict request preparation**

Write numeric images and canonical detections. Adapt top-level
`{classes,objects}` and `{skus:[...]}` inputs into canonical
`{skus:[...]}`. Use the current classification schema helper; do not copy
legacy model code or accept `features`.

- [ ] **Step 4: Implement Viewer ZIP packaging**

Read `CURRENT`; include only it and the selected run tree with
`zipfile.ZIP_STORED`. Reject missing fixed files. Return exactly:

```python
{
    "global_skus": json.loads(global_skus_path.read_text()),
    "viewer_bundle": zip_bytes,
}
```

- [ ] **Step 5: Run GREEN and commit**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_docker_mapping_processor.py -q
git add docker/__init__.py docker/processor.py tests/test_docker_mapping_processor.py
git commit -m "feat: adapt external classifier BSON inputs"
```

---

### Task 4: Implement isolated request runner and FastAPI shell

**Owner:** Terra worker owns runner/API files and tests; it consumes Tasks 1 and
3 interfaces without renaming them.

**Files:**
- Create: `docker/request_runner.py`
- Create: `docker/api.py`
- Create: `tests/test_docker_mapping_api.py`
- Modify: `docker/processor.py`

**Interfaces:**
- Consumes: `prepare_request()`, `build_success_response()`
- Produces: `run_mapping_request(dataset_dir, output_root, viewer_root, model_path) -> dict`
- Produces: FastAPI `app` with synchronous `POST /api`

- [ ] **Step 1: Write failing runner/API tests**

Inject fake pipeline/exporter dependencies. Require DA3 backend,
`classifier_enabled=False`, complete summary and explicit Viewer output. Send
BSON through TestClient and assert exact response keys. Use two threads with a
blocking fake and assert maximum active processing equals one.

- [ ] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_docker_mapping_api.py -q
```

- [ ] **Step 3: Implement request runner**

Construct `SKUDetectionMain`, set isolated save root, DA3 backend,
`classifier_enabled=False`, root config and local model path. Require success
for validation, reconstruction, matching, analysis, classification and dedup.
Call the current Viewer exporter with explicit paths.

- [ ] **Step 4: Implement synchronous BSON API**

Use a module-level `threading.Lock`. Decode BSON, create a
`TemporaryDirectory`, prepare the request, invoke the runner in a child
process, build and BSON-encode the response. Request contract errors return 400;
pipeline failures return 500 with exactly:

```json
{"stage": "<stage>", "message": "<message>"}
```

- [ ] **Step 5: Run GREEN and commit**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_docker_mapping_api.py \
  tests/test_docker_mapping_processor.py -q
git add docker/api.py docker/request_runner.py docker/processor.py \
  tests/test_docker_mapping_api.py
git commit -m "feat: serve isolated BSON mapping requests"
```

---

### Task 5: Add the offline self-contained Docker build

**Owner:** Terra worker owns deployment files only and must not run Docker
cleanup.

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/Dockerfile.dockerignore`
- Create: `docker/build.sh`
- Create: `docker/test_api.py`
- Create: `docker/README.md`
- Modify: `tests/test_shell_layout.py`
- Modify: `README.md`

**Interfaces:**
- Produces image `global-id-mapping:da3-self-contained` unless `IMAGE_TAG` is set.
- Produces command `python -m uvicorn docker.api:app --host 0.0.0.0 --port 80 --workers 1`.

- [ ] **Step 1: Write failing build-contract tests**

Require deployment files, executable `build.sh`, no detector/classifier COPY,
network-disabled build, named model/venv contexts and one Uvicorn worker.

- [ ] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_shell_layout.py -q
```

- [ ] **Step 3: Implement Dockerfile**

Use local base
`harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4`, remove its template,
copy only mapping source, one container-built unified venv, complete DA3 model
cache and SAM3 checkpoint. Set:

```dockerfile
ENV PATH=/app/.venv/bin:$PATH \
    DA3_VENV_PYTHON=/app/.venv/bin/python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DA3_MODEL_PATH=/opt/models/da3/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```

- [ ] **Step 4: Implement offline build script**

Validate local base, uv cache, DA3 cache, SAM3 checkpoint and candidate venv.
Build the Ubuntu-compatible venv in a temporary base container using the frozen
root lock/local uv cache, then use BuildKit named contexts with
`--network=none --pull=false`. Never download, prune, push or delete images.

- [ ] **Step 5: Implement client and README**

The client sends numeric images and sku-classifier output to
`http://127.0.0.1:8011/api`, writes `global_skus.json` and
`viewer_bundle.zip`, and verifies ZIP members. Document:

```bash
bash docker/build.sh
docker run --rm --gpus all -p 8011:80 \
  global-id-mapping:da3-self-contained
uv run python docker/test_api.py --dataset <path> --classifier-result <path>
```

- [ ] **Step 6: Check and commit**

```bash
bash -n docker/build.sh
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --offline pytest tests/test_shell_layout.py \
  tests/test_docker_mapping_api.py tests/test_docker_mapping_processor.py -q
git add docker/Dockerfile docker/Dockerfile.dockerignore docker/build.sh \
  docker/test_api.py docker/README.md tests/test_shell_layout.py README.md
git commit -m "build: package self-contained mapping service"
```

---

### Task 6: Build, inspect and GPU-validate the final image

**Owner:** Coordinator executes mutable host/Docker operations; Terra may
diagnose one bounded failure at a time. No worker deletes or pushes images.

**Files:**
- Modify only if evidence requires: `docker/README.md`, `README.md`,
  `docs/3d_core.md`

- [ ] **Step 1: Recheck resources**

Record root/Docker free space and exact reclaimable targets. If insufficient,
stop and request approval for named deletions; never run broad Docker prune.

- [ ] **Step 2: Build offline image**

```bash
DOCKER_BUILDKIT=1 IMAGE_TAG=global-id-mapping:da3-self-contained \
bash docker/build.sh
```

Require no network fetch and a loaded image.

- [ ] **Step 3: Inspect final image**

Verify one `.venv`, both weights, DA3/SAM3 source and absence of detector,
classifier, Pi3 and VGGT trees. Run `uv pip check` and core/DA3/SAM3 import
smoke without GPU.

- [ ] **Step 4: Run one GPU BSON smoke**

Start with `--gpus all -p 8011:80`, submit one real fd request, and verify HTTP
200 BSON, frame-aligned legacy `global_skus`, object IDs/classification, a
complete Viewer ZIP, cold DA3/SAM3 logs and serialized second-request behavior.

- [ ] **Step 5: Run final repository validation**

```bash
PYTHONPATH=. VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv \
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest tests perf/tests -q
npm --prefix modules/viewer_web test -- --run
npm --prefix modules/viewer_web run build
bash -n docker/build.sh scripts/3d/pipeline/video_to_viewer.sh
git diff --check
```

- [ ] **Step 6: Commit stable validation docs if changed**

```bash
git add docker/README.md README.md docs/3d_core.md
git commit -m "docs: record mapping docker validation"
```

# End-to-end performance harness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible cold and warm DA3-to-Three.js performance evidence for fd2–4 without modifying existing artifacts.

**Architecture:** A standard-library Python harness owns isolated output paths, stage subprocesses and `nvidia-smi` telemetry. A separate browser runner uses Playwright because the existing frontend is native Three.js. The reporter reads only stable JSON contracts and fails closed on a missing stage.

**Tech Stack:** Python 3.11 via `uv`, `nvidia-smi`, existing DA3/SAM3 commands, Node 22, Playwright Chromium, Markdown and JSON.

**Spec:** `perf/DESIGN.md`

## Global constraints

- Use GPU 2 through `CUDA_VISIBLE_DEVICES=2`; no CPU fallback.
- Keep model weights preinstalled; cold means no scene/cache artifacts, not a model download.
- Write generated files only below `perf/runs/`; never remove or mutate `Output/`.
- Run fd2, then fd3, then fd4 serially to avoid CUDA contention.
- A failed stage is an explicit failed result, never a zero-valued result.

### Task 1: Define and test the durable results schema

**Files:**
- Create: `perf/benchmark.py`
- Create: `perf/tests/test_benchmark.py`

- [ ] Write a failing test that a completed stage contains wall seconds, exit code and peak telemetry.
- [ ] Run `uv run --offline pytest perf/tests/test_benchmark.py -q` and observe the missing-module failure.
- [ ] Implement the dataclasses and JSON validation with no pipeline side effects.
- [ ] Re-run the test and confirm it passes.

### Task 2: Capture isolated cold/warm command and GPU evidence

**Files:**
- Modify: `perf/benchmark.py`
- Modify: `perf/tests/test_benchmark.py`

- [ ] Write failing tests for command construction and sampler-derived peak/baseline accounting.
- [ ] Run only the new tests and observe expected failure.
- [ ] Implement serial stage execution, 100 ms `nvidia-smi` sampler, stdout/stderr receipts and no-cache cold output paths.
- [ ] Re-run focused tests and confirm they pass.

### Task 3: Measure native Three.js loading

**Files:**
- Create: `perf/package.json`
- Create: `perf/browser-benchmark.mjs`
- Create: `perf/tests/browser-benchmark.test.mjs`

- [ ] Write a failing test for a normalized three-navigation browser receipt.
- [ ] Install the pinned Playwright dependency and Chromium under `perf/`.
- [ ] Implement the static-server browser runner with disabled cache, frame milestones and resource-byte accounting.
- [ ] Run the browser test against a tiny local fixture and confirm it passes.

### Task 4: Aggregate and document the benchmark

**Files:**
- Modify: `perf/benchmark.py`
- Modify: `perf/README.md`
- Test: `perf/tests/test_benchmark.py`

- [ ] Write a failing test for a three-dataset mean that excludes incomplete records.
- [ ] Implement summary JSON and Markdown rendering of cold/warm means, shares and peak VRAM.
- [ ] Run all harness tests, Black check, relevant existing Python tests and `viewer-web` build.

### Task 5: Run and review fd2–4

**Files:**
- Create: `perf/runs/<utc-run-id>/report.md`
- Create: `perf/runs/<utc-run-id>/summary.json`

- [ ] Run resource detection, then one fd2 cold smoke.
- [ ] Run the full serial cold/warm matrix on GPU 2.
- [ ] Verify every receipt and generate the final bottleneck report.
- [ ] Compare the final worktree diff against the explicit `perf/` allowlist.

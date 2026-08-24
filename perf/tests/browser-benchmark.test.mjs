import assert from "node:assert/strict";
import test from "node:test";

import {
  assertHardwareRenderer,
  chromiumLaunchArgs,
  viewerMountFailureMessage,
  summariseNavigations,
} from "../browser-benchmark.mjs";

test("summariseNavigations uses the median cache-disabled navigation", () => {
  const summary = summariseNavigations([
    { bundleLoadedMs: 140, firstFrameMs: 180, stableInteractiveMs: 220, bytes: 1_000 },
    { bundleLoadedMs: 100, firstFrameMs: 120, stableInteractiveMs: 150, bytes: 900 },
    { bundleLoadedMs: 120, firstFrameMs: 160, stableInteractiveMs: 200, bytes: 1_100 },
  ]);

  assert.deepEqual(summary, {
    sampleCount: 3,
    bundleLoadedMsMedian: 120,
    firstFrameMsMedian: 160,
    stableInteractiveMsMedian: 200,
    bytesMedian: 1_000,
  });
});

test("summariseNavigations rejects incomplete browser evidence", () => {
  assert.throws(
    () => summariseNavigations([{ bundleLoadedMs: 100, firstFrameMs: 120 }]),
    /stableInteractiveMs/,
  );
});

test("hardware renderer guard rejects SwiftShader receipts", () => {
  assert.throws(
    () => assertHardwareRenderer("ANGLE (Google, Vulkan (SwiftShader Device), SwiftShader driver)"),
    /software renderer/i,
  );
  assert.doesNotThrow(() => assertHardwareRenderer("NVIDIA GeForce RTX 4090 D/PCIe/SSE2"));
});

test("viewer mount failure keeps the page load error detail", () => {
  assert.equal(
    viewerMountFailureMessage("Error: SAM3 provenance mismatch"),
    "Viewer bundle did not mount: Error: SAM3 provenance mismatch",
  );
});

test("browser launch preserves the software fallback when hardware WebGL is absent", () => {
  const args = chromiumLaunchArgs();

  assert.ok(args.includes("--enable-gpu"));
  assert.equal(args.includes("--disable-software-rasterizer"), false);
});

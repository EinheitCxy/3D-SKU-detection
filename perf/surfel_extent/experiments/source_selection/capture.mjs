import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const baseUrl = "http://127.0.0.1:53110";
const outRoot = "runtime/viewer-extent-20261012/video12/followup/source-selection";
const receipt = JSON.parse(await fs.readFile("runtime/viewer-extent-20261012/video12/visual/capture-before-after.json", "utf8"));
const expected = receipt.screenshots.find(item => item.camera.scene === "video2" && item.camera.arm === "candidate"
  && item.camera.view === "close" && item.camera.radiusSize === 0.004)?.camera;
const errors = [];
const report = {
  baseUrl, scene: "video2", pointSize: 0.004, radius: 1.05, visibility: "all",
  renderer: "runtime/viewer-extent-20261012/video12/followup/source-selection-renderer.ts",
  comparison: "Only deterministic per-pixel source selection changes; full ZIP, all points, source-depth tolerance 0.005, cap 6, opaque output, and frozen camera remain fixed.",
  outputs: [], errors,
};
function sameArray(label, actual, wanted, tolerance = 0) {
  if (actual.length !== wanted.length || actual.some((value, index) => Math.abs(value - wanted[index]) > tolerance)) throw new Error(`${label} differs`);
}
function assertCamera(camera) {
  for (const key of ["position", "target", "up", "projection"]) sameArray(`camera.${key}`, camera[key], expected[key], 1e-12);
  for (const key of ["view", "radiusSize", "sourceFrameIndex", "fov"]) if (camera[key] !== expected[key]) throw new Error(`camera.${key} differs`);
}
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"],
});
try {
  const page = await browser.newPage({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(900000);
  page.on("pageerror", error => errors.push({ type: "pageerror", message: error.message }));
  page.on("console", message => { if (message.type() === "error") errors.push({ type: "console", message: message.text() }); });
  await page.goto(`${baseUrl}/?scene=video2&arm=candidate&view=close&radiusSize=0.004&frame=0&diagnostic=source-selection&sourceSelection=off`, { waitUntil: "domcontentloaded", timeout: 900000 });
  await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, { timeout: 900000 });
  const failure = await page.evaluate(() => window.__error);
  if (failure) throw new Error(failure);
  let firstCamera;
  for (const enabled of [false, true]) {
    if (enabled) await page.evaluate(() => window.__setDiagnosticSourceSelection(true));
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const camera = await page.evaluate(() => window.__cameraState());
    assertCamera(camera);
    if (firstCamera) for (const key of ["position", "target", "up", "projection"]) sameArray(`same.${key}`, camera[key], firstCamera[key]);
    else firstCamera = camera;
    const mode = enabled ? "on" : "off";
    const screenshot = `${mode}.png`;
    const coverage = await page.evaluate(() => window.__readDiagnosticCoverage());
    await fs.mkdir(outRoot, { recursive: true });
    await page.locator("canvas").screenshot({ path: path.join(outRoot, screenshot), timeout: 900000 });
    report.outputs.push({ mode, screenshot, camera, coverage });
  }
  for (const key of ["width", "height", "any", "total"]) {
    if (report.outputs[0].coverage[key] !== report.outputs[1].coverage[key]) throw new Error(`Accumulation occupancy ${key} changed across source-selection modes`);
  }
  await page.evaluate(() => window.__dispose?.());
  await page.close();
} catch (error) {
  report.failure = String(error?.stack ?? error);
  throw error;
} finally {
  await fs.mkdir(outRoot, { recursive: true });
  await fs.writeFile(path.join(outRoot, "receipt.json"), `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}
if (errors.length) throw new Error(`browser errors: ${JSON.stringify(errors)}`);
console.log(JSON.stringify(report));

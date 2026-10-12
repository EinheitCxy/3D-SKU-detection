import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : process.argv[index + 1];
}
const baseUrl = option("base-url", "http://127.0.0.1:53110");
const outRoot = path.resolve(option("out-dir", "runtime/viewer-extent-20261012/video12/followup/output-normalization"));
const receipt = JSON.parse(await fs.readFile("runtime/viewer-extent-20261012/video12/visual/capture-before-after.json", "utf8"));
const expected = receipt.screenshots.find(item => item.camera.scene === "video2" && item.camera.arm === "candidate"
  && item.camera.view === "close" && item.camera.radiusSize === 0.004)?.camera;
if (!expected) throw new Error("Frozen video2 candidate-close receipt is missing");

function sameArray(label, actual, wanted, tolerance = 0) {
  if (actual.length !== wanted.length || actual.some((value, index) => Math.abs(value - wanted[index]) > tolerance)) {
    throw new Error(`${label} differs`);
  }
}
function assertCamera(camera) {
  for (const key of ["position", "target", "up", "projection"]) sameArray(`camera.${key}`, camera[key], expected[key], 1e-12);
  for (const key of ["view", "radiusSize", "sourceFrameIndex", "fov"]) if (camera[key] !== expected[key]) throw new Error(`camera.${key} differs`);
}

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"],
});
const report = {
  baseUrl,
  scene: "video2",
  rendererBase: "runtime/viewer-extent-20261012/video12/followup/renderer-baseline.ts",
  rendererDiagnostic: "runtime/viewer-extent-20261012/video12/followup/diagnostic-renderer.ts",
  comparison: "Only final output normalization changes; accumulation, alpha, full ZIP, point visibility, radius, and frozen camera remain identical.",
  compositor: "old uses mix(white, rgb, smoothstep(0.03, 0.3, alpha)); opaque uses rgb for alpha > 0 and white for alpha == 0. Alpha 0.03 starts old-output contribution and 0.3 reaches full old-output weight.",
  outputs: [], errors: [],
};
try {
  const page = await browser.newPage({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(900000);
  page.on("pageerror", error => report.errors.push({ type: "pageerror", message: error.message }));
  page.on("console", message => { if (message.type() === "error") report.errors.push({ type: "console", message: message.text() }); });
  const url = new URL("/", baseUrl);
  for (const [key, value] of Object.entries({ scene: "video2", arm: "candidate", view: "close", radiusSize: "0.004", frame: "0", diagnostic: "source-frame", visibility: "all", output: "old" })) url.searchParams.set(key, value);
  await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 900000 });
  await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, { timeout: 900000 });
  const failure = await page.evaluate(() => window.__error);
  if (failure) throw new Error(failure);
  await page.evaluate(() => window.__setState("candidate", "close", 0.004, 0));
  let firstCamera;
  for (const mode of ["old", "opaque"]) {
    await page.evaluate(next => window.__setDiagnosticOutput(next), mode);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const camera = await page.evaluate(() => window.__cameraState());
    assertCamera(camera);
    if (firstCamera) for (const key of ["position", "target", "up", "projection"]) sameArray(`same.${key}`, camera[key], firstCamera[key]);
    else firstCamera = camera;
    const coverage = await page.evaluate(() => window.__readDiagnosticCoverage());
    const screenshot = `${mode}.png`;
    await fs.mkdir(outRoot, { recursive: true });
    await page.locator("canvas").screenshot({ path: path.join(outRoot, screenshot), timeout: 900000 });
    report.outputs.push({ mode, screenshot, camera, coverage });
  }
  if (JSON.stringify(report.outputs[0].coverage) !== JSON.stringify(report.outputs[1].coverage)) throw new Error("Accumulation alpha coverage changed across output modes");
  await page.evaluate(() => window.__showDiagnosticCoverageMask());
  report.coverageMask = "alpha-mask.png";
  await page.locator("canvas").screenshot({ path: path.join(outRoot, report.coverageMask), timeout: 900000 });
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
if (report.errors.length) throw new Error(`browser errors: ${JSON.stringify(report.errors)}`);
console.log(JSON.stringify(report));

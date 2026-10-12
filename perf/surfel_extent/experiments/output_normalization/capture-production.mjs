import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const argument = name => process.argv[process.argv.indexOf(name) + 1];
const scene = process.argv.includes("--scene") ? argument("--scene") : "video2";
if (scene !== "video1" && scene !== "video2") throw new Error("--scene must be video1 or video2");
const baseUrl = "http://127.0.0.1:53110";
const outRoot = "runtime/viewer-extent-20261012/video12/followup/output-normalization";
const receiptPath = scene === "video1" ? "runtime/viewer-extent-20261012/video12/visual/capture-candidate.json"
  : "runtime/viewer-extent-20261012/video12/visual/capture-before-after.json";
const receipt = JSON.parse(await fs.readFile(receiptPath, "utf8"));
const view = scene === "video1" ? "overview" : "close";
const expected = receipt.screenshots.find(item => item.camera.scene === scene && item.camera.arm === "candidate"
  && item.camera.view === view && item.camera.radiusSize === 0.004)?.camera;
const errors = [];
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"],
});
let report = { baseUrl, scene, renderer: "docker/viewer/src/surfel-renderer.ts", errors };
try {
  const page = await browser.newPage({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(900000);
  page.on("pageerror", error => errors.push({ type: "pageerror", message: error.message }));
  page.on("console", message => { if (message.type() === "error") errors.push({ type: "console", message: message.text() }); });
  await page.goto(`${baseUrl}/?scene=${scene}&arm=candidate&view=${view}&radiusSize=0.004&frame=0`, { waitUntil: "domcontentloaded", timeout: 900000 });
  await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, { timeout: 900000 });
  const failure = await page.evaluate(() => window.__error);
  if (failure) throw new Error(failure);
  const camera = await page.evaluate(() => window.__cameraState());
  for (const key of ["position", "target", "up", "projection"]) {
    if (camera[key].length !== expected[key].length || camera[key].some((value, index) => Math.abs(value - expected[key][index]) > 1e-12)) {
      throw new Error(`production camera.${key} differs from frozen receipt`);
    }
  }
  for (const key of ["view", "radiusSize", "sourceFrameIndex", "fov"]) if (camera[key] !== expected[key]) throw new Error(`production camera.${key} differs from frozen receipt`);
  await fs.mkdir(outRoot, { recursive: true });
  report = { ...report, camera, screenshot: `production-${scene}-opaque.png` };
  await page.locator("canvas").screenshot({ path: path.join(outRoot, report.screenshot), timeout: 900000 });
  await page.evaluate(() => window.__dispose?.());
  await page.close();
} catch (error) {
  report.failure = String(error?.stack ?? error);
  throw error;
} finally {
  await fs.writeFile(path.join(outRoot, `production-${scene}-receipt.json`), `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}
if (errors.length) throw new Error(`browser errors: ${JSON.stringify(errors)}`);
console.log(JSON.stringify(report));

import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

function option(name, fallback) { const i = process.argv.indexOf(`--${name}`); return i < 0 ? fallback : process.argv[i + 1]; }
const baseUrl = option("base-url", "http://192.168.2.6:53110");
const outRoot = path.resolve(option("out-dir", "runtime/viewer-extent-20261012/visual"));
const scenes = ["video3", "fd6"];
const requestedArm = option("arm", "");
if (requestedArm !== "" && requestedArm !== "major") throw new Error("--arm must be major when specified");
const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"] });
const report = { baseUrl, screenshots: [], errors: [] };
try {
  for (const scene of scenes) {
    const page = await browser.newPage({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
    page.setDefaultTimeout(900000);
    page.on("pageerror", error => report.errors.push({ scene, type: "pageerror", message: error.message }));
    page.on("console", message => { if (message.type() === "error") report.errors.push({ scene, type: "console", message: message.text() }); });
    const url = new URL("/", baseUrl);
    url.searchParams.set("scene", scene);
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 900000 });
    await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, { timeout: 900000 });
    const failure = await page.evaluate(() => window.__error);
    if (failure) throw new Error(`${scene}: ${failure}`);
    for (const radius of [0.004, 0.007619047619]) {
      const radiusLabel = radius === 0.004 ? "default" : "radius2";
      const views = requestedArm === "major" ? ["close"] : radius === 0.004 ? ["overview", "close", "side"] : ["close"];
      const arms = requestedArm === "major" ? ["major"] : ["baseline", "uniform"];
      for (const arm of arms) for (const view of views) {
        await page.evaluate(([a, v, r]) => window.__setState(a, v, r), [arm, view, radius]);
        const relative = path.join(scene, radiusLabel, `${arm}-${view}.png`);
        const output = path.join(outRoot, relative);
        await fs.mkdir(path.dirname(output), { recursive: true });
        await page.locator("canvas").screenshot({ path: output, timeout: 900000 });
        const state = await page.evaluate(() => window.__cameraState());
        report.screenshots.push({ path: relative, ...state });
        console.log(`saved ${output}`);
      }
    }
    await page.evaluate(() => window.__dispose?.());
    await page.close();
  }
} catch (error) {
  report.failure = String(error?.stack ?? error);
  throw error;
} finally {
  await fs.mkdir(outRoot, { recursive: true });
  const reportName = requestedArm === "major" ? "capture-major.json" : "capture-uniform.json";
  await fs.writeFile(path.join(outRoot, reportName), `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}
if (report.errors.length) throw new Error(`browser errors: ${JSON.stringify(report.errors)}`);
console.log(JSON.stringify(report));

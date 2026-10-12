import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : process.argv[index + 1];
}

const baseUrl = option("base-url", "http://192.168.2.6:53110");
const outRoot = path.resolve(option("out-dir", "runtime/viewer-extent-20261012/video12/visual"));
const scenes = option("scenes", "video1,video2").split(",").map(value => value.trim()).filter(Boolean);
const supportedScenes = new Set(["video1", "video2", "video3", "fd6"]);
if (scenes.length === 0 || scenes.some(scene => !supportedScenes.has(scene))) {
  throw new Error("--scenes must be a comma-separated subset of video1,video2,video3,fd6");
}
const requestedArm = option("arm", "");
if (requestedArm !== "" && !["baseline", "candidate", "uniform"].includes(requestedArm)) {
  throw new Error("--arm must be baseline, candidate, or uniform when specified");
}
const arms = requestedArm === "" ? ["baseline", "candidate"] : [requestedArm];
const allStates = [
  { radius: 0.004, radiusLabel: "default", view: "overview" },
  { radius: 0.004, radiusLabel: "default", view: "close" },
  { radius: 0.007619047619, radiusLabel: "radius2", view: "close" },
];
const requestedStates = new Set(option("states", "overview,default-close,radius2-close").split(",").map(value => value.trim()).filter(Boolean));
const stateKey = state => state.view === "overview" ? "overview" : state.radiusLabel === "default" ? "default-close" : "radius2-close";
const states = allStates.filter(state => requestedStates.has(stateKey(state)));
if (states.length !== requestedStates.size || states.length === 0) {
  throw new Error("--states must be a comma-separated subset of overview,default-close,radius2-close");
}

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"],
});
const report = {
  baseUrl,
  viewport: { width: 960, height: 600, deviceScaleFactor: 1 },
  scenes,
  screenshots: [],
  errors: [],
};
try {
  for (const scene of scenes) {
    const page = await browser.newPage({ viewport: { width: 960, height: 600 }, deviceScaleFactor: 1 });
    page.setDefaultTimeout(900000);
    page.on("pageerror", error => report.errors.push({ scene, type: "pageerror", message: error.message }));
    page.on("console", message => {
      if (message.type() === "error") report.errors.push({ scene, type: "console", message: message.text() });
    });
    const url = new URL("/", baseUrl);
    url.searchParams.set("scene", scene);
    url.searchParams.set("arm", arms[0]);
    url.searchParams.set("view", states[0].view);
    url.searchParams.set("radiusSize", String(states[0].radius));
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 900000 });
    await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, { timeout: 900000 });
    const failure = await page.evaluate(() => window.__error);
    if (failure) throw new Error(`${scene}: ${failure}`);
    let firstState = true;
    for (const state of states) {
      for (const arm of arms) {
        if (!(firstState && arm === arms[0])) {
          await page.evaluate(([nextArm, nextView, radius]) => window.__setState(nextArm, nextView, radius, 0), [arm, state.view, state.radius]);
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        }
        const relative = path.join(scene, state.radiusLabel, `${arm}-${state.view}.png`);
        const output = path.join(outRoot, relative);
        await fs.mkdir(path.dirname(output), { recursive: true });
        await page.locator("canvas").screenshot({ path: output, timeout: 900000 });
        const camera = await page.evaluate(() => window.__cameraState());
        report.screenshots.push({ path: relative, camera });
        console.log(`saved ${output}`);
        firstState = false;
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
  const suffix = requestedArm === "" ? "before-after" : requestedArm;
  await fs.writeFile(path.join(outRoot, `capture-${suffix}.json`), `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}
if (report.errors.length) throw new Error(`browser errors: ${JSON.stringify(report.errors)}`);
console.log(JSON.stringify(report));

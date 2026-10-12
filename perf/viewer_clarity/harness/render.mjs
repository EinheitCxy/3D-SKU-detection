import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

function option(name, fallback) {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? fallback : process.argv[index + 1];
}
const baseUrl = option("base-url", "http://127.0.0.1:4173");
const arm = option("arm", "V0");
const scene = option("scene", "video3");
const frame = Number(option("frame", "4"));
const zipPath = path.resolve(option("zip", ""));
const generation = path.resolve(option("generation-dir", path.dirname(zipPath)));
const out = path.resolve(option("out", `runtime/viewer-clarity-20261012/${scene}/${arm}/refview/${arm}/frame-${frame}.png`));
const radius = option("radius-size", "0.07");
const mode = option("mode", "surfel");
const browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args:["--no-sandbox","--use-angle=swiftshader","--enable-unsafe-swiftshader","--no-proxy-server"]});
try {
  const page = await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
  page.setDefaultTimeout(900000);
  page.on("pageerror", error => console.error(`pageerror: ${error.message}`));
  const url = new URL("/", baseUrl);
  url.searchParams.set("mode", mode);
  url.searchParams.set("frame", String(frame)); url.searchParams.set("scale", "2"); url.searchParams.set("radiusSize", radius);
  if (mode === "glb") {
    const generationUrl = absolute => `/@fs${path.resolve(absolute)}`;
    url.searchParams.set("glb", generationUrl(option("glb", "")));
    url.searchParams.set("manifest", generationUrl(path.join(generation,"manifest.json")));
    url.searchParams.set("surfel", generationUrl(path.join(generation,"surfel.json")));
  } else {
    url.searchParams.set("zip", "/assets/viewer_bundle.zip");
    await page.route("**/assets/viewer_bundle.zip", route => route.fulfill({path:zipPath,contentType:"application/zip"}));
  }
  await page.goto(url.href, {waitUntil:"networkidle",timeout:900000});
  await page.waitForFunction(() => window.__done === true || typeof window.__error === "string", null, {timeout:900000});
  const pageError = await page.evaluate(() => window.__error);
  if (pageError) throw new Error(`refview page failed: ${pageError}`);
  await fs.mkdir(path.dirname(out), {recursive:true});
  await page.locator("canvas").screenshot({path:out,timeout:900000});
  console.log(JSON.stringify({arm,scene,frame,out}));
  await page.evaluate(() => (window.__dispose?.()));
} finally { await browser.close(); }

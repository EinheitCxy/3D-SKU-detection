import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const argument = name => process.argv[process.argv.indexOf(name) + 1];
const baseUrl = process.argv.includes("--base-url") ? argument("--base-url") : "http://127.0.0.1:53110";
const pageName = process.argv.includes("--page") ? argument("--page") : "normalization-probe.html";
const output = path.resolve(process.argv.includes("--out") ? argument("--out")
  : "runtime/viewer-extent-20261012/video12/followup/output-normalization/probe.json");
const report = { errors: [] };
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-proxy-server"],
});
try {
  const page = await browser.newPage({ viewport: { width: 512, height: 512 }, deviceScaleFactor: 1 });
  page.on("pageerror", error => report.errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") report.errors.push(message.text()); });
  await page.goto(`${baseUrl}/${pageName}`, { waitUntil: "domcontentloaded", timeout: 120000 });
  await page.waitForFunction(() => window.__done || window.__error, null, { timeout: 120000 });
  Object.assign(report, await page.evaluate(() => ({ probe: window.__probe, failure: window.__error })));
  if (report.failure || report.errors.length || !Object.values(report.probe.checks).every(Boolean) || report.probe.glErrors.length) {
    process.exitCode = 1;
  }
} catch (error) {
  report.failure = String(error);
  process.exitCode = 1;
} finally {
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, JSON.stringify(report, null, 2) + "\n");
  await browser.close();
}
console.log(JSON.stringify(report));

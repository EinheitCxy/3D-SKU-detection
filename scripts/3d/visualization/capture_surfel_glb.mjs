import { access, mkdir, readdir, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const exists = async value => { try { await access(value, constants.R_OK); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
const { chromium } = await import(pathToFileURL(path.join(repo, 'perf/node_modules/playwright/index.mjs')));
const candidates = process.env.CHROMIUM_PATH ? [process.env.CHROMIUM_PATH] : [chromium.executablePath()];
for (const cache of ['/tmp/surfel-playwright', path.join(process.env.HOME || '', '.cache/ms-playwright')]) {
  if (!await exists(cache)) continue;
  for (const child of await readdir(cache)) {
    if (child.startsWith('chromium_headless_shell-')) candidates.push(path.join(cache, child, 'chrome-headless-shell-linux64/chrome-headless-shell'));
    if (child.startsWith('chromium-')) candidates.push(path.join(cache, child, 'chrome-linux64/chrome'));
  }
}
let executablePath;
for (const candidate of candidates) if (await exists(candidate)) { executablePath = candidate; break; }
if (!executablePath) throw Error('找不到 Chromium；设置 CHROMIUM_PATH。');
const url = process.argv[2];
if (!url) throw Error('用法: node scripts/3d/visualization/capture_surfel_glb.mjs <comparison URL> <output directory>');
const out = path.resolve(process.argv[3] || path.join(repo, 'runtime/surfel-glb-20261009/captures'));
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ executablePath, headless: true, args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1680, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [], requests = [], failedRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));
  page.on('requestfailed', request => failedRequests.push({ url: request.url(), error: request.failure()?.errorText }));
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => window.glbComparisonReady || window.glbComparisonError, {}, { timeout: 240000 });
  const error = await page.evaluate(() => window.glbComparisonError); if (error) throw Error(error);
  const views = [];
  for (const name of (process.argv[4] || 'front,side30,source0').split(',')) {
    const camera = await page.evaluate(name => window.setComparisonView(name), name);
    await page.screenshot({ path: path.join(out, `${name}.png`), timeout: 120000 });
    views.push({ name, camera });
  }
  const stats = await page.evaluate(() => window.glbComparisonStats);
  const finalError = await page.evaluate(() => window.glbComparisonError);
  const report = { url, executablePath, renderingRequested: 'software-swiftshader', performanceClaim: false, views, stats, requests, failedRequests, errors, finalError };
  await writeFile(path.join(out, 'browser-report.json'), JSON.stringify(report, null, 2));
  if (finalError || errors.length || failedRequests.length || stats.externalGlbResourceUrls.length) throw Error(`Browser validation failed; see ${out}/browser-report.json`);
  console.log(JSON.stringify({ out, triangles: stats.triangles, textures: stats.textures, externalGlbResourceUrls: stats.externalGlbResourceUrls, renderers: stats.renderers }, null, 2));
} finally { await browser.close(); }

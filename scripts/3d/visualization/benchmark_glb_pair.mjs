import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const [baseUrl, oldUrl, newUrl, output, cameraPath] = process.argv.slice(2);
if (!baseUrl || !oldUrl || !newUrl || !output) throw Error('Usage: node scripts/3d/visualization/benchmark_glb_pair.mjs <Vite base URL> <old GLB URL> <new GLB URL> <output directory> [camera JSON path]');
const cameraPose = cameraPath ? JSON.parse(await readFile(cameraPath, 'utf8')) : null;
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const { chromium } = await import(pathToFileURL(path.join(repo, 'perf/node_modules/playwright/index.mjs')));
const out = path.resolve(output);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || chromium.executablePath(), headless: true,
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 800, height: 600 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(240000);
  const errors = [], failedRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('requestfailed', request => failedRequests.push({ url: request.url(), error: request.failure()?.errorText }));
  await page.goto(new URL('glb-benchmark.html', baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`).href, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => window.glbBenchmarkReady);
  const results = [];
  for (const [label, url] of [['old', oldUrl], ['new', newUrl]]) {
    const result = await page.evaluate(({ url, cameraPose }) => window.runGlbBenchmark(url, cameraPose), { url, cameraPose });
    await page.locator('canvas').screenshot({ path: path.join(out, `${label}.png`), timeout: 120000 });
    results.push({ label, ...result });
  }
  const finalError = await page.evaluate(() => window.glbBenchmarkError);
  await writeFile(path.join(out, 'benchmark.json'), JSON.stringify({
    renderingRequested: 'software-swiftshader', realGpuPerformanceClaim: false,
    order: 'old then new; same renderer and camera; two warmup and five measured frames each',
    cameraPose, results, errors, failedRequests, finalError,
  }, null, 2));
  if (finalError || errors.length || failedRequests.length) throw Error(`Browser validation failed; see ${out}/benchmark.json`);
  console.log(JSON.stringify({ out, results: results.map(({ label, bytes, medianFrameMs, triangles, drawCalls }) => ({ label, bytes, medianFrameMs, triangles, drawCalls })) }, null, 2));
} finally { await browser.close(); }

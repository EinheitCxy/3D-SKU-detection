import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const exists = async value => { try { await access(value, constants.R_OK); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } };
const packages = [path.join(repo, 'perf/node_modules/playwright/index.mjs'), path.join(repo, 'modules/viewer_web/node_modules/playwright/index.mjs')];
let playwright;
for (const candidate of packages) if (await exists(candidate)) { playwright = await import(pathToFileURL(candidate)); break; }
if (!playwright) throw Error('找不到本地 Playwright；请在 perf 安装项目已有依赖。');
const candidates = process.env.CHROMIUM_PATH ? [process.env.CHROMIUM_PATH] : [];
for (const old of ['perf/surfel_compare/capture.mjs', 'perf/browser-benchmark.mjs']) {
  const source = await readFile(path.join(repo, old), 'utf8');
  for (const match of source.matchAll(/executablePath\s*:\s*['"]([^'"]+)['"]/g)) candidates.push(match[1]);
}
candidates.push(playwright.chromium.executablePath());
for (const cache of ['/tmp/surfel-playwright', path.join(process.env.HOME || '', '.cache/ms-playwright')]) {
  if (!await exists(cache)) continue;
  for (const child of await readdir(cache)) {
    if (child.startsWith('chromium_headless_shell-')) candidates.push(path.join(cache, child, 'chrome-headless-shell-linux64/chrome-headless-shell'));
    if (child.startsWith('chromium-')) candidates.push(path.join(cache, child, 'chrome-linux64/chrome'));
  }
}
let executablePath;
for (const candidate of candidates) if (await exists(candidate)) { executablePath = candidate; break; }
if (!executablePath) throw Error('找不到 Chromium；使用 CHROMIUM_PATH 指定已有浏览器路径。');
const url = process.argv[2] || 'http://127.0.0.1:5176/roi-fusion.html?assets=http://127.0.0.1:8767/';
const out = path.resolve(process.argv[3] || path.join(repo, 'runtime/roi-fusion-video3-gid3/captures'));
await mkdir(out, { recursive: true });
const hardware = process.env.ROI_CAPTURE_GPU === '1';
const args = hardware
  ? ['--no-sandbox', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--use-vulkan=native']
  : ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
console.log(hardware ? '请求 GPU 渲染；实际 renderer 将写入报告。' : '使用 SwiftShader 软件渲染，仅验证外观和交互，不代表 GPU 性能。');
const browser = await playwright.chromium.launch({ executablePath, headless: true, args });
try {
  const page = await browser.newPage({ viewport: { width: 1680, height: 1100 }, deviceScaleFactor: 1 });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForFunction(() => window.roiReady || window.roiError, {}, { timeout: 240000 });
  const error = await page.evaluate(() => window.roiError); if (error) throw Error(error);
  const presets = process.env.ROI_CAPTURE_PRESETS?.split(',') || ['source0', 'midpoint01', 'front', 'side30', 'side60'];
  const captures = [];
  for (const preset of presets) for (const sourceColor of [false, true]) {
    const result = await page.evaluate(({ preset, sourceColor }) => window.roiCapture(preset, sourceColor), { preset, sourceColor });
    const stem = `${preset.replace(':', '-')}-${sourceColor ? 'sources' : 'texture'}`;
    for (const [index, method] of result.methods.entries()) await writeFile(path.join(out, `${stem}-${method}.png`), Buffer.from(result.pngs[index], 'base64'));
    captures.push({ preset, sourceColor, methods: result.methods });
  }
  const contextAvailable = await page.locator('#context').isEnabled();
  let contextToggle = 'unavailable';
  if (contextAvailable) {
    await page.locator('#context').check();
    if (!await page.evaluate(() => window.roiCameraStates().every(p => p.contextVisible))) throw Error('场景开关未同步显示所有背景');
    await page.locator('#context').uncheck();
    if (!await page.evaluate(() => window.roiCameraStates().every(p => !p.contextVisible))) throw Error('场景开关未同步隐藏所有背景');
    contextToggle = 'passed';
  }
  await page.evaluate(() => window.roiCapture('front', false));
  const beforeDrag = await page.evaluate(() => window.roiCameraStates());
  const viewport = await page.locator('#columns .viewport canvas').first().boundingBox();
  await page.mouse.move(viewport.x + viewport.width / 2, viewport.y + viewport.height / 2);
  await page.mouse.down();
  await page.mouse.move(viewport.x + viewport.width / 2 + 50, viewport.y + viewport.height / 2 + 20, { steps: 5 });
  await page.mouse.up();
  const afterDrag = await page.evaluate(() => window.roiCameraStates());
  const vectorsEqual = (a, b) => a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) < 1e-9);
  const synchronizedDrag = afterDrag.every(p => ['position', 'quaternion', 'target'].every(key => vectorsEqual(p[key], afterDrag[0][key])) && p.zoom === afterDrag[0].zoom);
  const dragChangedCamera = !vectorsEqual(beforeDrag[0].position, afterDrag[0].position);
  if (!synchronizedDrag || !dragChangedCamera) throw Error('相机拖动同步验证失败');
  await page.evaluate(() => window.roiCapture('front', false));
  await page.evaluate(async () => {
    const references = [...document.querySelectorAll('#references img')];
    for (const image of references) image.loading = 'eager';
    await Promise.all(references.map(image => image.decode()));
  });
  await page.screenshot({ path: path.join(out, 'comparison.png'), fullPage: true });
  const finalError = await page.evaluate(() => window.roiError);
  if (finalError) throw Error(finalError);
  const stats = await page.evaluate(() => window.roiStats);
  const report = { url, executablePath, renderingRequested: hardware ? 'hardware' : 'software-swiftshader', performanceClaim: false, captures, interactionSmoke: { contextToggle, synchronizedDrag, dragChangedCamera }, stats, errors };
  await writeFile(path.join(out, 'browser-report.json'), JSON.stringify(report, null, 2));
  if (errors.length) throw Error(errors.join('\n'));
  console.log(JSON.stringify({ out, captures: captures.length, renderers: stats.renderers, errors }, null, 2));
} finally { await browser.close(); }

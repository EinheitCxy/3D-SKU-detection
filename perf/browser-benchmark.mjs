import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { basename, extname, normalize, resolve, sep } from "node:path";

const MIME_TYPES = new Map([
  [".bin", "application/octet-stream"],
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".jpeg", "image/jpeg"],
  [".jpg", "image/jpeg"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".webp", "image/webp"],
]);

export function validateNavigationReceipt(receipt) {
  const required = ["bundleLoadedMs", "firstFrameMs", "stableInteractiveMs", "bytes"];
  for (const key of required) {
    if (!Number.isFinite(receipt[key])) throw new Error(`navigation receipt missing ${key}`);
  }
  return receipt;
}

export function assertHardwareRenderer(renderer) {
  if (typeof renderer !== "string" || renderer.length === 0) {
    throw new Error("browser did not expose a WebGL hardware renderer");
  }
  if (/swiftshader|llvmpipe|software/i.test(renderer)) {
    throw new Error(`browser reported a software renderer: ${renderer}`);
  }
}

export function viewerMountFailureMessage(detail) {
  return `Viewer bundle did not mount: ${detail}`;
}

export function chromiumLaunchArgs() {
  return [
    "--enable-gpu",
    "--enable-gpu-rasterization",
    "--ignore-gpu-blocklist",
    "--disable-gpu-driver-bug-workarounds",
    "--use-angle=gl",
    "--use-gl=egl",
  ];
}

function parseArgs(argumentsList) {
  const values = new Map();
  for (let index = 0; index < argumentsList.length; index += 2) {
    const key = argumentsList[index];
    const value = argumentsList[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`invalid argument near ${key}`);
    values.set(key, value);
  }
  for (const key of ["--data-root", "--output"]) {
    if (!values.has(key)) throw new Error(`missing required argument ${key}`);
  }
  return {
    dataRoot: resolve(values.get("--data-root")),
    output: resolve(values.get("--output")),
    viewerDist: resolve(values.get("--viewer-dist") ?? new URL("../modules/viewer_web/dist", import.meta.url).pathname),
  };
}

function safeFile(root, requestedPath) {
  const relativePath = normalize(requestedPath).replace(/^([/\\])+/, "");
  const candidate = resolve(root, relativePath);
  if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) throw new Error("path traversal rejected");
  return candidate;
}

async function respondFile(response, root, pathname, fallbackToIndex) {
  const candidate = safeFile(root, pathname === "/" ? "index.html" : pathname);
  let filePath = candidate;
  try {
    const fileStat = await stat(filePath);
    if (fileStat.isDirectory()) filePath = resolve(filePath, "index.html");
  } catch {
    if (!fallbackToIndex) {
      response.writeHead(404).end();
      return;
    }
    filePath = resolve(root, "index.html");
  }
  try {
    const body = await readFile(filePath);
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-type": MIME_TYPES.get(extname(filePath).toLowerCase()) ?? "application/octet-stream",
      "content-length": body.length,
    });
    response.end(body);
  } catch {
    response.writeHead(404).end();
  }
}

async function startViewerServer({ viewerDist, dataRoot }) {
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
    if (pathname === "/data" || pathname.startsWith("/data/")) {
      const dataPath = pathname === "/data" ? "/" : pathname.slice("/data".length);
      await respondFile(response, dataRoot, dataPath, false);
      return;
    }
    await respondFile(response, viewerDist, pathname, true);
  });
  await new Promise((resolvePromise, rejectPromise) => {
    server.once("error", rejectPromise);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("could not obtain benchmark server port");
  return { server, origin: `http://127.0.0.1:${address.port}` };
}

async function collectNavigation(browser, origin) {
  const context = await browser.newContext({ cacheEnabled: false, viewport: { width: 1440, height: 960 } });
  const page = await context.newPage();
  await page.addInitScript(() => {
    window.__da3ViewerPerf = { canvasAttachedMs: null, firstFrameMs: null, stableInteractiveMs: null };
    const observer = new MutationObserver(() => {
      if (window.__da3ViewerPerf.canvasAttachedMs !== null || document.querySelector("canvas") === null) return;
      window.__da3ViewerPerf.canvasAttachedMs = performance.now();
      let frames = 0;
      const nextFrame = (timestamp) => {
        frames += 1;
        if (frames === 1) window.__da3ViewerPerf.firstFrameMs = timestamp;
        if (frames === 10) window.__da3ViewerPerf.stableInteractiveMs = timestamp;
        else requestAnimationFrame(nextFrame);
      };
      requestAnimationFrame(nextFrame);
      observer.disconnect();
    });
    observer.observe(document, { childList: true, subtree: true });
  });
  try {
    await page.goto(`${origin}/?data=/data/`, { waitUntil: "domcontentloaded", timeout: 120_000 });
    try {
      await page.waitForSelector(".viewer-shell canvas", { timeout: 120_000 });
    } catch (error) {
      const detail = await page.locator(".load-error").textContent().catch(() => null);
      if (detail !== null) throw new Error(viewerMountFailureMessage(detail));
      throw error;
    }
    await page.waitForFunction(() => window.__da3ViewerPerf?.stableInteractiveMs !== null, null, { timeout: 120_000 });
    return await page.evaluate(() => {
      const resources = performance.getEntriesByType("resource");
      const dataResources = resources.filter((entry) => entry.name.includes("/data/"));
      const bundleLoadedMs = Math.max(...dataResources.map((entry) => entry.responseEnd));
      const bytes = resources.reduce((total, entry) => total + (entry.transferSize || 0), 0);
      const canvas = document.querySelector("canvas");
      const context = canvas?.getContext("webgl2") ?? canvas?.getContext("webgl");
      const debugInfo = context?.getExtension("WEBGL_debug_renderer_info");
      return {
        bundleLoadedMs,
        firstFrameMs: window.__da3ViewerPerf.firstFrameMs,
        stableInteractiveMs: window.__da3ViewerPerf.stableInteractiveMs,
        bytes,
        renderer: !debugInfo || !context ? null : context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL),
        vendor: !debugInfo || !context ? null : context.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL),
      };
    });
  } finally {
    await context.close();
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const { chromium } = await import("playwright");
  const { server, origin } = await startViewerServer(options);
  const browser = await chromium.launch({
    headless: true,
    args: chromiumLaunchArgs(),
  });
  try {
    const navigation = validateNavigationReceipt(await collectNavigation(browser, origin));
    let rendererEvidence = "hardware";
    try {
      assertHardwareRenderer(navigation.renderer);
    } catch {
      rendererEvidence = "software_or_unavailable";
    }
    const output = {
      origin,
      navigation,
      rendererEvidence,
    };
    await import("node:fs/promises").then(({ mkdir, writeFile }) => mkdir(resolve(options.output, ".."), { recursive: true }).then(() => writeFile(options.output, `${JSON.stringify(output, null, 2)}\n`)));
  } finally {
    await browser.close();
    await new Promise((resolvePromise, rejectPromise) => server.close((error) => error ? rejectPromise(error) : resolvePromise()));
  }
}

if (basename(process.argv[1] ?? "") === basename(new URL(import.meta.url).pathname)) {
  main().catch((error) => {
    console.error(error.stack ?? error);
    process.exitCode = 1;
  });
}

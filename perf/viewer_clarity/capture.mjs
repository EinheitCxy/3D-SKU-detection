import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

function option(name, fallback) { const i = process.argv.indexOf(`--${name}`); return i < 0 ? fallback : process.argv[i + 1]; }
const baseUrl = option("base-url", "http://127.0.0.1:4173");
const zipPath = path.resolve(option("zip", ""));
const outDir = path.resolve(option("out-dir", "runtime/viewer-clarity-20261012/video3/V0/capture"));
const taskId = option("task-id", "viewer-clarity-smoke");
const browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args:["--no-sandbox","--use-angle=swiftshader","--enable-unsafe-swiftshader","--no-proxy-server"]});
const report = {taskId, screenshots:{}, errors:[]};
let page;
let lastStep = "startup";
try {
  page = await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
  page.setDefaultTimeout(600000);
  page.on("pageerror", error => report.errors.push(error.message));
  page.on("console", message => { if(message.type()==="error") report.errors.push(message.text()); });
  await page.addInitScript((replacement) => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = (input, init) => {
      const url = typeof input === "string" ? input : input instanceof Request ? input.url : "";
      return originalFetch(url.endsWith("/viewer_bundle.zip") ? replacement : input, init);
    };
  }, `${baseUrl}/assets/viewer_bundle.zip`);
  await fs.mkdir(outDir,{recursive:true});
  await page.route("**/assets/viewer_bundle.zip", route => route.fulfill({path:zipPath,contentType:"application/zip"}));
  const response = await page.goto(`${baseUrl}/?recognition_task_id=${encodeURIComponent(taskId)}`,{waitUntil:"domcontentloaded",timeout:600000});
  if (!response?.ok()) throw new Error(`Viewer preview returned ${response?.status()}`);
  const canvas = page.locator(".viewer-shell canvas");
  await page.waitForFunction(() => document.querySelector(".viewer-shell canvas")?.width > 0, null, {timeout:600000});
  await page.waitForTimeout(2000);
  lastStep = "overview";
  console.log("capture: overview loaded");
  await page.screenshot({path:path.join(outDir,"overview.png")});
  const canvasPng = await canvas.screenshot({path:path.join(outDir,"overview-canvas.png")});
  report.product_height_fraction = await page.evaluate(async (base64) => {
    const image = new Image();
    image.src = `data:image/png;base64,${base64}`;
    await image.decode();
    const bitmap = document.createElement("canvas"); bitmap.width=image.width; bitmap.height=image.height;
    const context = bitmap.getContext("2d",{willReadFrequently:true});
    if(!context) throw new Error("2D context unavailable for overview screenshot analysis");
    context.drawImage(image,0,0);
    const pixels=context.getImageData(0,0,bitmap.width,bitmap.height).data;
    let top=bitmap.height,bottom=-1;
    for(let y=0;y<bitmap.height;y++) for(let x=0;x<bitmap.width;x++) {
      const i=(y*bitmap.width+x)*4;
      if(pixels[i]!==255 || pixels[i+1]!==255 || pixels[i+2]!==255) {top=Math.min(top,y);bottom=Math.max(bottom,y);}
    }
    return bottom < top ? 0 : (bottom-top+1)/bitmap.height;
  }, canvasPng.toString("base64"));
  const rect = await canvas.boundingBox();
  lastStep = "close";
  await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height/2);
  await page.mouse.wheel(0,-1200);
  await page.waitForTimeout(250);
  await page.screenshot({path:path.join(outDir,"close.png")});
  console.log("capture: close saved");
  await page.mouse.move(rect.x+rect.width*.6,rect.y+rect.height*.5);
  await page.mouse.down(); await page.mouse.move(rect.x+rect.width*.73,rect.y+rect.height*.57,{steps:5}); await page.mouse.up();
  await page.screenshot({path:path.join(outDir,"side.png")});
  console.log("capture: side saved");
  lastStep = "switch-to-global-id";
  await page.locator(".selection-mode-buttons button").nth(1).click({timeout:600000});
  console.log("capture: global id click returned");
  lastStep = "wait-global-id-list";
  await page.waitForFunction(() => document.querySelector(".selection-mode-buttons button:nth-child(2)")?.getAttribute("aria-pressed") === "true", null, {timeout:600000});
  const item = page.locator(".global-pane .object-item").first();
  await item.waitFor({state:"visible",timeout:600000});
  lastStep = "select-first-global-id";
  await item.click();
  await page.waitForTimeout(250);
  await page.screenshot({path:path.join(outDir,"selected.png")});
  console.log("capture: selected saved");
  report.http_status = response.status();
  report.screenshots = {overview:"overview.png",close:"close.png",side:"side.png",selected:"selected.png",overview_canvas:"overview-canvas.png"};
  await fs.writeFile(path.join(outDir,"capture.json"),JSON.stringify(report,null,2)+"\n");
  if(report.errors.length) throw new Error(`browser errors: ${JSON.stringify(report.errors)}`);
  console.log(JSON.stringify(report));
} catch(error) {
  report.lastStep = lastStep;
  report.failure = String(error);
  await fs.writeFile(path.join(outDir,"capture.json"),JSON.stringify(report,null,2)+"\n");
  console.error(`capture failed at ${lastStep}: ${String(error)}`);
  try {
    report.dom = await Promise.race([page.evaluate(() => ({
      readyState: document.readyState,
      modes: [...document.querySelectorAll(".selection-mode-buttons button")].map(button => ({text:button.textContent,pressed:button.getAttribute("aria-pressed")})),
      globalHidden: document.querySelector(".global-pane")?.hidden,
      globalItems: document.querySelectorAll(".global-pane .object-item").length,
      bodyText: document.body.innerText.slice(0,1000),
    })), new Promise(resolve => setTimeout(() => resolve({diagnosticTimeout:true}),10000))]);
  } catch (diagnosticError) { report.diagnosticError = String(diagnosticError); }
  await fs.writeFile(path.join(outDir,"capture.json"),JSON.stringify(report,null,2)+"\n");
  throw error;
} finally { await browser.close(); }

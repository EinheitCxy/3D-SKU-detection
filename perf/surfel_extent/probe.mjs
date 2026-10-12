import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const baseUrl = process.argv.includes("--base-url") ? process.argv[process.argv.indexOf("--base-url") + 1] : "http://192.168.2.6:53110";
const out = path.resolve("runtime/viewer-extent-20261012/visual/probe.json");
const errors = [];
const browser = await chromium.launch({ headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  args:["--no-sandbox","--use-angle=swiftshader","--enable-unsafe-swiftshader","--no-proxy-server"] });
try {
  const page = await browser.newPage({viewport:{width:512,height:512},deviceScaleFactor:1});
  page.on("pageerror", error => errors.push({type:"pageerror",message:error.message}));
  page.on("console", message => { if(message.type()==="error") errors.push({type:"console",message:message.text()}); });
  await page.goto(`${baseUrl}/probe.html`,{waitUntil:"domcontentloaded",timeout:120000});
  await page.waitForFunction(() => window.__done === true || typeof window.__error === "string",null,{timeout:120000});
  const result = await page.evaluate(() => ({probe:window.__probe,failure:window.__error}));
  const report = {...result, errors};
  await fs.mkdir(path.dirname(out),{recursive:true});
  await fs.writeFile(out,JSON.stringify(report,null,2)+"\n");
  if(result.failure || errors.length || !Object.values(result.probe.checks).every(Boolean) || result.probe.glErrors.length) process.exitCode=1;
  console.log(JSON.stringify(report));
} finally { await browser.close(); }

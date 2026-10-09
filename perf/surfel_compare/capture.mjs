import {chromium} from '../node_modules/playwright/index.mjs';
import {mkdir,writeFile} from 'node:fs/promises';
const mode=process.argv[2]||'baseline', out=`modules/viewer_web/public/comparison/${mode}`;
await mkdir(out,{recursive:true});
const browser=await chromium.launch({executablePath:'/tmp/surfel-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless:true,args:['--no-sandbox','--enable-gpu','--ignore-gpu-blocklist','--use-angle=vulkan','--enable-features=Vulkan','--use-vulkan=native']});
const page=await browser.newPage({viewport:{width:560,height:1008},deviceScaleFactor:1});
page.on('pageerror',e=>console.error(e));
await page.goto(`http://127.0.0.1:5173/compare-render.html?mode=${mode}`,{waitUntil:'domcontentloaded',timeout:120000});
await page.waitForFunction(()=>window.comparisonReady,{},{timeout:180000});
console.log(await page.evaluate(()=>({renderer:window.comparisonReport.renderer,loadMs:window.comparisonReport.loadMs,firstRenderedMs:window.comparisonReport.firstRenderedMs,captureReadyMs:window.comparisonReport.firstFrameMs})));
const count=process.env.CAPTURE_LIMIT?Number(process.env.CAPTURE_LIMIT):70;
for(let i=0;i<count;i++){
 const r=await page.evaluate(i=>window.renderComparison(i),i);
 const stem=String(i).padStart(3,'0');
 await writeFile(`${out}/${stem}.png`,Buffer.from(r.color,'base64'));
 await writeFile(`${out}/${stem}-ids.png`,Buffer.from(r.ids,'base64'));
 if(i%10===0)console.log(`${mode} ${i}/${count}: ${r.renderMs.toFixed(1)} ms`);
}
await writeFile(`runtime/surfel-comparison/${mode}-${count===0?"loading":"browser"}.json`,JSON.stringify(await page.evaluate(()=>window.comparisonReport),null,2));
await browser.close();

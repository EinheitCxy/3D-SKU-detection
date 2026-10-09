import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '../node_modules/playwright/index.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const requestedTrial = process.argv[3];
if (requestedTrial && !['texture','adaptive','coverage','coverage_bounded'].includes(requestedTrial)) throw Error('Unknown capture trial');
const out = path.join(root,`runtime/surfel-coverage-20260924/${requestedTrial ? `captures-${requestedTrial}` : 'captures'}`);
await mkdir(out,{recursive:true});
const executablePath = '/tmp/surfel-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell';
const url = process.argv[2] || 'http://127.0.0.1:8769/web/surfel-coverage.html';
const browser = await chromium.launch({executablePath,headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const report = {url,renderingRequested:'software-swiftshader',performanceClaim:false,errors:[],captures:[],interactions:{}};
try {
  const page = await browser.newPage({viewport:{width:1680,height:1050},deviceScaleFactor:1});
  page.on('pageerror',error => report.errors.push(error.message));
  await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForFunction(() => window.coverageReady || window.coverageError,{}, {timeout:180000});
  if (await page.evaluate(() => window.coverageError)) throw Error(await page.evaluate(() => window.coverageError));
  report.renderers = await page.evaluate(() => window.coverageRenderers());
  for (const trial of requestedTrial ? [requestedTrial] : ['texture','adaptive','coverage']) {
    if (await page.locator('#trial').inputValue() !== trial) await page.evaluate(trial => window.coverageSelect(trial),trial);
    for (const preset of ['front','side60','close','close-side']) {
      const colors = ['side60','close'].includes(preset) ? [false,true] : [false];
      for (const source of colors) {
        const panels = await page.evaluate(({preset,source}) => window.coverageCapture(preset,source),{preset,source});
        const stem = `${trial}-${preset}-${source?'sources':'texture'}`;
        for (const [index,panel] of panels.entries()) await writeFile(path.join(out,`${stem}-${index?'candidate':'reference'}.png`),Buffer.from(panel.png,'base64'));
        report.captures.push({trial,preset,source,methods:panels.map(p=>p.method)});
      }
    }
    await page.evaluate(() => window.coverageCapture('side60',false));
    await page.screenshot({path:path.join(out,`${trial}-comparison.png`),fullPage:true});
    const states = await page.evaluate(() => window.coverageStates());
    if (states[1].method !== trial) throw Error('Trial selector loaded the wrong candidate');
    report.interactions[`${trial}_loaded`] = true;
  }
  const before = await page.evaluate(() => window.coverageStates());
  const box = await page.locator('#reference canvas').boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+55,box.y+box.height/2+20,{steps:5});await page.mouse.up();
  const after = await page.evaluate(() => window.coverageStates());
  const same = (a,b) => a.every((x,i)=>Math.abs(x-b[i])<1e-9);
  report.interactions.drag_changed_camera = !same(before[0].position,after[0].position);
  report.interactions.synchronized_drag = ['position','quaternion','target'].every(key=>same(after[0][key],after[1][key]));
  if (!report.interactions.drag_changed_camera || !report.interactions.synchronized_drag) throw Error('Camera synchronization failed');
  await page.locator('#observations').evaluate(image=>image.decode());
  const error = await page.evaluate(() => window.coverageError);if(error) throw Error(error);
  if(report.errors.length) throw Error(report.errors.join('\n'));
  report.completed = true;
} catch(error) {report.completed=false;report.failure=error.stack;throw error;}
finally {await writeFile(path.join(out,'browser-report.json'),JSON.stringify(report,null,2)+'\n');await browser.close();}
console.log(JSON.stringify({captures:report.captures.length,renderers:report.renderers,interactions:report.interactions,errors:report.errors}));

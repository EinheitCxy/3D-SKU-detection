import {chromium} from '../node_modules/playwright/index.mjs';
import {writeFile} from 'node:fs/promises';
const browser=await chromium.launch({executablePath:'/tmp/surfel-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1680,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('http://127.0.0.1:5173/comparison/index.html',{waitUntil:'domcontentloaded'});
await page.waitForFunction(()=>window.comparisonPlaybackReady===true);
const load=await page.evaluate(()=>({elapsedMs:performance.now(),bytes:performance.getEntriesByType('resource').reduce((a,r)=>a+r.encodedBodySize,0)}));
const clicks=[];
for(const [index,mode] of ['baseline','consistent','gaussian'].entries()){
 const point=await page.evaluate(async mode=>{const im=new Image();im.src=`${mode}/000-ids.png`;await im.decode();const c=document.createElement('canvas');c.width=560;c.height=1008;const ctx=c.getContext('2d');ctx.drawImage(im,0,0);const d=ctx.getImageData(0,0,560,1008).data;for(let y=500;y<800;y++)for(let x=140;x<420;x++){const o=(y*560+x)*4,id=d[o]+256*d[o+1];if(id)return {x,y,id};}throw Error('无可测试标签');},mode);
 const box=await page.locator('.viewport img').nth(index).boundingBox();await page.mouse.click(box.x+(point.x+.5)*box.width/560,box.y+(point.y+.5)*box.height/1008);
 const label=await page.locator('.result').nth(index).innerText();if(!label.includes(`商品编号 ${point.id} ·`))throw Error(`点击不一致 ${label}`);clicks.push({mode,...point,label});
}
await page.locator('#jump').selectOption('61');await page.waitForFunction(()=>[...document.querySelectorAll('.viewport img')].every(x=>x.src.endsWith('/061.png')));
await page.locator('#zoom').click();if(!await page.locator('#grid').evaluate(x=>x.classList.contains('zoom')))throw Error('缩放失败');await page.locator('#zoom').click();
await page.locator('#jump').selectOption('30');await page.waitForFunction(()=>[...document.querySelectorAll('.viewport img')].every(x=>x.src.endsWith('/030.png')));
await page.screenshot({path:'runtime/surfel-comparison/comparison.png',fullPage:true});
const result={load,clicks,errors,synchronizedJump:true,zoom:true};await writeFile('runtime/surfel-comparison/playback-browser.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));await browser.close();if(errors.length)throw Error(errors.join('\n'));

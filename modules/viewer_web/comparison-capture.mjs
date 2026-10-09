import {WebGLRenderer,Camera,Matrix4,BufferAttribute,SRGBColorSpace} from 'three';
import {loadViewerBundle} from './src/bundle-loader.ts';
import {loadSurfels} from './src/surfel-loader.ts';
import {createSurfelRenderer} from './src/surfel-renderer.ts';
const start=performance.now();
const q=new URLSearchParams(location.search), mode=q.get('mode')||'baseline';
const track=await (await fetch('/comparison/cameras.json')).json();
const bundle0=await loadViewerBundle(new URL('/data-video1-1fps/',location).href,fetch,'surfel');
const bundle={...bundle0,surfels:await loadSurfels(bundle0)};
const labels=new Uint16Array(bundle.pointCount);
for(const [id,obj] of Object.entries(bundle.objects))for(const [a,b] of obj.point_ranges)labels.fill(Number(id),a,b);
let visibility=new Uint8Array(bundle.pointCount).fill(1);
if(mode==='consistent')visibility=new Uint8Array(await(await fetch('/comparison/visibility.bin')).arrayBuffer());
if(visibility.length!==bundle.pointCount)throw new Error('实验可见性数据长度错误');
const renderer=new WebGLRenderer({antialias:false,preserveDrawingBuffer:true});
renderer.outputColorSpace=SRGBColorSpace;renderer.setPixelRatio(1);renderer.setSize(track.width,track.height);
document.body.append(renderer.domElement);
const camera=new Camera();camera.matrixAutoUpdate=false;
const surfel=createSurfelRenderer(renderer,camera,bundle,new Matrix4(),new BufferAttribute(visibility,1));
surfel.setSize(track.width,track.height);
const gl=renderer.getContext(), ext=gl.getExtension('WEBGL_debug_renderer_info');
const report={mode,pointCount:bundle.pointCount,visiblePoints:visibility.reduce((a,b)=>a+b,0),loadMs:performance.now()-start,renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):'unknown',frames:[]};
const originalRead=renderer.readRenderTargetPixels.bind(renderer);
let capture=false,idPixels=null;
renderer.readRenderTargetPixels=(target,...args)=>{
 if(capture){idPixels=new Uint8Array(track.width*track.height*4);originalRead(target,0,0,track.width,track.height,idPixels);}
 return originalRead(target,...args);
};
function pngIds(){
 const c=document.createElement('canvas');c.width=track.width;c.height=track.height;
 const ctx=c.getContext('2d'), im=ctx.createImageData(c.width,c.height);
 for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){
  const src=((c.height-1-y)*c.width+x)*4,dst=(y*c.width+x)*4;
  const slot=idPixels[src]+256*idPixels[src+1]+65536*idPixels[src+2]-1;
  const id=slot<0?0:labels[slot];im.data[dst]=id&255;im.data[dst+1]=id>>8;im.data[dst+3]=255;
 }
 ctx.putImageData(im,0,0);return c.toDataURL('image/png').split(',')[1];
}
window.renderComparison=async index=>{
 const f=track.frames[index], e=new Matrix4().set(...f.extrinsic.flat());
 const cvToGl=new Matrix4().makeScale(1,-1,-1);
 camera.matrixWorldInverse.copy(cvToGl.multiply(e));camera.matrixWorld.copy(camera.matrixWorldInverse).invert();camera.matrix.copy(camera.matrixWorld);
 const k=f.intrinsic,w=track.width,h=track.height,n=.01,far=100;
 camera.projectionMatrix.set(2*k[0][0]/w,0,1-2*k[0][2]/w,0, 0,2*k[1][1]/h,2*k[1][2]/h-1,0, 0,0,-(far+n)/(far-n),-2*far*n/(far-n), 0,0,-1,0);
 camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
 gl.finish();const t=performance.now();surfel.render();gl.finish();const renderMs=performance.now()-t;
 if(report.firstRenderedMs===undefined)report.firstRenderedMs=performance.now()-start;
 const color=renderer.domElement.toDataURL('image/png').split(',')[1];
 capture=true;surfel.pick(.5,.5);capture=false;
 const ids=pngIds();
 report.frames.push({index,renderMs});return {color,ids,renderMs};
};
await window.renderComparison(0);report.firstFrameMs=performance.now()-start;
report.resources=performance.getEntriesByType('resource').map(x=>({name:x.name,bytes:x.encodedBodySize,duration:x.duration}));
window.comparisonReport=report;window.comparisonReady=true;

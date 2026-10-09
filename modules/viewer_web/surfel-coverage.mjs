import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);
const defaultRoot = new URL(location.href); defaultRoot.port = '8769'; defaultRoot.pathname = '/'; defaultRoot.search = '';
const assets = new URL(new URLSearchParams(location.search).get('assets') || defaultRoot.href, location.href);
if (!assets.pathname.endsWith('/')) assets.pathname += '/';
const panels = [];
const sourceColor = frame => frame < 0 ? new THREE.Color('#888888') : new THREE.Color().setHSL((frame * .61803398875) % 1, .72, .58);
let manifest, synchronizing = false, busy = false;
window.coverageReady = false; window.coverageError = null;
function fail(error) {
  window.coverageError = error.stack || String(error); $('error').hidden = false;
  $('error').textContent = window.coverageError; $('status').textContent = 'Loading or rendering failed'; console.error(error);
}
window.addEventListener('error', event => fail(event.error || event.message));
window.addEventListener('unhandledrejection', event => fail(event.reason));
const render = () => panels.forEach(p => p.renderer.render(p.scene, p.camera));
function sync(source) {
  if (synchronizing) return;
  synchronizing = true;
  for (const p of panels) if (p !== source) {
    p.camera.position.copy(source.camera.position); p.camera.quaternion.copy(source.camera.quaternion);
    p.camera.up.copy(source.camera.up); p.camera.zoom = source.camera.zoom; p.camera.updateProjectionMatrix();
    p.controls.target.copy(source.controls.target); p.controls.update();
  }
  render(); synchronizing = false;
}
function dispose(object) {
  if (!object) return;
  const textures = new Set();
  object.traverse(mesh => {
    if (!mesh.isMesh) return;
    mesh.geometry.dispose();
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      if (m.userData.originalMap) textures.add(m.userData.originalMap); m.dispose();
    }
  });
  textures.forEach(texture => texture.dispose());
}
async function load(method) {
  const response = await fetch(new URL(`${method}/model.glb`, assets));
  if (!response.ok) throw Error(`${method}: HTTP ${response.status}`);
  const object = (await new GLTFLoader().parseAsync(await response.arrayBuffer(), assets.href)).scene;
  object.traverse(mesh => {
    if (!mesh.isMesh) return;
    const convert = original => {
      const frame = Number(original.userData?.source_frame ?? original.name.match(/frame_(-?\d+)/)?.[1] ?? -1);
      const m = new THREE.MeshBasicMaterial({color:original.color, map:original.map, side:original.side});
      m.userData = {frame, originalMap:original.map, originalColor:m.color.clone()}; original.dispose(); return m;
    };
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(convert) : convert(mesh.material);
  });
  return object;
}
function colorize(enabled) {
  for (const p of panels) p.object?.traverse(mesh => {
    if (!mesh.isMesh) return;
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      m.color.copy(enabled ? sourceColor(m.userData.frame) : m.userData.originalColor);
      m.map = enabled ? null : m.userData.originalMap; m.needsUpdate = true;
    }
  });
  $('source-color').checked = enabled; render();
}
function preset(name) {
  const close = name.startsWith('close');
  const center = new THREE.Vector3(...(close ? manifest.focus.center : manifest.center));
  const up = new THREE.Vector3(...manifest.up).normalize();
  const cameraOrigin = new THREE.Vector3(...manifest.camera_positions[0]);
  const direction = cameraOrigin.sub(center).normalize();
  if (name === 'side60' || name === 'close-side') direction.applyAxisAngle(up, Math.PI / 3);
  if (!['front','side60','close','close-side'].includes(name)) throw Error(`Unknown preset ${name}`);
  const distance = close ? manifest.focus.radius * 3 : manifest.radius * 2.3;
  const p = panels[0]; p.camera.position.copy(center).addScaledVector(direction, distance);
  p.camera.up.copy(up); p.camera.zoom = 1; p.camera.updateProjectionMatrix();
  p.controls.target.copy(center); p.controls.update(); sync(p); $('preset').value = name;
}
function metricText(method) {
  const r = manifest.results[method], p = r.evaluation.pooled;
  const pc = value => `${(100 * value).toFixed(2)}%`;
  return `${r.stats.points.toLocaleString('en-US')} points · Gray area ${pc(r.stats.texture.untextured_area_fraction)} · Depth-supported pixels ${pc(p.depth_supported_coverage)} · Textured supported pixels ${pc(p.textured_supported_coverage)}`;
}
async function selectTrial(method) {
  if (busy) throw Error('Trial load already in progress');
  if (!manifest.trials[method]) throw Error(`Unknown trial ${method}`);
  busy = true; window.coverageReady = false; $('trial').disabled = true; $('status').textContent = 'Loading…';
  try {
    const trial = manifest.trials[method];
    const methods = [trial.reference, method];
    for (const [i, p] of panels.entries()) {
      if (p.method === methods[i]) continue;
      if (p.object) { p.scene.remove(p.object); dispose(p.object); p.object = null; }
      p.object = await load(methods[i]); p.scene.add(p.object); p.method = methods[i];
    }
    $('reference-title').textContent = trial.reference_label; $('candidate-title').textContent = trial.label;
    $('description').textContent = trial.description;
    $('reference-metrics').textContent = metricText(trial.reference); $('candidate-metrics').textContent = metricText(method);
    $('details').textContent = JSON.stringify({reference:manifest.results[trial.reference], candidate:manifest.results[method]},null,2);
    $('trial').value = method; colorize($('source-color').checked); preset($('preset').value);
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    render(); window.coverageReady = true; $('status').textContent = 'Both views loaded';
  } finally { busy = false; $('trial').disabled = false; }
}
async function main() {
  const response = await fetch(new URL('manifest.json', assets));
  if (!response.ok) throw Error(`manifest: HTTP ${response.status}`);
  manifest = await response.json(); $('observations').src = new URL('observations.jpg', assets).href;
  for (const id of ['reference','candidate']) {
    const host = $(id), scene = new THREE.Scene(); scene.background = new THREE.Color('#15202d');
    const renderer = new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(1); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
    host.append(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost', e => {e.preventDefault(); fail(Error(`${id}: WebGL context lost`));});
    const camera = new THREE.PerspectiveCamera(40,1,.0001,1000);
    const controls = new OrbitControls(camera,renderer.domElement); controls.enableDamping = false; controls.minDistance = .02;
    const panel = {id,scene,renderer,camera,controls}; panels.push(panel);
    controls.addEventListener('change', () => sync(panel));
    new ResizeObserver(() => {renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();render();}).observe(host);
  }
  $('trial').onchange = e => selectTrial(e.target.value).catch(fail);
  $('preset').onchange = e => preset(e.target.value);
  $('source-color').onchange = e => colorize(e.target.checked);
  $('reset').onclick = () => preset($('preset').value);
  window.coverageSelect = selectTrial;
  window.coverageStates = () => panels.map(p => ({method:p.method,position:p.camera.position.toArray(),quaternion:p.camera.quaternion.toArray(),target:p.controls.target.toArray()}));
  window.coverageCapture = async (name, source) => {
    preset(name); colorize(source); await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))); render();
    return panels.map(p => ({method:p.method,png:p.renderer.domElement.toDataURL('image/png').split(',')[1]}));
  };
  window.coverageRenderers = () => panels.map(p => {const gl=p.renderer.getContext(),ext=gl.getExtension('WEBGL_debug_renderer_info');return ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);});
  await selectTrial(new URLSearchParams(location.search).get('trial') || 'adaptive');
}
main().catch(fail);

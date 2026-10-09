import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadViewerBundle } from './src/bundle-loader.ts';
import { loadSurfels } from './src/surfel-loader.ts';
import { createSurfelRenderer } from './src/surfel-renderer.ts';

const $ = id => document.getElementById(id);
const panels = [];
const camera = new THREE.PerspectiveCamera(40, 1, .001, 1000);
const stats = { glbResourceUrls: [], errors: [] };
let surfel, sync = false, center, sourcePosition, sourceForward, up, radius;
window.glbComparisonReady = false;
window.glbComparisonError = null;
function fail(error) {
  const message = error?.stack || String(error);
  window.glbComparisonError = message; stats.errors.push(message);
  $('error').hidden = false; $('error').textContent = message; $('status').textContent = '加载或渲染失败';
  console.error(error);
}
window.addEventListener('error', event => fail(event.error || event.message));
window.addEventListener('unhandledrejection', event => fail(event.reason));
function render() {
  if (!surfel) return;
  camera.updateMatrixWorld(); surfel.render();
  panels[1].renderer.render(panels[1].scene, camera);
}
function changed(panel) {
  if (sync) return;
  sync = true;
  for (const other of panels) if (other !== panel) {
    other.controls.target.copy(panel.controls.target); other.controls.update();
  }
  render(); sync = false;
}
function setView(name) {
  if (!['front', 'side30', 'source0'].includes(name)) throw Error(`Unknown view: ${name}`);
  const direction = sourcePosition.clone().sub(center).normalize();
  if (name === 'side30') direction.applyAxisAngle(up, Math.PI / 6);
  camera.up.copy(up); camera.zoom = 1;
  const target = center.clone();
  if (name === 'source0') {
    camera.position.copy(sourcePosition); target.copy(sourcePosition).add(sourceForward);
  } else camera.position.copy(center).addScaledVector(direction, radius * 3.2);
  sync = true;
  for (const panel of panels) panel.controls.target.copy(target);
  panels[0].controls.update(); sync = false;
  camera.updateProjectionMatrix(); render(); $('view').value = name;
}
window.setComparisonView = async name => {
  setView(name);
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  render();
  return { position: camera.position.toArray(), quaternion: camera.quaternion.toArray(), target: panels[0].controls.target.toArray() };
};
async function main() {
  const query = new URLSearchParams(location.search);
  if (!query.has('bundle') || !query.has('glb')) throw Error('需要 ?bundle=<含 CURRENT 的目录 URL>&glb=<GLB URL>[&global_id=ID]');
  const bundleUrl = new URL(query.get('bundle'), location.href).href;
  const glbUrl = new URL(query.get('glb'), location.href).href;
  $('download').href = glbUrl;
  $('download').download = new URL(glbUrl).pathname.split('/').pop();
  if (!bundleUrl.endsWith('/')) throw Error('bundle URL 必须以 / 结尾');
  const bundle = await loadViewerBundle(bundleUrl, fetch, 'surfel');
  bundle.surfels = await loadSurfels(bundle);
  const world = new THREE.Matrix4().set(...bundle.manifest.world_to_view);
  const visible = new Uint8Array(bundle.pointCount);
  const id = query.get('global_id');
  const ranges = id === null ? [[0, bundle.pointCount]] : bundle.objects[id]?.point_ranges;
  if (!ranges) throw Error(`Unknown global_id: ${id}`);
  const box = new THREE.Box3(), point = new THREE.Vector3();
  let visiblePoints = 0;
  for (const [start, end] of ranges) {
    visible.fill(1, start, end); visiblePoints += end - start;
    for (let i = start; i < end; i++) box.expandByPoint(point.fromArray(bundle.positions, i * 3).applyMatrix4(world));
  }
  if (!visiblePoints) throw Error('选择的 baseline 没有 Surfel');
  center = box.getCenter(new THREE.Vector3()); radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, .001);
  const extrinsic = bundle.surfels.metadata.frames[0].extrinsic;
  const pose = new THREE.Matrix4().set(...extrinsic.flat(), 0, 0, 0, 1).invert().premultiply(world);
  sourcePosition = new THREE.Vector3().setFromMatrixPosition(pose);
  sourceForward = new THREE.Vector3(0, 0, 1).transformDirection(pose);
  up = new THREE.Vector3(0, -1, 0).transformDirection(pose);
  camera.near = Math.max(radius / 10000, .00001); camera.far = Math.max(radius * 100, sourcePosition.distanceTo(center) * 10);
  for (const id of ['baseline', 'glb']) {
    const host = $(id), renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
    renderer.setClearColor(0xffffff, 1); host.append(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); fail(Error(`${id}: WebGL context lost`)); });
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
    const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = false;
    const panel = { host, renderer, scene, controls }; panels.push(panel);
    controls.addEventListener('change', () => changed(panel));
  }
  // A product comparison draws only its original slots. Hidden shelf slots
  // otherwise still execute the vertex shader, which is costly in SwiftShader.
  let renderBundle = bundle;
  let renderVisibility = visible;
  if (id !== null) {
    const select = (array, width) => {
      const result = new array.constructor(visiblePoints * width);
      let offset = 0;
      for (const [start, end] of ranges) {
        const values = array.subarray(start * width, end * width);
        result.set(values, offset); offset += values.length;
      }
      return result;
    };
    renderBundle = {
      ...bundle, pointCount: visiblePoints, positions: select(bundle.positions, 3),
      surfels: { ...bundle.surfels, u: select(bundle.surfels.u, 3), v: select(bundle.surfels.v, 3),
        scale: select(bundle.surfels.scale, 2), frames: select(bundle.surfels.frames, 1),
        metadata: { ...bundle.surfels.metadata, point_count: visiblePoints } },
    };
    renderVisibility = new Uint8Array(visiblePoints).fill(1);
  }
  surfel = createSurfelRenderer(panels[0].renderer, camera, renderBundle, world, new THREE.BufferAttribute(renderVisibility, 1));
  surfel.setRadius(.07);
  const manager = new THREE.LoadingManager();
  manager.setURLModifier(url => { stats.glbResourceUrls.push(url); return url; });
  manager.onError = url => fail(Error(`GLB resource failed: ${url}`));
  // The generic pane keeps every material and texture returned by GLTFLoader unchanged.
  const gltf = await new GLTFLoader(manager).loadAsync(glbUrl);
  panels[1].scene.add(gltf.scene);
  const glbBounds = new THREE.Box3().setFromObject(gltf.scene);
  if (glbBounds.isEmpty()) throw Error('GLB scene has no geometry bounds');
  center = glbBounds.getCenter(new THREE.Vector3());
  radius = Math.max(glbBounds.getSize(new THREE.Vector3()).length() / 2, .001);
  camera.near = Math.max(radius / 10000, .00001);
  camera.far = Math.max(radius * 100, sourcePosition.distanceTo(center) * 10);
  const textures = new Set(), materials = new Set();
  let triangles = 0, meshes = 0;
  gltf.scene.traverse(object => {
    if (!object.isMesh) return;
    meshes++; triangles += (object.geometry.index?.count ?? object.geometry.attributes.position.count) / 3;
    for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
      materials.add(material); if (material.map) textures.add(material.map);
    }
  });
  const resize = () => {
    for (const panel of panels) panel.renderer.setSize(panel.host.clientWidth, panel.host.clientHeight, false);
    const host = panels[0].host;
    camera.aspect = host.clientWidth / host.clientHeight; camera.updateProjectionMatrix();
    surfel.setSize(host.clientWidth, host.clientHeight); render();
  };
  new ResizeObserver(resize).observe(document.querySelector('.panels')); resize();
  Object.assign(stats, { bundleUrl, glbUrl, globalId: id, visiblePoints, meshes, triangles, textures: textures.size,
    materials: [...materials].map(m => ({ type: m.type, alphaTest: m.alphaTest, side: m.side, transparent: m.transparent })),
    externalGlbResourceUrls: stats.glbResourceUrls.filter(url => url !== glbUrl && !url.startsWith('blob:') && !url.startsWith('data:')),
    renderers: panels.map(({ renderer }) => { const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info'); return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); }) });
  window.glbComparisonStats = stats;
  $('report').textContent = JSON.stringify(stats, null, 2);
  $('view').onchange = event => setView(event.target.value); $('reset').onclick = () => setView($('view').value);
  await window.setComparisonView('front');
  window.glbComparisonReady = true; $('status').textContent = `${visiblePoints.toLocaleString()} Surfels / ${triangles.toLocaleString()} GLB 三角形`;
}
main().catch(fail);

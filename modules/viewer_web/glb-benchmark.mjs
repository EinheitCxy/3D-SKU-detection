import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(800, 600);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.setClearColor(0xffffff, 1);
document.getElementById('viewport').append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 800 / 600, .001, 1000);
let model, cameraSet = false, busy = false;
const gl = renderer.getContext();
const debug = gl.getExtension('WEBGL_debug_renderer_info');
const backend = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
renderer.domElement.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  window.glbBenchmarkError = 'WebGL context lost';
});
function disposeModel() {
  if (!model) return;
  scene.remove(model);
  const geometries = new Set(), materials = new Set(), textures = new Set();
  model.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    for (const material of object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : []) {
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  for (const texture of textures) { texture.dispose(); texture.source?.data?.close?.(); }
  for (const material of materials) material.dispose();
  for (const geometry of geometries) geometry.dispose();
  renderer.renderLists.dispose();
  model = undefined;
}
window.runGlbBenchmark = async (url, cameraPose = null) => {
  if (busy) throw Error('Benchmark already running');
  busy = true;
  try {
    disposeModel();
    window.glbBenchmarkError = null;
    const absoluteUrl = new URL(url, location.href).href;
    const start = performance.now();
    const response = await fetch(absoluteUrl, { cache: 'no-store' });
    if (!response.ok) throw Error(`GLB HTTP ${response.status}: ${absoluteUrl}`);
    const buffer = await response.arrayBuffer();
    const transferMs = performance.now() - start;
    const parseStart = performance.now();
    const gltf = await new GLTFLoader().parseAsync(buffer, new URL('.', absoluteUrl).href);
    const parseMs = performance.now() - parseStart;
    model = gltf.scene;
    scene.add(model);
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    if (box.isEmpty()) throw Error('GLB scene is empty');
    if (!cameraSet) {
      const center = box.getCenter(new THREE.Vector3());
      const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, .001);
      if (cameraPose) {
        for (const key of ['position', 'target', 'up']) {
          if (!Array.isArray(cameraPose[key]) || cameraPose[key].length !== 3 || !cameraPose[key].every(Number.isFinite)) throw Error(`Invalid camera ${key}`);
        }
        if (cameraPose.fov !== undefined && (!Number.isFinite(cameraPose.fov) || cameraPose.fov <= 0 || cameraPose.fov >= 180)) throw Error('Invalid camera fov');
        camera.position.fromArray(cameraPose.position);
        camera.up.fromArray(cameraPose.up);
        camera.fov = cameraPose.fov ?? 40;
        camera.lookAt(new THREE.Vector3().fromArray(cameraPose.target));
      } else {
        camera.position.copy(center).add(new THREE.Vector3(0, 0, radius / Math.sin(THREE.MathUtils.degToRad(20)) * 1.05));
        camera.lookAt(center);
      }
      camera.near = Math.max(radius / 10000, .00001);
      camera.far = Math.max(radius * 100, camera.position.distanceTo(center) + radius * 2);
      camera.updateProjectionMatrix();
      cameraSet = true;
    }
    const textures = new Set(), materialTypes = new Set();
    model.traverse(object => {
      for (const material of object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : []) {
        materialTypes.add(material.type);
        for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
      }
    });
    let estimatedTexturePixels = 0;
    for (const texture of textures) estimatedTexturePixels += (texture.image?.width || 0) * (texture.image?.height || 0);
    const pixel = new Uint8Array(4);
    function frame() {
      const before = performance.now();
      renderer.render(scene, camera);
      // A pixel readback forces completion across Chromium's GPU process;
      // gl.finish alone can return before the queued draws have completed.
      gl.readPixels(400, 300, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      if (gl.isContextLost()) throw Error('WebGL context lost');
      return performance.now() - before;
    }
    const warmupFrameMs = [frame(), frame()];
    const frameMs = [];
    for (let i = 0; i < 5; i++) frameMs.push(frame());
    const result = {
      url: absoluteUrl, bytes: buffer.byteLength, transferMs, parseMs, loadMs: transferMs + parseMs,
      warmupFrameMs, frameMs, medianFrameMs: [...frameMs].sort((a, b) => a - b)[2],
      timingDefinition: 'performance.now around renderer.render plus 1-pixel readPixels; software render completion including readback overhead',
      renderer: backend, realGpuPerformanceClaim: false,
      drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
      textures: textures.size, estimatedTexturePixels, materialTypes: [...materialTypes],
      camera: { position: camera.position.toArray(), quaternion: camera.quaternion.toArray(), near: camera.near, far: camera.far, fov: camera.fov },
      bounds: { min: box.min.toArray(), max: box.max.toArray() }, width: 800, height: 600,
    };
    document.getElementById('status').textContent = JSON.stringify(result, null, 2);
    return result;
  } finally { busy = false; }
};
window.glbBenchmarkReady = true;
const url = new URLSearchParams(location.search).get('glb');
if (url) window.runGlbBenchmark(url).catch(error => {
  window.glbBenchmarkError = error.stack || String(error);
  document.getElementById('status').textContent = window.glbBenchmarkError;
});

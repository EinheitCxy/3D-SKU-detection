import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const methods = [...document.querySelectorAll('#columns .viewport')].map(node => node.id);
if (!methods.length || methods.some(method => !['baseline', 'fused', 'tsdf'].includes(method))) throw Error('Invalid comparison panels');
const english = document.documentElement.lang.startsWith('en');
const tr = (zh, en) => english ? en : zh;
const panels = [];
const $ = id => document.getElementById(id);
const defaultAssets = new URL(location.href); defaultAssets.port = document.body.dataset.assetsPort || '8767'; defaultAssets.pathname = document.body.dataset.assetsPath || '/'; defaultAssets.search = ''; defaultAssets.hash = '';
const root = new URL(new URLSearchParams(location.search).get('assets') || defaultAssets.href, location.href);
if (!root.pathname.endsWith('/')) root.pathname += '/';
window.roiReady = false;
window.roiError = null;
function fail(error) {
  window.roiError = error.stack || String(error);
  $('error').textContent = window.roiError;
  $('error').hidden = false;
  $('status').textContent = tr('加载或渲染失败', 'Loading or rendering failed');
  console.error(error);
}
window.addEventListener('error', event => fail(event.error || event.message));
window.addEventListener('unhandledrejection', event => fail(event.reason));
const sourceColor = frame => frame < 0 ? new THREE.Color('#888888') : new THREE.Color().setHSL((frame * 0.61803398875) % 1, 0.72, 0.58);
const sourceFrame = material => Number(material.userData?.source_frame ?? material.name.match(/frame_(-?\d+)/)?.[1] ?? -1);
function unlit(object, contextual = false) {
  object.traverse(mesh => {
    if (!mesh.isMesh) return;
    const convert = original => {
      const material = new THREE.MeshBasicMaterial({
        color: original.color?.clone() || new THREE.Color('white'), map: original.map,
        vertexColors: original.vertexColors, side: original.side,
        transparent: original.transparent, opacity: original.opacity,
        alphaTest: original.alphaTest, depthWrite: original.depthWrite,
      });
      material.name = original.name;
      material.userData = { sourceFrame: sourceFrame(original), originalColor: material.color.clone(), originalMap: material.map, originalVertexColors: material.vertexColors, contextual };
      return material;
    };
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(convert) : convert(mesh.material);
  });
}
function render() { for (const p of panels) p.renderer.render(p.scene, p.camera); }
let synchronizing = false;
function synchronize(source) {
  if (synchronizing) return;
  synchronizing = true;
  try {
    for (const p of panels) {
      if (p === source) continue;
      p.camera.position.copy(source.camera.position);
      p.camera.quaternion.copy(source.camera.quaternion);
      p.camera.up.copy(source.camera.up);
      p.camera.zoom = source.camera.zoom;
      p.camera.updateProjectionMatrix();
      p.controls.target.copy(source.controls.target);
      p.controls.update();
    }
    render();
  } finally { synchronizing = false; }
}
async function loadGLB(name, optional = false) {
  const url = new URL(name, root);
  const response = await fetch(url);
  if (optional && response.status === 404) return null;
  if (!response.ok) throw Error(`${name}: HTTP ${response.status}`);
  return (await new GLTFLoader().parseAsync(await response.arrayBuffer(), root.href)).scene;
}
async function main() {
  const response = await fetch(new URL('meta.json', root));
  if (!response.ok) throw Error(`meta.json: HTTP ${response.status}`);
  const meta = await response.json();
  if (!english && meta.title) { document.querySelector('h1').textContent = meta.title; document.title = meta.title; }
  if (!english && meta.notice) document.querySelector('.notice').textContent = meta.notice;
  for (const method of methods) if (!english && meta.method_labels?.[method]) {
    $(method).closest('article').querySelector('h2').textContent = meta.method_labels[method];
  }
  if (!Array.isArray(meta.center) || meta.center.length !== 3 || !meta.center.every(Number.isFinite) || !Number.isFinite(meta.radius) || meta.radius <= 0) throw Error(tr('meta.json 需要有效 center[3] 和正数 radius', 'meta.json requires a valid center[3] and positive radius'));
  const center = new THREE.Vector3(...meta.center);
  const up = new THREE.Vector3(...(meta.up || meta.camera_up || [0, 1, 0])).normalize();
  const positions = meta.camera_positions || [];
  const ids = meta.source_image_ids || positions.map((_, i) => i);
  for (let i = 0; i < positions.length; i++) {
    const option = document.createElement('option');
    option.value = `source:${i}`; option.textContent = tr(`帧 ${ids[i]} 相机位置（朝向展示中心）`, `Frame ${ids[i]} camera position (looking at scene center)`); $('preset').append(option);
    const item = document.createElement('span');
    const swatch = document.createElement('i'); swatch.className = 'swatch'; swatch.style.background = `#${sourceColor(i).getHexString()}`;
    item.append(swatch, document.createTextNode(String(ids[i]))); $('legend').append(item);
  }
  if (positions.length >= 2) {
    const option = document.createElement('option'); option.value = 'midpoint01'; option.textContent = tr(`帧 ${ids[0]} / ${ids[1]} 相机位置中点`, `Midpoint of frames ${ids[0]} and ${ids[1]}`); $('preset').append(option);
  }
  const blank = document.createElement('span'); blank.textContent = tr('■ 灰色：未贴图', '■ Gray: no texture'); $('legend').append(blank);
  const objects = await Promise.all(methods.map(method => loadGLB(`${method}.glb`)));
  const context = await loadGLB('context.glb', true);
  const stats = { metadata: meta, methods: {}, renderers: [], rendererContract: 'Common unlit experimental disc/mesh renderer; not production surfel shader; no performance claim.' };
  for (const [index, method] of methods.entries()) {
    const host = $(method);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1); renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
    host.append(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); fail(Error(`${method}: WebGL context lost`)); });
    const scene = new THREE.Scene(); scene.background = new THREE.Color('#15202d');
    const camera = new THREE.PerspectiveCamera(40, 1, Math.max(meta.radius / 1000, 0.00001), Math.max(meta.radius * 1000, 100));
    camera.up.copy(up);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false; controls.target.copy(center); controls.minDistance = meta.radius * 0.15; controls.maxDistance = meta.radius * 100;
    const object = objects[index]; unlit(object); scene.add(object);
    const surroundings = context?.clone(true);
    if (surroundings) { unlit(surroundings, true); surroundings.visible = false; scene.add(surroundings); }
    let triangles = 0, meshes = 0;
    object.traverse(mesh => { if (mesh.isMesh) { meshes++; triangles += (mesh.geometry.index?.count ?? mesh.geometry.attributes.position.count) / 3; } });
    stats.methods[method] = { meshes, triangles, ...(meta.stats?.[method] || {}) };
    const methodStats = stats.methods[method];
    const area = methodStats.texture?.untextured_area_fraction;
    const spread = methodStats.geometry_proxy?.normal_spread_median_m;
    $(`${method}-summary`).textContent = english
      ? `Triangles ${triangles.toLocaleString('en-US')} · Untextured area ${Number.isFinite(area) ? `${(area * 100).toFixed(1)}%` : 'N/A'}${Number.isFinite(spread) ? ` · Median normal spread ${(spread * 1000).toFixed(2)} mm` : ''}`
      : `三角形 ${triangles.toLocaleString('zh-CN')} · 未贴图面积 ${Number.isFinite(area) ? `${(area * 100).toFixed(1)}%` : '未提供'} · 法向分散中位数 ${Number.isFinite(spread) ? `${(spread * 1000).toFixed(2)} mm` : '未提供'}`;
    $(`${method}-stats`).textContent = JSON.stringify(stats.methods[method], null, 2);
    const gl = renderer.getContext(), ext = gl.getExtension('WEBGL_debug_renderer_info');
    stats.renderers.push(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    const panel = { method, renderer, camera, controls, scene, object, context: surroundings }; panels.push(panel);
    controls.addEventListener('change', () => synchronize(panel));
    new ResizeObserver(() => {
      const width = host.clientWidth, height = host.clientHeight;
      renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); render();
    }).observe(host);
  }
  const front = positions.length ? new THREE.Vector3(...positions[0]).sub(center).normalize() : new THREE.Vector3(0, 0, 1);
  const preset = name => {
    if (name === 'source0') name = 'source:0';
    const p = panels[0];
    if (name === 'midpoint01') {
      if (positions.length < 2) throw Error(tr('midpoint01 需要两个来源相机位置', 'midpoint01 requires two source camera positions'));
      p.camera.position.set(...positions[0]).add(new THREE.Vector3(...positions[1])).multiplyScalar(0.5);
    } else if (name.startsWith('source:')) {
      const index = Number(name.slice(7));
      if (!Number.isInteger(index) || !positions[index]) throw Error(tr(`未知来源视角 ${name}`, `Unknown source view ${name}`));
      p.camera.position.set(...positions[index]);
    } else if (['front', 'side', 'side30', 'side60'].includes(name)) {
      const direction = front.clone();
      const degrees = { front: 0, side: 90, side30: 30, side60: 60 }[name];
      direction.applyAxisAngle(up, degrees * Math.PI / 180);
      p.camera.position.copy(center).addScaledVector(direction, meta.radius * (english ? 2.3 : 3.5));
    } else throw Error(tr(`未知视角 ${name}`, `Unknown view ${name}`));
    p.camera.up.copy(up); p.controls.target.copy(center); p.camera.zoom = 1; p.camera.updateProjectionMatrix(); p.controls.update(); synchronize(p);
    $('preset').value = name;
  };
  const colorize = enabled => {
    for (const p of panels) p.object.traverse(mesh => {
      if (!mesh.isMesh) return;
      for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        mat.color.copy(enabled ? sourceColor(mat.userData.sourceFrame) : mat.userData.originalColor);
        mat.map = enabled ? null : mat.userData.originalMap;
        mat.vertexColors = enabled ? false : mat.userData.originalVertexColors;
        mat.needsUpdate = true;
      }
    });
    $('source-color').checked = enabled; $('legend').hidden = !enabled; render();
  };
  $('preset').onchange = event => { try { preset(event.target.value); } catch (error) { fail(error); } };
  $('source-color').onchange = event => colorize(event.target.checked);
  $('context').onchange = event => { for (const p of panels) if (p.context) p.context.visible = event.target.checked; render(); };
  $('reset').onclick = () => preset($('preset').value);
  const references = [...(meta.reference_images || [])];
  if (meta.production_reference) references.push(typeof meta.production_reference === 'string' ? { url: meta.production_reference, label: tr('生产 Surfel 渲染参考', 'Production Surfel reference') } : meta.production_reference);
  for (const [i, reference] of references.entries()) {
    const item = typeof reference === 'string' ? { url: reference } : reference;
    const figure = document.createElement('figure'), img = document.createElement('img'), caption = document.createElement('figcaption');
    img.src = new URL(item.url, root).href; img.alt = english ? `Input frames ${i + 1}` : item.label || `原始参考 ${i + 1}`; img.loading = 'lazy'; img.onerror = () => fail(Error(tr(`参考图像加载失败：${img.src}`, `Reference image failed to load: ${img.src}`)));
    caption.textContent = img.alt; figure.append(img, caption); $('references').append(figure);
  }
  if (!references.length) $('references').textContent = tr('meta.json 暂未提供参考图像。', 'No reference images in meta.json.');
  $('environment').textContent = JSON.stringify({ renderers: stats.renderers, notice: stats.rendererContract }, null, 2);
  $('preset').disabled = false; $('source-color').disabled = false; $('reset').disabled = false; $('context').disabled = !context;
  $('status').textContent = english ? `${methods.length} full-shelf views loaded` : meta.preparation?.scope?.startsWith('whole') ? '三列整场景已加载' : context ? '三列已加载；可切换周围场景' : '三列已加载；未提供周围场景';
  preset('front');
  window.roiStats = stats;
  window.roiCameraStates = () => panels.map(p => ({ method: p.method, position: p.camera.position.toArray(), quaternion: p.camera.quaternion.toArray(), target: p.controls.target.toArray(), zoom: p.camera.zoom, contextVisible: Boolean(p.context?.visible) }));
  window.roiCapture = async (name = 'front', source = false) => {
    if (window.roiError) throw Error(window.roiError);
    preset(name); colorize(source);
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    render();
    return { pngs: panels.map(p => p.renderer.domElement.toDataURL('image/png').split(',')[1]), methods, preset: name, sourceColor: source, stats, context: $('context').checked };
  };
  await new Promise(resolve => requestAnimationFrame(resolve)); render(); window.roiReady = true;
}
main().catch(fail);

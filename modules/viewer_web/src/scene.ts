import {
  ACESFilmicToneMapping,
  AxesHelper,
  Box3,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  Fog,
  GridHelper,
  Group,
  Int8BufferAttribute,
  Matrix3,
  Matrix4,
  PerspectiveCamera,
  Points,
  Raycaster,
  Scene,
  SRGBColorSpace,
  ShaderMaterial,
  UniformsLib,
  UniformsUtils,
  Float32BufferAttribute,
  Uint8BufferAttribute,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Mesh } from "three";
import type { ViewerBundle } from "./bundle-loader";
import { POINTS_LAYER, createViewerPipeline, type ViewerPipeline } from "./edl";
import { createFootprintObjects } from "./footprints";
import { focusedCameraPosition } from "./focus";
import { cachedSelectionBox } from "./selection-bounds";
import { applySelectionColors, queueSelectionAttributeUpdates, type PointRange } from "./selection-colors";

const FOOTPRINT_CLICK_THRESHOLD_PX = 6;
const MIN_POINT_SIZE = 0.004;
const MAX_POINT_SIZE = 0.07;
const DEFAULT_POINT_SIZE = 0.015;
const MAX_SPLAT_PIXELS = 64;
const FOG_NEAR_RADII = 1.0;
const FOG_FAR_RADII = 8.0;
const LIGHT_DIRECTION = new Vector3(0.45, 0.75, 0.5).normalize();
const MIN_FOOTPRINT_OPACITY = 0.2;
const VIEW_TRANSITION_MS = 420;
const DEFAULT_BASE_FILL_OPACITY = 0.38;
const DEFAULT_BASE_OUTLINE_OPACITY = 0.95;
const FOCUS_OUTLINE_OPACITY = 1;
const SELECTED_FILL_OPACITY = 0.72;
const UNSELECTED_FILL_OPACITY = 0.12;
const SELECTION_BOX_PADDING_RATIO = 0.02;

export interface ViewerSceneController {
  selectGlobalId(globalId: string | null): void;
  focusGlobalId(globalId: string): void;
  setPointSize(pointSize: number): void;
  setFootprintOpacity(opacity: number): void;
  setViewPreset(preset: "fit" | "top" | "isometric"): void;
  setFootprintPickHandler(handler: ((globalId: string) => void) | null): void;
  dispose(): void;
}

export interface FootprintPointerPress {
  readonly pointerId: number;
  readonly clientX: number;
  readonly clientY: number;
}

export interface FootprintPointerRelease extends FootprintPointerPress {
  readonly button: number;
  readonly isPrimary: boolean;
}

interface FocusAnimation {
  readonly startedAt: number;
  readonly durationMs: number;
  readonly fromPosition: Vector3;
  readonly toPosition: Vector3;
  readonly fromTarget: Vector3;
  readonly toTarget: Vector3;
}

export function isFootprintClickRelease(press: FootprintPointerPress | null, release: FootprintPointerRelease): boolean {
  return press !== null
    && release.isPrimary
    && release.button === 0
    && release.pointerId === press.pointerId
    && Math.hypot(release.clientX - press.clientX, release.clientY - press.clientY) <= FOOTPRINT_CLICK_THRESHOLD_PX;
}

export function createViewerScene(container: HTMLElement, bundle: ViewerBundle): ViewerSceneController {
  const scene = new Scene();
  const background = new Color("#071015");
  scene.background = background;
  const camera = new PerspectiveCamera(50, 1, 0.01, 10000);
  const renderer = new WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.append(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;

  // All spatial content (points, footprints, helpers) lives under worldGroup in the
  // bundle's native DA3 OpenCV coordinates; the group applies manifest.world_to_view
  // once. three.js Matrix4.set() takes row-major arguments, matching the manifest
  // (row-major) convention; Matrix4.fromArray() would be column-major instead.
  const worldGroup = new Group();
  worldGroup.name = "world";
  const w2v = bundle.manifest.world_to_view;
  worldGroup.matrix.set(
    w2v[0], w2v[1], w2v[2], w2v[3],
    w2v[4], w2v[5], w2v[6], w2v[7],
    w2v[8], w2v[9], w2v[10], w2v[11],
    w2v[12], w2v[13], w2v[14], w2v[15],
  );
  worldGroup.matrixAutoUpdate = false;
  scene.add(worldGroup);

  const points = createPoints(bundle, worldGroup.matrix);
  points.layers.set(POINTS_LAYER);
  const pointMaterial = points.material as ShaderMaterial;
  const pointColorAttribute = points.geometry.getAttribute("aColor") as Uint8BufferAttribute;
  // bundle.colors stays pristine as the restore source; the attribute owns a copy.
  const originalPointColors = bundle.colors;
  worldGroup.add(points);
  const bounds = bundle.manifest.display_bounds;
  const box = new Box3(
    new Vector3(bounds[0], bounds[1], bounds[2]),
    new Vector3(bounds[3], bounds[4], bounds[5]),
  );
  const worldBox = box.clone().applyMatrix4(worldGroup.matrix);
  const sceneSpan = Math.max(worldBox.getSize(new Vector3()).length(), 1);
  const grid = new GridHelper(sceneSpan * 2, 20, "#294550", "#18313a");
  // grid lives under worldGroup in bundle (y-down) coordinates: the ground is
  // the LARGEST y here, not the smallest.
  grid.position.y = box.max.y;
  const axesHelper = new AxesHelper(sceneSpan * 0.2);
  worldGroup.add(grid, axesHelper);

  const footprints = createFootprintObjects(bundle.footprints);
  worldGroup.add(footprints.group);

  const sceneCenter = worldBox.getCenter(new Vector3());
  const sceneRadius = Math.max(worldBox.getSize(new Vector3()).length() * 0.5, 1);
  // Bounded far plane + distance fog from the scene bounding box give EDL a
  // workable depth precision and a depth cue; the fog color matches the
  // background so the fade blends into the void instead of a visible band.
  camera.far = Math.max(sceneRadius * 20, 100);
  camera.updateProjectionMatrix();
  scene.fog = new Fog(background, sceneRadius * FOG_NEAR_RADII, sceneRadius * FOG_FAR_RADII);
  const pipeline = createViewerPipeline(renderer, scene, camera);

  let currentSelection: string | null = null;
  let footprintOpacityScale = 1;
  let selectedGlobalIdForCamera: string | null = null;
  let pickHandler: ((globalId: string) => void) | null = null;
  let primaryPointerPress: FootprintPointerPress | null = null;
  let focusAnimation: FocusAnimation | null = null;
  let animationFrame = 0;
  let previousTintedRanges: readonly PointRange[] = [];
  const selectionBoxCache = new Map<string, Box3 | null>();

  const raycaster = new Raycaster();
  const pointer = new Vector2();
  applySelectionVisuals(currentSelection);
  updateSelectionPointTint(currentSelection);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    pipeline.setPixelRatio(renderer.getPixelRatio());
    pipeline.setSize(width, height);
    const drawingSize = renderer.getDrawingBufferSize(new Vector2());
    pointMaterial.uniforms.uResolution.value.copy(drawingSize);
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  const onPointerDown = (event: PointerEvent) => {
    if (event.isPrimary && event.button === 0) {
      primaryPointerPress = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY };
    }
  };
  const onPointerCancel = (event: PointerEvent) => {
    if (event.isPrimary && primaryPointerPress?.pointerId === event.pointerId) primaryPointerPress = null;
  };
  const onPointerUp = (event: PointerEvent) => {
    const press = primaryPointerPress;
    if (event.isPrimary && press?.pointerId === event.pointerId) primaryPointerPress = null;
    if (!isFootprintClickRelease(press, event)) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(footprints.pickMeshes, false)[0];
    const globalId = hit?.object.userData.globalId;
    if (typeof globalId === "string") pickHandler?.(globalId);
  };
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("pointercancel", onPointerCancel);
  controls.addEventListener("start", () => { focusAnimation = null; });

  const animate = (time: number) => {
    if (focusAnimation !== null) {
      const progress = Math.min((time - focusAnimation.startedAt) / focusAnimation.durationMs, 1);
      const eased = 1 - (1 - progress) ** 3;
      camera.position.lerpVectors(focusAnimation.fromPosition, focusAnimation.toPosition, eased);
      controls.target.lerpVectors(focusAnimation.fromTarget, focusAnimation.toTarget, eased);
      if (progress === 1) focusAnimation = null;
    }
    controls.update();
    pipeline.composer.render();
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);

  setViewPreset("fit", false);

  return {
    selectGlobalId(globalId) {
      currentSelection = globalId;
      selectedGlobalIdForCamera = globalId;
      applySelectionVisuals(globalId);
      updateSelectionPointTint(globalId);
    },
    focusGlobalId(globalId) {
      const footprint = footprints.focusTargets.get(globalId);
      let target: Vector3;
      let targetRadius: number;
      let minDistance: number;
      if (footprint !== undefined) {
        const targetBox = new Box3().setFromObject(footprint);
        target = targetBox.getCenter(new Vector3());
        targetRadius = Math.max(targetBox.getSize(new Vector3()).length() * 0.8, sceneSpan * 0.08, 0.3);
        minDistance = 0.9 * sceneRadius;
      } else {
        // Rejected/missing footprints leave focusTargets empty; fall back to the
        // selected points' box (bundle coords -> world) so camera focus still
        // works — footprint status must not gate selection focus.
        const pointBox = computeSelectionBox(globalId);
        if (pointBox === null) return;
        pointBox.applyMatrix4(worldGroup.matrix);
        target = pointBox.getCenter(new Vector3());
        targetRadius = Math.max(pointBox.getSize(new Vector3()).length() * 0.8, 0.15);
        minDistance = 0.45 * sceneRadius;
      }
      const direction = camera.position.clone().sub(controls.target).lengthSq() > 0
        ? camera.position.clone().sub(controls.target).normalize()
        : sceneCenter.clone().sub(target).normalize();
      const distance = Math.max(targetRadius / Math.sin((camera.fov * Math.PI) / 360), minDistance);
      const focused = focusedCameraPosition(
        [camera.position.x, camera.position.y, camera.position.z],
        [controls.target.x, controls.target.y, controls.target.z],
        [target.x, target.y, target.z],
        distance * 1.1,
      );
      const toPosition = new Vector3(...focused);
      animateToView(toPosition, target, VIEW_TRANSITION_MS, "focus");
      selectedGlobalIdForCamera = globalId;
    },
    setPointSize(size) {
      const next = clamp(size, MIN_POINT_SIZE, MAX_POINT_SIZE);
      pointMaterial.uniforms.uSize.value = next;
    },
    setFootprintOpacity(opacity) {
      footprintOpacityScale = clamp(opacity, 0, 1);
      applySelectionVisuals(currentSelection);
    },
    setViewPreset(preset) {
      setViewPreset(preset, true);
    },
    setFootprintPickHandler(handler) {
      pickHandler = handler;
    },
    dispose() {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointercancel", onPointerCancel);
      controls.dispose();
      pipeline.dispose();
      scene.traverse((object) => {
        const geometry = (object as { geometry?: BufferGeometry }).geometry;
        geometry?.dispose();
        const material = (object as { material?: unknown }).material;
        const materials = Array.isArray(material) ? material : [material];
        materials.forEach((item) => (item as { dispose?: () => void } | undefined)?.dispose?.());
      });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };

  /**
   * 1/99-percentile bounding box of the selected global ID's points, in bundle
   * coordinates. Used only as the camera-focus fallback when the footprint has
   * no focus target (rejected/missing).
   */
  function computeSelectionBox(globalId: string): Box3 | null {
    return cachedSelectionBox(selectionBoxCache, globalId, () => computeSelectionBoxUncached(globalId));
  }

  function computeSelectionBoxUncached(globalId: string): Box3 | null {
    const entry = bundle.objects[globalId];
    if (entry === undefined) return null;
    const axes: number[][] = [[], [], []];
    for (const instance of entry.instances) {
      const [start, end] = instance.point_index_range;
      for (let index = start * 3; index < end * 3; index += 3) {
        axes[0].push(bundle.positions[index]);
        axes[1].push(bundle.positions[index + 1]);
        axes[2].push(bundle.positions[index + 2]);
      }
    }
    if (axes[0].length === 0) return null;
    const min: number[] = [];
    const max: number[] = [];
    for (const axis of axes) {
      axis.sort((left, right) => left - right);
      min.push(quantile(axis, 0.01));
      max.push(quantile(axis, 0.99));
    }
    const box = new Box3(
      new Vector3(min[0], min[1], min[2]),
      new Vector3(max[0], max[1], max[2]),
    );
    box.expandByScalar(box.getSize(new Vector3()).length() * SELECTION_BOX_PADDING_RATIO);
    return box;
  }

  function updateSelectionPointTint(globalId: string | null): void {
    const colorArray = pointColorAttribute.array as Uint8Array;
    const entry = globalId === null ? undefined : bundle.objects[globalId];
    const ranges = entry?.instances.map((instance) => instance.point_index_range).filter(([start, end]) => end > start) ?? [];
    const changed = applySelectionColors(colorArray, originalPointColors, previousTintedRanges, ranges);
    previousTintedRanges = ranges;
    queueSelectionAttributeUpdates(pointColorAttribute, changed);
  }

  function applySelectionVisuals(selectedId: string | null): void {
    for (const [footprintId, visual] of footprints.selectionVisuals) {
      const isSelected = selectedId !== null && footprintId === selectedId;
      const fillOpacity = selectedId === null ? DEFAULT_BASE_FILL_OPACITY : isSelected ? SELECTED_FILL_OPACITY : UNSELECTED_FILL_OPACITY;
      const outlineOpacity = selectedId === null ? DEFAULT_BASE_OUTLINE_OPACITY : isSelected ? FOCUS_OUTLINE_OPACITY : UNSELECTED_FILL_OPACITY * 1.5;
      const fillScale = fillOpacity * Math.max(footprintOpacityScale, MIN_FOOTPRINT_OPACITY);
      const outlineScale = outlineOpacity * Math.max(footprintOpacityScale, MIN_FOOTPRINT_OPACITY);
      visual.fills.forEach((material) => {
        material.opacity = Math.min(fillScale, 1);
      });
      visual.outlines.forEach((material) => {
        material.opacity = Math.min(outlineScale, 1);
      });
    }
  }

  function setViewPreset(preset: "fit" | "top" | "isometric", animate: boolean): void {
    if (preset === "fit") {
      setView(controlsTarget(), cameraPositionForDirection(new Vector3(1, 0.7, 1)), animate);
      return;
    }
    if (preset === "top") {
      setView(controlsTarget(), cameraPositionForDirection(new Vector3(0, 1, 0), sceneRadius * 1.15), animate);
      return;
    }
    setView(controlsTarget(), cameraPositionForDirection(new Vector3(1, 1, 0.6)), animate);
  }

  function controlsTarget(): Vector3 {
    const target = selectedGlobalIdForCamera === null ? sceneCenter : getSelectionTarget();
    return target;
  }

  function cameraPositionForDirection(direction: Vector3, distanceMultiplier = 1.5): Vector3 {
    const base = controlsTarget();
    const normalized = direction.lengthSq() === 0 ? new Vector3(1, 0.7, 1) : direction.clone().normalize();
    const distance = Math.max(sceneRadius / Math.tan((camera.fov * Math.PI) / 360) * distanceMultiplier, 1);
    const offset = normalized
      .clone()
      .multiplyScalar(distance === 0 ? 1 : distance);
    if (normalized.y > 0.99) {
      offset.x = sceneSpan * 0.01;
      offset.z = sceneSpan * 0.01;
    }
    return base.clone().add(offset);
  }

  function setView(target: Vector3, position: Vector3, doAnimate: boolean): void {
    if (doAnimate) {
      animateToView(position, target, VIEW_TRANSITION_MS, "preset");
      return;
    }
    controls.target.copy(target);
    camera.position.copy(position);
    camera.lookAt(target);
    controls.update();
  }

  function getSelectionTarget(): Vector3 {
    if (selectedGlobalIdForCamera === null) return sceneCenter;
    const targetObject = footprints.focusTargets.get(selectedGlobalIdForCamera);
    if (targetObject === undefined) return sceneCenter;
    return targetObject.getWorldPosition(new Vector3());
  }

  function animateToView(position: Vector3, target: Vector3, durationMs: number, source: "focus" | "preset"): void {
    const hasSelection = source === "focus" && selectedGlobalIdForCamera !== null;
    focusAnimation = {
      startedAt: performance.now(),
      durationMs: Math.max(durationMs, 1),
      fromPosition: camera.position.clone(),
      toPosition: position.clone(),
      fromTarget: controls.target.clone(),
      toTarget: target.clone(),
    };
    if (!hasSelection && source === "preset") {
      selectedGlobalIdForCamera = null;
    }
  }
}

const POINT_VERTEX_SHADER = /* glsl */ `
attribute vec3 aColor;
attribute vec3 aNormal;
uniform float uSize;
uniform vec2 uResolution;
uniform mat3 uNormalMatrix;
varying vec3 vColor;
varying vec3 vNormal;
#include <fog_pars_vertex>
// colors.u8.bin holds sRGB bytes from the camera JPEGs; three's working space
// is linear, so decode here before lighting and the OutputPass re-encode.
vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}
void main() {
  vColor = srgbToLinear(aColor);
  vNormal = uNormalMatrix * aNormal;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  // World-size splats: pixels = size * viewportHeight * proj[1][1] / clipW,
  // clamped so close-up points cannot explode (three PR#29474 semantics).
  gl_PointSize = clamp(
    uSize * uResolution.y * projectionMatrix[1][1] / gl_Position.w,
    1.0, ${MAX_SPLAT_PIXELS.toFixed(1)}
  );
  #include <fog_vertex>
}
`;

const POINT_FRAGMENT_SHADER = /* glsl */ `
uniform vec3 uLightDir;
varying vec3 vColor;
varying vec3 vNormal;
#include <fog_pars_fragment>
void main() {
  // Circular splat: discard outside the inscribed circle (three points_waves).
  vec2 centered = gl_PointCoord - vec2(0.5);
  if (dot(centered, centered) > 0.25) discard;
  vec3 normal = normalize(vNormal);
  float halfLambert = dot(normal, uLightDir) * 0.5 + 0.5;
  gl_FragColor = vec4(vColor * (0.6 + 0.4 * halfLambert), 1.0);
  #include <fog_fragment>
}
`;

export function createPoints(bundle: ViewerBundle, worldMatrix: Matrix4): Points {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(bundle.positions, 3));
  // The attribute takes a copy of the bundle colors: selection tinting mutates
  // the attribute array in place, while bundle.colors remains the restore source.
  geometry.setAttribute("aColor", new Uint8BufferAttribute(bundle.colors.slice(), 3, true).setUsage(DynamicDrawUsage));
  // int8-quantized unit normals; normalized attributes decode to [-1, 1].
  geometry.setAttribute("aNormal", new Int8BufferAttribute(bundle.normals, 3, true));
  // worldGroup.matrixAutoUpdate is false, so the fixed world_to_view rotation is
  // precomputed once as the normal matrix (rotation is orthonormal).
  const normalMatrix = new Matrix3().setFromMatrix4(worldMatrix);
  const material = new ShaderMaterial({
    uniforms: UniformsUtils.merge([
      UniformsLib.fog,
      {
        uSize: { value: DEFAULT_POINT_SIZE },
        uResolution: { value: new Vector2(1, 1) },
        uNormalMatrix: { value: normalMatrix },
        uLightDir: { value: LIGHT_DIRECTION.clone() },
      },
    ]),
    vertexShader: POINT_VERTEX_SHADER,
    fragmentShader: POINT_FRAGMENT_SHADER,
    fog: true,
  });
  const points = new Points(geometry, material);
  points.name = "da3-rgb-points";
  return points;
}

function quantile(sorted: readonly number[], q: number): number {
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  if (base + 1 < sorted.length) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

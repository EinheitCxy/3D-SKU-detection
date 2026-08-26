import {
  ACESFilmicToneMapping,
  AxesHelper,
  Box3,
  BufferGeometry,
  Color,
  DynamicDrawUsage,
  Float32BufferAttribute,
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
  ShaderMaterial,
  SRGBColorSpace,
  UniformsLib,
  UniformsUtils,
  Uint8BufferAttribute,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { ViewerBundle } from "./bundle-loader";
import type { ObjectIndex } from "./contracts";
import { POINTS_LAYER, createViewerPipeline } from "./edl";
import { focusedCameraPosition } from "./focus";
import {
  applyVisibilityUpdates,
  buildPointRangeLookup,
  buildVisibilityDelta,
  firstVisiblePointGlobalId,
} from "./point-picking";
import { cachedSelectionBox } from "./selection-bounds";
import {
  applySelectionColors,
  mergePointRanges,
  queueSelectionAttributeUpdates,
  type PointRange,
} from "./selection-colors";

const CLICK_THRESHOLD_PX = 6;
const MIN_POINT_SIZE = 0.004;
const MAX_POINT_SIZE = 0.07;
const DEFAULT_POINT_SIZE = 0.015;
const MAX_SPLAT_PIXELS = 64;
const FOG_NEAR_RADII = 1;
const FOG_FAR_RADII = 8;
const LIGHT_DIRECTION = new Vector3(0.45, 0.75, 0.5).normalize();
const VIEW_TRANSITION_MS = 420;
const SELECTION_BOX_PADDING_RATIO = 0.02;

export interface ViewerSceneController {
  selectGlobalId(globalId: string | null): void;
  selectGlobalIds(globalIds: ReadonlySet<string>): void;
  focusGlobalId(globalId: string): void;
  setPointSize(pointSize: number): void;
  setVisibleGlobalIds(ids: ReadonlySet<string>): void;
  setViewPreset(preset: "fit" | "top" | "isometric"): void;
  setPointPickHandler(handler: ((globalId: string) => void) | null): void;
  dispose(): void;
}

export interface PointPointerPress {
  readonly pointerId: number;
  readonly clientX: number;
  readonly clientY: number;
}

export interface PointPointerRelease extends PointPointerPress {
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

export function isPointClickRelease(press: PointPointerPress | null, release: PointPointerRelease): boolean {
  return press !== null
    && release.isPrimary
    && release.button === 0
    && release.pointerId === press.pointerId
    && Math.hypot(release.clientX - press.clientX, release.clientY - press.clientY) <= CLICK_THRESHOLD_PX;
}

export function selectionRangesForGlobalIds(objects: ObjectIndex, globalIds: ReadonlySet<string>): readonly PointRange[] {
  return [...globalIds]
    .flatMap((globalId) => objects[globalId]?.point_ranges ?? [])
    .filter(([start, end]) => end > start)
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
}

export function createViewerScene(container: HTMLElement, bundle: ViewerBundle): ViewerSceneController {
  const scene = new Scene();
  const background = new Color("#071015");
  scene.background = background;
  const camera = new PerspectiveCamera(50, 1, 0.01, 10000);
  const renderer = new WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.append(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;
  const worldGroup = new Group();
  const w2v = bundle.manifest.world_to_view;
  worldGroup.matrix.set(
    w2v[0], w2v[1], w2v[2], w2v[3], w2v[4], w2v[5], w2v[6], w2v[7],
    w2v[8], w2v[9], w2v[10], w2v[11], w2v[12], w2v[13], w2v[14], w2v[15],
  );
  worldGroup.matrixAutoUpdate = false;
  scene.add(worldGroup);

  const points = createPoints(bundle, worldGroup.matrix);
  points.layers.set(POINTS_LAYER);
  worldGroup.add(points);
  const pointMaterial = points.material as ShaderMaterial;
  const pointColorAttribute = points.geometry.getAttribute("aColor") as Uint8BufferAttribute;
  const pointVisibilityAttribute = points.geometry.getAttribute("aVisible") as Uint8BufferAttribute;
  const bounds = bundle.manifest.display_bounds;
  const box = new Box3(
    new Vector3(bounds[0], bounds[1], bounds[2]),
    new Vector3(bounds[3], bounds[4], bounds[5]),
  );
  const worldBox = box.clone().applyMatrix4(worldGroup.matrix);
  const sceneCenter = worldBox.getCenter(new Vector3());
  const sceneSpan = Math.max(worldBox.getSize(new Vector3()).length(), 1);
  const sceneRadius = Math.max(sceneSpan * 0.5, 1);
  const grid = new GridHelper(sceneSpan * 2, 20, "#294550", "#18313a");
  grid.position.y = box.max.y;
  worldGroup.add(grid, new AxesHelper(sceneSpan * 0.2));
  camera.far = Math.max(sceneRadius * 20, 100);
  camera.updateProjectionMatrix();
  scene.fog = new Fog(background, sceneRadius * FOG_NEAR_RADII, sceneRadius * FOG_FAR_RADII);
  const pipeline = createViewerPipeline(renderer, scene, camera);

  let selectedGlobalIdForCamera: string | null = null;
  let pickHandler: ((globalId: string) => void) | null = null;
  let visibleGlobalIds = new Set(Object.keys(bundle.objects));
  let previousVisibleGlobalIds = new Set(visibleGlobalIds);
  let primaryPointerPress: PointPointerPress | null = null;
  let focusAnimation: FocusAnimation | null = null;
  let animationFrame = 0;
  let previousTintedRanges: readonly PointRange[] = [];
  let currentPointSize = DEFAULT_POINT_SIZE;
  const pointRangeLookup = buildPointRangeLookup(bundle.objects);
  const selectionBoxCache = new Map<string, Box3 | null>();
  const raycaster = new Raycaster();
  raycaster.layers.enable(POINTS_LAYER);
  const pointer = new Vector2();
  const updateRaycasterThreshold = () => {
    raycaster.params.Points.threshold = currentPointSize * sceneSpan;
  };
  updateRaycasterThreshold();

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    pipeline.setPixelRatio(renderer.getPixelRatio());
    pipeline.setSize(width, height);
    pointMaterial.uniforms.uResolution.value.copy(renderer.getDrawingBufferSize(new Vector2()));
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  const onPointerDown = (event: PointerEvent) => {
    if (event.isPrimary && event.button === 0) {
      primaryPointerPress = {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
      };
    }
  };
  const onPointerCancel = (event: PointerEvent) => {
    if (event.isPrimary && primaryPointerPress?.pointerId === event.pointerId) {
      primaryPointerPress = null;
    }
  };
  const onPointerUp = (event: PointerEvent) => {
    const press = primaryPointerPress;
    if (event.isPrimary && press?.pointerId === event.pointerId) {
      primaryPointerPress = null;
    }
    if (!isPointClickRelease(press, event)) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const globalId = firstVisiblePointGlobalId(
      raycaster.intersectObject(points, false).flatMap((hit) => hit.index === undefined ? [] : [hit.index]),
      pointRangeLookup,
      visibleGlobalIds,
    );
    if (globalId !== null) pickHandler?.(globalId);
  };
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("pointercancel", onPointerCancel);
  controls.addEventListener("start", () => {
    focusAnimation = null;
  });

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
      selectedGlobalIdForCamera = globalId;
      updateSelectionPointTint(globalId === null ? new Set() : new Set([globalId]));
    },
    selectGlobalIds(globalIds) {
      selectedGlobalIdForCamera = null;
      updateSelectionPointTint(globalIds);
    },
    focusGlobalId(globalId) {
      const pointBox = computeSelectionBox(globalId);
      if (pointBox === null) return;
      pointBox.applyMatrix4(worldGroup.matrix);
      const target = pointBox.getCenter(new Vector3());
      const targetRadius = Math.max(pointBox.getSize(new Vector3()).length() * 0.8, 0.15);
      const distance = Math.max(
        targetRadius / Math.sin((camera.fov * Math.PI) / 360),
        0.45 * sceneRadius,
      );
      const focused = focusedCameraPosition(
        [camera.position.x, camera.position.y, camera.position.z],
        [controls.target.x, controls.target.y, controls.target.z],
        [target.x, target.y, target.z],
        distance * 1.1,
      );
      animateToView(new Vector3(...focused), target, VIEW_TRANSITION_MS);
      selectedGlobalIdForCamera = globalId;
    },
    setPointSize(size) {
      currentPointSize = clamp(size, MIN_POINT_SIZE, MAX_POINT_SIZE);
      pointMaterial.uniforms.uSize.value = currentPointSize;
      updateRaycasterThreshold();
    },
    setVisibleGlobalIds(ids) {
      const next = new Set(ids);
      const delta = buildVisibilityDelta(bundle.objects, previousVisibleGlobalIds, next);
      const changedRanges = applyVisibilityUpdates(
        pointVisibilityAttribute.array as Uint8Array,
        delta.ranges,
      );
      queueVisibilityAttributeUpdates(pointVisibilityAttribute, changedRanges);
      previousVisibleGlobalIds = next;
      visibleGlobalIds = next;
    },
    setViewPreset(preset) {
      setViewPreset(preset, true);
    },
    setPointPickHandler(handler) {
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
        (object as { geometry?: BufferGeometry }).geometry?.dispose();
        const material = (object as { material?: unknown }).material;
        (Array.isArray(material) ? material : [material]).forEach((item) => {
          (item as { dispose?: () => void } | undefined)?.dispose?.();
        });
      });
      renderer.dispose();
      renderer.domElement.remove();
    },
  };

  function computeSelectionBox(globalId: string): Box3 | null {
    return cachedSelectionBox(selectionBoxCache, globalId, () => {
      const axes: number[][] = [[], [], []];
      const ranges = bundle.objects[globalId]?.point_ranges ?? [];
      for (const [start, end] of ranges) {
        for (let index = start * 3; index < end * 3; index += 3) {
          axes[0].push(bundle.positions[index]);
          axes[1].push(bundle.positions[index + 1]);
          axes[2].push(bundle.positions[index + 2]);
        }
      }
      if (axes[0].length === 0) return null;
      const min = axes.map((axis) => quantile(axis.sort((left, right) => left - right), 0.01));
      const max = axes.map((axis) => quantile(axis.sort((left, right) => left - right), 0.99));
      const pointBox = new Box3(
        new Vector3(min[0], min[1], min[2]),
        new Vector3(max[0], max[1], max[2]),
      );
      pointBox.expandByScalar(pointBox.getSize(new Vector3()).length() * SELECTION_BOX_PADDING_RATIO);
      return pointBox;
    });
  }

  function updateSelectionPointTint(globalIds: ReadonlySet<string>): void {
    const ranges = selectionRangesForGlobalIds(bundle.objects, globalIds);
    const changed = applySelectionColors(
      pointColorAttribute.array as Uint8Array,
      bundle.colors,
      previousTintedRanges,
      ranges,
    );
    previousTintedRanges = ranges;
    queueSelectionAttributeUpdates(pointColorAttribute, changed);
  }

  function setViewPreset(preset: "fit" | "top" | "isometric", animateView: boolean): void {
    const target = selectedGlobalIdForCamera === null
      ? sceneCenter
      : computeSelectionBox(selectedGlobalIdForCamera)
        ?.applyMatrix4(worldGroup.matrix)
        .getCenter(new Vector3()) ?? sceneCenter;
    const direction = preset === "top"
      ? new Vector3(0, 1, 0)
      : preset === "isometric"
        ? new Vector3(1, 1, 0.6)
        : new Vector3(1, 0.7, 1);
    const distance = Math.max(
      sceneRadius / Math.tan((camera.fov * Math.PI) / 360) * (preset === "top" ? 1.15 : 1.5),
      1,
    );
    const offset = direction.normalize().multiplyScalar(distance);
    if (direction.y > 0.99) {
      offset.x = sceneSpan * 0.01;
      offset.z = sceneSpan * 0.01;
    }
    const position = target.clone().add(offset);
    if (animateView) animateToView(position, target, VIEW_TRANSITION_MS);
    else {
      controls.target.copy(target);
      camera.position.copy(position);
      camera.lookAt(target);
      controls.update();
    }
  }

  function animateToView(position: Vector3, target: Vector3, durationMs: number): void {
    focusAnimation = {
      startedAt: performance.now(),
      durationMs: Math.max(durationMs, 1),
      fromPosition: camera.position.clone(),
      toPosition: position.clone(),
      fromTarget: controls.target.clone(),
      toTarget: target.clone(),
    };
  }
}

const POINT_VERTEX_SHADER = /* glsl */ `
attribute vec3 aColor; attribute vec3 aNormal; attribute float aVisible;
uniform float uSize; uniform vec2 uResolution; uniform mat3 uNormalMatrix;
varying vec3 vColor; varying vec3 vNormal; varying float vVisible;
#include <fog_pars_vertex>
vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
void main() { vColor = srgbToLinear(aColor); vNormal = uNormalMatrix * aNormal; vVisible = aVisible; vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mvPosition; gl_PointSize = clamp(uSize * uResolution.y * projectionMatrix[1][1] / gl_Position.w, 1.0, ${MAX_SPLAT_PIXELS.toFixed(1)});
#include <fog_vertex>
}
`;

const POINT_FRAGMENT_SHADER = /* glsl */ `
uniform vec3 uLightDir; varying vec3 vColor; varying vec3 vNormal; varying float vVisible;
#include <fog_pars_fragment>
void main() { if (vVisible < 0.5) discard; vec2 centered = gl_PointCoord - vec2(0.5); if (dot(centered, centered) > 0.25) discard; float halfLambert = dot(normalize(vNormal), uLightDir) * 0.5 + 0.5; gl_FragColor = vec4(vColor * (0.6 + 0.4 * halfLambert), 1.0);
#include <fog_fragment>
}
`;

export function createPoints(bundle: ViewerBundle, worldMatrix: Matrix4): Points {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(bundle.positions, 3));
  geometry.setAttribute(
    "aColor",
    new Uint8BufferAttribute(bundle.colors.slice(), 3, true).setUsage(DynamicDrawUsage),
  );
  geometry.setAttribute(
    "aVisible",
    new Uint8BufferAttribute(
      new Uint8Array(bundle.positions.length / 3).fill(1),
      1,
      false,
    ).setUsage(DynamicDrawUsage),
  );
  geometry.setAttribute("aNormal", new Int8BufferAttribute(bundle.normals, 3, true));
  const material = new ShaderMaterial({
    uniforms: UniformsUtils.merge([
      UniformsLib.fog,
      {
        uSize: { value: DEFAULT_POINT_SIZE },
        uResolution: { value: new Vector2(1, 1) },
        uNormalMatrix: { value: new Matrix3().setFromMatrix4(worldMatrix) },
        uLightDir: { value: LIGHT_DIRECTION.clone() },
      },
    ]),
    vertexShader: POINT_VERTEX_SHADER,
    fragmentShader: POINT_FRAGMENT_SHADER,
    fog: true,
  });
  return new Points(geometry, material);
}

function quantile(sorted: readonly number[], q: number): number {
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return base + 1 < sorted.length ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max));
}

function queueVisibilityAttributeUpdates(attribute: Uint8BufferAttribute, changed: readonly PointRange[]): void {
  if (changed.length === 0) return;
  const pending = attribute.updateRanges.map(({ start, count }) => [start, start + count] as PointRange);
  attribute.clearUpdateRanges();
  for (const [start, end] of mergePointRanges([...pending, ...changed])) attribute.addUpdateRange(start, end - start);
  attribute.needsUpdate = true;
}

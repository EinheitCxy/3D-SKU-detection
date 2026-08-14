import {
  AxesHelper,
  Box3,
  BufferGeometry,
  Color,
  GridHelper,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Raycaster,
  Scene,
  SRGBColorSpace,
  Float32BufferAttribute,
  Uint8BufferAttribute,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Mesh } from "three";
import type { ViewerBundle } from "./bundle-loader";
import { createFootprintObjects } from "./footprints";

export interface ViewerSceneController {
  selectGlobalId(globalId: string | null): void;
  focusGlobalId(globalId: string): void;
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

const FOOTPRINT_CLICK_THRESHOLD_PX = 6;

export function isFootprintClickRelease(press: FootprintPointerPress | null, release: FootprintPointerRelease): boolean {
  return press !== null
    && release.isPrimary
    && release.button === 0
    && release.pointerId === press.pointerId
    && Math.hypot(release.clientX - press.clientX, release.clientY - press.clientY) <= FOOTPRINT_CLICK_THRESHOLD_PX;
}

interface FocusAnimation {
  readonly startedAt: number;
  readonly fromPosition: Vector3;
  readonly toPosition: Vector3;
  readonly fromTarget: Vector3;
  readonly toTarget: Vector3;
}

export function createViewerScene(container: HTMLElement, bundle: ViewerBundle): ViewerSceneController {
  const scene = new Scene();
  scene.background = new Color("#071015");
  const camera = new PerspectiveCamera(50, 1, 0.01, 10000);
  const renderer = new WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.append(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;
  const points = createPoints(bundle);
  scene.add(points);
  const pointBox = new Box3().setFromObject(points);
  const box = pointBox.isEmpty()
    ? new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1))
    : pointBox;
  const sceneSpan = Math.max(box.getSize(new Vector3()).length(), 1);
  const grid = new GridHelper(sceneSpan * 2, 20, "#294550", "#18313a");
  grid.position.y = box.min.y;
  scene.add(grid, new AxesHelper(sceneSpan * 0.2));

  const footprints = createFootprintObjects(bundle.footprints);
  scene.add(footprints.group);
  frameCamera(camera, controls, box);

  const raycaster = new Raycaster();
  const pointer = new Vector2();
  let pickHandler: ((globalId: string) => void) | null = null;
  let primaryPointerPress: FootprintPointerPress | null = null;
  let focusAnimation: FocusAnimation | null = null;
  let animationFrame = 0;
  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
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

  const render = (time: number) => {
    if (focusAnimation !== null) {
      const progress = Math.min((time - focusAnimation.startedAt) / 360, 1);
      const eased = 1 - (1 - progress) ** 3;
      camera.position.lerpVectors(focusAnimation.fromPosition, focusAnimation.toPosition, eased);
      controls.target.lerpVectors(focusAnimation.fromTarget, focusAnimation.toTarget, eased);
      if (progress === 1) focusAnimation = null;
    }
    controls.update();
    renderer.render(scene, camera);
    animationFrame = requestAnimationFrame(render);
  };
  animationFrame = requestAnimationFrame(render);

  return {
    selectGlobalId(globalId) {
      for (const [footprintId, visual] of footprints.selectionVisuals) {
        const isSelected = globalId !== null && footprintId === globalId;
        const fillOpacity = globalId === null ? 0.38 : isSelected ? 0.72 : 0.12;
        const outlineOpacity = globalId === null ? 0.95 : isSelected ? 1 : 0.18;
        visual.fills.forEach((material) => { material.opacity = fillOpacity; });
        visual.outlines.forEach((material) => { material.opacity = outlineOpacity; });
      }
    },
    focusGlobalId(globalId) {
      const footprint = footprints.focusTargets.get(globalId);
      if (footprint === undefined) return;
      const targetBox = new Box3().setFromObject(footprint);
      const target = targetBox.getCenter(new Vector3());
      const radius = Math.max(targetBox.getSize(new Vector3()).length() * 0.8, sceneSpan * 0.08, 0.3);
      const direction = camera.position.clone().sub(controls.target).normalize();
      focusAnimation = {
        startedAt: performance.now(), fromPosition: camera.position.clone(), toPosition: target.clone().addScaledVector(direction, radius * 3),
        fromTarget: controls.target.clone(), toTarget: target,
      };
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
}

function createPoints(bundle: ViewerBundle): Points {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(bundle.positions, 3));
  geometry.setAttribute("color", new Uint8BufferAttribute(bundle.colors, 3, true));
  const material = new PointsMaterial({ size: 0.015, sizeAttenuation: true, vertexColors: true });
  const points = new Points(geometry, material);
  points.name = "da3-rgb-points";
  return points;
}

function frameCamera(camera: PerspectiveCamera, controls: OrbitControls, box: Box3): void {
  const center = box.isEmpty() ? new Vector3() : box.getCenter(new Vector3());
  const radius = Math.max(box.getSize(new Vector3()).length() * 0.5, 1);
  const distance = radius / Math.tan((camera.fov * Math.PI) / 360);
  controls.target.copy(center);
  camera.position.copy(center).add(new Vector3(1, 0.7, 1).normalize().multiplyScalar(distance * 1.5));
  camera.lookAt(center);
  controls.update();
}

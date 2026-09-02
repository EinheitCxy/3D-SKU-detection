import {
  DepthTexture,
  HalfFloatType,
  ShaderMaterial,
  Vector2,
  WebGLRenderTarget,
} from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { FullScreenQuad, Pass } from "three/addons/postprocessing/Pass.js";
import { SMAAPass } from "three/addons/postprocessing/SMAAPass.js";
import type { PerspectiveCamera, Scene, WebGLRenderer } from "three";

/** Points render on this camera/scene layer; every other object stays on layer 0. */
export const POINTS_LAYER = 1;

export const DEFAULT_EDL_STRENGTH = 0.4;
export const DEFAULT_EDL_RADIUS = 1.4;

const EDL_VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

/**
 * Eye-Dome Lighting composite (potree-style): the point cloud is rendered into a
 * color+depth render target, then each pixel is darkened by how much closer its
 * diagonal neighbours are (scale-invariant relative depth response), producing
 * local shading on the sparse splat cloud without any geometry normals.
 */
const EDL_FRAGMENT_SHADER = /* glsl */ `
#include <packing>
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform vec2 uTexelSize;
uniform float uCameraNear;
uniform float uCameraFar;
uniform float uEdlStrength;
uniform float uEdlRadius;
varying vec2 vUv;

/** Positive view-space distance, or 0.0 for pixels without point-cloud coverage. */
float viewDistance(vec2 uv) {
  float depth = texture2D(tDepth, uv).x;
  if (depth >= 1.0) return 0.0;
  return -perspectiveDepthToViewZ(depth, uCameraNear, uCameraFar);
}

float edlResponse(float dist, float neighbour) {
  if (neighbour <= 0.0 || neighbour >= dist) return 0.0;
  return 1.0 - exp(-4.0 * (dist - neighbour) / dist);
}

void main() {
  vec4 color = texture2D(tColor, vUv);
  float dist = viewDistance(vUv);
  if (dist <= 0.0) {
    gl_FragColor = color;
    return;
  }
  vec2 offset = uTexelSize * uEdlRadius;
  float occlusion =
    edlResponse(dist, viewDistance(vUv + vec2(-offset.x, -offset.y))) +
    edlResponse(dist, viewDistance(vUv + vec2( offset.x, -offset.y))) +
    edlResponse(dist, viewDistance(vUv + vec2(-offset.x,  offset.y))) +
    edlResponse(dist, viewDistance(vUv + vec2( offset.x,  offset.y)));
  float shade = exp(-uEdlStrength * occlusion);
  gl_FragColor = vec4(color.rgb * shade, color.a);
}
`;

class EDLPass extends Pass {
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly renderTarget: WebGLRenderTarget;
  private readonly material: ShaderMaterial;
  private readonly fsQuad: FullScreenQuad;

  constructor(
    scene: Scene,
    camera: PerspectiveCamera,
    strength: number,
    radius: number,
  ) {
    super();
    this.scene = scene;
    this.camera = camera;
    this.needsSwap = false;
    this.renderTarget = new WebGLRenderTarget(1, 1, {
      type: HalfFloatType,
      depthBuffer: true,
      depthTexture: new DepthTexture(1, 1),
    });
    this.renderTarget.texture.name = "EDLPass.points";
    this.material = new ShaderMaterial({
      uniforms: {
        tColor: { value: null },
        tDepth: { value: null },
        uTexelSize: { value: new Vector2(1, 1) },
        uCameraNear: { value: 0.01 },
        uCameraFar: { value: 10000 },
        uEdlStrength: { value: strength },
        uEdlRadius: { value: radius },
      },
      vertexShader: EDL_VERTEX_SHADER,
      fragmentShader: EDL_FRAGMENT_SHADER,
      depthTest: false,
      depthWrite: false,
    });
    this.fsQuad = new FullScreenQuad(this.material);
  }

  setSize(width: number, height: number): void {
    this.renderTarget.setSize(width, height);
  }

  render(renderer: WebGLRenderer, _writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget): void {
    // 1. Points-only pass into the color+depth target (layer-scoped, scene
    //    background + fog still apply through the point material).
    this.camera.layers.set(POINTS_LAYER);
    renderer.setRenderTarget(this.renderTarget);
    try {
      renderer.render(this.scene, this.camera);
    } finally {
      this.camera.layers.set(0);
    }

    // 2. Composite EDL shading in place into the composer buffer.
    const uniforms = this.material.uniforms;
    uniforms.tColor.value = this.renderTarget.texture;
    uniforms.tDepth.value = this.renderTarget.depthTexture;
    uniforms.uTexelSize.value.set(1 / this.renderTarget.width, 1 / this.renderTarget.height);
    uniforms.uCameraNear.value = this.camera.near;
    uniforms.uCameraFar.value = this.camera.far;
    renderer.setRenderTarget(readBuffer);
    this.fsQuad.render(renderer);
  }

  dispose(): void {
    this.renderTarget.dispose();
    this.material.dispose();
    this.fsQuad.dispose();
  }
}

/**
 * Renders the remaining layer-0 scene (grid, axes, selection box)
 * on top of the EDL-composited image without touching its color, exactly like
 * potree draws helpers above the shaded point cloud.
 */
class OverlayPass extends Pass {
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;

  constructor(scene: Scene, camera: PerspectiveCamera) {
    super();
    this.scene = scene;
    this.camera = camera;
    this.needsSwap = false;
  }

  render(renderer: WebGLRenderer, _writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget): void {
    this.camera.layers.set(0);
    renderer.setRenderTarget(readBuffer);
    renderer.clearDepth();
    // three r185 WebGLBackground: a Color scene.background sets forceClear=true,
    // which clears the color buffer whenever autoClearColor is on — even with
    // autoClear=false. Detach the background while drawing the helpers so the
    // EDL-composited image in the readBuffer is preserved.
    const autoClear = renderer.autoClear;
    const background = this.scene.background;
    renderer.autoClear = false;
    this.scene.background = null;
    try {
      renderer.render(this.scene, this.camera);
    } finally {
      renderer.autoClear = autoClear;
      this.scene.background = background;
    }
  }
}

export interface ViewerPipeline {
  readonly composer: EffectComposer;
  setPixelRatio(pixelRatio: number): void;
  setSize(width: number, height: number): void;
  dispose(): void;
}

export interface PipelineOptions {
  readonly edlStrength?: number;
  readonly edlRadius?: number;
}

/**
 * Full frame pipeline: EDL (points only) -> overlay helpers -> SMAA (linear
 * space, required before OutputPass) -> OutputPass (ACES tone mapping + sRGB).
 */
export function createViewerPipeline(
  renderer: WebGLRenderer,
  scene: Scene,
  camera: PerspectiveCamera,
  options: PipelineOptions = {},
): ViewerPipeline {
  const edlPass = new EDLPass(
    scene,
    camera,
    options.edlStrength ?? DEFAULT_EDL_STRENGTH,
    options.edlRadius ?? DEFAULT_EDL_RADIUS,
  );
  const overlayPass = new OverlayPass(scene, camera);
  const smaaPass = new SMAAPass();
  const outputPass = new OutputPass();
  const composer = new EffectComposer(renderer);
  composer.addPass(edlPass);
  composer.addPass(overlayPass);
  composer.addPass(smaaPass);
  composer.addPass(outputPass);
  return {
    composer,
    setPixelRatio(pixelRatio) {
      composer.setPixelRatio(pixelRatio);
    },
    setSize(width, height) {
      composer.setSize(width, height);
    },
    dispose() {
      edlPass.dispose();
      smaaPass.dispose();
      outputPass.dispose();
      composer.dispose();
    },
  };
}

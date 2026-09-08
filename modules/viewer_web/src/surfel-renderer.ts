import {
  AddEquation, BufferAttribute, Camera, Color, CustomBlending, DoubleSide, DynamicDrawUsage, Float32BufferAttribute,
  FloatType, GLSL3, HalfFloatType, InstancedBufferAttribute, InstancedBufferGeometry,
  Matrix3, Matrix4, Mesh, NearestFilter, NoBlending, OneFactor, OrthographicCamera, PlaneGeometry,
  Scene, ShaderMaterial, UnsignedByteType, Vector2, Vector4, WebGLRenderer, WebGLRenderTarget,
} from "three";
import type { ViewerBundle } from "./bundle-loader";

// View-space depth tolerance in scene units: 1 mm plus half a native grid step.
// Source visibility separately permits 1.5% of the sampled source-camera depth.
const BLEND_DEPTH_ABSOLUTE_TOLERANCE = 0.001;
const BLEND_DEPTH_GRID_STEP_RATIO = 0.5;
const SOURCE_DEPTH_RELATIVE_TOLERANCE = 0.015;

// Three.js uses this marker to upload Uint16 bits as GL_HALF_FLOAT, not integers.
class HalfFloatInstancedAttribute extends InstancedBufferAttribute {
  readonly isFloat16BufferAttribute = true;
}

const vertexShader = /* glsl */ `
in vec3 aCenter; in vec3 aU; in vec3 aV; in float aFrame; in float aSlot;
in float aVisible; in float aSelected;
uniform float uRadius;
out vec2 vDisk; out vec3 vWorld; out float vViewDepth;
flat out int vFrame; flat out float vSlot;
flat out float vVisible; flat out float vSelected; flat out float vTolerance;
void main() {
  vec3 p = aCenter + uRadius * (position.x * aU + position.y * aV);
  vec4 view = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  vDisk = position.xy; vWorld = p; vViewDepth = -view.z;
  vFrame = int(aFrame); vSlot = aSlot;
  vVisible = aVisible; vSelected = aSelected;
  vTolerance = ${BLEND_DEPTH_ABSOLUTE_TOLERANCE} + ${BLEND_DEPTH_GRID_STEP_RATIO} * min(length(aU), length(aV));
}
`;

const fragmentShader = /* glsl */ `
precision highp sampler2DArray;
uniform sampler2DArray uTextures; uniform sampler2DArray uDepths;
uniform sampler2D uFront; uniform vec2 uResolution; uniform vec2 uGrid;
uniform mat4 uExtrinsic[FRAME_COUNT]; uniform mat3 uIntrinsic[FRAME_COUNT];
uniform mat3 uInverseAffine[FRAME_COUNT]; uniform vec4 uSizes[FRAME_COUNT];
uniform vec2 uTextureSize; uniform int uPass;
in vec2 vDisk; in vec3 vWorld; in float vViewDepth;
flat in int vFrame; flat in float vSlot;
flat in float vVisible; flat in float vSelected; flat in float vTolerance;
out vec4 outColor;
vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
void main() {
  float radius2 = dot(vDisk, vDisk);
  if (vVisible < 0.5 || radius2 > 1.0) discard;
  vec3 sourceCamera = (uExtrinsic[vFrame] * vec4(vWorld, 1.0)).xyz;
  if (sourceCamera.z <= 0.0) discard;
  vec3 projected = uIntrinsic[vFrame] * sourceCamera;
  vec2 processed = projected.xy / projected.z;
  if (any(lessThan(processed, vec2(0.0))) || any(greaterThan(processed, uGrid - 1.0))) discard;
  float sourceDepth = texture(uDepths, vec3((processed + 0.5) / uGrid, float(vFrame))).r;
  if (sourceDepth <= 0.0 || abs(sourceDepth - sourceCamera.z) > ${SOURCE_DEPTH_RELATIVE_TOLERANCE} * sourceDepth + 0.001) discard;
  vec2 source = (uInverseAffine[vFrame] * vec3(processed, 1.0)).xy;
  vec4 sizes = uSizes[vFrame];
  if (any(lessThan(source, vec2(0.0))) || any(greaterThan(source, sizes.xy - 1.0))) discard;
  // Native quad rasterization supplies perspective-correct per-fragment plane depth.
  if (uPass == 0) { outColor = vec4(vViewDepth, 0.0, 0.0, 1.0); return; }
  if (uPass == 2) {
    uint id = uint(vSlot) + 1u;
    outColor = vec4(float(id & 255u), float((id >> 8u) & 255u), float((id >> 16u) & 255u), 255.0) / 255.0;
    return;
  }
  float front = texelFetch(uFront, ivec2(gl_FragCoord.xy), 0).r;
  if (front <= 0.0 || vViewDepth > front + vTolerance) discard;
  vec2 texturePixel = (source + 0.5) / sizes.xy * sizes.zw;
  vec3 color = texture(uTextures, vec3(texturePixel / uTextureSize, float(vFrame))).rgb;
  color = vSelected > 0.5 ? vec3(1.0, 0.0, 1.0) : color;
  float weight = exp(-2.0 * radius2);
  outColor = vec4(srgbToLinear(color) * weight, weight);
}
`;

export function createSurfelRenderer(
  renderer: WebGLRenderer, camera: Camera, bundle: ViewerBundle, worldMatrix: Matrix4,
  visibility: BufferAttribute,
) {
  const data = bundle.surfels;
  if (!data) throw new Error("Surfel mode requires Surfel sidecar");
  if (!renderer.extensions.has("EXT_color_buffer_float")) throw new Error("Surfel requires EXT_color_buffer_float");
  if (Math.max(...data.textureSize) > renderer.capabilities.maxTextureSize) throw new Error("Surfel textures exceed GPU texture limit");
  const geometry = new InstancedBufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0], 3));
  geometry.setIndex([0,1,2,0,2,3]);
  geometry.instanceCount = bundle.pointCount;
  const attribute = (name: string, values: ArrayLike<number>, size: number) =>
    geometry.setAttribute(name, new InstancedBufferAttribute(values instanceof Float32Array ? values : new Float32Array(values), size));
  attribute("aCenter", bundle.positions, 3);
  geometry.setAttribute("aU", new HalfFloatInstancedAttribute(data.u, 3));
  geometry.setAttribute("aV", new HalfFloatInstancedAttribute(data.v, 3));
  geometry.setAttribute("aFrame", new InstancedBufferAttribute(data.frames, 1));
  attribute("aSlot", Float32Array.from({ length: bundle.pointCount }, (_, i) => i), 1);
  const sharedVisibility = new InstancedBufferAttribute(visibility.array, 1).setUsage(visibility.usage);
  const selected = new InstancedBufferAttribute(new Uint8Array(bundle.pointCount), 1).setUsage(DynamicDrawUsage);
  geometry.setAttribute("aVisible", sharedVisibility);
  geometry.setAttribute("aSelected", selected);
  const frameMatrices = data.metadata.frames.map(frame => {
    const e = frame.extrinsic;
    return new Matrix4().set(...[...e[0], ...e[1], ...e[2], 0,0,0,1] as Parameters<Matrix4["set"]>);
  });
  const mat3 = (values: number[][]) => new Matrix3().set(...values.flat() as Parameters<Matrix3["set"]>);
  const uniforms = {
    uTextures: { value: data.textures }, uDepths: { value: data.depths }, uFront: { value: null },
    uResolution: { value: new Vector2(1, 1) }, uGrid: { value: new Vector2(...data.metadata.grid_size) },
    uExtrinsic: { value: frameMatrices }, uIntrinsic: { value: data.metadata.frames.map(f => mat3(f.intrinsic)) },
    uInverseAffine: { value: data.metadata.frames.map(f => mat3(f.processed_to_source)) },
    uSizes: { value: data.metadata.frames.map(f => new Vector4(...f.source_size, ...f.texture_size)) },
    uTextureSize: { value: new Vector2(...data.textureSize) }, uPass: { value: 0 },
    uRadius: { value: 1.05 },
  };
  const material = new ShaderMaterial({
    glslVersion: GLSL3, defines: { FRAME_COUNT: data.metadata.frames.length }, uniforms,
    vertexShader, fragmentShader, side: DoubleSide, toneMapped: false,
  });
  const mesh = new Mesh(geometry, material);
  mesh.matrix.copy(worldMatrix); mesh.matrixAutoUpdate = false; mesh.frustumCulled = false;
  const scene = new Scene(); scene.add(mesh);
  const front = new WebGLRenderTarget(1, 1, { type: FloatType, depthBuffer: true, minFilter: NearestFilter, magFilter: NearestFilter });
  const accumulation = new WebGLRenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false, minFilter: NearestFilter, magFilter: NearestFilter });
  const ids = new WebGLRenderTarget(1, 1, { type: UnsignedByteType, depthBuffer: true, minFilter: NearestFilter, magFilter: NearestFilter });
  uniforms.uFront.value = front.texture as never;
  const outputMaterial = new ShaderMaterial({
    uniforms: { uAccumulation: { value: accumulation.texture } },
    vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
    fragmentShader: `uniform sampler2D uAccumulation; varying vec2 vUv;
      void main() { vec4 c = texture2D(uAccumulation, vUv);
        gl_FragColor = vec4(c.a > 0.0 ? c.rgb / c.a : vec3(1.0), 1.0);
        #include <colorspace_fragment>
      }`,
    depthTest: false, depthWrite: false, toneMapped: false,
  });
  const outputScene = new Scene();
  const quad = new Mesh(new PlaneGeometry(2, 2), outputMaterial); outputScene.add(quad);
  const outputCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  let visibilityVersion = -1;
  const sync = () => {
    if (visibility.version !== visibilityVersion) { sharedVisibility.needsUpdate = true; visibilityVersion = visibility.version; }
  };
  const savedColor = new Color();
  const renderPass = (pass: number, target: WebGLRenderTarget) => {
    uniforms.uPass.value = pass;
    // Even an unexecuted shader branch must not bind the current color attachment.
    uniforms.uFront.value = (pass === 1 ? front.texture : accumulation.texture) as never;
    material.depthWrite = material.depthTest = pass !== 1;
    material.blending = pass === 1 ? CustomBlending : NoBlending;
    material.blendEquation = AddEquation; material.blendSrc = material.blendDst = OneFactor;
    renderer.setRenderTarget(target); renderer.setClearColor(0, 0); renderer.clear(); renderer.render(scene, camera);
  };
  return {
    render() {
      sync();
      renderer.getClearColor(savedColor); const alpha = renderer.getClearAlpha();
      renderPass(0, front); renderPass(1, accumulation);
      renderer.setRenderTarget(null); renderer.setClearColor(savedColor, alpha);
      renderer.render(outputScene, outputCamera);
    },
    setSize(width: number, height: number) {
      front.setSize(width, height); accumulation.setSize(width, height); ids.setSize(width, height);
      uniforms.uResolution.value.set(width, height);
    },
    setRadius(size: number) { uniforms.uRadius.value = Math.min(2.0, Math.max(0.6, 1.05 * size / 0.004)); },
    select(ranges: readonly (readonly [number, number])[]) {
      (selected.array as Uint8Array).fill(0);
      for (const [start, end] of ranges) (selected.array as Uint8Array).fill(1, start, end);
      selected.needsUpdate = true;
    },
    pick(x: number, y: number): number | null {
      sync(); renderer.getClearColor(savedColor); const alpha = renderer.getClearAlpha();
      renderPass(2, ids);
      const pixel = new Uint8Array(4);
      renderer.readRenderTargetPixels(ids, Math.min(ids.width-1, Math.max(0, Math.floor(x * ids.width))),
        Math.min(ids.height-1, Math.max(0, Math.floor((1-y) * ids.height))), 1, 1, pixel);
      renderer.setRenderTarget(null); renderer.setClearColor(savedColor, alpha);
      const id = pixel[0] + pixel[1] * 256 + pixel[2] * 65536;
      return id === 0 ? null : id - 1;
    },
    dispose() {
      front.dispose(); accumulation.dispose(); ids.dispose(); geometry.dispose(); material.dispose();
      quad.geometry.dispose(); outputMaterial.dispose(); data.textures.dispose(); data.depths.dispose();
    },
  };
}

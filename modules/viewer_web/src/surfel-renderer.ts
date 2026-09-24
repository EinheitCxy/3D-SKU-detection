import {
  AddEquation, BufferAttribute, Camera, Color, CustomBlending, DoubleSide, DynamicDrawUsage, Float32BufferAttribute,
  DataTexture, FloatType, GLSL3, HalfFloatType, InstancedBufferAttribute, InstancedBufferGeometry,
  Matrix3, Matrix4, Mesh, NearestFilter, NoBlending, OneFactor, OrthographicCamera, PlaneGeometry,
  RGBAFormat, Scene, ShaderMaterial, UnsignedByteType, Vector2, Vector3, Vector4, WebGLRenderer, WebGLRenderTarget,
} from "three";
import type { ViewerBundle } from "./bundle-loader";

// View-space depth tolerance in scene units: 1 mm plus half a native grid step.
// Source visibility separately permits 1.5% of the sampled source-camera depth.
const BLEND_DEPTH_ABSOLUTE_TOLERANCE = 0.001;
const BLEND_DEPTH_GRID_STEP_RATIO = 0.5;
const SOURCE_DEPTH_RELATIVE_TOLERANCE = 0.015;
const MAX_NATIVE_STRETCH = 2.0;
const MAX_FOOTPRINT_ASPECT = 4.0;

// Three.js uses this marker to upload Uint16 bits as GL_HALF_FLOAT, not integers.
class HalfFloatInstancedAttribute extends InstancedBufferAttribute {
  readonly isFloat16BufferAttribute = true;
}

const vertexShader = /* glsl */ `
precision highp sampler2DArray;
in vec3 aCenter; in vec3 aU; in vec3 aV; in vec2 aScale; in float aFrame; in float aSlot;
in float aVisible; in float aSelected;
uniform float uRadius; uniform float uPixelCenterOffset;
uniform sampler2DArray uDepths; uniform vec2 uGrid;
uniform sampler2D uCameras;
out vec2 vDisk; out vec3 vWorld; out float vViewDepth;
flat out int vFrame; flat out float vSlot;
flat out float vVisible; flat out float vSelected; flat out float vTolerance;
flat out vec4 vSourceBounds;

// Clamp singular values, not individual axes: nearly parallel U/V can both
// be long in the same direction. Preserve the surface plane and the short axis.
void boundFootprint(inout vec3 u, inout vec3 v, vec2 nativeStep) {
  vec3 a = u / nativeStep.x, b = v / nativeStep.y;
  float aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
  float gap = length(vec2(aa - bb, 2.0 * ab));
  float major = sqrt(max(0.0, 0.5 * (aa + bb + gap)));
  float minor = sqrt(max(0.0, 0.5 * (aa + bb - gap)));
  float cap = min(${MAX_NATIVE_STRETCH.toFixed(1)}, ${MAX_FOOTPRINT_ASPECT.toFixed(1)} * minor);
  float majorScale = min(1.0, cap / max(major, 1e-8));
  float minorScale = min(1.0, ${MAX_NATIVE_STRETCH.toFixed(1)} / max(minor, 1e-8));
  mat2 correction = mat2(minorScale);
  if (gap > 1e-6) {
    float low = minor * minor;
    mat2 majorProjector = mat2(aa - low, ab, ab, bb - low) / gap;
    correction += (majorScale - minorScale) * majorProjector;
  } else {
    correction = mat2(majorScale);
  }
  u = (a * correction[0][0] + b * correction[0][1]) * nativeStep.x;
  v = (a * correction[1][0] + b * correction[1][1]) * nativeStep.y;
}

vec4 cameraTexel(int column, int frame) { return texelFetch(uCameras, ivec2(column, frame), 0); }
mat4 cameraExtrinsic(int frame) {
  return mat4(cameraTexel(0, frame), cameraTexel(1, frame), cameraTexel(2, frame), cameraTexel(3, frame));
}
mat3 cameraMatrix3(int column, int frame) {
  vec4 c0 = cameraTexel(column, frame), c1 = cameraTexel(column + 1, frame), c2 = cameraTexel(column + 2, frame);
  return mat3(c0.xyz, c1.xyz, c2.xyz);
}

bool continuousNeighbor(ivec2 pixel, int frame, float predictedDepth, float tolerance) {
  if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, ivec2(uGrid)))) return false;
  float depth = texelFetch(uDepths, ivec3(pixel, frame), 0).r;
  return depth > 0.0 && abs(depth - predictedDepth) <= tolerance;
}
void main() {
  int frame = int(aFrame);
  mat4 extrinsic = cameraExtrinsic(frame);
  mat3 rotation = mat3(extrinsic);
  mat3 inverseIntrinsic = cameraMatrix3(7, frame);
  mat3 intrinsic = cameraMatrix3(4, frame);
  vec3 center = (extrinsic * vec4(aCenter, 1.0)).xyz;
  vec2 nativeStep = max(vec2(1e-6), center.z * vec2(
    length(inverseIntrinsic[0]), length(inverseIntrinsic[1])));
  vec3 u = aU, v = aV;
  boundFootprint(u, v, nativeStep);
  vec3 projected = intrinsic * center;
  ivec2 pixel = ivec2(floor(projected.xy / projected.z - vec2(uPixelCenterOffset) + 0.5));
  vec2 depthStep = vec2((rotation * aU).z, (rotation * aV).z);
  vec2 tolerance = vec2(0.001) + 0.5 * nativeStep;
  // At a discontinuity, stop at this source pixel's cell boundary. Smooth
  // interiors retain the requested radius, including their texture detail.
  vSourceBounds = vec4(-1e6, 1e6, -1e6, 1e6);
  if (!continuousNeighbor(pixel + ivec2(-1, 0), frame, center.z - depthStep.x, tolerance.x)) vSourceBounds.x = float(pixel.x) - 0.5;
  if (!continuousNeighbor(pixel + ivec2( 1, 0), frame, center.z + depthStep.x, tolerance.x)) vSourceBounds.y = float(pixel.x) + 0.5;
  if (!continuousNeighbor(pixel + ivec2(0, -1), frame, center.z - depthStep.y, tolerance.y)) vSourceBounds.z = float(pixel.y) - 0.5;
  if (!continuousNeighbor(pixel + ivec2(0,  1), frame, center.z + depthStep.y, tolerance.y)) vSourceBounds.w = float(pixel.y) + 0.5;
  // Sampling scale expands only the disk; continuity and blend tolerances keep
  // the native one-pixel tangents and must not grow with the sampling footprint.
  vec3 p = aCenter + uRadius * (position.x * aScale.x * u + position.y * aScale.y * v);
  vec4 view = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  vDisk = position.xy; vWorld = p; vViewDepth = -view.z;
  vFrame = int(aFrame); vSlot = aSlot;
  vVisible = aVisible; vSelected = aSelected;
  vTolerance = ${BLEND_DEPTH_ABSOLUTE_TOLERANCE} + ${BLEND_DEPTH_GRID_STEP_RATIO} * min(length(u), length(v));
}
`;

const fragmentShader = /* glsl */ `
precision highp sampler2DArray;
uniform sampler2DArray uTextures; uniform sampler2DArray uDepths;
uniform sampler2D uFront; uniform sampler2D uWinner; uniform vec2 uResolution; uniform vec2 uGrid;
uniform float uPixelCenterOffset;
uniform sampler2D uCameras;
uniform vec2 uTextureSize; uniform vec3 uCameraPosition;
uniform int uPass; uniform bool uSourceSelection;
in vec2 vDisk; in vec3 vWorld; in float vViewDepth;
flat in int vFrame; flat in float vSlot;
flat in float vVisible; flat in float vSelected; flat in float vTolerance;
flat in vec4 vSourceBounds;
out vec4 outColor;
vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec4 cameraTexel(int column, int frame) { return texelFetch(uCameras, ivec2(column, frame), 0); }
mat4 cameraExtrinsic(int frame) {
  return mat4(cameraTexel(0, frame), cameraTexel(1, frame), cameraTexel(2, frame), cameraTexel(3, frame));
}
mat3 cameraMatrix3(int column, int frame) {
  vec4 c0 = cameraTexel(column, frame), c1 = cameraTexel(column + 1, frame), c2 = cameraTexel(column + 2, frame);
  return mat3(c0.xyz, c1.xyz, c2.xyz);
}
void main() {
  // Derivatives must precede every fragment-dependent discard or branch.
  // Use the real texture pixel grid: values below one mean magnification.
  mat4 extrinsic = cameraExtrinsic(vFrame);
  mat3 intrinsic = cameraMatrix3(4, vFrame);
  mat3 inverseAffine = cameraMatrix3(10, vFrame);
  vec3 sourceCamera = (extrinsic * vec4(vWorld, 1.0)).xyz;
  float projectionDepth = max(abs(sourceCamera.z), 1e-8) * (2.0 * step(0.0, sourceCamera.z) - 1.0);
  vec3 projected = intrinsic * sourceCamera;
  vec2 processed = projected.xy / projectionDepth - vec2(uPixelCenterOffset);
  vec2 source = (inverseAffine * vec3(processed, 1.0)).xy;
  vec4 sizes = cameraTexel(13, vFrame);
  vec2 texturePixel = (source + 0.5) / sizes.xy * sizes.zw;
  vec2 textureDx = dFdx(texturePixel), textureDy = dFdy(texturePixel);
  float aa = dot(textureDx, textureDx), ab = dot(textureDx, textureDy), bb = dot(textureDy, textureDy);
  float sigmaMin = sqrt(max(0.0, 0.5 * (aa + bb - length(vec2(aa - bb, 2.0 * ab)))));
  float resolution = clamp(sigmaMin, 0.0, 1.0);
  float radius2 = dot(vDisk, vDisk);
  if (vVisible < 0.5 || radius2 > 1.0) discard;
  if (sourceCamera.z <= 0.0) discard;
  if (processed.x < vSourceBounds.x || processed.x > vSourceBounds.y
    || processed.y < vSourceBounds.z || processed.y > vSourceBounds.w) discard;
  if (any(lessThan(processed, vec2(0.0))) || any(greaterThan(processed, uGrid - 1.0))) discard;
  float sourceDepth = texture(uDepths, vec3((processed + 0.5) / uGrid, float(vFrame))).r;
  if (sourceDepth <= 0.0 || abs(sourceDepth - sourceCamera.z) > ${SOURCE_DEPTH_RELATIVE_TOLERANCE} * sourceDepth + 0.001) discard;
  if (any(lessThan(source, vec2(0.0))) || any(greaterThan(source, sizes.xy - 1.0))) discard;
  // Because the winner pass overrides depth below, every other pass must
  // explicitly retain the rasterized geometric depth.
  gl_FragDepth = gl_FragCoord.z;
  // Native quad rasterization supplies perspective-correct per-fragment plane depth.
  if (uPass == 0) { outColor = vec4(vViewDepth, 0.0, 0.0, 1.0); return; }
  if (uPass != 2 || uSourceSelection) {
    float front = texelFetch(uFront, ivec2(gl_FragCoord.xy), 0).r;
    if (front <= 0.0 || vViewDepth > front + vTolerance) discard;
  }
  if (uPass == 3) {
    vec3 sourceRay = normalize(vWorld - cameraTexel(14, vFrame).xyz);
    vec3 currentRay = normalize(vWorld - uCameraPosition);
    float alignment = max(0.0, dot(sourceRay, currentRay));
    float quality = 0.85 * alignment + 0.15 * resolution;
    // Quantize quality and prefer the lower frame index for exact ties so a
    // still camera has deterministic ownership without temporal hysteresis.
    float rank = floor(quality * 4095.0 + 0.5) * float(FRAME_COUNT) + float(FRAME_COUNT - vFrame);
    outColor = vec4(float(vFrame + 1), 0.0, 0.0, 1.0);
    gl_FragDepth = 1.0 - rank / (4096.0 * float(FRAME_COUNT));
    return;
  }
  if (uSourceSelection && abs(texelFetch(uWinner, ivec2(gl_FragCoord.xy), 0).r - float(vFrame + 1)) > 0.5) discard;
  if (uPass == 2) {
    uint id = uint(vSlot) + 1u;
    outColor = vec4(float(id & 255u), float((id >> 8u) & 255u), float((id >> 16u) & 255u), 255.0) / 255.0;
    return;
  }
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
  const frameCount = data.metadata.frames.length;
  const gl = renderer.getContext() as WebGL2RenderingContext;
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (data.textures.image.depth > maxLayers || data.depths.image.depth > maxLayers) {
    throw new Error(`Surfel 纹理数组层数超过显卡限制 ${maxLayers}`);
  }
  if (frameCount > maxLayers || frameCount > renderer.capabilities.maxTextureSize) {
    throw new Error(`Surfel 帧数 ${frameCount} 超过显卡相机纹理限制`);
  }
  const geometry = new InstancedBufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0], 3));
  geometry.setIndex([0,1,2,0,2,3]);
  geometry.instanceCount = bundle.pointCount;
  const attribute = (name: string, values: ArrayLike<number>, size: number) =>
    geometry.setAttribute(name, new InstancedBufferAttribute(values instanceof Float32Array ? values : new Float32Array(values), size));
  attribute("aCenter", bundle.positions, 3);
  geometry.setAttribute("aU", new HalfFloatInstancedAttribute(data.u, 3));
  geometry.setAttribute("aV", new HalfFloatInstancedAttribute(data.v, 3));
  geometry.setAttribute("aScale", new HalfFloatInstancedAttribute(data.scale, 2));
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
  const cameraData = new Float32Array(frameCount * 16 * 4);
  const writeMat3 = (offset: number, matrix: Matrix3) => {
    for (let column = 0; column < 3; column += 1) {
      cameraData.set(matrix.elements.slice(column * 3, column * 3 + 3), offset + column * 4);
    }
  };
  data.metadata.frames.forEach((frame, index) => {
    const offset = index * 64;
    const extrinsic = frameMatrices[index];
    cameraData.set(extrinsic.elements, offset);
    writeMat3(offset + 16, mat3(frame.intrinsic));
    writeMat3(offset + 28, mat3(frame.intrinsic).invert());
    writeMat3(offset + 40, mat3(frame.processed_to_source));
    cameraData.set([...frame.source_size, ...frame.texture_size], offset + 52);
    cameraData.set(new Vector3().setFromMatrixPosition(new Matrix4().copy(extrinsic).invert()).toArray(), offset + 56);
  });
  const cameras = new DataTexture(cameraData, 16, frameCount, RGBAFormat, FloatType);
  cameras.minFilter = NearestFilter; cameras.magFilter = NearestFilter; cameras.generateMipmaps = false; cameras.needsUpdate = true;
  const uniforms = {
    // Pi3X fits K to pixel centers x+0.5/y+0.5; depth arrays use integer indices.
    uPixelCenterOffset: { value: bundle.manifest.backend === "Pi3X" ? 0.5 : 0.0 },
    uTextures: { value: data.textures }, uDepths: { value: data.depths }, uFront: { value: null }, uWinner: { value: null },
    uResolution: { value: new Vector2(1, 1) }, uGrid: { value: new Vector2(...data.metadata.grid_size) },
    uCameras: { value: cameras },
    uTextureSize: { value: new Vector2(...data.textureSize) },
    uCameraPosition: { value: new Vector3() }, uPass: { value: 0 }, uSourceSelection: { value: false },
    uRadius: { value: 1.05 },
  };
  const material = new ShaderMaterial({
    glslVersion: GLSL3, defines: { FRAME_COUNT: frameCount }, uniforms,
    vertexShader, fragmentShader, side: DoubleSide, toneMapped: false,
  });
  const mesh = new Mesh(geometry, material);
  mesh.matrix.copy(worldMatrix); mesh.matrixAutoUpdate = false; mesh.frustumCulled = false;
  const scene = new Scene(); scene.add(mesh);
  const front = new WebGLRenderTarget(1, 1, { type: FloatType, depthBuffer: true, minFilter: NearestFilter, magFilter: NearestFilter });
  const accumulation = new WebGLRenderTarget(1, 1, { type: HalfFloatType, depthBuffer: false, minFilter: NearestFilter, magFilter: NearestFilter });
  // R stores frame index + 1; its depth buffer retains the best local source quality.
  const winner = new WebGLRenderTarget(1, 1, { type: FloatType, depthBuffer: true, minFilter: NearestFilter, magFilter: NearestFilter });
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
  const inverseWorld = new Matrix4().copy(worldMatrix).invert();
  const cameraWorldPosition = new Vector3();
  let visibilityVersion = -1;
  const sync = () => {
    if (visibility.version !== visibilityVersion) { sharedVisibility.needsUpdate = true; visibilityVersion = visibility.version; }
  };
  const savedColor = new Color();
  const syncCameraPosition = () => {
    camera.getWorldPosition(cameraWorldPosition);
    uniforms.uCameraPosition.value.copy(cameraWorldPosition).applyMatrix4(inverseWorld);
  };
  const renderPass = (pass: number, target: WebGLRenderTarget) => {
    uniforms.uPass.value = pass;
    // Even an unexecuted shader branch must not bind the current color attachment.
    uniforms.uFront.value = (pass === 0 ? accumulation.texture : front.texture) as never;
    uniforms.uWinner.value = (pass === 3 ? accumulation.texture : winner.texture) as never;
    material.depthWrite = material.depthTest = pass !== 1;
    material.blending = pass === 1 ? CustomBlending : NoBlending;
    material.blendEquation = AddEquation; material.blendSrc = material.blendDst = OneFactor;
    renderer.setRenderTarget(target); renderer.setClearColor(0, 0); renderer.clear(); renderer.render(scene, camera);
  };
  return {
    render() {
      sync();
      syncCameraPosition();
      renderer.getClearColor(savedColor); const alpha = renderer.getClearAlpha();
      renderPass(0, front);
      if (uniforms.uSourceSelection.value) renderPass(3, winner);
      renderPass(1, accumulation);
      renderer.setRenderTarget(null); renderer.setClearColor(savedColor, alpha);
      renderer.render(outputScene, outputCamera);
    },
    setSize(width: number, height: number) {
      front.setSize(width, height); accumulation.setSize(width, height); winner.setSize(width, height); ids.setSize(width, height);
      uniforms.uResolution.value.set(width, height);
    },
    setRadius(size: number) { uniforms.uRadius.value = Math.min(2.0, Math.max(0.6, 1.05 * size / 0.004)); },
    setSourceSelection(enabled: boolean) { uniforms.uSourceSelection.value = enabled; },
    select(ranges: readonly (readonly [number, number])[]) {
      (selected.array as Uint8Array).fill(0);
      for (const [start, end] of ranges) (selected.array as Uint8Array).fill(1, start, end);
      selected.needsUpdate = true;
    },
    pick(x: number, y: number): number | null {
      sync(); syncCameraPosition(); renderer.getClearColor(savedColor); const alpha = renderer.getClearAlpha();
      if (uniforms.uSourceSelection.value) { renderPass(0, front); renderPass(3, winner); }
      renderPass(2, ids);
      const pixel = new Uint8Array(4);
      renderer.readRenderTargetPixels(ids, Math.min(ids.width-1, Math.max(0, Math.floor(x * ids.width))),
        Math.min(ids.height-1, Math.max(0, Math.floor((1-y) * ids.height))), 1, 1, pixel);
      renderer.setRenderTarget(null); renderer.setClearColor(savedColor, alpha);
      const id = pixel[0] + pixel[1] * 256 + pixel[2] * 65536;
      return id === 0 ? null : id - 1;
    },
    dispose() {
      front.dispose(); accumulation.dispose(); winner.dispose(); ids.dispose(); geometry.dispose(); material.dispose();
      quad.geometry.dispose(); outputMaterial.dispose(); cameras.dispose(); data.textures.dispose(); data.depths.dispose();
    },
  };
}

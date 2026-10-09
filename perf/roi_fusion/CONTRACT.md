# Video3 single-product experiment

Isolated experiment; do not edit production files or existing comparison assets.
Input: runtime/roi-fusion-video3-gid3/input.npz, allow_pickle=False.

Arrays: points[N,3], u[N,3], v[N,3], normals[N,3], confidence[N], frame[N] (local 0..F-1), xy[N,2] (integer processed x,y); depth[F,H,W] (full scene camera Z metres), mask[F,H,W] (product bool), K[F,3,3], E[F,4,4] (world-to-camera), affine[F,3,3] (original to processed pixel), image_paths[F] (absolute Unicode paths), image_ids[F], spacing scalar (median valid U/V lengths).
All points are in original world coordinates. Camera intrinsics/poses immutable.

Output root runtime/roi-fusion-video3-gid3. Methods save baseline.glb, fused.glb, tsdf.glb, plus per-method stats JSON. GLBs have original world coordinates and unlit display in an independent Three page. Baseline and fused use SAME 8-sided radius=1.05 surfel discs tessellated to triangles and source-image projection to UV. This common experimental renderer is NOT claimed pixel-identical to production splatting. Include original production screenshot/reference separately.

Shared texture helper owned by TSDF worker: texture.py export_textured_mesh(vertices, faces, face_sources, data, output_path) -> stats dict. face_sources[T] local frame indices, -1 explicitly grey/untextured. Projection uses inverse affine and source image dimensions; original images <=1920 long/1080 short, no upscaling. Shared choose_sources(vertices,faces,data) -> face_sources[T],stats: spatially connected normal-consistent patches, select visible source once per patch; untextured reported explicitly. No dynamic view-dependent source switching. Validate front-facing visibility against original full-scene depth (tolerance recorded).

Surfel worker owns fusion.py and test_fusion.py. Exports baseline and fused using shared texture.py; output fused.npz with points,u,v,frame plus membership diagnostic. TSDF worker owns texture.py, tsdf.py, test_texture.py. Main owns prepare.py, evaluate.py, README/report and orchestration. UI worker owns modules/viewer_web/roi-fusion.html and modules/viewer_web/roi-fusion.mjs, perf/roi_fusion/capture.mjs only. No commits/push. Preserve others' changes.

UI assets served by dedicated local HTTP server rooted at runtime/roi-fusion-video3-gid3 on port 8767 (CORS localhost); page on Vite 5173 or independent 5175, query ?assets=http://127.0.0.1:8767/. meta.json includes center[3], radius, camera_positions source world centers, source_image_ids, stats; load baseline.glb/fused.glb/tsdf.glb and optional context.glb. Orbit controls synchronized, select source/front/side presets, source-color toggle (material userData.source_frame or material.name frame_N), isolated product/context toggle. Export window.roiReady and window.roiCapture(preset,sourceColor=false) returning three PNG base64, plus window.roiStats. All errors shown.

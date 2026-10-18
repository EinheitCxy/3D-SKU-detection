# Findings

- Current source is local main bf9bde8 with pre-existing documentation changes. Target production source files were clean at task start.
- Six-video profile: DA3 24.199s, constructor 11.320s; point ranges video3 3.429s/video6 13.950s. Those are prior instrumented observations, not this optimization's control.
- Actual DA3 Hub loader defaults to non-strict copy loading. Full-meta construction has `.item()` and nonpersistent-buffer obstacles. Selective initialization skipping requires actual key/shape coverage.
- Ground currently uses full post-SOR/DBSCAN point domain for candidate evaluation, 3-point hypotheses and max1000 iterations; retains 8% support, side/protection/remaining guards. Proxy proposals change plane selection and need separate quality reporting.
- GLB already reuses final Viewer geometry and source JPEG. Its expensive 64-texel sampling is new work. Preserve f16 round-trip and JPEG decode. Current measured encoding workers=1; bounded 2/4-worker implementation already exists.
- GPU2 is the only authorized experiment device. Existing GPU residents must remain untouched.
- Baseline core completed successfully in131.816s (includes imports, diagnostic copies/saves and process lifecycle); delivery still running. Exact post-DBSCAN cloud, product protection and final sorted labels were captured for same-domain component tests.
- Worker focused checks: DA3 nine tests, filter39 tests, mesh/GLB/LOD89 tests passed. These are implementation checks, not actual checkpoint/real-data/performance acceptance.
- DA3 candidate retains standard CPU tensors and nonpersistent buffer initialization. It skips only large Linear/Conv weight Kaiming fills (numel>=65536), audits all actual state keys/shapes/dtypes before loading, and requires strict checkpoint load.
- Correction to first audit: raw missing six keys are aliases of the same LayerNorm weight/bias reused in four auxiliary sequences, not six independent uncovered parameters. Alias-aware metadata/identity audit and native strict loading passed:1383 state keys,1377 checkpoint keys/physical parameter tensors,6 covered aliases,0 uncovered,1 native-initialized nonpersistent buffer. Skipped406 weights/6737780736bytes.
- Sequential fresh CPU pair: constructor11.826516s→0.651774s; CPU construct+metadata audit+load12.993895s→2.045339s. Loaded samples and nonpersistent buffer equal; this is not yet GPU forward/full-pipeline equivalence. Receipts in runtime/latency-optimization-20261018/da3-cpu-pair-alias-aware/.
- Real point ranges same-array comparison:2402589int32 labels/711 observations,3.489543s→0.000964s, all endpoints exactly equal. Ten relevant point-range/schema3 tests passed.
- Ground same-domain3425675point component:10.879015s→0.980081s; candidate removes51286background/0product, retains all1902589products. Candidate normal is87.7733deg from orientation floor normal;81.72% of its280535inliers are protected product points. No confirmed-floor/visual acceptance: keep cap0 default,50K explicit experiment.
- Initial camera-basis probe:29×1024inputs,29316paired surviving quads,geometry/order same;6alpha texels differ,9318RGB texels differ,4443both-opaque RGB texels differ,all-RGB max41. These are sampled pre-palette results, not final-image equivalence. Preserve source-world projection default; retain camera-basis as explicit experimental option. Frame constants reuse can remain default.

## Final observations — 2026-10-18 21:39 CST

- Safe default candidate (selective DA3 initialization + typed ranges + shared frame constants + encoder2, source_world/ground0) completed253.138706→186.894435s; full savings66.244271s/26.169%. Viewer129.854374→109.162660s, ZIP138.096691→117.362010s, GLB114.041383→68.770372s. These final observations supersede earlier in-progress statements above without erasing their history.
- Same-run DA3constructor12.624388→.914212s and CPUload13.562327→2.424568s, reconstruction25.132625→13.706324s; same-run ranges3.939481→.001608s. The separately timed CPU/point/ground components remain separate evidence.
- Direct full output comparison PASS: all15DA3 numeric arrays, global mapping,8Viewer binary arrays, objects, manifest and entireGLB bytes exact. Both2402589Viewerpoints/1282759baked/31pages. Both sampled task GPU peaks14.777GiB; no memory reduction claim.
- Workerencode cumulative wall43.640→45.379s is overlapped withbake; join43.649→.900s is main-thread waiting. Do not add concurrent/nested spans or infer most encoding computation disappeared.
- Three medium final reviews PASS. Production defaults exclude the two experiments whose output/quality gates did not pass. Final results do not establish multi-video speed distribution or global numerical equivalence. All task GPU workers exited; pre-existing residents untouched.

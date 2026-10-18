# DA3 selective initialization optimization

Status: implementation, focused tests, real-checkpoint CPU pair and the coordinator's fixed-video3 GPU2 full-pipeline validation are complete. Alias-aware audit and native strict loading passed. The implementation worker ran CPU checks only; the coordinator ran the GPU comparison below.

`src/da3_model_loading.py` temporarily intercepts native Linear / `_ConvNd` reset methods in the calling DA3 process/thread. Only their weight's Kaiming random fill is skipped when numel >= 65,536. Bias initialization, small weights, token values, LayerNorm and buffers retain their constructor behavior. Methods and init functions are restored even on failure. Other Python threads retain native fills.

The native Hugging Face config parser and safetensors loader are retained through a local loader subclass. Before any checkpoint copy, `safe_open.get_slice` metadata must cover every physical state tensor with matching shape/dtype; omitted state keys are accepted only when they refer to the identical Parameter/buffer object as a present key. Distinct shared-storage/strided views are not guessed equivalent. All skipped tensors must map to covered model parameters. The inherited loader retains **strict=True** with existing safetensors alias handling and copy semantics, without meta tensors or `assign=True`. There is no arbitrary partial-loading fallback. The model carries a scalar `_da3_loading_report`, and logs raw missing keys, covered aliases, uncovered missing keys, key/skip counts, bytes and elapsed CPU construct/audit/load time. Normal runner and CPU standby use this shared entry.

### Six omitted state names are two shared physical LayerNorm parameters

The first baseline failed before checkpoint copy/inference because the initial audit incorrectly required one checkpoint entry per state key. `Depth-Anything-3/src/depth_anything_3/model/dualdpt.py:133–147` creates one `ln_seq` with one `LayerNorm(32)` and expands those **same module objects** into four Sequential heads. The omitted `model.da3.head.scratch.output_conv2_aux.{1,2,3}.2.{weight,bias}` entries are therefore six aliases of `.0.2.weight/bias`, not six independent random weights. Each alias shape is `[32]` (32 elements), below the 65,536 skip threshold; constructor defaults are ones/zeros. Metadata-only checkpoint header inspection confirmed `.0.2.weight/bias` are present, shape `[32]`, F32; the six alias names are absent.

Forward (`dualdpt.py:260`) uses the last head, so the shared LayerNorm **does participate** in inference. Loading the present `.0.2.*` keys updates that same object used in the last head. The real-model CPU audit confirmed **1,383 state names, 1,377 checkpoint keys, 1,377 physical state tensors/parameters**, with **6 raw missing names resolved as 6 identical-object aliases and 0 uncovered missing tensors**. There are 0 physical persistent buffers. Unexpected keys and shape/dtype mismatches were empty; native safetensors strict loading succeeded.

The **one physical nonpersistent buffer** is `model.da3.gs_adapter.sh_mask`, F32 `[9]`, 36 bytes. Baseline/candidate values were exactly equal: one `1.0`, three float32 representations of `0.025`, and five float32 representations of `0.00625`. It retains native constructor initialization because it is not serialized. **406** large weight tensors skip random fill, accounting for **6,737,780,736 bytes**; every skipped physical parameter passed coverage checks.

### Measured CPU component pair

Evidence: `runtime/latency-optimization-20261018/da3-cpu-pair-alias-aware/{pair,baseline,candidate}.json`. The coordinator ran sequential fresh CPU processes with CUDA hidden and OMP/MKL/OpenBLAS=4 against the same existing snapshot. The earlier `da3-cpu-pair/baseline.json` audit failure remains retained as the rejected one-state-name-per-checkpoint-key assumption.

| Interval | Native construction + audited strict load | Selective random-fill skipping + audited strict load |
| --- | ---: | ---: |
| Constructor | 11.826516 s | 0.651774 s |
| Metadata audit | 0.294871 s | 0.280600 s |
| CPU construction + audit + load total | 12.993895 s | 2.045339 s |

The constructor wall decreased by 11.174742s; the containing CPU load interval decreased by 10.948556s. Audit is a child interval, not additive to the total. These are one controlled CPU component comparison, not a GPU-forward or full-video speed claim. Both processes reported `cuda_initialized=false`. Loaded samples from eight eligible weight tensors (three elements each) and all values of the small nonpersistent buffer matched exactly. **Sample equality is neither full tensor equality nor output equivalence.**

Focused validation: `test/test_da3_model_loading.py` — **11 passed** (2.12s), using the existing uv test environment with CUDA hidden. Checks cover selective scope, failure restoration, another thread, full tiny-safetensors loading, key/shape/dtype rejection, deterministic buffer preservation, native HF config resolution, four Sequential heads sharing one LayerNorm saved with omitted safetensors aliases and strict native load, and explicit rejection of distinct strided Parameter views. No full-regression suite or GPU run was executed here.

CPU comparison command used by the coordinator (a repeat must use a new output directory):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv MPLCONFIGDIR=/tmp/3d-profile-20261018-mpl \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python \
  python reports/latency-optimization-20261018/da3_probe.py --mode pair \
  --out runtime/latency-optimization-20261018/da3-cpu-pair-alias-aware
```

The probe uses the existing local b2359bdf checkpoint without copy/hash. The earlier six-video mean constructor wall was 11.320s under a different in-run profiling window; this new component difference must not be subtracted from that historical pipeline mean. Skipped random fills change CPU RNG advancement; same-seed output validation is necessary rather than assumed.

### Coordinator's final GPU2 video3 pair

Same29frames/711detections, seed42 and4CPUthreads, fresh model/result processes per arm; one run each. In this same full pipeline, constructor12.624388→0.914212s, containing CPUload13.562327→2.424568s and reconstruction25.132625→13.706324s. Overall253.138706→186.894435s includes the other accepted optimizations and is not attributed solely to DA3. [Full pair](pair-summary.json).

All15entire numeric DA3 arrays (including depth/confidence/worldpoints/cameras/images/frame bindings) were exactly equal; mapping, all8Viewer arrays and metadata, and complete finalGLB payload also equal. This is actual fixed-video output evidence, separate from sampled parameter checks above. Both task driver200ms sampled GPU peaks15132MiB=14.777GiB. [Validation](pair-validation.json). These results do not establish all-input equality or an optimized multi-video average.

# Request prototype

`request_probe.py` is isolated; production and upstream files are unchanged.

These notes describe the prototype author's checks. Later coordinator-run actual-input, GPU, fault and continuous-delivery results are summarized in [README.md](README.md); the pending statements below describe the original handoff stage.

```python
predict_prefetch(model, images, sensor, device,
                 batch_size=2, precision="fp16", max_size=1008,
                 reduced=False, prefetch=True, delayed_d2h=False)
cpu_input_check(images, sensor, batch_size=2, max_size=1008)
```

- Pure prefetch: defaults. Producer uses the original RGB arithmetic, AREA resize, rint/clip uint8 packing, and even zero-width `np.pad`. Torch stack/cat/prompt interpolation remain in the caller thread.
- Copy/allocation experiment independently: `reduced=True, prefetch=False`. Removes zero-padding copy only when padding is zero; writes CHW batches/colors into preallocated NumPy arrays; divides float32 RGB in place and rounds/clips the separate saved-RGB temporary in place. Never quantizes network RGB. Nonzero padding still calls original edge pad.
- Combined: `reduced=True, prefetch=True`.
- D2H experiment independently: `prefetch=False, delayed_d2h=True`. Retains float32 cropped GPU depth until all predict batches finish, then returns all CPU arrays. On the current 20 x 756 x 1008 output, retained depth payload is ~58.14 MiB; allocator peak may grow by more. Can combine switches, but report each standalone measurement before combined attribution.

All variants preserve metadata and native prompt rules. `predict_prefetch` does not reset seeds, matching `predict_promptda`; the timing caller must replicate `predict_candidate`'s Python/NumPy/Torch/CUDA seed=42 reset **before each variant/request**, after model construction, and apply the same finite-positive output check if replacing `predict_candidate`.

One CPU executor worker, at most one submitted future, at most consumed RGB batch plus next RGB batch. Depth prompts and output RGB/depth still accumulate for the request as native. No pinned-memory pool, CUDA stream, CUDA call or Torch operation in the producer. Torch CPU pool setup is therefore not induced by the worker. Executor creation, PNG reads, joining the producer, final stacking and all D2H copies stay within the measured request. No worker persists between calls.

Exceptions propagate from `future.result()`. `finally` cancels pending work and waits for any running finite batch to finish before returning/raising; there is no producer queue `put` that can block after consumer failure. Cancellation does not interrupt a running PNG/filesystem read, so pathological blocked I/O can still delay shutdown, like a synchronous read.

## Local checks

- `uv` AST parse: PASS.
- CPU toy check (single Python command, timeout 5s, exit 0): 3 frames each at 28x28 (pad=0) and 29x27 (nonzero pad), native max_size=0, batch sizes [2,1]. Network RGB, prompt arrays, depth mock, saved RGB, frame IDs (nonmonotonic 8,3,11) and every metadata field strictly array-equal for pure prefetch, reduced serial and combined variants.
- Missing-image producer exception: propagated FileNotFoundError; no remaining `promptda-rgb` thread.
- No GPU launch or model load; actual 20-frame resize inputs, FP16/depth equivalence, delayed-D2H, request timings and memory peaks remain for coordinator verification. `cpu_input_check` is a separate process/check: it retains native input arrays and performs CPU Torch packing, so do not run it before timing in the same process.

`cpu_input_check` imports `candidates` from the existing scanner path, calls actual native `predict_promptda` with a CPU capture mock, and compares pre-model batch image/prompt values plus all returned fields. Importing the prototype itself starts no model, test or worker.

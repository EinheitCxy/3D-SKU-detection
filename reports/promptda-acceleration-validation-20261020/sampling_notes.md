# Sampling probe（隔离 CPU 原型）

`sampling_probe.py` 仅改变报告目录，不修改生产文件、上游代码或 dense cache。

以下保留原型阶段说明；coordinator 后续真实输入、组件计时、连续交付与直接产物比对见 [README.md](README.md)。通用未抽中异常点的失败语义差异仍然存在。

- `select_before_backprojection(prediction, sensor, max_points=1_000_000, seed=42)` 返回 `(world_points, colors, source_indices, stats)`。坐标尚未经过 display transform；source_indices 是原 depth tensor 按 frame、帧内 C-order flatten 后的位置，保留原 RNG choice 返回顺序。
- 主候选仍使用原 `column_stack(x,y,ones).astype(float32)`。`select_before_backprojection_float32_pixels` 才额外去掉 float64 临时数组；CLI 用 `--include-float32-pixels` 单独运行该 variant。
- baseline 调用当前 `scanner_io.backproject`，保留采样前全量世界坐标有限检查。候选只检查抽中点，因此全域失败语义尚未保持，不能默认替换。
- `certify_finite_domain` 是真实输入的充分条件检查：对实际 float32 inverse-K/c2w、完整 valid depth 最大值和像素范围建立绝对值上界。3 项 dot 乘以保守因子 8，depth product/add 乘以 2，加入 tiny cushion，再要求距离 FLT_MAX 仍有 1024 倍余量。它返回 true/unknown，不通过时没有 fallback；独立计时外运行。它只证明当前输入的 world 运算有限，不修复通用失败语义，也不宣称坐标 bitwise。
- `unselected_overflow_check()` 构造未抽中的 finite-positive 极大深度；可导致原版全量投影 overflow 并拒绝，而候选接受。此反例保留在结果 JSON。

CLI 示例（由 coordinator 在正确的现有 uv 环境串行执行）：

```bash
uv run --no-sync python reports/promptda-acceleration-validation-20261020/sampling_probe.py \
  --prediction Output/promptda-da3-d4d5b1bcd2-20261020/promptda-full-ready/artifacts/predictions.npz \
  --sensor Output/promptda-da3-d4d5b1bcd2-20261020/prepared/camera_inputs.npz \
  --output reports/promptda-acceleration-validation-20261020/sampling_results.json \
  --include-float32-pixels
```

`--export-glb <path>` 可选，仅导出主候选一次，复用 native trimesh PointCloud/Scene 写法并回读一次；默认不产生大文件。

每个 variant 的 A/B 顺序固定 AB、BA、AB，各侧 3 次；加载数组后才计时，包含 mask、世界点计算、采样和 source-ID mapping；display transform、比较、证书和 GLB 均不计入。reference 的全域 source-ID 列表预先计算，以免为原版加入不存在的全域工作；reference 的 selected-ID gather 仍计入，JSON 明示这一小量 instrumentation。坐标每轮先报告 bitwise，再报告数值差异 component 数和 maxabs；world/display 分开。RGB/sourceIDs 同顺序必须相等，否则直接失败。结果仅标量 JSON，不保存数组。

已通过 AST parse 和小 toy（2×3×4，含一个无效深度；max_points=1/7/100；两种 candidate 与 reference 的 sourceIDs/RGB/顺序及 world 坐标 bitwise 均一致）。toy 的 finite certificate 为 true（23 个 valid，world 保守上界约 7006，接受界约 3.323e35）；未抽中异常 source=1、抽中 source=0，reference 拒绝而 candidate 接受，确认失败语义减弱。运行使用 coordinator 指定的既有环境：

`UV_CACHE_DIR=/tmp/promptda-probe-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-project --offline --python runtime/viewer-latency-20261017/test-env/bin/python python ...`

真实输入验证和性能结果由 coordinator 追加；本原型作者没有读取真实缓存或执行重 CPU 测试。

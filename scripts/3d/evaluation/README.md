# SAM3.1 PVS 小样本试验

在仓库根运行，先选择有足够空闲显存的 GPU。以下示例使用 GPU2、fd2 第0帧全部检测框，输出目录必须尚不存在：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/3d/evaluation/sam31_pvs_trial.py \
  --dataset imdata/floor_display2 --frames 0 --batch-size 4 \
  --gpu-memory-limit-gib 9 --output runtime/sam31-pvs-sample
```

`--frames 0 1 2` 可指定多帧。图像按最长边1008等比缩小，框同步缩放；原生 encoder 输入为1008×1008。每帧先编码一次，再按 batch 解码独立框；输出 masks 位于等比缩小后的图像空间，**不是 DA3 processed space**。原始 `object_id` 是 detection JSON 展平后的0-based顺序，不过滤或重排检测框。

实现使用 `sam31/` 原生 multiplex tracker，从同一份 `sam3.1_multiplex.pt` 加载 tracker 参数及共享的 detector vision backbone，并以 `strict=True` 验证完整匹配。PVS 使用 interactive neck、box corner labels 2/3、原生 object-presence 处理、最后裁到对应 bbox。`multimask_output=False` 每框返回一个 mask；原生 interactive decoder 仍启用 stability selection（delta=0.05，threshold=0.98），低稳定性时可选择其他候选。空 mask 会原样报告，不追加中心点重试。

输出：

- `<frame>_input.jpg` / `<frame>_pvs.jpg`：输入与带检测框/ID的 mask 叠加图。
- `<frame>_masks.npz`：按原始 object_id 排序的 bool `masks` 和同空间 `boxes_xyxy`。
- `<frame>_objects.json`：每框面积、predicted IoU、presence logit；这些模型分数不是人工真值准确率。
- `results.json`：实际权重加载、参数、逐帧耗时与 PyTorch allocated/reserved 峰值。

计时在 GPU synchronize 后覆盖 encoder、box decoder、mask CPU回传和 bbox clipping；不含输入预处理、模型加载或文件写入，加载单列。没有预热，首帧可能包含首次执行开销。显存限制只约束本进程 PyTorch allocator，不代表设备总显存上限。

此入口只做独立试验，不修改默认分割模式，不发布 matching cache。非空 mask 数不代表准确率；切换生产路径前仍需同图同框的 self-exemplar 对比及 matching 验证。

## 同条件 PVS / self-exemplar 比较

`sam31_pvs_compare.py` 接收 JSON 样本清单，格式为 `{"samples":[{"dataset":"imdata/floor_display2","frame":"0","boxes":75}]}`。指定框数用于发现输入变化，全部原始检测框保留。

```bash
CUDA_VISIBLE_DEVICES=2 uv run --no-sync python scripts/3d/evaluation/sam31_pvs_compare.py \
  --arm pvs --samples runtime/sam31-pvs-compare-20261016/samples.json \
  --output runtime/pvs-comparison-new/pvs --batch-size 4 --repeats 3
```

完成后以独立进程运行 `--arm self-exemplar --output runtime/pvs-comparison-new/self-exemplar`，其余参数保持一致。已执行的后台串行入口保存在 `runtime/sam31-pvs-compare-20261016/run.sh`，通过 `nohup bash ... > nohup.log 2>&1 < /dev/null &` 启动；每个输出目录必须尚不存在。

每个方法各预热同一张图一次；每图3次完整重跑API，取中位数。统一外层计时从已读取的 PIL 图像和框开始，覆盖API内预处理、传输、推理与CPU mask后处理；不含读图、加载模型、结果校验和落盘。baseline使用当前优化后的self-exemplar、threshold=0.5，无mask缓存。两者都使用FP32权重、BF16 autocast、TF32和4个CPU线程。

`results.json` 保留各次耗时、非空mask、空mask的object_id、重复间差异像素数及allocated/reserved峰值；另由nvidia-smi按本进程PID采样显存，观测峰值可能漏过短峰。每张图最后一次结果保存在 `<dataset>/<frame>/masks.npz`，不会发布到生产缓存。

总耗时节省百分比为 `100 × (1 − ΣPVS每图中位耗时 / Σself-exemplar每图中位耗时)`。共同batch=4的结果不能替代生产batch=32或完整3D流水线的性能结论。

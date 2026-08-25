# Personalcare Classifier

此模块只保留可运行的 `source/` 和一个 canonical 权重：`source/model/model.bin`。`model_best.pth.tar` 与历史 checkpoint/layer snapshots 不在模块内保留。

`model.bin` 是恢复出的原始个人护理模型格式；`PersonalcarePredictor` 只在实例化时按既有可逆字节变换读取它，并且每个分类进程只加载一次 MobileNetV3。模型必须使用显式可用的 CUDA device；不会自动回退到 CPU。

## 本地数据集分类

从仓库根运行：

```bash
uv run --project modules/personalcare_classifier python \
  modules/personalcare_classifier/source/classify_dataset.py \
  --dataset imdata/floor_display6 \
  --output-root Output \
  --device cuda:0
```

输入是 `images/<数字帧>` 与 `detections_results/<数字帧>.json`。两组数字帧 ID 必须完全一致。runner 按原始 object 顺序读取 bbox，最多每 32 个有效 crop 调用一次 predictor；无效、反向、空或越界 bbox 会发布 `classification.status: "unavailable"` 和 `reason: "invalid_bbox"`，不会进入模型。

成功时 stdout 只输出一个 JSON 结果，包含 `success`、run ID、enriched detections 目录、result 路径和计数。产物以原子指针发布：

```text
<output-root>/<dataset>/personalcare_classification/
├── runs/<time_ns>-<pid>/detections/<frame>.json
├── runs/<time_ns>-<pid>/result.json
└── CURRENT
```

每个 enriched JSON 保留完整原始 payload，并为有效对象添加 `classes.cls`、`confidences.cls` 和规范化的 `classification`。临时 run 完整写入并校验后才重命名并替换 `CURRENT`，因此失败不会替换已有已完成 run。不会发布深度 feature、HTTP/BSON 服务或模型副本。

该 CLI 同样是根 `main.py --mode pipeline` 的唯一分类实现。pipeline 在 dataset validation 后异步启动它；它可与 DA3 reconstruction 或 matching 并行，但实际时间重叠取决于 cache、可视化与运行 receipt。本次 fd6 cache-reuse receipt 中分类在 matching 开始前已完成。等 matching 和此 CLI 都成功后，才把 stdout 返回的 `detection_dir` 显式交给 dedup。原始 `detections_results/` 从不被写入。输出没有 hash、签名、加密、内容指纹或分类 cache；run ID 只是 `<time_ns>-<pid>` 的发布代号。

GPU 必须显式选择。若要使用物理 GPU 2，请用 `CUDA_VISIBLE_DEVICES=2` 并传入进程内 `--device cuda:0`；不设置 mask 时传入实际可见 index。CUDA/模型失败为失败结果，绝不转用 CPU 或替代权重。

分类记录的 confidence 仅供下游确定性聚合与审计：同一 global ID 的 candidate 按 aggregate confidence、support count、max confidence 和 SKU 字段排序。所有 conflict candidate 会传递给 Viewer，但每个 global ID 只以 primary candidate 计入 Total/SKU facet，Viewer UI 不显示 confidence。

原始 `sku-classifier-personalcare-20250510-recovery/` 目录的属主为 `nobody`，因此其重复副本无法由当前用户安全删除；完成确认后请由拥有权限的用户清理该原目录，而不要删除本模块内的 canonical source。

# da3 vs pi3x 准确率与性能对比 (floor_display4..8)

生成时间: 2026-09-04 (pi3x 首次接入, commit 工作树)

> 同一 config.yaml（pairing=next, batch_all_refs, sam3_max_batch_size=32）、同一评测器
> （accuracy_annotation.py + scripts/3d/evaluation/accuracy_evaluation.sh）、同一代码版本。
> pi3x 为当日全量 pipeline 新跑；da3 为当日 concise 重跑（da3_cache 为 2026-08-19/20 的
> DA3 1.1 metric 缓存，SAM3 v2 mask 缓存复用，匹配确定性等价于全新跑）。
> 阈值按后端语义各自标定：da3 用原始 conf(1.5) + 米制几何；pi3x 用 sigmoid conf(0.05) +
> 米制几何（min_depth 0.3 / max_depth 8.0 / max_3d_distance 0.5m，近似米制）。

## 逐数据集对比（micro 聚合，口径同 docs/accuracy_da3_vs_pi3.md）

| 数据集 | pi3x Recall | da3 Recall | pi3x Precision | da3 Precision | pi3x F1 | da3 F1 | ΔF1 |
|---|---|---|---|---|---|---|---|
| floor_display4 | **81.2%** (125/154) | 67.5% (104/154) | **94.7%** (125/132) | 78.8% (104/132) | **87.4%** | 72.7% | **+14.7** |
| floor_display5 | 79.8% (91/114) | **81.6%** (93/114) | 90.1% (91/101) | **90.3%** (93/103) | 84.7% | **85.7%** | -1.0 |
| floor_display6 | 79.3% (157/198) | **87.9%** (174/198) | 87.1% (155/178) | **96.6%** (172/178) | 83.0% | **92.0%** | **-9.0** |
| floor_display7 | **86.9%** (305/351) | 80.6% (283/351) | **95.6%** (304/318) | 92.2% (282/306) | **91.0%** | 86.0% | **+5.0** |
| floor_display8 | 77.9% (113/145) | **80.7%** (117/145) | 87.8% (108/123) | **89.6%** (112/125) | 82.6% | **84.9%** | -2.3 |
| **TOTAL** | **82.2%** (791/962) | 80.1% (771/962) | **91.9%** (783/852) | 90.4% (763/844) | **86.8%** | 85.0% | **+1.8** |

- pi3x 总体净胜：Recall +2.1pt、Precision +1.5pt、F1 +1.8pt。
- pi3x 大幅胜 fd4（+14.7 F1）与 fd7（+5.0）；大幅输 fd6（-9.0）；fd5/fd8 基本打平。
- 预测数接近（pi3x 1017 vs da3 1003），差异来自匹配质量而非匹配数量。

## fd6 失败定位

pi3x 的 fd6 劣势集中在环形配对的前两对，其余 pair 与 da3 相当或更好：

| pair | pi3x R / P | da3 R / P |
|---|---|---|
| 1_to_2 | 40.0% / 42.9% | 93.3% / 100% |
| 2_to_3 | **0.0%** / 0.0% | 55.6% / 83.3% |
| 3_to_4 | 79.2% / 95.0% | 91.7% / 100% |
| 7_to_8 | **96.1%** / **98.0%** | 94.1% / 97.9% |

→ 假设：fd6 相机未摆平（历史 sky-noise 数据集），pi3x 对 view 1-3 的位姿/尺度估计
在该大视角差 pair 上失稳（近似米制 scale 跨视角不一致时投影全部落空）。
待查：pi3x fd6 的 camera_poses/metric 逐帧一致性。da3 1.1 在该场景位姿更稳。

## 性能对比

### 3D 重建（fd4, 11 图, 含模型加载, 单次冷跑）

| 阶段 | pi3x | da3 (2026-08-19 log) |
|---|---|---|
| 模型加载 | 15.5s | (subprocess 内, 计入推理) |
| 推理 | **2.0s** | 30.1s |
| 缓存保存 | 8.5s | (runner 内) |
| GLB 导出 | 1.7s | - |
| **总流程** | **32.1s** | 33.6s |

- 单次冷跑总耗时基本持平；pi3x 纯推理快 ~15×（2.0s vs 30.1s），若模型常驻
  （批量多数据集）pi3x 重建约 16s/数据集 vs da3 约 34s/数据集。
- pi3x RoPE2D CUDA kernel 未编译，当前走 PyTorch 慢速路径（编译后推理还会更快）。

### SKU 匹配（batch_all_refs, SAM3 mask 缓存 warm, 秒）

| 数据集 | refs | pi3x | da3 | Δ |
|---|---|---|---|---|
| fd4 | 11 | 123.1 | 106.6 | +15% |
| fd5 | 8 | 91.6 | 76.0 | +21% |
| fd6 | 11 | 140.5 | 98.0 | +43% |
| fd7 | 16 | 194.3 | 192.5 | +1% |
| fd8 | 8 | **49.1** | 72.8 | **-33%** |

- pi3x 匹配 4/5 数据集慢 ~1-43%（fd8 反快 33%），主因是 pi3 数据路径每 ref 真实加载
  解码全部图像（da3 用零存储 image descriptor，只在可视化时解码）；次要因素是
  pi3x 处理分辨率 574×434（249k px）高于 da3 504×378（190k px, +31%）。
  但 mask 采样受 `max_3d_points_per_bbox` 上限约束，不能由像素数直接推导投影与 NN
  工作量增加 31%；具体耗时贡献需要分阶段计时，以上归因不是已完成的因果验证。
- warm cache 命中时不运行 SAM3 推理；冷跑才包含 SAM3 推理开销。
- pi3x 冷跑（首次建 SAM3 mask 缓存）匹配耗时：fd4 133.8 / fd5 98.4 / fd6 145.7 /
  fd7 286.0 / fd8 96.3（fd7 SAM3 首次推理量大）；da3 的 warm 缓存同理是此前冷跑建的。

## 结论

2026-09-07 实现更新：Pi3/Pi3X matching 已使用零存储 meta image descriptor，
transforms 只读取图片尺寸，并按后端、原始帧顺序及 pixel_limit 跨 reference 复用。
匹配前不再解码整批 RGB 或将其上传 GPU；生成可视化时仍会在 CPU 解码，并保持原有
LANCZOS resize。上面的耗时是优化前结果；本次仅完成 CPU 路径验证，未重新测量 GPU
匹配耗时或全数据集 assignments/F1，不能将原有时间表当作优化后结果。

- **准确率：pi3x 总体优于 da3 1.1**（F1 86.8% vs 85.0%，fd4-8 micro），且胜在
  fd4/fd7 这种多视角数据集；唯 fd6 显著回退（前两对环形 pair 失稳），修复 fd6
  位姿稳定性后总 F1 还有 ~1.8pt 空间。
- **性能：重建打平（常驻后 pi3x 快 ~2×）；匹配互有胜负**（fd4-7 pi3x 慢 1-43%，
  fd8 pi3x 快 33%），优化方向明确（zero-storage descriptor + RoPE2D CUDA kernel）。
- pi3x 是 approximate metric（metric scale 逐 batch 估计），不接 da3 的
  is_metric==1 硬门；ground-stack-area / facing-area 等米制计量阶段不对 pi3x 启用。

## 复现

```bash
# pi3x（全量）
CUDA_VISIBLE_DEVICES=1 uv run python main.py --mode pipeline --floor N \
    --algorithm 3d --match_backend pi3x --recon_backend pi3x
# da3（缓存复用）
CUDA_VISIBLE_DEVICES=0 uv run python main.py --mode concise --floor N \
    --algorithm 3d --match_backend da3
# 评测
bash scripts/3d/evaluation/accuracy_evaluation.sh floor_displayN --backend pi3x --save-root Output
```

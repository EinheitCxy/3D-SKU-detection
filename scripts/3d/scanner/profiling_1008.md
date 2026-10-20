# PromptDA 长边1008与batch扫描（2026-10-19）

当前默认RGB长边为1008。本扫描1920×1440通过官方float32 AREA路径得到1008×756，两边均被patch14整除，无需补边。K应用半像素resize：sx=sy=0.525，offset=−0.2375；E保留Scanner w2c，LiDAR保留原生256×192网格及完整米制值，不进行事后深度缩放。

## batch扫描与默认选择

上一轮原图batch2是GPU2当时约19GiB可用显存下的保守测试起点，没有证明最优。本轮测FP32的1/4/8，以及FP16的1/2/4/8/14；数据只有14帧，batch14已经一次容纳整段，更大的上限不会增加这次请求的实际并行帧数。

实测batch收益不单调。FP16 batch2与8在独立进程与后续交错复测中都非常接近，batch14并未提高稳定吞吐，不能把batch8约0.2%的中位数优势称为显著提速。当前默认采用长边1008、FP16 autocast、batch2；FP32对照可显式选择。

## 输入、计时与环境

同一capture `572dd9f6fb`、1Hz的14帧，物理GPU2（RTX4090D 24GiB），容器内为cuda:0，Torch2.7.1+cu126。PyTorch线程为8，OMP/BLAS线程环境为8，OpenCV保留默认设置。约5044MiB其他GPU进程占用保留不动。DA3沿用已接受的504×378、Nested metric深度/E修复、原生BF16多帧联合；PromptDA是FP32权重、可选FP16 autocast。PromptDA的输出像素数为DA3的4倍，这不是等分辨率的架构比较。

每个独立进程一次load、一次cold和3次warm。预测请求从prepared PNG路径与常驻模型到CPU dense输出，含RGB读取/resize、H2D、计算、D2H与打包；CUDA在窗口前后同步。加载、抽帧、NPZ/GLB落盘和matching不计入请求。页缓存未清空，模型与allocator在同一进程内复用，3样本不支持尾延迟或普遍最优结论。

## 独立进程结果

| 配置 | 输出 W×H | 加载/s | Cold/s | Warm中位数/s | Warm范围/s | FPS | Allocated峰/GiB | Reserved峰/GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DA3 + K/E / BF16 / 14帧联合 | 504×378 | 9.014 | 2.237 | 1.334 | 1.333–1.489 | 10.496 | 11.336 | 15.123 |
| PromptDA FP32 / b1 | 1008×756 | 8.431 | 4.141 | 3.618 | 3.614–3.629 | 3.869 | 2.881 | 3.518 |
| PromptDA FP32 / b4 | 1008×756 | 8.182 | 3.965 | 3.549 | 3.523–3.573 | 3.945 | 7.316 | 9.727 |
| PromptDA FP32 / b8 | 1008×756 | 7.971 | 4.007 | 3.575 | 3.527–3.586 | 3.917 | 14.055 | 16.682 |
| PromptDA FP16 / b1 | 1008×756 | 7.979 | 2.570 | 1.906 | 1.902–1.930 | 7.347 | 3.123 | 3.471 |
| PromptDA FP16 / b2 | 1008×756 | 8.219 | 2.482 | 1.792 | 1.791–1.802 | 7.814 | 3.863 | 4.467 |
| PromptDA FP16 / b4 | 1008×756 | 8.321 | 2.467 | 1.844 | 1.841–1.870 | 7.593 | 5.802 | 6.998 |
| PromptDA FP16 / b8 | 1008×756 | 8.201 | 2.399 | 1.787 | 1.786–1.796 | 7.836 | 9.667 | 12.133 |
| PromptDA FP16 / b14 | 1008×756 | 8.049 | 2.475 | warm1 OOM | 未完成 | — | 15.477（cold） | 17.090（cold） |
| PromptDA FP16 / b14 / expandable | 1008×756 | 7.968 | 2.425 | 1.892 | 1.756–1.919 | 7.398 | 15.466 | 18.133 |

峰值是predict窗口前reset后、包含常驻模型的本进程PyTorch绝对峰；allocated与reserved不相加，也不包含其他进程。FP32 matmul=`highest`；cuDNN保留环境原有TF32设置；xFormers保持启用。未加compile或CUDA Graph。

batch14默认allocator的cold成功并导出GLB，warm1在官方DPT最后双线性上采样申请5.09GiB块时OOM，失败记录完整保留。作为单独配置，设置`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`后完成cold+3warm，allocated峰15.466GiB；该行与默认allocator行需连同分配设置阅读。此设置旨在减少分配碎片，依据[PyTorch2.7 allocator文档](https://docs.pytorch.org/docs/2.7/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。没有添加自动降batch或精度降级。

## 近似持平配置的交错复测

为避免把运行顺序或分配器预热当成batch提升，另用同一个常驻模型和同一expandable allocator：先按14/8/2预热3轮，再以交错顺序对2/8/14各测5次完整14帧请求。保留allocator状态、不释放模型、不写NPZ/GLB；分开报告，不混进上表的新进程统计。

| FP16 batch | n | 中位数/s | Mean/s | 范围/s |
|---:|---:|---:|---:|---:|
| 2 | 5 | 1.855 | 1.856 | 1.835–1.870 |
| 8 | 5 | 1.852 | 1.843 | 1.810–1.861 |
| 14 | 5 | 1.952 | 1.921 | 1.811–2.023 |

batch8相对2的中位数差约0.19%，落在本次运行波动范围内；本扫描没有更大batch带来明显吞吐收益的证据。batch14中位数更慢，显存需求更高，且本次时延波动较大。当前默认选择FP16/batch2，保留batch8作为吞吐近似相同的对照。

## 数值与输出

下表是每组完整14×756×1008预测相对同分辨率FP32/batch1的差异；processed RGB、K/E、frame IDs、网络尺寸均一致。它描述数值一致性，不是对独立LiDAR真值的准确率。

| 配置 | 平均绝对差/mm | P99/mm | 最大差/mm | >1cm像素占比 |
|---|---:|---:|---:|---:|
| PromptDA FP32 / b4 | 0.0174 | 0.1731 | 24.689 | 0.000066% |
| PromptDA FP32 / b8 | 0.0174 | 0.1745 | 18.685 | 0.000066% |
| PromptDA FP16 / b1 | 0.3633 | 2.2511 | 93.669 | 0.009308% |
| PromptDA FP16 / b2 | 0.3654 | 2.3098 | 103.210 | 0.013263% |
| PromptDA FP16 / b4 | 0.3605 | 2.2192 | 76.530 | 0.009608% |
| PromptDA FP16 / b8 | 0.3608 | 2.2280 | 80.469 | 0.009673% |
| PromptDA FP16 / b14 | 0.3607 | 2.2097 | 64.557 | 0.010545% |
| PromptDA FP16 / b14 / expandable | 0.3607 | 2.2097 | 64.557 | 0.010545% |

当前默认FP16 autocast，模型权重与保存的depth仍为float32；FP32可显式选择。数值差异如上，未证明商品边缘、matching、面积结果完全等价。

## 冷态交付分段

| 配置 | NPZ保存/s | GLB导出与回读/s | Cold请求→NPZ+GLB实际连续墙钟/s |
|---|---:|---:|---:|
| DA3 + K/E / BF16 / 14帧联合 | 1.762 | 0.889 | 4.892 |
| PromptDA FP32 / b1 | 0.101 | 2.147 | 6.404 |
| PromptDA FP32 / b4 | 0.103 | 2.200 | 6.283 |
| PromptDA FP32 / b8 | 0.101 | 2.197 | 6.320 |
| PromptDA FP16 / b1 | 0.099 | 2.180 | 4.864 |
| PromptDA FP16 / b2 | 0.101 | 2.156 | 4.753 |
| PromptDA FP16 / b4 | 0.099 | 2.209 | 4.790 |
| PromptDA FP16 / b8 | 0.106 | 2.179 | 4.698 |
| PromptDA FP16 / b14 | 0.103 | 2.159 | 4.751 |
| PromptDA FP16 / b14 / expandable | 0.098 | 2.166 | 4.703 |

PromptDA使用当前默认无压缩NPZ，DA3保留原DEFLATE；交付时间受输出格式与点数影响，不代表纯模型成本。GLB均为POINTS，保留1,000,000点上限，实际回读通过；dense NPZ保留10,668,672个深度像素。batch14失败配置的交付行只代表它成功的cold请求，不代表后续服务可用。完整流程未运行SAM3/PVS或matching system。

## 使用与记录

```bash
CUDA_VISIBLE_DEVICES=2 uv run python scripts/3d/scanner/reconstruct.py \
  --base Output/scanner_572dd9f6fb_1fps --candidate promptda \
  --promptda-repo runtime/PromptDA --checkpoint runtime/models/promptda/vitl/model.ckpt \
  --promptda-max-size 1008 --promptda-precision fp16 --promptda-batch-size 2 \
  --output Output/promptda_1008_fp16_new
```

默认`--promptda-max-size 1008 --promptda-precision fp16 --promptda-batch-size 2`；`--promptda-max-size 0`保留原图模式，504复现早期尺寸。共享cache按显式method/resolution检查并绑定实际网格/半像素affine。既有504/native cache保留原元数据，不自动升级。

复现测量时把reconstruct.py替换为profile_candidates.py，使用新输出目录，即执行一次cold+3warm；batch14调优配置在进程启动前设置上面的allocator环境变量。实际argv、GPU前后状态与退出码见[launch.json](../../../Output/scanner_1008_profile_20261019/launch.json)。完整逐次计时、RSS、GPUscope与失败原因在对应profile.json/run.log。交错复测代码在`runtime/promptda-1008-20261019/confirm_batches.py`，日志与JSON均保留。

数字见[profiling_1008_results.json](profiling_1008_results.json)。30项定向CPU检查通过，覆盖1008网络float32 RGB/原生prompt、K、batch顺序、metadata、转换及footprint输入。

[默认FP16/batch2 GLB](../../../Output/scanner_1008_profile_20261019/promptda_1008_fp16_b2/scene.glb) · [batch8对照GLB](../../../Output/scanner_1008_profile_20261019/promptda_1008_fp16_b8/scene.glb) · [FP32 GLB](../../../Output/scanner_1008_profile_20261019/promptda_1008_fp32_b1/scene.glb) · [DA3+K/E GLB](../../../Output/scanner_1008_profile_20261019/da3_ke/scene.glb)

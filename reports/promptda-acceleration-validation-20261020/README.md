# PromptDA 加速验证：d4d5b1bcd2，2026-10-20

本轮调用三位 medium subagents 分别制作初始化、请求预取和反投影采样的隔离原型；coordinator 串行执行实际计时与 GPU 实验，另一位 medium subagent 独立审查证据。生产源码、上游 PromptDA/DINO、模型权重与默认配置均未改动。之前的[静态审查](../promptda-da3-d4d5b1bcd2-20261020/ACCELERATION_PATHS.md)保留为历史依据。

**当前输入可以加速。组合原型的原视频→GLB 连续观测为 23.779→17.680 秒，少 6.099 秒（25.65%）。** 新产生的 9 个 NPZ 字段逐元素完全一致；百万点 GLB 的坐标、颜色与节点变换完全一致。两臂任务显存采样峰都是 4,996 MiB（约 4.879 GiB）。

这组连续对照每臂 n=1，顺序执行，GPU 共享负载还发生变化，因此只支持本次观测；不能当作稳定服务收益或跨视频结论。提前采样存在通用失败语义差异，仅在当前输入通过有限范围检查后执行。

## 产物与口径

- [原生 GLB](../../Output/promptda-acceleration-validation-20261020/delivery-native-raw/cloud.glb) / [组合原型 GLB](../../Output/promptda-acceleration-validation-20261020/delivery-optimized-raw/cloud.glb)：各 1,000,000 彩色 POINTS，16,000,952 bytes。没有纹理 Surfel，也没有运行 Viewer 的 SOR、DBSCAN、删地面或 SKU matching。
- [机器可读汇总](summary.json)、[完整交付时间分区 CSV](delivery_partition.csv)、[请求汇总 CSV](request_variants.csv)、[逐请求 CSV](request_runs.csv)、[加载逐轮 CSV](loader_runs.csv)、[采样逐轮 CSV](sampling_runs.csv)。
- 输入 `fd-videos/d4d5b1bcd2`，与原 profiling 相同的 20 帧；原 RGB 1920×1440，网络 1008×756，FP32 权重、FP16 autocast、batch2、原始 LiDAR prompt、原始相机与 frame IDs、seed42。
- 保存的 dense NPZ 仍是全部 20×756×1008 网格，约 106.7 MB。百万点采样仅作用于独立显示 GLB。
- Git 提交仅包含原型、文档、CSV 与精简 JSON 汇总。本文 `Output/` 链接是本机保留的原始凭据与产物，不随提交发布；重现仍需已有模型环境和输入数据。
- 计时总窗口从新进程开始准备原视频，到新模型进程完成 dense NPZ / GLB 并退出。包含准备过程、导入、模型加载、首个请求、写文件、GLB 回读和进程边界开销；不包含 launcher 前置资源查询、COS、浏览器可用时间或完整 SKU 流水线。

## 1. 初始化：跳过随后会被 checkpoint 覆盖的大权重填充

原实现先给 Linear / Conv weight 做 Kaiming 填充，DINO 又给 Linear 做 trunc-normal，随后加载已有权重将它们覆盖。原型 [load_probe.py](load_probe.py) 只跳过当前调用线程中真实 module.weight 且大小不少于 65,536 elements 的这两类填充；bias、tokens、norm 与 buffers 保持原流程。退出和异常时恢复临时 hooks。

加载仍沿用官方 `.ckpt` 的 `state_dict` 与 `key[9:]` 映射，先核对实际解码的名称、shape、dtype 与被跳过 weight 的覆盖，再 strict copy。没有 mmap、assign、meta、dtype 改动或全局关闭 init_weights。

修正了首次试验的边界：只预热 hubconf 会漏掉工厂函数延迟导入的 vision_transformer，导致首轮原生侧多计 import。`load/` 的首次结果仅留作初步记录；正式表使用 `load-imports-warmed/`，显式先导入依赖，再按 native→skip、skip→native 测两对。

| CPU 构造＋读取 checkpoint＋strict copy | 第 1 对 | 第 2 对 | 中位数 |
|---|---:|---:|---:|
| 原生 | 5.370 s | 5.084 s | 5.227 s |
| 跳过覆盖初始化 | 0.915 s | 0.706 s | 0.811 s |

每对最终 433 个 state entries 和 2 个 buffers 全部相同；没有 nonpersistent buffer。跳过 338,210,816 个 parameter elements，均被实际 checkpoint 覆盖。候选的覆盖审计成本包含在表中，约 0.003 秒。依赖预热约 1.535 秒另列，不能宣称这部分消失；本表排除 import、GPU 转移和比较成本。

GPU 实测另通过了 skip-init 模型完整请求与原生输出的逐字段比较。它是当前 checkpoint / Torch 2.7.1 的支持证据；临时内部函数 hook 仍须在生产实现时限制使用范围，不能据此宣称支持任意架构。详见[加载结果](../../Output/promptda-acceleration-validation-20261020/load-imports-warmed/result.json)与[原型说明](load_notes.md)。

## 2. 请求：下一批 RGB 与当前批 GPU 推理重叠

原流程逐 batch 执行 PNG 读取→float32/255→OpenCV AREA→RGB 打包→stack / prompt→H2D→predict→D2H，CPU 准备与 GPU 工作大部分串行。

[request_probe.py](request_probe.py) 的纯预取保留原数值操作：一个 CPU producer 准备下一批 RGB，最多一个 future；consumer 顺序完成 Torch packing、prompt、CUDA 与结果收集。producer 不执行 Torch / CUDA。没有改变网络 float32 RGB、保存 uint8 RGB 的 rint / clip 边界或零宽 padding 行为。线程创建、PNG decode、等待、最终完整 CPU 输出和退出都计入请求。

实际 20 帧 CPU 输入检查中，network RGB、prompt 及全部结果字段相同，CUDA 未初始化。生产者读取异常和消费者异常都正确抛出，退出后无遗留 producer 线程。阻塞文件系统 I/O 仍可能延迟 join，未承诺能够中断外部 I/O。

GPU 对照各变体一次 warmup 后三轮交错，第二轮反序、第三轮轮转，无变体间 empty_cache。25 次请求含一个 cold reference、五次 variant warmup、18 次正式计时和一个 skip-loader 检查，共 24 次对参考输出的比较，所有字段完全相同。

| 变体 | n | 平均秒数 | 中位秒数 | 最小–最大 | 请求 allocated 峰 GiB |
|---|---:|---:|---:|---:|---:|
| native | 3 | 2.986 | 2.977 | 2.914–3.068 | 3.863 |
| prefetch | 3 | 1.811 | 1.822 | 1.760–1.850 | 3.855 |
| reduced_serial | 3 | 3.096 | 3.093 | 3.092–3.102 | 3.855 |
| reduced_prefetch | 3 | 1.839 | 1.808 | 1.801–1.908 | 3.855 |
| delayed_d2h | 3 | 2.010 | 2.012 | 2.002–2.015 | 3.911 |
| combined_delayed | 3 | 1.781 | 1.800 | 1.738–1.804 | 3.911 |

纯预取中位数 2.977→1.822 秒，下降 38.79%。这是真实 resident-model＋PNG 路径→全部 CPU depth/RGB/metadata 窗口，不是只计 GPU kernel。

- `reduced_serial` 只减少 padding / batch 临时副本，单独启用反而略慢。没有做算子级因果定位，因此不能将原因断言为某个 NumPy / Torch 操作。
- `delayed_d2h` 留 GPU depth 到最后再取回 CPU，比原生快，但慢于纯预取，allocated 峰还比原生高约 49 MiB。
- `reduced_prefetch`、`combined_delayed` 的小差距与 n=3 的波动规模重叠；不足以证明额外复杂度值得。连续交付试验因此只用纯预取。
- 所有变体 reserved 峰约 4.467 GiB，受共享 allocator 历史影响，不能当成每臂独立冷分配。整个请求实验的自有任务 driver 采样峰为 4,996 MiB。

详见[GPU 请求结果](../../Output/promptda-acceleration-validation-20261020/requests/result.json)、[输入检查](../../Output/promptda-acceleration-validation-20261020/input/result.json)、[异常退出检查](../../Output/promptda-acceleration-validation-20261020/fault/result.json)与[实现说明](request_notes.md)。旧 profiling 的 2.603 秒不是本轮对照，不用它减新候选时间。

## 3. GLB：相同选点提前到反投影之前

原流程先把全部 15,240,960 个有效深度点变成世界坐标，检查有限性，再随机保留 1,000,000 点。原型 [sampling_probe.py](sampling_probe.py) 保持相同 valid mask、逐帧 C-order、总 N、default_rng(42).choice 结果及返回顺序；把原扁平序号映射回 frame/y/x，只为选中点执行原来的 pixel→inv(K)→depth→c2w 运算，按 choice 顺序写回 points / RGB。

| 已加载数组后的组件窗口，n=3 | 原生中位数 | 提前采样中位数 |
|---|---:|---:|
| mask→世界坐标选点与 source IDs | 1.691 s | 0.482 s |

三对 AB/BA/AB 中，100 万 source IDs、输出顺序、RGB、世界与 display 坐标全部相同，坐标逐位比较相同，最大差 0。直接分配 float32 pixel 临时数组的额外变体约 0.482 秒，没有显著新增收益。

表中不包含读 NPZ、display transform、GLB writer / readback、比较或有限范围检查。source-domain 的参考仪器开销已在计时外预计算，选中 ID gather 仍计入；候选 mapping 在计时内。不能把这个组件降幅按比例外推到完整导出。

**通用默认替换仍不通过。** 保留的反例中，一个未抽中但有限正值的巨大深度，原生全量投影发生 overflow 并拒绝；候选只检查抽中点，会接受。因此当前输入产物相同不等于全输入域报错语义相同。

连续实验额外检查了当前 float32 相机、inverse-K、全部有效 depth 最大值和像素范围的保守运算上界。当前 world 中间值上界约 6,673，远小于保守接受界 3.323e35，返回 true 才运行候选；unknown 直接停止，没有增加替代路径。这个检查只证明当前 world 计算有限，不宣称通用语义等价或单独证明坐标逐位相同。其成本 0.062 秒已计入下面的完整 GLB 导出时间。

详见[组件结果及反例](../../Output/promptda-acceleration-validation-20261020/sampling/result.json)、[原型说明](sampling_notes.md)。下一步若落地，应先确定如何保持全域失败语义；不能将此原型直接设为默认。

## 4. 三项组合后的连续交付

两臂都重新从原视频准备相同 20 帧，并在新容器/新 Python 进程加载模型，保存完整 dense NPZ 和百万彩色点 GLB。组合侧只有 selective skip-init、纯 RGB 预取、通过当前输入有限范围检查后的同索引提前投影。

| 实测窗口 | 原生秒数 | 组合秒数 |
|---|---:|---:|
| 原视频准备进程 | 4.767 | 4.871 |
| 模型加载：含首次模型依赖与转 GPU | 8.668 | 4.448 |
| 完整首个请求：PNG→CPU 输出 | 3.962 | 2.727 |
| 保存完整 dense NPZ | 0.145 | 0.145 |
| GLB 导出，含格式回读 | 2.694 | 1.767 |
| 脚本开始→GLB 完成（父窗口） | 17.167 | 10.818 |
| 容器调用→退出（父窗口） | 19.012 | 12.808 |
| 原视频准备开始→容器退出（总窗口） | 23.779 | 17.680 |

父窗口与子窗口存在包含关系，不可直接逐行相加。[delivery_partition.csv](delivery_partition.csv) 单独给出互不重叠的完整总窗口分区，含导入/准备和容器启动退出的剩余时间。其他表是嵌套观察窗口，不与它混加。

两臂保存的 NPZ 都是 106,691,298 bytes，GLB 都是 16,000,952 bytes。自有任务显存 driver 采样峰均 4,996 MiB，树 RSS 采样峰约 3.206 / 2.670 GiB；树 RSS 可能重复计共享页，采样可能漏掉瞬时峰，因此只能称观测峰。

独立运行的 [evaluate_outputs.py](evaluate_outputs.py) 直接读取这两次新输出，而非复用旧缓存的结论：9 个 NPZ 字段的 key、dtype、shape、值全部相同；GLB 的百万点坐标、颜色、geometry 关联和节点变换相同。最大坐标差 0。这证明当前输入的可视化载荷相同，没有进行额外渲染目测或独立重建精度评测，也没有比较 ZIP / GLB 文件的逐字节身份。

原始凭据：[原生 result](../../Output/promptda-acceleration-validation-20261020/delivery-native-raw/result.json)、[原生 launch](../../Output/promptda-acceleration-validation-20261020/delivery-native-raw/launch.json)、[组合 result](../../Output/promptda-acceleration-validation-20261020/delivery-optimized-raw/result.json)、[组合 launch](../../Output/promptda-acceleration-validation-20261020/delivery-optimized-raw/launch.json)、[直接产物比较](../../Output/promptda-acceleration-validation-20261020/evaluate/result.json)。

## 5. 共享负载、结论边界与后续顺序

实际运行在物理 GPU1（容器逻辑 cuda:0），使用原有本地 Docker 镜像，network=none，CPU / BLAS / Torch 线程数 8，容器 RAM 限额 24 GiB。已有服务 PID7262 与其他用户 PID3217393 保留；原生交付期间又出现无关 PID3856119，占用由约 1.2 GiB 增至约 4 GiB，组合臂开始时该进程已驻留。完整 before / samples / after 留在各 launch JSON，未触碰这些任务。

因此完整交付只能报告这一对运行的观测降幅；GPU 请求的三轮交错增强了局部比较可信度，但仍是小样本共享 GPU。没有多视频均值、P95、隔离设备复测或长期服务并发证据。不同窗口的收益不叠加成预测值。

| 方向 | 本轮判断 | 下一步 |
|---|---|---|
| 跳过 checkpoint 覆盖初始化 | 当前模型状态与请求等价，CPU / 新进程观测均更快 | 优先设计为隔离 loader 实现，保留当前支持边界 |
| 一个 batch 的 RGB 预取 | 当前输入精确相同，resident 请求明显更快 | 优先落在 Scanner producer，不改 root pipeline 的 cache 解耦 |
| 同索引提前投影 | 当前输入相同且更快；通用失败语义不通过 | 先解决或明确有限输入域，暂不默认替换 |
| 单独减少副本 | 实测稍慢 | 不作为加速发布 |
| 更多预取 / D2H 组合 | 相比纯预取没有可靠新增收益 | 暂缓，避免增加维护与显存负担 |
| 常驻 GPU / CPU 待命服务 | 本轮没有设计或实施生命周期 | 可进一步研究，但涉及并发、退出和资源所有权 |
| compile、batch 扫描、mmap/assign、GPU resize、自写 GLB writer | 未实现 / 未测量 | 排在已验证的结构性优化之后；不能承诺收益 |

生产源码和默认配置未修改；本目录仅发布隔离实验与研究记录，未部署。本轮研究已完成，独立证据审查见 [review.md](review.md)。

## 复用与重现

只重新生成汇总（不启动模型）：

```bash
UV_CACHE_DIR=/tmp/3d-profile-uv PYTHONDONTWRITEBYTECODE=1 \
uv run --no-project --offline \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/promptda-acceleration-validation-20261020/summarize.py
```

实际实验由 [run_experiment.py](run_experiment.py) 串行启动，各次实际命令、设备和资源采样保存在 `Output/promptda-acceleration-validation-20261020/*/launch.json`。原目录禁止覆盖；任何重测需先检查实时资源并用新的 `--tag` 留档。这里的报告生成脚本仅消费现有 JSON，没有模型或测试执行副作用。

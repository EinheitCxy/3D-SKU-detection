# PromptDA 加速路径审查（2026-10-20）

本文保留实现前的静态审查。后续隔离原型、实际 CPU/GPU 对照、连续交付及输出等价验证见[加速验证报告](../promptda-acceleration-validation-20261020/README.md)；其中单次共享 GPU 交付观测下降约25.65%，提前采样仍不具备通用默认替换条件。

本次由三位 medium、只读 subagents 分别审查加载、请求执行和点云交付，coordinator 综合源码与既有 profiling。没有修改生产算法、权重、分辨率、精度或采样策略，没有启动新 GPU 实验。下列收益均为候选工作量或现有时间窗口；没有新的加速结果。

## GLB 下载

- [PromptDA：1008 / FP16 / batch2](../../Output/promptda-da3-d4d5b1bcd2-20261020/promptda-full-ready/artifacts/cloud.glb)，16,000,952 bytes。
- [DA3：504 / 原生 BF16 / 20帧联合](../../Output/promptda-da3-d4d5b1bcd2-20261020/da3-full/artifacts/cloud.glb)，16,000,948 bytes。

两份各100万彩色 POINTS。原始 Scanner 导出路径没有运行 Viewer 的 SOR/DBSCAN/平面清理或纹理 Surfel。dense prediction 分别保留1524万/381万深度像素，GLB采样不改变dense NPZ。

## 当前测量基线

| 口径 | PromptDA |
|---|---:|
| 原视频→NPZ/GLB完成并进程退出，n=1 | 23.124452 s |
| 该完整运行中 prepared-input + model load | 9.194604 s |
| 独立 baseline model load，含首次依赖导入 | 8.463265 s |
| 常驻模型完整请求 warm median，n=3 | 2.602624 s |
| 独立诊断反投影父区间 | 1.311909 s |
| 实际配置 | 20帧、1008×756、FP32权重、FP16 autocast、batch2 |

完整数据与所有窗口定义见 [profiling 报告](README.md)。组件、父子区间、不同请求之间的秒数不能相加成新端到端收益。

## 推荐顺序

| 优先级 | 方向 | 现有证据 | 预计影响范围 | 必须先证明的内容 |
|---|---|---|---|---|
| 1 | 选择性跳过被权重覆盖的初始化 | uniform_ self累计4.120s；Kaiming累计2.285s、DINO final init累计1.985s | 主要影响每次新进程的冷启动；可触及数秒窗口，实际收益未知 | 被跳过的真实weight全部被checkpoint覆盖；最终参数、buffers和推理保持要求 |
| 2 | GLB在反投影前选择原来相同的source indices | 反投影1524万点后只交付100万；父区间1.312s，后置point/color gather约0.125s | 目标点级计算量降约93.4%；时间不会按相同比例下降 | 同一valid域、seed、source IDs、RGB和输出顺序；坐标浮点差异与原有失败语义 |
| 3 | 单CPU producer、有界预取下一batch | PNG decode1.030s，GPU predict stream合计1.013s | 主要改善2.603s warm请求；理想可覆盖约0.9s GPU窗口，实际更低 | 请求仍包含PNG读取和完整CPU输出；运算相同、帧顺序/异常传播正确、RAM有界 |
| 4 | pad=0绕过副本、预分配及减少CPU临时数组 | float32/255约0.189s，RGB pack0.151s，np.pad0.040s，batch stack0.022s | 相关窗口约0.40s；必须的运算仍在，且可能被预取隐藏 | 网络float32 RGB与保存的uint8 RGB不变；不改变AREA/量化边界 |
| 5 | 减少逐batch D2H同步 | host等待0.533s，CUDA复制区间0.016s | 可改变CPU/GPU重叠；0.533s不能当纯传输可省额 | 输出完整、CPU读取时copy已完成；新增显存和实际交付延迟测量 |
| 服务化方向 | CPU预加载待命worker，或GPU常驻模型 | baseline load8.463s；模型resident allocated约1.268GiB | 可摊薄后续请求的import/构造/权重加载；一次全新CLI无此现成收益 | 明确请求生命周期、并发与资源所有权，再测完整第二请求及空闲占用 |

## 1. 初始化：需分别处理两轮填充

`load.dinov2_network_construct=7.376584s` 包住 `torch.hub.load`。CPU profile进一步拆分：

- 真正 `DinoVisionTransformer.__init__`：4.069739s，[cpu_profiles.csv](cpu_profiles.csv) 第312行。
- `vision_transformer` 首次模块 import：3.301674s，同文件第319行；涉及xformers/torch._dynamo导入。

所以7.377s不能全当作随机初始化成本。把import提前只改变计时边界；复用进程才能在后续请求摊薄它。

现有 [DA3 selective skip-init](../../src/da3_model_loading.py) 选择性跳过大Linear/Conv的Kaiming weight填充，保留bias等初始化。PromptDA DINO还会对Linear执行第二轮 `trunc_normal_`，源码在：

`runtime/promptda-counterpart-20261019/PromptDA/torchhub/facebookresearch_dinov2_main/vision_transformer.py:170`（模型初始化入口）、`:331`（Linear初始化）。

需要按真实module.weight身份分别控制两条路径，保留pos_embed、cls_token、norm、bias及buffers。禁止整个关闭 `init_weights()`。当前DA3 coverage审计读取safetensors，PromptDA加载 `.ckpt` 中的 `state_dict`，审计不能不加修改直接复用。

Kaiming 2.285s与DINO final init 1.985s是可调查的两个观察窗口；其中还包含必须保留的初始化及调用开销。它们不是保证可消除的4.27s。`uniform_`的4.120s是另一层self统计，不与这些父区间再相加。

最低成本验证：先在CPU做原生loader/选择性loader一次A/B，严格加载，比较最终参数、persistent/nonpersistent buffers和状态覆盖，同时计构造、读权重、赋值与RSS。通过后再在本20帧上测真实输出与冷交付，不先跑完整多场景回归。

## 2. 同一百万点先采样，再反投影

当前 [scanner_io.py](../../scripts/3d/scanner/scanner_io.py):84–119 先逐帧构建全部有效世界点及RGB，concatenate后才 `default_rng(42).choice(N,1_000_000,replace=False)`。

等价选点方案：

1. 保留原 `prediction_mask`，取得各帧valid点数与累计偏移；顺序仍为逐帧、帧内C-order。
2. 对相同总N调用原来的 `choice`，保留原随机返回顺序。
3. 将选中扁平序号映射成 frame/y/x，按帧计算所需点。
4. 按其在原choice结果中的位置写入预分配points/colors，保持输出顺序。
5. 保留原 `pixel@inv(K).T → multiply depth → c2w → 最终display transform` 的运算顺序。

不能换成每帧定额采样或按帧重新拼接。小矩阵batch可能令BLAS选择不同kernel，所以数学等价仍不能提前承诺坐标bitwise相同。先对照source IDs/RGB/顺序和坐标，若有浮点差异单独报告再判断。

还需保持原来“所有dense世界点有限”的失败语义。当前函数在采样前检查全量点；新版本仅检查抽中的点会改变未抽中异常点的报错行为。这一边界尚未解决，当前方案不能直接称为完全等价实现。

像素临时数组也可改进：当前 `column_stack(x,y,np.ones)` 先创建float64，再cast float32；后续matmul已经float32。可以直接分配float32 `(n,3)` 并填x/y/1，减少临时数组。若已前移采样，该收益随点数缩小，不能再次宣称省下全量0.264s pixels构造窗口。

本项只优化独立POINTS GLB的显示点投影。dense NPZ、[PromptDA cache bridge](../../src/promptda_cache.py) 和后续SKU几何仍需要完整网格，不随百万点GLB一起缩减。

最低成本验证：复用已保存prediction/sensor，在CPU比较原导出与候选，无需模型/GPU。固定原seed与valid域，检查原source IDs、颜色、顺序、坐标和失败边界；保持一次必要GLB格式回读。

## 3. 有界预取与CPU去重

[candidates.py](../../scripts/3d/scanner/candidates.py):141–194 当前每batch串行执行decode、float32/255、OpenCV AREA、RGB打包、stack、H2D、predict、D2H。

先保留原数值操作，使用1个CPU producer、最多1个排队batch，在当前batch执行GPU时准备下一batch。producer只操作CPU数据；consumer负责CUDA与按原frame IDs收集输出。异常需传回consumer，退出需结束producer，不能遗留阻塞队列或后台任务。

一个batch2的float32 CHW约17.4MiB，uint8 RGB约4.4MiB，另外有原图解码和float32 scratch；额外RAM应为几十到约百MiB并实际测量。不能仅把预处理移出请求计时。对照仍为resident模型+PNG路径→完整CPU depth/RGB；同时补raw→GLB实际交付。

然后单独调查pad=0时跳过np.pad、预分配batch/最终输出、浮点scratch复用。网络继续使用float32 `/255` 后的OpenCV `INTER_AREA`，保存RGB继续走原rint/clip/uint8。metadata只需尺寸/通道但当前完整decode，同一PNG在预测又decode；重复读取可研究，原图绑定与cache RGB核对仍保留。

## 4. D2H、batch和GPU预处理的优先级

每batch立刻 `.cpu()` 限制了主线程继续做后续准备。可单独测量先保留10个GPU深度结果、循环结束统一取回CPU；20帧float32深度新增约58.1MiB存储，cat等操作还可能再要同量暂存。实际峰值必须重测。它与预取覆盖相同的部分等待，收益不能相加。

当前H2D RGB stream只有约0.047s。单独加non_blocking不应预期巨大收益；若采用pinned buffer和独立copy stream，必须在copy完成后读取CPU结果或复用源buffer，遵守stream/event生命周期。[PyTorch2.7 CUDA说明](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution)。

本20帧若扫batch，先测2/4，两者均整除20，再判断是否需要8。历史14帧视频的batch2/8近似持平只属于原场景；增batch会增加激活显存，也可能改变kernel与浮点输出。

GPU resize暂不优先：当前AREA仅约0.066s，GPU实现与OpenCV AREA是否等价未证实，原图H2D和保存RGB D2H还会新增开销。当前encoder已使用FlashAttention。compile的首次成本、图断点、显存和完整请求收益均未知，排在上述结构性工作之后。

## 5. 服务生命周期与现有解耦边界

Coordinator另检查了 [DA3重建器](../../src/da3_3d_reconstructor.py):60–109、[DA3 worker](../../src/da3_worker.py):1–19：现有待命worker预加载模型到CPU，单次请求才转GPU，请求结束进程退出。它为PromptDA提供了“CPU预加载、空闲不占GPU模型内存”的可参考方案。

PromptDA本次测得模型allocated约1.268GiB；经过请求后allocator reserved约4.467GiB。GPU常驻时这两种数字应分开报告，还要与其他stage的同一时间线核对完整峰值。CPU预加载方案仍会付出每次to(cuda)的约0.517s观察窗口，实际请求收益需要实验。

当前 [main.py](../../main.py):1069–1080 的 `match_backend=promptda` 消费显式预转换场景cache；模型运行由Scanner producer完成。加速应先落在producer的loader/predict/export中，保留这条解耦边界。引入生产worker生命周期需要单独设计并按已有规则确认，当前审查不改根pipeline接口。

## 暂缓项

- **mmap/assign**：当前checkpoint读取约0.778s、赋值约0.100s、完整checkpoint约0.895s；比初始化小。mmap的触页成本可能转移至消费阶段，assign会改变参数对象语义。官方recipe提供机制，不是本项目收益证明。[PyTorch checkpoint加载说明](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)。
- **重写GLB writer**：当前scene.export约0.415s，回读约0.015s；直接写POINTS可能减少通用库分配，但增加二进制格式维护。优先解决1.312s全量反投影，不能把整个0.415s视为可省。
- **减帧、降分辨率、换精度或减显示点数**：属于质量/工作量取舍，不包含在当前保持效果的优先方案中。

## 结论与状态

一次新CLI优先调查selective skip-init；独立GLB交付优先调查同source-index的采样前移；常驻请求优先调查有界CPU预取。三项分别影响加载、导出和请求执行，可分开验证后再测组合。服务连续请求可另外评估CPU待命worker。

三位subagent只读审查完成；coordinator已核对下载文件与运行边界。尚未实现候选、未启动新性能实验，未宣称新增加速或视觉等价结果。

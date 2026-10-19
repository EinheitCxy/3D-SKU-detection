# 2026-10-19 发布及可视化影响审查

三个medium reviewers分别只读审查模型/匹配及Git、几何过滤、Surfel交付；coordinator核对分组和发布边界。范围为10月17–19日相关提交及本轮待提交内容，不是全仓库安全审计或浏览器画质评测。

## 本次发布选择

| 模块 | 纳入的实现 | 默认/显式 |
|---|---|---|
| PVS | 跳过checkpoint覆盖的大weight初始化，保留其余构造和strict load | 默认开启 |
| Matching | 同一次投影的bbox hit counts复用/批量计算 | 默认采用 |
| NN/SOR | 同完整float64域的一棵cKDTree、共享KNN结果 | Viewer准备默认采用；低层原Open3D控制保留 |
| GLB | 正式CUDA采样/mask/RGB延展、固定shape、encoder4 | 显式CUDA/fixed shape/4；CPU默认2 |

不纳入今日process参数/CLI/worker、GPU ground scorer或native DBSCAN helper；这些源码保留本地实验。`main.py`本次不改。混合的Viewer准备文件只选择共享KNN调用，混合的geometry测试只选择共享KNN测试。

Git初态为main、HEAD `d38f12a`、暂存区为空，29个tracked dirty路径及4055个untracked文件已记录在本地receipt。已有ahead2：`1edf6ff`、`d38f12a`，仅历史DA3 allocator/GPU bake实验与正反结果，无生产默认显存限制。origin实时核验为公开`EinheitCxy/3D-SKU-detection`，默认分支main；目标只做普通快进推送，不改历史。原始视频、图片、模型、GLB/NPZ、环境、采样CSV和日志不进入新提交。

发布前以HEAD源码加精确选择的31个文件构成独立smoke目录，未带入process worker/GPU ground，使用原`main.py`。相关PVS、hit-count、共享KNN/过滤/Viewer、GPU采样边界、CLI与编码队列/失败清理测试共**77 passed，12.16秒**；视频脚本`bash -n`及`--help`通过。第一次smoke仅因导出目录漏复制已有tracked `da3_defaults.py`而停止收集，补齐原依赖后通过；没有新增用户untracked模块或修改生产逻辑。未重复真实GPU实验。

## 为什么GPU ground暂不通过

固定相同1000个NumPy三点候选、3425675个全域点时，GPU与参考的内点数、胜出候选542、winner mask及同CPU公式refine完全相同；RMSE最大绝对差`2.3418766925686896e-16`。这种float64规约舍入差本次没有改变胜出结果，但在近乎并列的候选间仍需检查择优。

主要缺口是原Open3D随机候选、并行早停及候选选择未复现。加速评分器只证明“给定同一候选列表”的结果，不能证明整个RANSAC会给出同一列表和地面。换地面可能改变后续删点，所以原Open3D默认保留。这不是观察到画质退化后的判负。

`85.947→2.217s`是自定义Python/NumPy参考与GPU固定候选评分对照，不能作为原Open3D约10.654秒ground的提速比。后续若继续，应先取得原候选与停止/择优语义，再做完整过滤mask与交付验证。

## 为什么进程隔离暂不通过

| 同输入组件截止点，秒 | thread | process |
|---|---:|---:|
| PVS就绪 | 3.646652 | 3.008257 |
| 几何准备就绪 | 14.982830 | 20.370489 |
| 双方全部就绪 | 14.983108 | 20.370611 |

全部10项输出一致，问题是截止点慢了5.387504秒。隔离略改善PVS，但准备支线成为更慢的瓶颈。源码包含临时npy写入、新uv/Python进程、npz结果回传和父进程PCD重建；这些子段没有独立秒数，不能把差额全部归因于IPC或GIL。此对照使用原Open3D统计，发生在共享KNN接线之前，不能冒充process与新共享线程的配对测试。

本次采用共享KNN减少实际邻域工作和互阻；不额外发布更慢的process维护接口。

## 哪些效果验证已经成立

- 10/18安全默认前后：`reports/latency-optimization-20261018/pair-validation.json`记录DA3、mapping、Viewer全部输出及GLB payload一致。
- 10/19完整video3、29帧、DA3子进程seed42：三个臂15项DA3数组、8份Viewer二进制、manifest、objects和mapping一致；GLB所有accessors及31页128295600个decoded RGBA texels一致。
- video6、48帧固定generation：GPU dynamic2/dynamic4/fixed4的几何及49页200736000个decoded RGBA texels均与CPU参考一致。
- video3/6原raw域4092480/6773760点：共享KNN全部NN距离、median及raw SOR mask一致。video6不是共享KNN的完整流水线对照。

本次保留原source_world投影、f16/JPEG边界、radius2、tile8、LOD、palette128、二值alpha、6倍native长边限制、商品保护、20%guard、原DBSCAN与ground。video3最终仍为2402589个Viewer点，其中1902589商品点加500000背景点；没有提前缩减过滤点域。

完整video3核心交付从183.533016到137.642563秒，driver采样峰均15132MiB。时间从帧就绪计，不含检测/分类、COS或浏览器；每臂n=1，共享主机，不代表新多视频均值/P95或瞬时显存上限。

## 两天所有历史改动不能统称纯加速

| 历史变化 | 效果边界 |
|---|---|
| `21ea29c` DA3 skip-init、`092047d` typed point ranges、`b59d0ff` 有界编码 | 按已测默认条件保持输出的加速；见各自对照 |
| `741a52d`显式`camera_basis` | 实测会改变少量边界alpha/RGB；默认仍source_world |
| `bb5e276`显式ground proposal cap>0 | 改变候选/过滤域；50K实验比基线多删51286背景点，默认cap=0 |
| `6d07c9a`显式proxy/early background | 改变邻居计数或过滤输入域；默认均0 |
| `6d07c9a`视频脚本输出 | 接入完整Surfel bundle，是输出能力变化 |
| `2b65e0f`默认quiet outputs | 不再生成辅助图片/可选报告，是用户已要求的产物行为变化 |
| `1edf6ff`、`d38f12a` allocator实验 | 仅实验探针；video6的12GiB cap曾OOM，不作为本轮要求 |

`6d07c9a`没有新增0.4m厚度拒绝，也未改变matching模块；该规则不能归因于本次加速。

PVS跳初始化会减少构造期torch RNG消费，不能说内部随机轨迹不变。生产SAM cache miss/warmup已有`_preserve_cache_boundary_rng`恢复Python/NumPy/torch CPU及目标CUDA状态，隔离后续matching采样；直接调用loader本身没有该守卫。DA3在独立子进程中，构造期消费改变不影响父matching RNG。严格checkpoint覆盖和固定seed实验不证明任意输入/调用方式绝对相同。

首个新control只seed父进程，遗漏DA3子进程seed42，确有depth/scale及画面payload差异，已经排除。成功结果使用修正后的研究probe；没有放宽比较标准或改变生产重建算术。

因此，本轮发布目标和已测结果支持“保持既有点云及GLB效果的加速”；不支持“最近两天所有可选参数、任意输入及浏览器渲染均绝对不变”。未做新的浏览器人工对照或部署验证。

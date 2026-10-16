# RGB 完整对照：已接受并集成

RGB-02 已完成一次固定 video3 全场景 B1/C1/C2/B2 对照，独立评估通过全部 hard gates；Coordinator 已接受并集成 `_dilate_rgb` 的 `np.roll` → slices 替换。
两组完整 CPU exporter wall-time 分别节省 36.89526513963938 s、31.11267403513193 s，锁定主指标为其中较小值。
该结论仅覆盖本场景、固定配置和本次序列；旧轮仍为 inconclusive，不能据此推断稳定百分比、GPU、完整流水线或 Viewer 首屏收益。

## Evidence

下表先绑定 claim 与证据；E/O/S/I/D/P 分别对应文末登记 artifact。全部输入均来自本任务 context manifest；未读取未登记结果。历史失败记录作为限制证据保留。

| claim_id | Claim | 支持 artifact IDs | 反向/不充分证据 IDs | Metric/run evidence | 外部 citation IDs | 状态 | 限制与未消除 confounds |
|---|---|---|---|---|---|---|---|
| C1 | 本次完整 exporter 观察到正收益 | E: artifact-d651eb67cc7d-2708ea6a; O: artifact-632f12dfc23b-920d8bd3 | 无本轮反例；P: artifact-0cbcbd2fe686-87ec5fa2 限制历史合并 | O: 两组差 36.89526513963938 / 31.11267403513193 s；min=31.11267403513193 s | 不适用 | accepted，两个 pair 均 >0 | 单场景、seed=0、共享主机、同进程 cache/order |
| C2 | 本轮比较范围内产物等价 | E; O | 无本轮反例 | O: 每 arm 31 atlas / 121513344 bytes / 2564994 triangles；全部完整比较 true | 不适用 | accepted | 对本轮 B1 比较，未使用历史 GLB；无真实 Viewer 质量新实验 |
| C3 | 集成保留既有语义且通过工程验证 | I: artifact-39b4195fa58d-6a958c10; D: artifact-995cbf170a7f-8ea9977c; artifact-faf57a6294b7-67f60741; S: artifact-91400176e6d3-ef54995d | 无记录反例 | I: 59 passed；D: 新增 25 cases；S: alpha/diameter 等 smoke | 不适用 | implemented + verified | 只集成局部函数，无部署或远程发布 |
| C4 | 正式运行满足资源合同 | E; O | 无超限记录 | O: CPU1323.846443 s、RSS2.251983643 GiB、GPU0；disk≤0.5625 GiB | 不适用 | accepted，测量范围明确 | disk 为结构上界；RSS 采样不能排除采样间峰值 |
| C5 | 旧轮失败不能被新结果或 continuation 修复 | P | 无 | P: 原 B2 缺失；不同采集成本、争用；continuation 无 baked reference | 不适用 | 历史 inconclusive 保留 | 失败针对旧执行与证据设计，不证明 slices 算法普遍失败 |
| C6 | Encoding、NN/SOR、DBSCAN 尚无新完整收益结论 | P; artifact-dcf68415693d-6bf6ea2e | 无完整正式对照支持收益 | P: encoding 完整对照取消；DBSCAN smoke orientation 写入问题；NEXT: NN/SOR 依赖分析 | 不适用 | 仅后续方案 | pilot、数量变化或依赖分析均不能替代正式计时与质量验证 |

正式序列与计时（O；完整 exporter outer wall，单位 s）：

| 顺序 | Arm | 原始 wall-time | 完整 export / 比较 |
|---|---|---:|---|
| 1 | B1，原 roll | 157.39404551684856 | 通过 / 通过 |
| 2 | C1，slices | 120.49878037720919 | 通过 / 通过 |
| 3 | C2，slices | 118.5134458206594 | 通过 / 通过 |
| 4 | B2，原 roll | 149.62611985579133 | 通过 / 通过 |

`B1−C1=36.89526513963938 s`，`B2−C2=31.11267403513193 s`；主指标 `min=31.11267403513193 s`。不扣诊断成本，不从成功子集取值，不报告 population CI 或稳定百分比。
一次进程完成全部顺序，固定 seed=0 与 CPU affinity `{3,22,27,29}`；共享主机并非独占。锁定合同见 [evaluation-spec](evaluation-spec.json)。
输入为 generation `c9c3be6802144fcf97a942085b118422` 的完整 29 帧、2402589 点，`global_id=None`；radius2、tile8、atlas2048、batch1024、LOD=true、palette128 固定。

每个 raw timer 均从加载/导出前开始，至完整输出关闭后结束；都在 palette quantization 前用相同 hook 复制全部页面。
每 arm 复制数组均为 670092600 bytes，复制成本保留在 raw timer 内。页面与最终 GLB 的比较均在该 arm timer 外，未减去比较开销再构造主指标。
所有 31 页的完整量化前 RGBA（包括 alpha=0 像素的 RGB）、positions、UV、indices、dtype/shape、页面顺序及 source/object/LOD 输入与选择均一致。
最终 typed accessor payload、decoded RGBA、primitive/page/material/texture/source/sampler 与 scene/object associations、文件大小均一致；每 arm 均为 121513344 bytes、2564994 triangles。
参照为本轮 B1，未借历史 GLB 补证。比较覆盖语义和表示字段；此处不额外宣称整个 GLB 文件逐字节相同。

集成仅改变 `src/surfel_mesh.py::_dilate_rgb`，保持扩散轮次、四方向顺序和 first-direction tie 规则，不改变 alpha。
`FINAL_DIAMETER_MAX_NATIVE=6.0`、半轴 cap `3*max(native axes)`、source-depth tolerance 与 coverage 规则保持不变（E、patch）。
已有 mesh + GLB suite 合计 **59 passed**，其中新增 **25** 个原 roll 对照、空/单维/非方形、多 batch、透明 RGB、in-place 和 tie 回归 cases（I、patch）。
测试命令为 `uv run --no-sync --project /home/xingyu/3D_Recognization python -m pytest -q test/test_surfel_mesh.py test/test_surfel_glb.py --import-mode=importlib`。
I 记录 returncode=0、CPU9.387438 s、wrapper wall8.578560 s；pytest 自身显示 59 passed in5.18s，三者计量边界不同。
最终 smoke 验证 synthetic 边界、真实 1024-point batch、双方既有 mesh alpha/diameter 行为及采集接口；smoke 不作为正式性能样本。

资源解释（E、O、S、initial smoke、I、D）：

| 项目 | 数值 / 范围 | 解释 |
|---|---:|---|
| Official aggregate process CPU | 1323.846443 s | 含比较及 children 计量；不等于 exporter wall saving |
| Official runner / outer command wall | 604.536592 / 605.378 s | 后者包含 launcher 开销 |
| Official RSS | 2.251983643 GiB | 0.1s self/direct-child 采样与 self ru_maxrss 的较大值；未观察到 children |
| Official 新增磁盘 | ≤0.5625 GiB | 强制结构上界：两份≤256MiB 文件 +64MiB metadata；不是实测 peak |
| GPU | 0 s | 空 CUDA device 环境及 CPU-only 路径，无 GPU 性能结论 |
| 两次 smoke CPU | 11.033340 +13.318294 =24.351634 s | initial 与 final 都计入开销；未混入 official 主指标 |
| 实测执行 CPU 合计 | 1357.585515 s | official + 两次 smoke + integration；D 明确的执行范围 |

另有 readiness 10 CPU s 保守预留，未测量，不能并入“实测 CPU”。这也不是所有分析任务的全覆盖 CPU 账单。
硬件预算保持 CPU2400 s、RAM12 GiB、新存储1 GiB、GPU0；worker 次数与累计 worker wall 按最新用户授权不设总限，最多3个 medium leaf 并发，每项任务仍有限。
RAM 中最多保留 B1/current 页面；baseline GLB BIN RAM 为121431912 bytes。RLIMIT_AS 是 guard，不能替代 RSS 测量；目录可见磁盘量不能证明 unlinked temp peak。
本报告无外部科学 claim；外部 citation 核验不适用，所有工程数值均绑定上述本地登记 artifact。

## Ideas Considered

保留 slices：它只移除 roll 的整数组搬运方式，完整场景两组均正收益，并通过产物等价及集成验证；Coordinator 的 keep/current-best 只适用于已锁定证据范围。
下一候选为 bounded parallel encoding，应从当前已接受 RGB baseline 增量对照，避免把两种干预混为一项收益。
NN median/SOR 可探索提前算 raw stats/mask，与 matching 重叠；这仍是依赖分析，尚未测得加速。

## Rejected Ideas

不接受以旧 continuation 修复旧结果：原 B2 缺失、CPU 争用、timer 内验证不一致以及缺失 baked reference 都保留，旧 verdict 不改。
不把 encoding pilot 外推到完整导出；旧完整 encoding 实验已取消且未集成。DBSCAN-off smoke 曾误调用可写 orientation API，无可接受质量/正式计时结论。
不将本轮收益从历史流水线总耗时中直接扣减：历史 all-chain267.565 s、Viewer 可用121.214 s、GLB137.389 s 仍只是旧基线。
已有 Surfel prep 与 matching 重叠2.999 s 在本轮前已实现，不能再次记作 RGB-02 收益（P、pipeline summary）。

## Risks and Confounds

仅一个诊断场景、一个 seed、两个 paired observations，没有 heldout 或跨场景复现。反向端点 baseline 减轻但不消除 warm-cache/order 与共享主机干扰。
完整页面复制使该测量合同可比，但不能等同无 capture 的部署耗时；本轮没有新的 GPU pipeline、浏览器首屏或交付时延测量。
逐页/最终语义等价与回归测试支持这项局部集成；不构成新真实 Viewer 视觉验收，也不能替代后续过滤默认变化的质量验证。
资源结论受 RSS 采样与磁盘结构上界边界限制；不将它们改写为连续全树 RSS 或磁盘实测 peak。
本轮成功解决旧轮关键证据设计缺口；它不改写旧失败，也不支持对任一算法作普遍成败判断。

## Recommended Next Action

Coordinator 完成 portable 证据归档与文档提交；本任务只提供报告，不进行 push、部署或其他发布操作。
优先另立 encoding 增量实验：≤2 encoder threads、bounded queue、单 writer；保持 MEDIANCUT128、dither=NONE、optimize=True，检查 PNG bytes 与 decoded 语义。
继续使用完整 B1/C1/C2/B2、相同 timer/capture/比较边界，禁止并发 benchmark；执行前另行批准硬件预算，不能沿用本轮剩余量自动启动。
NN/SOR 提前 raw stats 最多隐藏实际重叠的 matching 窗口；最终 protected apply 必须等待 labels，随后才能 DBSCAN，不能整段提前。
DBSCAN 应先固定只读 frozen orientation，再按预定真实 Viewer 质量条件对照；在此之前不改默认过滤策略。

## 归档与证据

本目录从已接受的 synthesis report 精简导出，仅调整标题与证据链接。原始结果未经重跑或改写；旧轮失败保留在[历史报告](../README.md)，最新方向见[下一步](../NEXT.md)。

- [完整四臂原始结果](official.json)、[正式命令回执](official-command.json)、[独立评估 E](independent-evaluation.json)。
- [最终 smoke S](smoke.json)、[初始 smoke](smoke-initial.json)、[集成测试 I](integration.json)、[精确源码/测试补丁](source.patch)。
- [固定参数](binding.json)、[锁定接受标准](evaluation-spec.json)、[实际运行脚本](full-export-runner.py)。脚本保留原始绝对路径和 artifact ID，重放依赖本机 runtime 中的冻结输入及 Viewer generation，不是独立可移植 CLI。
- [历史流水线汇总](../results/pipeline-summary.json)、[组件表](../results/pipeline-detailed-tables.json)。表中的 D 为本机 registry 内已接受的 coordinator integration decision，P 为历史报告快照。

完整 registry、冻结依赖和 task reports 位于 `runtime/rgb-dilation-validation-20261016/.research/`。各 arm 的新 GLB 已在完整比较后按批准方案释放；模型、generation、cache 和大型数组不进入本归档。代码与文档仅在本地提交，未 push 或部署。

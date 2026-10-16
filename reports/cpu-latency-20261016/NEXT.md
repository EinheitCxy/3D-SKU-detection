# 下一轮耗时测试方案（未执行）

旧实验仍为 inconclusive；原失败、缺失 B2、CPU 争用及验证开销不一致的结论保留。
本方案不修改旧 EvaluationSpec，不恢复旧实验，不代表新预算或执行已获批准。

## 先完成 RGB 完整导出对照

- 使用独占、显式指定的四个 CPU；先检查可用集合与占用，禁止与 encoding 并发。
- 固定 seed=0，使用旧 generation `c9c3be6802144fcf97a942085b118422`。
- 新合同先固定输入、候选、CPU、采集边界、预算、失败条件和接受标准，再运行。
- 两个相反顺序的 pair 为 B1→C1、C2→B2，共四次完整 export。
- 每次 timer 都覆盖完整 export 与完全相同的 page-copy 采集 hook。
- prequantization RGBA/geometry 与 final GLB 比较均移至各次 timer 停止后。
- raw wall 原样保留，不扣除验证开销；逐次保留次数、原始耗时及失败原因。
- 仅四次完整导出且比较全部通过，才报告 `min(B1-C1, B2-C2)`。
- 任一次失败立即停止并保留 partial，primary 留空，禁止成功子集接受或续跑补齐。
- 历史 B 单次约 370–396 CPU s、C 约 272–283 CPU s；四次约 1284–1358 CPU s。
- 建议新循环总额度 2400 CPU s：RGB完整对照与比较2000 s、工程smoke100 s、余量300 s；0 GPU。
- 可分两条各不超过 600 s wall 的命令，每条完成一个 pair；否则给出充分 timeout。
- 分命令必须显式传递全部原始记录和参照；不能把两次独立 core 调用合并当作完整序列。
- 新循环另申请最多3次medium leaf调用、累计worker wall1200 s、峰值RAM12 GiB、新文件不超过1 GiB；只做RGB验证与独立评估，encoding/DBSCAN本轮只保留方案。
- 最多一次完整官方四导出序列，不默认重试；代码、CPU选择、指标与资源在结果前固定。通过独立验收才集成既有RGB补丁；不发布。

## 随后探索

- Encoding：先做串行/有界线程编码的完整 export 对照，使用相同采集、比较和资源规则。
- 首张页 pilot 只作组件诊断，不能代替 full-export 指标。
- DBSCAN：先固定任务内只读 orientation JSON，禁止触发原 cache 的重算或写回。
- 再按预先固定的相机与质量条件做真实 Viewer 对比，不能仅凭过滤数量接受。

`perf/surfel_latency.py` 仅提供依赖注入的四次计时核心及 CPU 列表校验。
它还不是完整 scene runner；caller 负责产物、page-copy、比较、亲和性及中断持久化。
本次仅用假 clock/export 验证工程行为，没有新的真实场景耗时或性能收益结论。

后续可探索 [pointcloud_filter.py](../../utils/pointcloud_filter.py) 的 NN median/SOR 原始 mask
提前计算并与 matching 重叠；实际 apply 的 min_remaining_ratio 和商品保护 OR 仍等待最终标签。
DBSCAN 的输入依赖该 apply 结果，不能整段提前；这只是依赖分析，尚未验证收益。
另见已有 [COS 渐进交付草案](../../docs/superpowers/specs/2026-10-16-progressive-cos-delivery-design.md)。

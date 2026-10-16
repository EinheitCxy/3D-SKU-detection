# 后续耗时优化

RGB 完整对照已按新合同执行并通过独立验收，仓库源码已集成切片实现，相关测试 59 passed。完整导出两对分别节省 36.895 s、31.113 s，见[新结果与限制](rgb-full-validation/README.md)。首轮 inconclusive、原 B2 缺失及 continuation 不能补证的结论保留在[历史报告](README.md)。

Rick 已取消累计 worker 调用数和 worker wall 配额；仍采用 medium leaf、最多 3 个并行 worker，每项任务有有限 timeout。本轮批准的 2,400 CPU s、0 GPU、12 GiB RAM、1 GiB 新文件范围只覆盖 RGB 对照、评估与集成。下面的新候选需要单独确定硬件预算与接受标准，尚未执行。

## 优先：atlas 编码并行

- 以已经接受的 RGB 切片版本作为 baseline，只增加编码并行这一项变化。
- 沿用候选中的最多两个编码线程、有界 page 队列和单 BIN writer；页面交给编码线程后保持只读，按原 page 顺序收尾，限制内存，异常直接传播。
- 保持 MEDIANCUT、128 色、dither=NONE、PNG optimize=True、纹理采样、Surfel 长边约束及 GLB 结构。比较编码结果、完整 decoded RGBA、几何/关联和文件大小。
- 复用已验证的完整 B1/C1/C2/B2 计时流程：每臂相同 capture、timer 外比较、全部完成且两对都改善才接受；保留原始耗时和失败 partial，不默认重试。
- 单独运行 benchmark，预先固定 CPU 集合与线程数；固定亲和性不等于独占共享主机。先跑小 smoke，再执行一次完整序列并独立评估。
- 首张页 pilot 的约 2.952/1.517 s 仅是串行/并行组件观察，不能推断完整导出收益。建议另设 2,400 CPU s、0 GPU、12 GiB RAM、1 GiB 临时文件预算，执行前固定是否足够覆盖共享主机波动及比较成本。

## 随后：NN/SOR 与 matching 重叠

沿已有依赖分析，提前计算初始 NN 距离统计和 SOR 原始 mask；它们是现有清理流程，不是新增步骤。最终商品保护 OR、`min_remaining_ratio` 与实际筛选仍等待 SAM 标签，DBSCAN 的输入也依赖该筛选，不能把整段过滤提前。

独立诊断的原生 NN/SOR 分别为 3.811/6.420 s；实际可隐藏时间受 matching 窗口、CPU 争用和数据准备限制，尚无重叠收益证据。验证应以最终保留点/标签等价和 Viewer 可用时间为准。几何/纹理提前准备已与 matching 重叠 2.999 s，是此前实现，不能再次计作新增收益。

## DBSCAN 与渐进交付

DBSCAN-off 对照应先从任务内冻结 JSON 只读取得 orientation，禁止重算或写回原 sidecar；随后在固定相机下比较真实 Viewer 的背景噪点与主体完整性。首轮没有可用的正式质量/耗时结论，因此默认暂不改。

GLB、匹配结果分别交付 COS 并异步加载仍是[渐进交付设计草案](../../docs/superpowers/specs/2026-10-16-progressive-cos-delivery-design.md)。这涉及服务接口和前端状态流，尚未在本轮实现或部署；没有首屏提速实测。

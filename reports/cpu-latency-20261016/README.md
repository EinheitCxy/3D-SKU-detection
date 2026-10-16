# video3 CPU latency experiments — 2026-10-16

本目录保存可随 Git 审查的精简结果与候选补丁；模型、Viewer generation、GLB、cache、controller 工作状态和过程日志保留在本机 runtime，不进入提交。该归档复制已有结果，没有重跑实验。

最新状态：新的 RGB 完整对照已通过独立验收并集成，详见 [RGB 新对照](rgb-full-validation/README.md)。两对完整 GLB 导出节省 **36.895 s、31.113 s**，集成测试 **59 passed**。以下保留首轮失败与未完成实验的原始结论。

完整流水线基线：video3、29 帧、711 observations、385 global IDs、2,402,589 Viewer 点；总耗时 **267.565 s**，Viewer 可用 **121.214 s**，ZIP 就绪 **129.371 s**，GLB 阶段 **137.389 s**。完整运行仅一次，设备和参数见 [pipeline summary](results/pipeline-summary.json) 与 [组件表](results/pipeline-detailed-tables.json)；组件独立诊断不拼接或替换正式总量。

## 首轮结果与限制（历史）

| 方向 | 实现 / 验证 | 本轮决定 |
|---|---|---|
| RGB 延展 | task-local 切片替换 np.roll，保留扩散轮次、四方向优先顺序、alpha 和既有 Surfel 直径限制 | inconclusive，未集成 |
| Atlas 编码 | 至多两个编码线程、有界队列、单 BIN writer；保持既有 MEDIANCUT/128色/dither=NONE/optimize=True | 完整实验取消，未集成 |
| DBSCAN-off | research-only 对照；smoke 调用可写 orientation API，导致 sidecar 重算 | debuggable，无正式计时或 Viewer 质量结论；原 JSON 内容已恢复，metadata 未恢复 |

[独立评估](results/independent-evaluation.json) 未接受任何候选。已实现的几何/纹理提前准备与 matching 重叠 2.999 s，发生在本轮之前。

## 原始 GLB 观察

| 运行 | 完整 export raw wall (s) | 说明 |
|---|---:|---|
| B1 | 169.464647 | 与 encoding 争用 CPU0–3 |
| C1 | 133.610064 | 完成，timer 内含 page 比较 |
| C2 | 126.081133 | 完成，timer 内含 page 比较 |
| 原 B2 | 缺失 | SIGXCPU / rc3 |
| B2 continuation | 156.874763 | 单独新进程成功；最终比较在 timer 外 |

35.854583 s 和 30.793630 s 是直接算术差，只能作诊断。原始顺序中断、CPU 争用、不同验证开销、缺少 continuation baked-page reference，以及 dilation 累计 1522.963927 CPU s 超过原单候选 1200 s 条件，阻止正式性能接受。单独 continuation 在额外 500 CPU s 范围内成功，不覆盖原始失败。没有新的完整 GPU 流水线、Viewer 首屏或稳定百分比收益结论。

RGB smoke：31 个合成边界/并列/多 batch case、15 个既有 mesh tests、真实 batch 1024 点（baked 1020）比较通过。C1/C2 与 B1 完整 baked RGBA/geometry、最终 typed geometry/UV/indices、associations、31 张 decoded RGBA atlas（包含透明 RGB）一致，已完成 GLB 大小均为 121,513,344 bytes。Continuation 对本机保留的历史 GLB 的 279 个 accessor、31 张 decoded RGBA 和 associations 比较通过；它没有补证缺失的量化前 baked 参照。

Encoding 32 项相关测试通过；首张真实页两份副本的串行/线程编码 pilot 约 2.952/1.517 s，反向顺序约 2.927/1.535 s，PNG 和 decoded RGBA 一致。完整实验完成 0 次，不能由 pilot 推断整场景收益。

## 证据与后续

- [RGB smoke](results/dilation-smoke.json)、[原始 partial](results/dilation-partial.json)、[单独 continuation](results/dilation-continuation.json)、[候选补丁](candidates/dilation.patch)。
- [Encoding smoke](results/encoding-smoke.json)、[中断记录](results/encoding-interruption.json)、[候选补丁](candidates/encoding.patch)。
- [资源记账](results/resource-accounting.json)：9 次 medium worker 调用、0 GPU；已知实测 CPU 1580.893551 s，另有 1168 s 未测量任务的保守额度。两者之和是预算记账组成，不能称为完整实测 CPU；并发 RSS、临时磁盘峰值未完整测量。

下一步先修正计时工具：统一 timer 范围、baseline/candidate 相同验证动作、显式且不重叠的 CPU 分配、保留失败 partial，orientation 从冻结 JSON 读取。新的完整对照需先确定足够预算并预先固定规则；本归档不修改历史阈值或原始结果。生产源码和过滤默认均未因本轮改变。

## 计时工具修订（新对照之前）

新增 `perf/surfel_latency.py` 与 15 项工程测试：固定 B1/C1/C2/B2，两个 arm 使用同一采集 hook，比较发生在 export timer 结束后，失败保留 partial 且 primary 为空，CPU IDs 显式校验。测试使用假 clock/export，15 passed in 0.05 s；没有新的真实场景计时。该 core 尚未接入完整 scene runner，也不负责实际亲和性、page copy、逐次持久化和信号处理。见 [测试记录](results/test-repair.json) 与 [下一轮方案](NEXT.md)。

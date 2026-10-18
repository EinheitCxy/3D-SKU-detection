# Video3 latency optimization — 2026-10-18

状态：实现、组件实验、GPU2完整前后对照与三位medium subagent最终复核已完成。协调记录：[task_plan.md](task_plan.md)、[progress.md](progress.md)；正式计时入口：[profiling_breakdown.md §12](../../docs/profiling_breakdown.md#12-后续已实现的优化固定video3前后完整对照)。此前六视频均值保持原实验条件，不混入本轮。

| video3同次完整运行 | 修改前 | 安全默认候选 |
|---|---:|---:|
| 完整流水线 | 253.139s | 186.894s（缩短26.2%） |
| Viewer可用累计 | 129.854s | 109.163s |
| ZIP就绪累计 | 138.097s | 117.362s |
| DA3重建自身 | 25.133s | 13.706s |
| ↳ 网络构造 | 12.624s | 0.914s |
| ↳ CPU构造+加载 | 13.562s | 2.425s |
| Point ranges | 3.939481s | 0.001608s |
| GLB导出自身 | 114.041s | 68.770s |
| 任务显存采样峰 | 14.777GiB | 14.777GiB |
| Viewer点数 | 2,402,589 | 2,402,589 |

每臂一次，GPU2/相同29帧与711检测/seed42/4CPU线程；累计前缀与嵌套区间不能相加。[完整pair数据](pair-summary.json)。15项DA3数值数组、global mapping、8项Viewer二进制、objects、manifest及最终GLB完整payload均精确相等；两臂386globalIDs、1282759baked surfels、31atlas页。[输出验证](pair-validation.json)。

## 已验证的组件

| 项目 | 对照 | 候选 | 证据与范围 |
|---|---:|---:|---|
| DA3 CPU constructor | 11.826516s | 0.651774s | 新CPU进程各一次，尚非GPU/完整流水线收益 |
| CPU构造+覆盖审核+加载 | 12.993895s | 2.045339s | 同checkpoint，1377physical参数全部覆盖 |
| point ranges | 3.489543s | 0.000964s | 同一2402589个int32 labels/711observations，所有端点相等 |
| ground组件 | 10.879015s | 0.980081s | 同一3425675点域；50K改变背景，因此不推广默认 |

DA3：[加载说明](da3_loading_report.md)，实际JSON在`runtime/latency-optimization-20261018/da3-cpu-pair-alias-aware/`。1383个state键中的6个是已覆盖的共享LayerNorm别名，真实未覆盖数0；1个nonpersistent buffer保留原生初始化。跳过406个大weight的随机填充。初次把别名误判missing的失败审核保留在`da3-cpu-pair/`，已纠正。

Point ranges：[真实对照](point_ranges_real.json)。Ground：[实现与不推广理由](ground_report.md)，成功结果`runtime/latency-optimization-20261018/ground-probe-writable/result.json`；初次只读mmap不能传Open3D的诊断失败保留在`ground-probe/`。不改变NN/SOR、DBSCAN和原保护规则。

## GLB模式与流水

默认`projection_mode='source_world'`保持旧world→camera→projection的float32运算顺序，按frame共享相机和网格常量。显式`camera_basis`先变换两个basis再生成texels，是实验选项；没有错误后自动切换。

[29帧真实样本](bake_modes_real_probe.json)：每帧最多1024个LOD后surfel，共29696输入/29316baked。source_world的几何、alpha、RGBA逐元素相等；camera_basis几何相等，但6个alpha texel改变，9318个RGB texel改变，共同不透明处RGB最大差1。这不是最终palette后的画面接受证据；camera_basis不推广默认。[细节](bake_report.md)。

同一baseline generation的source_world完整导出1/2/4 workers分别108.396069/67.627083/66.638055秒；采样RSS峰约913.0/993.3/1034.2MiB。旧GLB→新1、新1→2、新1→4的document、所有geometry accessor和128295600个decodedRGBA texel完全相等。每臂一次、固定顺序和同进程，比较工作在timer外，不代表尾延迟分布。

默认选择2workers，显式4仍支持。2在本次比1快37.6%；4仅额外快0.989秒(1.46%)且多约41MiB采样RSS，未证明稳定额外收益。[原始结果](../../runtime/latency-optimization-20261018/bake-export-compare/result.json)。

## 完整流水线

修改前源码快照：`runtime/latency-optimization-20261018/baseline-code/`。src/utils/main/defaults/config是实际拷贝，模型/数据/依赖只链接复用。固定video3既有29帧/711检测，seed42；OMP/OpenBLAS/MKL=4，GPU2 RTX4090D。模型页缓存/OS缓存不清空，保留原GPU2两个resident。

控制组已完成：total253.138706s、Viewer129.854374s、ZIP138.096691s、GLB114.041383s，Viewer2402589点。[summary](baseline-summary.json)。本轮使用function-level spans和两臂相同的诊断点域捕获，含导入/进程生命周期及诊断copy/save时间；不与旧AST仪器的单video3时间计算提速率。

安全完整候选已经完成：total186.894435s、Viewer109.162660s、ZIP117.362010s、GLB68.770372s。保留ground cap0/source_world，使用2atlas workers；正式结果不含50K ground或camera_basis的实验收益。最终源码快照在`runtime/latency-optimization-20261018/candidate-code/`，生产文件冻结后执行本轮对照。

GLB encode累计调用wall为43.640→45.379s，主线程encoding join为43.649→0.900s：编码计算与bake重叠，不能将0.900s解释为编码只花了这么久。未修改的matching/filter波动不作优化归因。

## 使用与复现

Python使用项目既有uv环境；完整配置/命令在各receipt。GPU只由协调者调度；CPU组件不与完整计时并发。

```python
from src.surfel_glb import export_scene_glb
from utils.pointcloud_filter import PointCloudFilterConfig

# 保留原投影，使用两页有界编码流水；显式1仍可作串行对照。
export_scene_glb(generation, output, projection_mode="source_world", encoder_workers=2)

# 显式实验，不改变缺省行为。
ground_experiment = PointCloudFilterConfig(ground_proposal_max_points=50_000)
export_scene_glb(generation, another_output, projection_mode="camera_basis")
```

独立CPU探针：`da3_probe.py`、`point_ranges_probe.py`、`ground_probe.py`、`bake_probe.py`、`bake_export_compare.py`。`run_case.py/profile_*.py/detail_common.py`复用既有scalar profiler做本轮完整计时；`summarize_pair.py`只汇总本轮，不重复相加嵌套时间。

[已有PyTorch重计算方向调查](pytorch_candidates.md)：优先研究有界GPU纹理采样+mask+延展的融合；未新增本轮GPU算法，不以全量cdist替代百万点DBSCAN。

定向测试和三位medium最终复核见[验证与复核](review.md)。没有commit/push/发布或修改服务。任务GPU进程均已退出，原GPU2 resident保留。未测优化后的多视频均值、重复运行分布/P95、热服务、网络/浏览器或外部准确率；本轮单个固定video3的完整输出精确相等不能代表所有输入上的数值保证。

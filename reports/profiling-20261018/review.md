# GPU2 主跑只读复核

本次复核未运行模型、benchmark 或额外 hash。仅写本文件。**最终结论：accept，在下述已声明测量条件和限制内可作为当前 canonical。** 主跑 receipt 已 complete，core/delivery returncode 均为0；本次独立最终审查覆盖主表、资源采样、delivery与overlap结果，未重新执行summarizer。

## 最终结果独立核对

- **主表闭合。** `summary.json.phases`六个互斥业务段加未归属差额，原始精度求和与248.8995973669953秒之差为0。`build_summary.py:37–45`采用父阶段而非嵌套子项，并独立算prepare/matching交集；两进程wall之外间隙为0.001437699秒，不存在把prepare的15.106秒再加进主总量的问题。`docs/profiling_breakdown.md:36–61`的主表和128.747/128.750/137.148/248.267秒里程碑与summary一致。
- **GPU任务峰有原始PID依据。** 独立按receipt六个seen PIDs筛选`gpu-processes.csv`、同timestamp求和，复得2026/10/18 19:33:01.679的15,132MiB，成员为DA3 PID2234393。实际GPU CSV仅出现两个task CUDA PID2234137/2234393及resident PID6221/8550；resident每采样合计5,030MiB。`build_summary.py:46,54–75,110–121`过滤及聚合逻辑与结果相符，未把20,181MiB整卡峰冒充task峰。
- **GPU采样窗口明确。** 1243个device样本跨度248.777秒；首样相对receipt.start_unix约+0.026518秒，末样相对预期结束约−0.096080秒。两端小于200ms，文档`:305`如实保留短峰可能漏测。device记录绑定index2和指定UUID。不同CSV采样器时间戳不用于推算“整卡峰减任务峰”的同刻占用。
- **RSS可复核且定义正确。** 原始host峰位于core offset24.755569秒，四PID RSS分别40,775,680、768,004,096、42,614,784、14,009,307,136字节，和为14,860,701,696字节，与summary完全一致。文档`:299–305`保留共享页重复计、进程生命周期HWM和非同时峰的区别。
- **delivery默认与包含关系正确。** `results/delivery.json`为complete，generation与主summary完全相同，记录encoder_workers=1及实际radius2/tile8/LOD/128色默认。主report bake66.421464秒、compact44.694603秒，join42.881788秒+write1.812814秒恰好闭合compact；两父区间之和比完整production wall少约0.001800秒，是外层少量控制开销，不是漏掉一个阶段。文档`:259–285`未重复叠加encoder或子函数。
- **独立delivery细分数值相符。** 七个AST block合计59.491381秒，小于_bake_batch父区间59.624199秒；默认31页，1280批。ZIP元数据独立分组复得8个binary/740个JPEG/3个JSON，字节与文档`:247–255`一致；按成员payload统计未混入ZIP headers。诊断保留源码位置及配置，不保存数组主体。
- **重叠解释有界。** `results/overlap.json`为success，24.110781秒结束，prepare_complete和pvs_load_complete均true；模型构造17.362315秒、当前线程CPU6.607440秒与文档一致。文档`:135–158`明确线程CPU不等于全部工作线程CPU，wall−CPU不等于精确GIL等待；该机制证据不冒充可回收提速。
- **旧新差值没有因果夸大。** 文档`:313`同时保留总时长−18.665秒及Viewer返回+7.536秒，并列出GPU/代码/页缓存/共享负载差异，不称受控加速率。主链路、独立diagnostics、CUDA events和allocator/RSS峰没有混加。
- **已解决文档一致性问题。** 曾发现delivery算法文仍写“未测量”；协调者已修。重新读取确认`algorithms_delivery.md:3,14,104`写实测完成并链接`results/delivery.json`及主summary。该问题已关闭。

本次没有另读并逐项复算DA3/PVS/Viewer独立result；它们的全部内部细分不属于本次最终审查read scope。这里的accept不扩展为准确率、视觉等价、热服务或多视频泛化结论。

## 已确认

- 重建使用指定本地 snapshot `b2359bdf726fb44ef62acca04d629dcf158053e7`。路径经 `profile_core.py:50–54` → `main.py:1078–1079,837` → `src/da3_3d_reconstructor.py:167–181` 传入 runner；`src/da3_runner.py:224–229` 明确加载该路径并迁移至 CUDA，不允许 CPU fallback。实际 `runtime/profiling-20261018/core.log:13–14` 和 `process-2234393.json` 的 argv 同时支持该路径；log 记录本地加载、metric gate、partial落盘与原子发布。
- matching 不再加载另一个 DA3 model。`core.log:88` 首次读取本轮新生成的 `outputs/dataset/da3_cache/predictions.npz`，`:127` 及后续 reference复用同一cache；因此“重建/匹配同模型”的准确表述是 matching 使用指定 snapshot 所生成的新几何缓存。
- 最新默认调用路径得到保留：`main.py:434` parallel_refs默认1，`:1106–1109` 未覆盖；日志记录29个reference且pairing_3d=next。`:1091–1099` 的Surfel准备与matching并行，`:1130–1139`复用prepared。`profile_delivery.py:18`调用当前生产默认compact GLB，encoder_workers仍为1。
- GPU绑定用UUID，经launcher env传入core及DA3子进程。`gpu-device.csv`起始有效记录同时给出index=2和相同UUID；CUDA进程报告logical_device=0与UUID隔离后的预期一致。审核环境单独执行nvidia-smi失败，未据此否定主跑中已生成的有效CSV。
- 输入为29帧已有JPEG与711个现成检测/分类对象，输出目录新建；classifier关闭、accuracy关闭、quiet_outputs=True。不得将该测量称作含检测/分类或视频抽帧的端到端耗时。HF离线、BLAS/OMP线程4也应保留在条件标签中。

## 必须保留的测量边界

1. GPU整卡memory.used包含既有resident；任务显存应按receipt见到的task PIDs过滤process CSV，再在同采样时点求和；整卡峰值单列。200ms采样可能漏瞬态峰值。各进程torch allocated/reserved生命周期peak分别列出，不能相加当同时峰值。
2. host-processes的rss_sum是同时采样进程树RSS之和，共享页可能重复计；并非独占物理RAM。单进程ru_maxrss也不能跨进程直接相加作为同时峰值。
3. monitor Popen没有检查提前退出；最终接纳必须确认两份GPU CSV有有效记录且覆盖主跑窗口。当前起始CSV有效不代替完整覆盖检查。
4. receipt.elapsed_s包括两个uv进程启动、imports、退出与轮询；core_pipeline不含core imports，zip/glb spans不含delivery imports。应明确canonical采用哪条边界，不将launcher总wall与pipeline净段互换。
5. Surfel prepare与matching重叠，父子span亦存在嵌套。按start/end时间轴解释；不得累加父子inclusive或将独立diagnostics加到主跑总耗时。
6. DA3 inference等wrapper记录host API wall，未新增CUDA同步。可以报告API/阶段墙钟，不能称纯GPU kernel时间；完整输出落盘的主墙钟仍有意义。
7. worker/CPU参数、当前copy的config及source-at-run记录应与结果一并保留。该单次GPU2实测可建立带条件的本轮基线，不能据不同GPU/环境旧跑直接作受控因果加速结论。

## 初审待项（最终复核已关闭）

- receipt status=complete、各stage returncode=0、完整pipeline与GLB report。
- 29个reference、PVS/采样/过滤及GLB计数，主边界与资源窗口对齐。
- 独立diagnostics及summary中的nested计时解释；CPU诊断不能替换主跑wall。
- 协调者提供summary后的核对结果见上文；本节保留初审检查清单，不表示仍有阻塞。

## 协调者收尾 — 2026-10-18 19:58:51 CST

另外两位medium源码/结果复核：GPU分支、CUDA/wall、模型shape与GIL证据措辞accept；geometry的所有阶段数字/保护/随机性边界accept，并提出两处小映射问题。协调者已修正算法入口的实际serialize→point_ranges顺序、valid bool/valid_indices整数区分，以及GPU算法末句head来自AST wall而非CUDA event。未追加benchmark。最终validation.json通过；当前无未解决的实质复核问题。

# Bounded Torch bake experiment

Rick授权：先将上一轮本人优化提交并push到origin/main；同时测试有界RGB采样、mask和固定方向RGB延展的现有PyTorch实现。更新：2026-10-18 23:38 CST。当前：实现、GPU2实验及验收完成，两位独立结果和最终文档复核PASS；本目录记录随独立实验提交发布。

- Coordinator：Git快照、精确暂存/提交/推送、GPU2调度、最终结论。
- Medium delivery worker：仅本目录torch_bake.py、probe.py、implementation.md，先实现CPU source_world几何＋有界Torch颜色/mask/延展，不改生产默认。
- Medium geometry worker：仅numerics_review.md、test_torch_bake.py，独立抽取数值契约并做微型reference对照。
- Medium GPU worker：只读审计旧改动/报告归属，完成后独立评审实验。

验收：保留f16/JPEG、投影次序、depth/source/round、binary alpha和固定old-known方向传播。先微型测试，再GPU2真实29frame均匀样本；通过后才做同generation完整GLB组件对照。每frame最多保留其RGB/depth，一个surfel batch≤1024；不提前烘焙原始点云。所有GPU运行由coordinator串行调度，包含context、H2D/D2H与导出收尾；不从纯算子时间推断全链路收益。

结果：完整GLB66.589322→44.671011s（减少32.916%）；含import脚本至就绪69.928963→48.084271s。相同generation各臂一次，新进程、encoder2；GPU计时包含首context和上传下载。完整document／accessors／31页128295600decodedRGBA texels一致；sample29批全部2931600rawRGBA texels及corners顺序一致。生产CPU baker默认未改，未重跑完整pipeline或多视频均值。[报告](README.md) · [汇总](summary.json)。

资源：GPU2原6221/8550residents保留，23:36收尾快照仍19038MiB空闲／util0，实验GPU进程均退出。Torch allocator allocated峰37.119MiB、reserved54MiB，不含全context/driver。大产物仅runtime；不拷贝模型/环境，不运行DA3/PVS。本次实验文件作为单独提交。

Git起点：main@bf9bde8，暂存区空；初始diff/status/worktree快照在runtime/commit-helper-20261018-latency。其他cleanup/accuracy/camera/历史研究文件保留。上一轮5个新功能提交＋4个原本地提交已push，origin/main@37019bd。首次自动审批因公开元数据拒绝，Rick明确批准后才重试；直连HTTP无响应，检查远端未变化并终止本次挂起进程后，通过已验证本机代理成功快进，无force-push。

后续与事件见[JOURNAL.md](JOURNAL.md)。CPU30项、where后受影响10项、12tiny smoke通过；最终静态／提交结果见[completion.json](completion.json)。限制是单generation/n1、最终全页量化后结果而非所有full raw tiles、未测浏览器／准确率／P95／多视频稳定收益。

# Surfel 覆盖实验

- 目标：分别尝试纹理来源保留、边缘自适应采样、覆盖优先点预算，并由 coordinator 审查和验证。
- 更新时间：2026-09-24 16:43，Asia/Shanghai。
- 当前：实验及后续授权的生产部署、四分支提交/push均完成。Rick于15:51授权部署方案2并push，15:56指定中英文Viewer由低成本subagents更新；两名Luna完成对应更新。
- 实验条件：复用29帧504分辨率缓存，仅CPU；实验阶段生产代码和既有产物只读。后续生产接入按新增授权修改。
- 证据：`perf/surfel_coverage/CONTRACT.md`；`runtime/surfel-coverage-20260924/REPORT.md`、`comparison.json` 和各方法目录。
- 结果：方案2优先保留为研究候选（深度支持覆盖+5.11个百分点、贴图支持覆盖+5.12个百分点）；方案1相对固定fused几何贴图支持覆盖+7.19个百分点，但仍有接缝。方案3初版漏空严重；一次保守修复仍深度支持覆盖-0.367个百分点、贴图支持覆盖-0.304个百分点，两版均不采用。
- 验证：24项针对性测试通过；24组固定视角、48张单面板截图；加载/选择/相机同步通过，最终页面默认方案2验证通过。仅CPU缓存实验和SwiftShader外观验证，不是生产准确率或GPU性能证据。
- 实验产物：交付时543MiB左右；原数据和原实验保留。实验阶段没有训练、模型推理或生产发布；后续发布证据单独记录。
- 查看：Rick随后授权增加局域网入口 `http://192.168.2.6:8769/web/surfel-coverage.html`；本机127.0.0.1入口保留。均只服务本轮产物，实验进程已结束。
- 新任务范围：main正式Surfel v3采样/尺度，Docker打包；visualization_en与visualization_zn读取新尺度，保留现有v2。保留商品/背景预算及global ID；原生U/V与深度阈值不变。
- 发布验证：57项定向测试、三套TypeScript/Vite构建通过；新镜像实际导出2,363,298点v3包，CPU导出78.37秒、打包0.41秒。中英文Viewer的18,771点真实商品切片显示/选择/拾取/隐藏通过；WebGL单点尺度1→3对应可见宽度54→162px，原生深度连续性裁切仍生效。
- 服务：后端192.168.2.6:8011，中文5175，英文5178；实际LAN HTML/JS和API均HTTP 200。保留旧容器供回滚，清理本轮三个候选容器。
- Git：origin/main=df8e001，origin/docker=ac35bcd，gitlab/visualization_en=d23c166，gitlab/visualization_zn=a7a07ee，均push成功；其他untracked及本地实验不入提交。
- 限制：完整236万点场景在SwiftShader截图超时，未验证完整场景交互性能；未重跑GPU模型推理或上传COS。生产预算/渲染与实验不同，不沿用实验覆盖/性能增益；旧任务包需重新导出才能获得优化。
- 发布回执：`runtime/surfel-coverage-20260924/DEPLOYMENT.md`，原始JSON/日志位于`production-smoke/`和`scale-probe/`。无剩余已授权发布步骤。

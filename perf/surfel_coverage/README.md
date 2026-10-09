# Surfel 覆盖实验（2026-09-24）

2026-10-09 收尾清理已删除 `runtime/surfel-coverage-20260924/` 下的 `uniform/`、`coverage/`、`adaptive/`、`fused/`、`texture/`、`coverage_bounded/`。报告、截图、最终部署与 `production-smoke/` 验收记录保留；下文是历史实验说明，六组原始模型、缓存和逐方法统计已不在本地。比较页面需重新生成数据后才能回放这些方案，浏览器工具需按 [perf 安装说明](../README.md#运行) 重装。

这是复用 video3 的29帧、504×280 DA3缓存的隔离CPU实验。生产Viewer、原缓存、相机、旧GLB和SKU匹配/计数均不修改。输入和验收条件见 [CONTRACT.md](CONTRACT.md)。历史结果与负例报告位于 `runtime/surfel-coverage-20260924/`。

本轮已完成：方案2优先保留为研究候选（深度支持覆盖+5.11个百分点、贴图支持覆盖+5.12个百分点）；方案1贴图支持覆盖+7.19个百分点，但主要来自恢复primary来源。方案3初版和保守修正版都未通过收益验收。相对各自匹配基线，不能直接把增益相加。完整结果与截图见 [REPORT.md](../../runtime/surfel-coverage-20260924/REPORT.md)。

三个独立方向：

- `texture`：相对既有 `fused` 几何，保留合法主来源，仅为连续残余区域选择整块严格可见来源。
- `adaptive`：相对 `uniform`，保持217049点，向边缘分配更多小圆盘，平坦区域更稀；采样和footprint共同构成一个候选。
- `coverage`：同点数、scale4，按空间/法向单元与质量重新分配。首轮出现全局质量排名导致覆盖丢失，作为负例保留。
- `coverage_bounded`：上述机制失败后仅一次纠偏；至少保留每个16×16图块中15/16基线锚点，只交换少量受支持的冗余点。不是预先独立提出的第四个假设。

独立评估从固定7个原相机位置发射像素射线，比较首交面与原始深度。`depth_supported_coverage` 是原有效像素中，首交深度落在 `max(2*spacing,0.005*Z)` 内的比例；`textured_supported_coverage` 再要求该面有纹理来源。mask只是全场景深度/置信度有效域。指标不等于物理真值、held-out质量、SKU准确率或文字清晰度。灰面面积会被圆盘大小和重叠影响，必须配合固定视角截图。

## 查看

按Rick要求，局域网入口为 `http://192.168.2.6:8769/web/surfel-coverage.html`；同一局域网可直接打开。原本机入口 `http://127.0.0.1:8769/web/surfel-coverage.html` 仍保留给本机截图工具。默认展示方案2，下拉切换各方案，左侧始终是匹配基线。旋转、缩放、平移同步；可切换整货架、商品特写与纹理来源色。

重启局域网静态服务（显式绑定指定网卡，不绑定所有接口）：

```bash
UV_CACHE_DIR=/tmp/codex-uv-cache uv run --no-sync python perf/roi_fusion/serve.py \
  --root runtime/surfel-coverage-20260924 --port 8769 --host 192.168.2.6
```

本机截图服务使用同一命令，将 `--host` 改为 `127.0.0.1`；两者监听不同地址，可同时运行。

## 实际运行方式

从仓库根目录，用现有环境，无依赖安装或模型运行：

```bash
export UV_CACHE_DIR=/tmp/codex-uv-cache
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run uniform
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run fused
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run texture
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run adaptive
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run coverage
timeout 900 uv run --no-sync python -u -m perf.surfel_coverage.run coverage_bounded
uv run --no-sync python -m perf.surfel_coverage.evaluate uniform
# evaluate 对每个已生成的方法分别执行一次。
uv run --no-sync python -m perf.surfel_coverage.summarize --include-bounded
node perf/surfel_coverage/build_viewer.mjs
node perf/surfel_coverage/capture.mjs
node perf/surfel_coverage/capture.mjs \
  'http://127.0.0.1:8769/web/surfel-coverage.html?trial=coverage_bounded' coverage_bounded
```

这些命令记录本轮用法；已完成的方法目录会拒绝覆盖，重新试验应先在 `common.py` 显式设置新的实验输出根目录，并同步build/capture路径。不要删除既有结果来复跑。构建显式关闭Vite的public目录复制，只输出本页及其脚本。浏览器使用已有Chromium和SwiftShader，只验证外观和交互，不作GPU性能结论。

每个方法目录包含 `stats.json`、`evaluation.json`、`model.glb`、`mesh.npz`；采样候选另有 `selection.npz`。`comparison.json` 包含相对各自基线的差异，纹理候选还逐面区分新增、丢失、换来源及保持。所有统计由实际产物产生；总时长仅含本轮缓存读取/选择/贴图/导出及保存，不含DA3推理，既有融合几何也未重新计算。

最小验证由各worker针对所属模块执行，主线程验证ray camera-Z和灰色前景遮挡；回执见各模块报告和最终结果报告。不要求无关全量测试。

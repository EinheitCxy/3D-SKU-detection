# 默认关闭辅助图片实现记录

当前状态：源码完成，89 个相关工程测试通过。下方早期“未执行测试”记录属于实现阶段，已由文末测试结果更新；未运行 GPU、benchmark、删除历史文件或发布。独立工程环境的离线依赖恢复见 [environment.md](environment.md)。

## 实现

- `utils/config.py:296`：`SKUMatchingConfig.quiet_outputs` 默认 `True`；`extract_main_settings` 保留 YAML `main.quiet_outputs`。
- `main.py:437,518,993,1173`：直接调用 matching、完整与 concise pipeline 默认 quiet。
- `main.py:560`：匹配入口始终向 inference 透传 `--quiet_outputs` 或 `--no-quiet_outputs`，显式 False 可以覆盖 YAML True。
- `main.py` 的 `--quiet_outputs` parser：`BooleanOptionalAction` 保留旧正向开关并提供明确反向开关。主入口优先级为 CLI > `main.quiet_outputs` > `inference.quiet_outputs` > True；完整、concise 与 interactive 都透传最终值。
- `src/inference.py:103,133,345`：直接 CLI 的 quiet 参数默认 None，仅显式参数覆盖配置，保留 YAML False；可选 correspondences JSON 仍须 `--save_json` 且非 quiet。
- `main.py:1184`：concise quiet 时跳过可选 accuracy 报告，完整 pipeline 继续使用原有 quiet guards 跳过匹配前后带框图、分析和 accuracy 报告。
- `modules/video_to_dedup/run.sh:2,155`：仅本次更新默认输出文案。

既有 `utils/sku_matching_system.py:401,415,422` 已控制可选图与 JSON，并始终保存 dedup 所需 `matching_summary.txt`，因此未修改此文件。canonical masks/cache、源帧、Viewer textures/thumbs 和机器可读 dedup 结果保持原有路径。未修改 geometry 或 matching thresholds；0.4 m 厚度轴定义仍待决定。

## 测试准备与验证

- `test/test_main_pipeline.py:22` 起：新增默认 quiet、False 恢复完整/concise 输出、三个 CLI mode 的 YAML/CLI 优先级、interactive 菜单两个 pipeline 透传、直接 inference YAML False 和 CLI 反向覆盖、matching 两向透传测试。
- 既有依赖辅助输出的两个 pipeline 测试显式 `quiet_outputs=False`。获主线程授权后，`test/test_matching_viewer_identity.py:13` 同样显式 False，保留其原有可视化路径。
- 目录中没有 `test_config*.py` 或 `test_inference*.py`；没有新建对应测试文件。
- 通过：`UV_CACHE_DIR=/tmp/uv-quiet-outputs-cache uv run --no-project --python /usr/bin/python3.11 python -c '…ast.parse…'`，解析五个改动 Python 文件。
- 通过：`bash -n modules/video_to_dedup/run.sh`。
- 首次 AST 命令未设置临时 UV cache，因 `/home/xingyu/.cache/uv` 只读失败；仅改用 `/tmp` cache 重试成功，未安装任何依赖。
- 未验证：pytest、真实图片落盘行为、完整 pipeline/Viewer 集成和耗时。当前 `.venv` dangling、系统 3.11 缺项目依赖，遵守本轮禁止安装/pytest/benchmark 的限制。

开始时 `main.py`、shell 和 main 测试已有 dirty edits；本次保留其 Surfel preparation 等原有内容，不能把完整 Git diff 都归因于本工作项。

## 独立静态 review 后修正

- 修复 interactive 丢失 resolved quiet 设置：`main()` 将最终 quiet 传入 `interactive_mode`，菜单完整与 concise 流程继续透传，不重置为默认 True。
- 修复 YAML `inference: null` 导致 eager `.get()` 异常：仅在 inference 为 dict 时读取其 quiet 字段，其他已有合法 section 仍可单独设置 quiet。
- CLI mode 矩阵增加 interactive，以及 `inference: null` 有/无 `main.quiet_outputs` 的情况；菜单行为测试覆盖 True/False 向两条 pipeline 传递。全部新增测试现为 34 cases，未执行。
- 本次修正后仅对 `main.py`、`test/test_main_pipeline.py` 运行上述 UV/系统 Python AST 检查，通过。没有新增 pytest、benchmark 或依赖安装。

## Addendum：独立 Surfel CLI 实验入口

已在 `scripts/export_surfel_viewer.py` 接入既有导出 API 的三个实验参数：

- `--no-dbscan` → `PointCloudFilterConfig.keep_main_clusters=False`。
- `--cluster-proxy-voxel-scale FLOAT` → 同名 filter config 字段，默认 `0.0`。
- `--early-background-candidates INT` → 同名 `ViewerExportTuning` 字段，默认 `0`。

脚本构造 `filter_config` 与 `tuning` 并传给 `export_web_viewer_bundle`。不带这些参数时配置与原默认一致，未新增 fallback 或其他验证策略。参数值约束由现有 dataclass 执行。

环境就绪后，可分别运行以下命令；路径占位值须换为已批准输入及独立输出目录，三个候选不组合：

```bash
uv run --no-sync python scripts/export_surfel_viewer.py --dataset /path/to/dataset --cache /path/to/predictions.npz --mapping /path/to/global_mapping.json --mask-cache-root /path/to/sam3_mask_cache_sam31_pvs/v2 --output /path/to/dbscan-off-output --no-dbscan
uv run --no-sync python scripts/export_surfel_viewer.py --dataset /path/to/dataset --cache /path/to/predictions.npz --mapping /path/to/global_mapping.json --mask-cache-root /path/to/sam3_mask_cache_sam31_pvs/v2 --output /path/to/proxy-output --cluster-proxy-voxel-scale 1.0
uv run --no-sync python scripts/export_surfel_viewer.py --dataset /path/to/dataset --cache /path/to/predictions.npz --mapping /path/to/global_mapping.json --mask-cache-root /path/to/sam3_mask_cache_sam31_pvs/v2 --output /path/to/early-background-output --early-background-candidates 500000
```

`1.0` 仅展示显式 proxy scale 的调用形式，正式实验 scale 仍由锁定方案指定。相同输入基线使用同一命令但删除最后一个实验开关并选择独立 output。三个候选保持相同 `--texture-edge`、`--voxel-size`、masterdata 和环境；上述示例使用脚本既有默认值。

验证：仅用 `UV_CACHE_DIR=/tmp/uv-quiet-outputs-cache uv run --no-project --python /usr/bin/python3.11 python -c '…ast.parse…'` 解析脚本，通过。未运行项目测试、`--help`、实际导出或计时，参数 runtime wiring 待可用环境验证。

## 后续获准工程测试结果

独立 3.11 环境恢复后，主线程另行授权最小工程功能测试；本节覆盖前文“未执行 pytest”的历史状态。使用 GPU 隐藏、OMP/OpenBLAS/MKL 各 2 threads，执行 `test/test_main_pipeline.py` 和 `test/test_matching_viewer_identity.py`：**89 passed in 4.96 s**，首次运行退出码 0，无修复、重试或重复测试。

整体命令 wall 7.407 s，user CPU 7.016 s，system CPU 1.908 s。使用 180 s timeout、200 s CPU 限制与更严格的 8 GiB virtual address-space 限制，未单独采集 RSS 峰值。没有 benchmark 或 GPU 工作，正式 CPU 性能实验预算仍 pending。

完整命令、结果与限制：[outputs-tests.json](outputs-tests.json)；原始输出：[outputs-tests.log](outputs-tests.log)。本次两文件通过不覆盖 Surfel CLI 实际导出或完整 Viewer 集成。

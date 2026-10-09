# 项目收尾清理（2026-10-09）

## 目标与执行计划

Rick 已授权删除当前项目用不到的 legacy 文件和目录，并要求低成本 subagents 只读盘点，主线程负责决策和删除。

1. 分区核查过期代码、legacy 副本、生成产物和引用关系：完成。
2. 汇总精确删除清单，核对 Git 改动、当前运行任务、软链和容器挂载：完成。
3. 删除确认无用的对象，并修正直接相关的配置引用：完成。
4. 检查删除结果、断链和必要的最小测试：完成。

根目录 `task_plan.md`、`findings.md`、`progress.md` 属于其他工作，继续保留；本文件承接 planning-with-files 的清理计划、发现和执行记录。

## 已确认的当前状态

- 三个只读 `gpt-6-luna` subagents 分别核查代码、产物和文档。
- 原有 README、核心文档、perf README 的改动和旧报告迁移继续保留。
- 当前存在未跟踪的 Surfel GLB、compact、mesh、LOD 源码和测试，属于新工作。
- 根 `.venv` 为指向 `/data/www/comfyui/3d-recognition-build/runtime/root-venv` 的软链。
- 删除前文件系统可用约 16 GiB；其变化受同机其他工作影响。
- Git worktrees 已登记，均不作为普通临时目录删除。
- Host 只读检查确认根目录有 Python 任务、根 Viewer 有 Vite 服务运行；当前运行容器没有挂载待审历史产物，映射容器仅挂载 `docker/.env`。通用 Export 容器挂载整机根目录为只读。
- 已保存四个 Viewer `CURRENT` 指针；五个 Viewer public 软链均正常。
- 文档分区审计完成：历史设计/性能记录不是按日期删除的对象；classifier planning、论文素材、独立项目和当前 Surfel 实验保留。

## 审计过程中的调整

首轮代码和产物 subagents 未交付足够的文件证据，没有按其结果删除。已将任务拆成带明确读取命令和最多两次 shell 调用的三个小任务：legacy 副本、旧代码、旧演示生成物；新建读取任务使用 `gpt-6-luna` 的 low effort。

## 删除清单与结果

| 已删除对象 | 判断依据 | 删除前占用 |
|---|---|---:|
| `legacy/`，含旧 Pi3 和 VGGT 源树 | 当前 Pi3/Pi3X 使用根目录 `Pi3/`；VGGT 使用根目录 `vggt-main/`。旧目录无当前调用，相关测试只检查 ignore 规则。Pi3 副本存在独有旧源码，按用户明确的 legacy 淘汰授权移除，并非完全相同副本 | 约 50 MiB |
| `src/facing_area_stage.py`、`utils/facing_area.py`、`utils/bbox_3d_extractor.py` | 已退出入口的旧正面面积功能及专属 helper；当前 `ground-stack-area` 使用独立 DA3 footprint 实现 | 36,864 B |
| `src/analyze_accuracy_metrics.py` | 旧 accuracy 汇总脚本，无当前调用；现有 batch evaluation 脚本继续保留 | 12,288 B |
| `test/legacy_api_client.py` | 对应已移除的 localhost:8000 BSON API | 4,096 B |
| `modules/viewer_web/public/comparison-da3-2s/` | 无当前页面、脚本或文档引用的历史展示数据 | 154,644,480 B |
| `modules/viewer_web/public/comparison-da3-896/` | 同上 | 151,896,064 B |
| `modules/viewer_web/public/comparison-pi3x/` | 同上 | 167,628,800 B |
| `modules/viewer_web/public/comparison-pi3x-2s/` | 同上 | 164,823,040 B |
| `modules/viewer_web/public/comparison-precision/` | 同上 | 215,289,856 B |
| `modules/viewer_web/public/comparison-video1-31frames-896/` | 同上 | 5,988,352 B |
| `sam3/test_dir/self_exemplar_output/` | 2026-01-12 示例生成的报告和展示图片，不含源码或权重 | 201,289,728 B |
| `Depth-Anything-3/output/` | 2026-07-08 示例生成的 GLB、深度可视化和图片 | 18,345,984 B |

同时移除四个已删 Python 模块的五份字节码缓存（77,824 B），并删除 `config.yaml` 中无运行时消费者的 `batch_accuracy` 旧配置段。

除 `legacy/` 的约 50 MiB 外，精确记录的删除前磁盘占用为 1,080,037,376 B；本轮合计约 **1.05 GiB**。文件系统可用空间约由 16 GiB 增至 17 GiB，其他任务仍会影响 `df`。

执行分工偏差：legacy 审计子代理执行了该目录删除，超出其只读委托；主线程随后关闭子代理，核验当前根源码和其他工作状态。其余对象由主线程按明确清单删除。没有提交或推送。

## 保留依据

保留当前根 Pi3/VGGT、模型权重、`.venv` 软链、四个 CURRENT run、现用 `public/comparison/` 与 `data-video1-1fps/`、当前 runtime 实验、GLB/LOD 新源码和测试、数据集、论文素材、独立子项目及已登记 worktrees。

`utils/track_utils.py`、`utils/extract_frames.py` 仍可作为手工工具；`batch_accuracy_evaluation_all.sh` 等脚本仍被现有测试或流程使用。历史设计和性能记录也继续保留。

## 验证

- 当前 footprint CPU smoke：`test_ground_stack_area_cli_calls_da3_footprint_stage` 与 `test_ground_stack_area_cli_rejects_removed_bbox_anchor_options`，**6 passed in 6.78s**。没有执行 GPU pipeline 或全量回归。
- 所有已删除对象与对应缓存均确认不存在；当前根 Pi3/VGGT、Pi3X 权重、GLB/LOD 新源码、环境软链及现用 comparison 数据仍在。
- 四个 CURRENT 的 run ID 未变化、目录存在；五个原有 public 软链均有效，没有新增断链。
- `git diff --check` 通过。与删除前 Git 状态相比，新增状态仅为四个旧 Python 文件删除、`config.yaml` 修改和本清理记录；原有状态条目无丢失。
- 删除前 host 进程检查未发现候选文件被可读 Python/Node 进程直接调用或打开；51 个其他进程无读取权限，因此不将句柄检查表述为完整主机证明。

## 第二轮 review：A–E 已确认执行

Rick 要求进一步激进检查，但本轮新增删除必须先确认。当前只读审计三个分区：runtime/Output 历史实验和缓存、工具环境和 vendored 辅助资产、孤立源码和旧文档。

初始目录占用：runtime 约 22 GiB、Output 约 3.3 GiB、perf 约 675 MiB、modules 约 791 MiB；项目合计约 32 GiB。这些是盘点范围，不是授权删除清单。

三组低成本子代理完成了只读 review，审阅阶段没有新增删除。随后 Rick 确认 A–E；执行回执见本节末尾，F、G 保留。

### 待确认清单

| 组 | 精确范围 | 删除前占用 | 删除影响 |
|---|---|---:|---|
| A，推荐 | `runtime/surfel-comparison/` 下 `gs-model/`、`gs/`、`deps/`、`torch-extensions/` | 8,243,093,504 B，约 7.68 GiB | 失去旧 Gaussian Splatting 权重、导出结果和实验依赖；旧 GS 对比不能直接复跑。权重目录仅有 config 和独立权重文件，没有嵌套 Git 或共享硬链接；当前核心代码及新 GLB/LOD 未发现路径引用 |
| B，推荐 | `runtime/surfel-coverage-20260924/` 下 `uniform/`、`coverage/`、`adaptive/`、`fused/`、`texture/`、`coverage_bounded/` | 550,457,344 B，约 525 MiB | 失去六组旧对照原始 mesh/GLB/cache。最终部署、production-smoke 验收与其余报告保留；当前 GLB/LOD 和运行服务未发现直接引用 |
| C，推荐但可选择保留 | `perf/.playwright/`、`perf/node_modules/`、根 `node_modules/` | 706,301,952 B，约 674 MiB | 再运行浏览器采集/视觉验收需重新安装 npm 依赖和 Chromium。根 node_modules 仅有 Vitest 结果缓存，无软链；正在运行 Vite 的 `modules/viewer_web/node_modules/` 保留 |
| D，推荐 | 下方逐项列出的上游 demo 和旧输出 | 158,150,656 B，约 151 MiB | 失去本地上游演示数据、notebook、演示文档图片与旧测试展示图；不整删 SAM3 assets，保留 BPE 词表及自定义测试脚本 |
| E，推荐 | `utils/track_utils.py`、`utils/extract_frames.py`、`config.sam31.yaml`、`src/vggt_3d_reconstructor.py` | 40,960 B | 退休独立 RAFT 工具、旧抽帧 CLI、旧 SAM3.1 配置快照及禁用的 VGGT 重建器。需同步清理重建器恢复提示、注释注册项与 reconstruction CLI 选项；VGGT point_tracking 支持保留 |
| F，单独确认 | 根 `task_plan.md`、`findings.md`、`progress.md`，以及 `docs/superpowers/` | 344,064 B | 当前 classifier、dedup 元数据和 Viewer 功能已落地，这些成为旧过程记录；删除会失去规划、探索和历史设计内容 |
| G，单独确认 | 下方列出的 11 个非 DA3 重建缓存目录 | 1,227,010,048 B，约 1.14 GiB | Pi3、Pi3X、MapAnything 后端仍受支持；删除仅用于释放缓存空间，后续使用或直接复核其缓存需重新推理。保留 DA3 缓存、原图、匹配摘要和准确率报告 |

推荐 A–E 共约 **8.99 GiB**；若批准 A–G，全清单约 **10.14 GiB**。这些是删除前占用合计，不保证同机 `df` 会增加完全相同的数量。

D 的精确路径：

- `sam3/examples/`
- `vggt-main/examples/`（80 个文件由主仓库跟踪，删除会形成 tracked 改动）
- `Depth-Anything-3/assets/`
- `sam3/assets/dog.gif`
- `sam3/assets/images/`
- `sam3/assets/model_diagram.png`
- `sam3/assets/player.gif`
- `sam3/assets/sa_co_dataset.jpg`
- `sam3/assets/saco_gold_annotation.png`
- `sam3/assets/veval/`
- `sam3/assets/videos/`
- `sam3/test_dir/exemplar_find_all.jpg`
- `sam3/test_dir/test_self_exemplar.jpg`
- `sam3/test_dir/test_sam3_mask_output.jpg`
- `sam3/test_dir/test_self_exemplar_output.jpg`
- `sam3/test_dir/__pycache__/`

G 的精确路径：

- `Output/floor_display2/pi3_cache/`
- `Output/floor_display4/pi3x_cache/`
- `Output/floor_display4/mapanything_cache/`
- `Output/floor_display5/pi3x_cache/`
- `Output/floor_display5/mapanything_cache/`
- `Output/floor_display6/pi3x_cache/`
- `Output/floor_display6/mapanything_cache/`
- `Output/floor_display7/pi3x_cache/`
- `Output/floor_display7/mapanything_cache/`
- `Output/floor_display8/pi3x_cache/`
- `Output/floor_display8/mapanything_cache/`

### review 中排除的对象

- `sam3/assets/bpe_simple_vocab_16e6.txt.gz` 是 tokenizer 的实际依赖；BPE 词表不是 demo，必须保留。
- 当前 SAM3/Pi3X 等核心权重、外部 `.venv` 入口、detector 环境、数据集、DA3 缓存、当前 Viewer bundles、当前新 GLB/LOD 工作和 Git worktrees 保留。
- `scripts/3d/evaluation/batch_accuracy_evaluation_all.sh` 仍有批评估用途与明确测试契约，不放入推荐清单。
- `sam3/test_dir/` 有四个自定义 Python 脚本，不整删目录。
- `knowledge/`、`auto-research-loop/` 与独立的 `controllable-autoresearch-agent/` 含用户或研究资料，不凭大小和无入口删除。

本轮验证仅为文件盘点、引用、Git 状态、软链、权重硬链接和运行进程/容器挂载检查。没有新增代码行为改动，没有重复运行测试。确认删除后才执行获批组，并修正直接相关的入口、说明或引用。

### A–E 执行（用户于 16:56 确认）

Rick 已明确确认“删推荐的 A–E”。执行范围锁定为上表 A–E 及 E 中退休模块的对应字节码；F 的旧规划/设计资料、G 的非 DA3 缓存继续保留。删除前重新核对精确路径、Git 改动与运行任务，主线程独占所有删除和编辑。

执行完成：A–E 共 33 个目标和一份对应的退休字节码已删除，删除前磁盘占用合计 **9,658,056,704 B（8.9948 GiB）**。项目约由 32 GiB 降至 23 GiB；所在文件系统可用约由 17 GiB 增至 26 GiB（同机其他工作会影响可用空间）。

联动编辑：

- `main.py` 将重建选项与匹配选项分开，重建仅提供 Pi3/Pi3X/DA3/MapAnything；CLI、交互菜单和程序调用默认使用 DA3。VGGT matching/point_tracking 继续保留。
- 移除 `src/__init__.py` 中退休重建器的注释导入与导出，以及旧 VGGT 专用 `mask_*` 参数、配置项和提取键。
- 更新根 README、核心文档、perf 安装说明和两个历史实验 README，标明已清理产物与浏览器依赖重装要求。
- 在现有 `test/test_main_pipeline.py` 增加四个有行为意义的检查：CLI 拒绝退休 VGGT 重建选项、无 YAML backend 时默认 DA3、point_tracking 仍接受 VGGT matching、程序调用按 DA3 注册表分发。

完成验证：

- 上述四个 CPU stub 检查 **4 passed in 5.53s**；没有运行 GPU 推理、全量回归或重装浏览器环境。
- 34 个删除目标均不存在；32 个 F/G 与当前依赖记录仍在，元数据未变化。四个 CURRENT 的 run ID 与目录完整，五个原有 public 软链正常。
- BPE 词表、当前权重/环境、GLB/LOD 新工作、现用 batch accuracy 入口，以及两份历史实验报告均保留。
- `git diff --check` 通过；新增 Git 状态为 80 个获批 VGGT demo 文件和三个旧源码删除，以及本次必要的源码/测试编辑。原有状态条目中仅获批 `config.sam31.yaml` 和根 `node_modules/` 的未跟踪条目消失。
- 本批删除和编辑由主线程执行；低成本子代理只读定位。没有提交或推送。

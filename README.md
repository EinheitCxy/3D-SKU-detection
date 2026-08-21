# 3D SKU 跨图匹配与全局 ID 映射

多视图货架/地堆场景的 SKU 跨图匹配、去重与全局 ID 计数系统。给定同一场景的多张照片及每图 SKU 检测框，系统判定哪些框是**同一物理物体**，经并查集去重后分配跨图唯一的 `global_id`，使每件商品只计一次。

详细架构与开发指引见 [CLAUDE.md](CLAUDE.md)；`code/` 子项目用法见 [code/README.md](code/README.md)。

## 核心能力

- **跨图 SKU 匹配**：点追踪（`point_tracking`）与 3D→2D 投影（`3d`）两套算法，可独立或并行
- **顺序去重 + 全局 ID**：并查集连通分量聚类，跨图传递性匹配 -> 唯一 `global_id`
- **多 3D 重建后端**：Pi3（缓存式，批量推荐）/ DA3（多视图高精度，subprocess 隔离）/ VGGT（实时，可选）
- **TypeScript/Three.js 静态 viewer**：Python 导出严格 bundle（同源点云过滤），浏览器端只负责校验、渲染与交互审查
- **准确性评估**：对照人工标注计算 Precision/Recall/F1
- **DA3 地堆 footprint 面积**：对去重 `global_id` 融合多视图 metric 点云，将各 carton 的支撑平面 OBB 做二维 polygon union，输出 `m²`；无需尺寸锚点

## 项目结构

```
3D_Recognization/
├── code/                   # 核心 R&D 系统（pyproject.toml + .venv，Python 3.11）
│   ├── main.py             # 统一 CLI 入口（SKUDetectionMain）
│   ├── modules/            # 流水线阶段（匹配/去重/重建/分析/viewer-web 导出）
│   ├── utils/              # 复用库（算法/几何/配置/可视化）
│   ├── scripts/            # 批量/评估脚本（batch.sh / k.sh 等）
│   └── config.yaml         # 单一可调参数源
├── bbox_gen.py             # YOLO SKU 检测 CLI（生成 detections_results，code/ 上游）
├── Pi3/                    # Pi3 重建模型库（核心源码已入库，sys.path 注入）
├── sam3/                   # SAM3 分割模型库（核心源码已入库，mask 引导采样）
├── Depth-Anything-3/       # DA3 模型库（核心源码已入库；独立 .venv 被 code/ subprocess 调用）
├── vggt-main/              # VGGT 模型库（后端已禁用，整体 gitignore 不入库）
├── imdata/                 # 数据集（floor_display*/，images/ + detections_results/）
├── auto-research-loop/     # autoresearch 循环工作区（program/progress/results，本地不入库）
└── frame_sampler/          # 抽帧 Docker 服务（其余 Docker 服务已在 2026-07-16 清理，commit 9d9503f）
```

> **git 跟踪策略**：三个 vendored 模型库（sam3 / Pi3 / Depth-Anything-3）的**核心源码已入库**（约 374 文件 / 12MB），库内环境与非核心内容（`.venv/`、`checkpoints/`、权重、`assets/`、`examples/`、构建与运行产物）由 `.gitignore` 排除；权重需按各库 README 另行下载。`vggt-main/`（后端已禁用）与 `auto-research-loop/`（循环工作区）整体不入库。

## 环境与依赖

项目有三套独立 Python 环境（均 Python 3.11）：

| 环境 | 位置 | 用途 | numpy |
|---|---|---|---|
| 核心 | `code/.venv` | 匹配/重建/Pi3/SAM3/VGGT | 1.26.x |
| DA3 | `Depth-Anything-3/.venv` | DA3 推理（依赖 omegaconf/e3nn/evo 等 code/ 未装包） | 1.26.x |
| bbox_gen | 根 `.venv` | YOLO 检测（ultralytics） | 2.3.5 |

```bash
cd code && uv sync                 # 核心环境
cd code && uv sync --extra gpu     # faiss-gpu / cupy（CUDA 12.x）
uv pip install -e .                # bbox_gen（根目录）
```

GPU（CUDA）为匹配/重建必需。`uv` 是唯一 Python 工具。

### 视频到去重结果

`code/scripts/video_to_dedup.sh` 会先用根目录 `.venv` 中的 `bbox_gen.py` 在 CPU 上生成每帧 `detections_results/`，再用 `code/.venv` 运行 Pi3 匹配和去重；不会创建新的虚拟环境。重复运行会只重置该脚本工作目录中的数字帧和检测 JSON，避免旧帧被重复计入。若主机没有可通信的 NVIDIA GPU，脚本会在检测完成后停止，并且不会写出不可信的 `global_mapping.json`。

```bash
cd code
bash scripts/video_to_dedup.sh ../small_fd_video/fd_area_test.mp4 2.0 0
```

可用 `DETECTOR_ROOT`、`DETECTOR_ENV`、`CODE_ENV` 覆盖已有检测器与两个环境的位置，`DETECTOR_DEVICE` 默认为 `cpu`。

## 快速开始

```bash
cd code

# 完整流水线（Pi3 后端，3D 匹配 —— 推荐）
uv run python main.py --mode pipeline --dataset ../imdata/floor_display2 \
    --algorithm 3d --match_backend pi3 --recon_backend pi3

# --floor N 是 --dataset ../imdata/floor_displayN 的快捷方式
uv run python main.py --mode pipeline --floor 2

# 交互模式
uv run python main.py --mode interactive

# 仅 3D 重建 / 仅匹配 / 仅去重 / 仅分析
uv run python main.py --mode reconstruct --recon_backend pi3
uv run python main.py --mode concise   --match_backend pi3 --algorithm 3d
uv run python main.py --mode dedup
uv run python main.py --mode analyzer

# 导出 TypeScript/Three.js 静态 viewer bundle（只读已有正式产物，不启动 Node/浏览器）
uv run python main.py --mode viewer-web \
  --dataset ../imdata/my_stack --save_root ../Output

# DA3 地堆 footprint 面积（要求 DA3 cache、global_mapping.json 与本地 SAM3 checkpoint）
uv run python main.py --mode ground-stack-area \
    --dataset ../imdata/my_stack --save_root ../Output

# 批量评估（floor_display2..12）
bash batch_accuracy_evaluation.sh 2 12
```

**`--mode`**: `interactive` | `pipeline` | `concise` | `analyzer` | `dedup` | `ground-stack-area` | `reconstruct` | `viewer-web`
**`--algorithm`**: `point_tracking` | `3d` | `both`
**`--recon_backend` / `--match_backend`**: `vggt` | `pi3` | `da3`（默认来自 `config.yaml`）

## TypeScript/Three.js 静态 viewer

Python 是唯一的数据与证据生产端，负责 DA3/SAM3、匹配、去重、正式 ground footprint 面积和 provenance；TypeScript 只负责严格加载 bundle、渲染点云/正式 footprint 与交互式审查，不重新计算面积。两步使用方式如下：

```bash
cd code
uv run python main.py --mode viewer-web \
  --dataset ../imdata/my_stack --save_root ../Output

cd ../viewer-web
npm run dev
```

默认 bundle 输出到 `viewer-web/public/data/`，也可用 `--viewer-web-output` 指定目录；`--viewer-web-voxel-size` 默认 `0.005`，`--viewer-web-max-points` 默认 `1500000`。bundle 使用不可变 `CURRENT -> runs/<run_id>/` 布局，loader 对 schema、provenance、数组 byte length、`world_to_view`（行主序 4×4，DA3 CV 坐标 -> viewer Y-up）和 accepted/rejected 关系严格校验，异常输入直接 fail closed。

Vite 本地 `/data/` 只会直接对应默认的 `viewer-web/public/data/`。因此只有使用默认 output 时，CLI 打印的 `npm --prefix <repo>/viewer-web run dev` 才能直接启动并读取 bundle；`<repo>` 会替换为当前 checkout 的绝对路径。使用 `--viewer-web-output <custom-output>` 时，CLI 不打印该默认 npm 命令，而会明确提示 custom output 必须在前端启动前部署，或挂载/serve 到浏览器 URL `/data/`。

正式 report 绑定生成时读取的 raw `global_mapping.json` 字节快照 SHA-256（`global_mapping_sha256`）。viewer exporter 会在构建 object index 前后重新计算该 digest；mapping 不同或在导出期间变化会 fail closed，且 `accepted` generation 的 object ID 集与 footprint geometry ID 集不一致也会拒绝发布。没有 `global_mapping_sha256` 的历史 formal generation 同样 fail closed，必须先重新运行 `--mode ground-stack-area`，再执行 viewer export；不提供 fallback。

`accepted` 的数值只表示正式 `da3_ground_footprint_union`；`rejected` 或 `value_m2: null` 表示 unavailable，界面显示 `—`，绝不显示为 `0 m²`。实验性 front-facing area 不接入 v1，青色仍保留给未来该指标；正式 ground footprint 使用琥珀色。

当前展示入口已加入以下 v1 体验增强：

- 上下文 status bar（dataset/backend/points/footprint 概况）
- 左侧 global ID 列表搜索 + Prev/Next/Clear
- 中台 scene 控制（点云大小、footprint 不透明度、相机预设）
- 右侧 evidence drawer（全局统计 + per-ID 观测与状态）
- `H` 键隐藏/显示 evidence 抽屉，`Esc` 取消选择，快捷键不依赖后端重跑

前端开发验证在 `viewer-web/` 执行：

```bash
npm test -- --run
npm run build
```

DA3 在 Git worktree 中运行时，可复用主 checkout 已有的 DA3 环境，无需创建 worktree 专用环境：

```bash
DA3_VENV_PYTHON=/home/xingyu/3D_Recognization/Depth-Anything-3/.venv/bin/python \
uv run python main.py --mode pipeline --algorithm 3d \
  --recon_backend da3 --match_backend da3 --dataset <dataset> --save_root <save_root>
```

## DA3 地堆 footprint 并集面积

`ground-stack-area` 是锚点无关的只读计量阶段：它读取 schema-v2 DA3 cache 中的 metric `world_points`、`world_points_conf`、逐帧原图→处理网格 affine、缓存时的原图尺寸与去重后的 `global_mapping.json`，用本地 SAM3 checkpoint 对每个检测框生成 mask。对每一物理 `global_id`，把它全部有效观测的 3D 点沿拟合支撑平面法向投影到该平面并恢复其 OBB；最终取所有 carton OBB 投影的**多边形并集面积**（`m²`），指标 `da3_ground_footprint_union`。若任一 global ID 几何不完整（缺 mask、有效点不足、OBB 退化等），整体拒绝并输出 `status: rejected` 与 `value_m2: null`，不会以部分结果冒充总面积。旧 runner 生成的 cache 不满足该 schema/provenance 合约，必须先用 DA3 reconstruction 重新生成 cache。DA3 尺度是模型估计，现场 reference 仍可作为 QA，而非运行前提。

SAM3 的 `predict_inst` 在自动混合精度环境下会根据模型参数的 `dtype` 对齐推理上下文：模型为 `float32` 时会显式关闭 `torch.autocast`，模型为 `half`/`bfloat16` 时会启用对应精度的 `autocast`。该逻辑用于避免 `ground-stack-area` 中的 input/weight dtype 混用导致的 BF16 mismatch。

公开输出根为 `<save_root>/<dataset>/ground_stack_footprint/`。每次运行先把完整结果写入并 fsync 到不可变的 `runs/<run_id>/`，其中严格只有下列三个产物和 `manifest.json`；全部完成后才原子替换 `CURRENT`。generation 的创建/fsync 可以并行，但 publisher 会在固定 `locks/publication.lock` 上持有 exclusive `flock`，把 `CURRENT` replace 与“按本次 expected run ID 解析自身 generation”作为一个串行区；公开 reader 持有同一 inode 的 shared lock 后解析。expected run ID 不符会报 `OSError`，不会返回竞争运行的路径。`CURRENT` replace 前的 write/fsync/rename/temp/replace 失败会向调用方报错并保留上一完整 generation 或不创建 CURRENT；成功 replace 是逻辑发布点，之后 output-root directory fsync 失败只产生明确的 durability warning，新 generation 仍作为成功结果返回且不尝试回滚。读取方始终只看到一个完整的旧或新 generation，不会读到跨运行混合的 JSON/GeoJSON/PNG；历史 generation 保持不变。

- `runs/<run_id>/measurement_report.json`：指标 `da3_ground_footprint_union`、单位 `m²`、状态（`accepted`/`rejected`）、缓存 provenance、支撑平面候选与各门、逐 global ID 观测/voxel/分量诊断、并集代数与精度敏感性、库版本与 generation-relative 产物名；每个支撑平面候选还记录 `ransac.trial_count` 与 `ransac.early_exit`。`performance.stages_seconds` 以 `time.monotonic` 记录 `validation_and_io`、`sam3_source_masks`、`select_support_plane`、`per_id_obb_union`、`shadow_evidence`、`artifact_creation`，未进入的阶段为 `null`；`total_seconds_pre_publication` 从 stage 入口计到 GeoJSON/PNG artifact 已创建、正式 report **唯一一次序列化之前**，因此不包含 report/manifest fsync、generation rename、`CURRENT` 锁等待或 replace。这些 timing 仅用于性能审计，不放宽任何门，也不改变 m² 定义；
- `footprints.geojson`：每个 global ID OBB 一个 feature + 一个 `union` feature，坐标系为支撑平面局部米制 `(u,v)`，含 `global_id`/`area_m2`/`observations_used`；
- `top_down_footprint.png`：俯视复核图，各 OBB 轮廓 + 并集填充边界，标注米制轴。

结果是**每个 carton OBB 投影到支撑平面的多边形并集**：它包含悬垂（overhang），不是包装表面积、正面/接触面积或地面接触面积，也不会估计未检测/被遮挡商品。要求现有 DA3 cache、global mapping 与本地 SAM3 checkpoint，缺少任一即拒绝。输入文件不会被改写。

`ground-stack-area` 在全部 DA3/image/detection/mapping 合约验证通过后，使用公开缓存根 `<save_root>/<dataset>/sam3_mask_cache/v1/` 读写逐源帧不可变 mask bundle。key 覆盖 image ID 与原图 bytes/size、完整有序 object ID + 精确 binary64 bbox prompts、真实 checkpoint SHA-256、stage/cache/SAM3 code fingerprint、Python/NumPy/PyTorch/SAM3/CUDA/cuDNN/device/precision runtime fingerprint、完整 `predict_inst` contract 及 source-mask shape/dtype；任一项变化都会 miss 并重新计算。cache producer 和 hit validation 都直接复用 SAM3 正式后处理的 `clip_mask_to_bbox`，包括 fractional、边缘和完全超出图像 prompt 的边界像素行为；本次 semantic schema 已升级，旧裁剪语义 bundle 只会 miss，且从不自行实现第二套 clip 或 bbox fallback。stage 在一次运行入口和全部 frame cache access 完成后各校验一次 checkpoint digest；真实 SAM3 miss 仍保留 producer 与 model-load 边界的 before/after 校验。`measurement_report.json` 的 `sam3_mask_cache.frames[]` 按帧记录 key、payload/checkpoint digest、code provenance 与唯一字段 `cache_events`（没有 `events` alias）：正常冷启动为 `miss,written`，复用为 `hit`，损坏 bundle 会移入 `corrupt/` 后以 `invalid,written` 重算，非致命写失败为 `miss,cache_write_failed` 并仅使用当次完整 fresh masks。缓存命中永远不会提供 bbox fallback；空/无效 mask 仍触发整次 `rejected`/`value_m2: null`，也绝不发布 partial total。

缓存目录创建或临时 bundle 目录创建若发生 `OSError`，同样属于非致命 cache write failure：stage 保留已验证、已 canonical clipping 的 fresh boolean masks，逐帧记录 `miss,cache_write_failed`（损坏 bundle 重算时为 `invalid,cache_write_failed`），`payload_sha256` 为 `null`，且不会删除既有 final entry。

`ground-stack-area` 现会在正式 status/value/plane/per-ID diagnostics/polygons/union/rejection reason 完全冻结后，调用内部 `code/utils/footprint_evidence.py` 附加 `evidence.mode=shadow` 报告。每个 observation 保留真实 source mask、正式 `source_to_processed_affine` 和 processed mask，并分别报告 `source_mask_pixel_count` 与 `processed_mask_pixel_count`。camera 分支二次读取并验证 optional DA3 camera tensors，计算 source-world reconstruction residual、最多 512 个确定性采样点的双向 occlusion-aware reprojection，以及固定正式支撑平面的 per-ID leave-one-observation-out OBB diagnostics。独立的 `mask_robustness` 分支即使缺少 depth/K/E 仍会运行：先在 source pixels 上以 3×3 kernel 做一次 1-pixel erosion/original/dilation，再与正式路径共同调用同一个 nearest-neighbour CV2 affine warp，并使用冻结 plane、每 observation 32 点、每 ID 64 elevated 点、5 mm voxel、component/OBB、0.1 mm precision union 和全 ID 规则重算。任一 variant 失败只记录 `value_m2: null`、reason 与 rejection transition，不输出 partial polygons；仅三种 variant 全部 accepted 时才给出经验性 `area_interval_m2`，且永不替换正式 mask/面积。双向 pair 只在不同 source image 之间生成：投影点比 target depth 远超过 20 mm 记为 occluded（遮挡中性，不是冲突），近超过 20 mm 记为 foreground conflict，中间才是 visible-consistent；20 mm 只是 shadow diagnostic tolerance，绝不是正式面积门。只有一个 distinct source image 时明确标记 `single_observation_insufficient_cross_view_evidence`，不会冒充多视角证据。formal plane 不存在时，`evidence.status` 与 `evidence.mask_robustness.status` 均为 `unavailable_no_formal_geometry`；camera fields 缺失、malformed 或任一 evidence 计算失败只产生 additive `unavailable_*`/`failed_*`/variant rejection，不能改变正式面积或 generation geometry。该报告只是 internal consistency/sensitivity，不是 calibrated uncertainty、accuracy claim、confidence interval 或 error bound；任何 hard gate/error bound 都必须先在独立 calibration set 上完成校准与验证。

## 3D 重建后端

| 后端 | 速度 | 精度 | 缓存 | 适用 |
|---|---|---|---|---|
| `pi3` | 快（读缓存） | 高 | `pi3_cache/` | 批量生产（推荐） |
| `da3` | 中（subprocess） | 更高（多视图） | `da3_cache/` | 高精度场景 |
| `vggt` | 慢（每次推理） | 高 | 无 | 单次调试（当前默认禁用） |

新增后端只需：① 继承 `ReconstructorBase` ② `@register_reconstructor("name")` ③ 在 `modules/__init__.py` 导入——无需改 `main.py`。

DA3 因依赖集与 `code/` 不同（需 omegaconf/e3nn 等，code/ 与 DA3 均为 numpy<2，无 numpy 冲突），通过 subprocess 调用 `Depth-Anything-3/.venv` 运行 `modules/da3_runner.py`，输出与 Pi3 schema 一致的 `da3_cache/predictions.npz`。权重 `DA3NESTED-GIANT-LARGE` 为 **CC BY-NC 4.0（非商用）**。

### 匹配阶段性能优化（2026-07，da3）

`--mode concise` batch_all_refs（每图作参考，N ref 串行匹配）经 per-stage profiling（`utils/profiling.py` + `--enable_profiling`）定位瓶颈并优化：**fd5/6/7 全量 720s -> 180s（-75%）**，R/P 全程等价（R83.56%/P93.71%，gate ±0.5pt 内，多 cycle 位级一致验证）。

| Cycle | 优化 | wall-clock | 等价性 |
|---|---|---|---|
| baseline | max_batch_size=5，无缓存 | 720s | R84.01/P94.06 |
| C2 | `sam3_max_batch_size` 5->32（避免 >5 bbox/ref 时 N×forward） | 565s | R-0.45/P-0.35pt |
| C3 | `_DA3_IMAGE_CACHE`（图像 tensor 跨 ref 复用） | 402s | 0pt 位级 |
| C4 | 诊断（定位 build_transforms 36s 真凶） | - | - |
| C5 | `_DA3_TRANSFORMS_CACHE`（transforms_info 跨 ref 复用） | 240s | 0pt 位级 |
| C6 | build_da3 `.size` 懒读 + 移除每 ref `PI3_SCENE_CACHE.clear()` + DIAG 降级 | 180s | 0pt 位级 |

剩余瓶颈 `sam3_mask` 占 64%（GPU compute-bound，单次 forward/ref，CUDA 跨线程串行不可并行化）。`--parallel_refs` 对 SAM3 **无效**（GPU-bound）且破坏等价性（RNG 非确定）。详见 `docs/profiling_breakdown.md`；循环工作记录见 `auto-research-loop/program_time.md` / `progress_time.md`（本地未入库）。

## 检测数据格式

每图一个 JSON（`detections_results/<i>.json`）：

```json
{
  "skus": [
    {
      "classes": { "det": ["8926^bottle"] },
      "objects": [
        { "position": [x1, y1, x2, y2], "classes": { "det": 0 }, "confidences": { "det": 0.93 } }
      ]
    }
  ]
}
```

可用 `bbox_gen.py` 从图片生成该格式（用法见 `uv run python bbox_gen.py -h`）。

## 输出与日志

- 每次运行生成一个日志：`<save_root>/run_YYYYMMDD_HHMMSS.log`（控制台 INFO，文件 DEBUG）
- 去重产物：`<save_root>/<dataset>/dedup_detections/{<i>.json, global_mapping.json, global_skus.json}`
- 重建产物：`<save_root>/<dataset>/{pi3_cache,da3_cache}/predictions.npz` + `reconstruction_<backend>.glb`

更多细节见 [code/README.md](code/README.md)。

# 3D SKU 跨图匹配与全局 ID 映射

多视图货架/地堆场景的 SKU 跨图匹配、去重与全局 ID 计数系统。给定同一场景的多张照片及每图 SKU 检测框，系统判定哪些框是**同一物理物体**，经并查集去重后分配跨图唯一的 `global_id`，使每件商品只计一次。

详细架构与开发指引见 [CLAUDE.md](CLAUDE.md)；`code/` 子项目用法见 [code/README.md](code/README.md)。

## 核心能力

- **跨图 SKU 匹配**：3D→2D 投影（`3d`，唯一算法）
- **顺序去重 + 全局 ID**：并查集连通分量聚类，跨图传递性匹配 -> 唯一 `global_id`
- **单一 3D 重建后端**：DA3（Depth-Anything-3，多视图高精度，subprocess 隔离运行于 `Depth-Anything-3/.venv`，缓存 `da3_cache/predictions.npz`）
- **交互式 3D viewer**：基于 Viser，GPU 加速 KNN 与点云下采样
- **准确性评估**：对照人工标注计算 Precision/Recall/F1

## 项目结构

```
3D_Recognization/
├── code/                   # 核心 R&D 系统（pyproject.toml + .venv，Python 3.11）
│   ├── main.py             # 统一 CLI 入口（SKUDetectionMain）
│   ├── modules/            # 流水线阶段（匹配/去重/重建/分析/viewer）
│   ├── utils/              # 复用库（算法/几何/配置/可视化）
│   ├── viewer/             # Viser 3D viewer 子系统
│   ├── scripts/            # 批量/评估脚本（batch.sh / k.sh 等）
│   └── config.yaml         # 单一可调参数源
├── bbox_gen.py             # YOLO SKU 检测 CLI（生成 detections_results，code/ 上游）
├── sam3/                   # SAM3 分割模型库（核心源码已入库，mask 引导采样）
├── Depth-Anything-3/       # DA3 模型库（核心源码已入库；独立 .venv 被 code/ subprocess 调用）
├── imdata/                 # 数据集（floor_display*/，images/ + detections_results/）
├── auto-research-loop/     # autoresearch 循环工作区（program/progress/results，本地不入库）
└── frame_sampler/          # 抽帧 Docker 服务（其余 Docker 服务已在 2026-07-16 清理，commit 9d9503f）
```

> **git 跟踪策略**：两个 vendored 模型库（sam3 / Depth-Anything-3）的**核心源码已入库**（约 374 文件 / 12MB），库内环境与非核心内容（`.venv/`、`checkpoints/`、权重、`assets/`、`examples/`、构建与运行产物）由 `.gitignore` 排除；权重需按各库 README 另行下载。`auto-research-loop/`（循环工作区）整体不入库。Pi3 与 vggt-main 源码树已从仓库删除（da3 为唯一 3D 重建后端）。

## 环境与依赖

项目有三套独立 Python 环境（均 Python 3.11）：

| 环境 | 位置 | 用途 | numpy |
|---|---|---|---|
| 核心 | `code/.venv` | 匹配/重建/SAM3 | 1.26.x |
| DA3 | `Depth-Anything-3/.venv` | DA3 推理（依赖 omegaconf/e3nn/evo 等 code/ 未装包） | 1.26.x |
| bbox_gen | 根 `.venv` | YOLO 检测（ultralytics） | 2.3.5 |

```bash
cd code && uv sync                 # 核心环境
cd code && uv sync --extra gpu     # faiss-gpu / cupy（CUDA 12.x）
uv pip install -e .                # bbox_gen（根目录）
```

GPU（CUDA）为匹配/重建必需。`uv` 是唯一 Python 工具。

## 快速开始

```bash
cd code

# 完整流水线（da3 后端，3D 匹配 -- 唯一后端，参数可省略）
uv run python main.py --mode pipeline --floor 2
# 等价于 --dataset ../imdata/floor_display2 --algorithm 3d --match_backend da3 --recon_backend da3

# 交互模式
uv run python main.py --mode interactive

# 仅 3D 重建 / 仅匹配 / 仅去重 / 仅分析 / 3D viewer
uv run python main.py --mode reconstruct
uv run python main.py --mode concise   --algorithm 3d
uv run python main.py --mode dedup
uv run python main.py --mode analyzer
uv run python main.py --mode viewer

# 批量评估（floor_display2..12）
bash batch_accuracy_evaluation.sh 2 12
```

**`--mode`**: `interactive` | `pipeline` | `concise` | `analyzer` | `dedup` | `reconstruct` | `viewer`
**`--algorithm`**: `3d`（唯一算法）
**`--recon_backend` / `--match_backend`**: `da3`（唯一可选，默认 da3，可省略）

## 3D 重建后端

唯一后端为 **DA3（Depth-Anything-3）**，Pi3 与 VGGT 源码树已从仓库删除。

DA3 因依赖集与 `code/` 不同（需 omegaconf/e3nn 等，code/ 与 DA3 均为 numpy<2，无 numpy 冲突），通过 subprocess 调用 `Depth-Anything-3/.venv` 运行 `modules/da3_runner.py`（自包含，不 import `code/`）。DA3 多视图批量推理后反投影出 `world_points`，写入 `da3_cache/predictions.npz`（depth/extrinsics(w2c)/intrinsics/world_points/image_ids）。匹配阶段不加载任何模型，仅读 npz 缓存。权重 `depth-anything/DA3NESTED-GIANT-LARGE`（6.3GB，米制）为 **CC BY-NC 4.0（非商用）**。

`--recon_backend`/`--match_backend` 参数保留但仅接受 `da3`（默认 da3，可省略）。新增后端只需：① 继承 `ReconstructorBase` ② `@register_reconstructor("name")` ③ 在 `modules/__init__.py` 导入（CLI `choices` 列表同步更新）--无需改 `main.py`。

### 匹配阶段性能优化（2026-07，da3）

`--mode concise` batch_all_refs（每图作参考，N ref 串行匹配）经 per-stage profiling（`utils/profiling.py` + `--enable_profiling`）定位瓶颈并优化：**fd5/6/7 全量 720s -> 180s（-75%）**，R/P 全程等价（R83.56%/P93.71%，gate ±0.5pt 内，多 cycle 位级一致验证）。

| Cycle | 优化 | wall-clock | 等价性 |
|---|---|---|---|
| baseline | max_batch_size=5，无缓存 | 720s | R84.01/P94.06 |
| C2 | `sam3_max_batch_size` 5->32（避免 >5 bbox/ref 时 N×forward） | 565s | R-0.45/P-0.35pt |
| C3 | `_DA3_IMAGE_CACHE`（图像 tensor 跨 ref 复用） | 402s | 0pt 位级 |
| C4 | 诊断（定位 build_transforms 36s 真凶） | - | - |
| C5 | `_DA3_TRANSFORMS_CACHE`（transforms_info 跨 ref 复用） | 240s | 0pt 位级 |
| C6 | build_da3 `.size` 懒读 + 移除每 ref `SCENE_CACHE.clear()` + DIAG 降级 | 180s | 0pt 位级 |

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
- 重建产物：`<save_root>/<dataset>/da3_cache/predictions.npz` + `reconstruction_da3.glb`

更多细节见 [code/README.md](code/README.md)。

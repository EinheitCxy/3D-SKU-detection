# 3D SKU 跨图匹配与全局 ID 映射

多视图货架/地堆场景的 SKU 跨图匹配、去重与全局 ID 计数系统。给定同一场景的多张照片及每图 SKU 检测框，系统判定哪些框是**同一物理物体**，经并查集去重后分配跨图唯一的 `global_id`，使每件商品只计一次。

详细架构与开发指引见 [CLAUDE.md](CLAUDE.md)；`code/` 子项目用法见 [code/README.md](code/README.md)。

## 核心能力

- **跨图 SKU 匹配**：点追踪（`point_tracking`）与 3D→2D 投影（`3d`）两套算法，可独立或并行
- **顺序去重 + 全局 ID**：并查集连通分量聚类，跨图传递性匹配 -> 唯一 `global_id`
- **多 3D 重建后端**：Pi3（缓存式，批量推荐）/ DA3（多视图高精度，subprocess 隔离）/ VGGT（实时，可选）
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
├── Depth-Anything-3/       # DA3 模型库（独立 .venv；被 code/ subprocess 调用）
├── Pi3/, sam3/, vggt-main/ # vendored 模型库（sys.path 注入）
├── imdata/                 # 数据集（floor_display*/，images/ + detections_results/）
└── Global-ID-Mapping/, Dockered_GlobalIDMapping/, frame_sampler/, docker_template/  # Docker 服务/模板
```

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

# 仅 3D 重建 / 仅匹配 / 仅去重 / 仅分析 / 3D viewer
uv run python main.py --mode reconstruct --recon_backend pi3
uv run python main.py --mode concise   --match_backend pi3 --algorithm 3d
uv run python main.py --mode dedup
uv run python main.py --mode analyzer
uv run python main.py --mode viewer

# 批量评估（floor_display2..12）
bash batch_accuracy_evaluation.sh 2 12
```

**`--mode`**: `interactive` | `pipeline` | `concise` | `analyzer` | `dedup` | `reconstruct` | `viewer`
**`--algorithm`**: `point_tracking` | `3d` | `both`
**`--recon_backend` / `--match_backend`**: `vggt` | `pi3` | `da3`（默认来自 `config.yaml`）

## 3D 重建后端

| 后端 | 速度 | 精度 | 缓存 | 适用 |
|---|---|---|---|---|
| `pi3` | 快（读缓存） | 高 | `pi3_cache/` | 批量生产（推荐） |
| `da3` | 中（subprocess） | 更高（多视图） | `da3_cache/` | 高精度场景 |
| `vggt` | 慢（每次推理） | 高 | 无 | 单次调试（当前默认禁用） |

新增后端只需：① 继承 `ReconstructorBase` ② `@register_reconstructor("name")` ③ 在 `modules/__init__.py` 导入——无需改 `main.py`。

DA3 因依赖集与 `code/` 不同（需 omegaconf/e3nn 等，code/ 与 DA3 均为 numpy<2，无 numpy 冲突），通过 subprocess 调用 `Depth-Anything-3/.venv` 运行 `modules/da3_runner.py`，输出与 Pi3 schema 一致的 `da3_cache/predictions.npz`。权重 `DA3NESTED-GIANT-LARGE` 为 **CC BY-NC 4.0（非商用）**。

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

可用 `bbox_gen.py` 从图片生成该格式（见 [README_bbox_gen.md](README_bbox_gen.md)）。

## 输出与日志

- 每次运行生成一个日志：`<save_root>/run_YYYYMMDD_HHMMSS.log`（控制台 INFO，文件 DEBUG）
- 去重产物：`<save_root>/<dataset>/dedup_detections/{<i>.json, global_mapping.json, global_skus.json}`
- 重建产物：`<save_root>/<dataset>/{pi3_cache,da3_cache}/predictions.npz` + `reconstruction_<backend>.glb`

更多细节见 [code/README.md](code/README.md)。

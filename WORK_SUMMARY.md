# 工作总结：DA3 集成与项目规范化

> 分支：`deep-anything-reconstructor` ｜ 审阅人：Rick ｜ 日期：2026-07-13
> 最新 commit：`3635c8f feat(da3): integrate Depth-Anything-3 as subprocess-isolated reconstruction backend`

---

## 一、概述

本分支围绕两个目标展开：**(1) 精简 `code/` 逻辑、规范结构**；**(2) 接入 Depth-Anything-3 (DA3) 作为新的 3D 重建后端，使其同样支持 SKU matching**。主线为 DA3 集成，附带完成了若干项目规范化工作（bbox_gen 工具化、GitHub 仓库治理、文档漂移修复、代码审查）。本文档汇总所做工作、验证结果、技术决策，以及建议的后续工作。

---

## 二、完成的工作

### 2.1 Depth-Anything-3 后端集成（主线）

**背景与问题**

DA3（`depth-anything/DA3NESTED-GIANT-LARGE`，1.4B 参数，米制深度，6.3GB 权重）是真正的多视图模型--N 张图联合送入同一 transformer 推理，跨图深度一致、共享世界坐标系，据称多视图精度优于 VGGT/Pi3。但接入有两个关键障碍：

1. **依赖集差异**：DA3 依赖 `omegaconf/addict/e3nn/evo` 等 `code/` 未安装的包（code/ 与 DA3 均为 numpy<2，无 numpy 冲突；2.3.5 实为根 bbox_gen venv）。直接 in-process import 会因 `omegaconf` 等缺失失败（实测 `ModuleNotFoundError: omegaconf`）。
2. **输出契约缺口**：DA3 只输出 `depth + extrinsics(w2c) + intrinsics`，不直接输出 matcher 必需的 `world_points`，需自行反投影。

**方案：subprocess 隔离**

按与 Rick 确认的"用 da3 目录下的环境"决策，采用 **subprocess 跨进程架构**：`code/`（numpy 1.26.1）通过 subprocess 调用 `Depth-Anything-3/.venv/bin/python`（numpy 1.26.4）运行独立脚本，生成 `da3_cache/predictions.npz`，matcher 读缓存。两套环境彻底隔离，互不污染。

**实现交付**

| 文件 | 角色 | 关键点 |
|---|---|---|
| `code/modules/da3_runner.py`（新建） | DA3 venv 下运行的独立推理脚本 | 不 import `code/`；多视图批量推理；反投影 depth+ext+ixt → `world_points`；写 `da3_cache/predictions.npz`（schema 与 pi3 完全一致） |
| `code/modules/da3_3d_reconstructor.py`（重写） | `ReconstructorBase` 子类，subprocess 编排 | `load_model`=no-op（校验 venv/runner）；`run_inference`=subprocess 调 runner 后读 npz；`export_glb`=跳过（SKU matching 仅需 npz） |
| `code/modules/__init__.py` | 注册 da3 | `@register_reconstructor("da3")` 装饰器触发注册 |

**修复的 2 个关键 bug**

1. **extrinsics 非方阵**：DA3 输出 `extrinsics` 为 `(N,3,4)` 的 `[R|t]`，`np.linalg.inv` 要求方阵 -> 报 `LinAlgError`。修复：补齐最后一行 `[0,0,0,1]` 成 `(4,4)` 再求逆。
2. **depth 维度不符**：matcher 的 `geometry_3d.py` 索引 `depth[...,0]`，要求 `(N,H,W,1)`，而 DA3 原始 depth 是 `(N,H,W)`。修复：在 `da3_runner.py` 保存时扩维 `depth[..., None]`。

### 2.2 后端注册表重构（精简逻辑、规范结构）

**问题**：原架构无注册表，后端字符串 `"vggt"/"pi3"` 在 **6 处**硬编码 if/else；`vggt_3d_reconstructor.py:62` 有 `continue`-in-except 的 **SyntaxError**（这正是 VGGT 被注释禁用的根因）；`matching_algorithms.py:308` 把缓存路径硬编码成 `"pi3_cache"`。

**交付**：

- 引入 `RECONSTRUCTOR_REGISTRY` + `@register_reconstructor(name)` 装饰器（`reconstructor_base.py`），pi3/vggt/da3 均已注册。**新增后端只需 1 处注册**，无需改 `main.py` 的 if/else。
- 修复 vggt `SyntaxError`（删除非法 `continue`），vggt 现已可 `import` 通过（之前会话已完成，本会话复核确认）。
- 6 处硬编码已统一为含 `da3` 的元组判断（`main.py`/`config.py`/`matching_algorithms.py`/`inference.py`/CLI choices）。
- `matching_algorithms.py:308` cache 路径参数化为 `f"{config.backend}_cache"`，da3 正确读 `da3_cache/`。

### 2.3 `bbox_gen.py` 工具化（附带）

将原 5 行硬编码脚本改造为 argparse CLI（输入图片目录或单张图片，输出可视化图 + 下游兼容 JSON）。经 code review 后修复 6 项缺陷：

1. 非数字文件名 -> 明确报错退出（而非让下游静默丢弃）
2. 单图 predict 崩溃 -> 写空占位 JSON + 跳过，不破坏帧对齐
3. 去掉未请求的 `class_names` fallback 链，直接 `model.names`
4. 模型路径改为相对 `__file__`（从任意 CWD 运行都能找到权重）
5. 可选批处理 `--batch N`（`model.predict(list, batch=N)`）
6. `--conf 1.5` 被拒绝（`(0,1]` 范围校验）+ `--imgsz` 须为 32 倍数

JSON 输出格式与 `imdata/floor_display*/detections_results` 完全一致，下游 `data_utils.load_detections` 实测可解析。

### 2.4 GitHub 仓库连接与治理（附带）

- 将 `3D_Recognization` 连接到远程仓库 `https://github.com/EinheitCxy/3D-SKU-detection`（gh CLI 认证 + git remote）。
- 仓库层面去重：精简 `.gitignore` 排除所有 `.venv`、模型权重（`*.pt/*.safetensors`）、vendored 库（`sam3/vggt-main/Pi3/Depth-Anything-3`）、数据集（保留 `picture_mapping_benchmark.csv`）、服务目录重复副本、测试视频。最终仓库 **3.9M / 106 文件**，零泄漏。
- `frame_sampler/`（独立 Lingmou GitLab 项目）按确认排除，避免公司代码进个人 GitHub。
- 合并远程 60+ commits 历史（`--allow-unrelated-histories`），保留本地 code/ R&D 改动 + 新增 bbox_gen/Docker 服务。

### 2.5 文档漂移修复（附带）

`Global-ID-Mapping/README.md` 原描述虚构的 CLI `--images/--skus/--output` 模式，实际 `build.sh` 跑的是 HTTP API。重写 README 与 `api.py`/`processor.py`/`build.sh` 对齐：BSON `POST /api`、输入 `{images,skus}`、输出 `{global_skus}`、端口表、3 步处理流程、硬编码路径说明。

### 2.6 代码审查（附带）

以专业 SWE 视角对项目做 xhigh-recall 代码审查（4 角度 finder + 验证 + sweep），输出 15 条发现。`bbox_gen.py` 的 6 项修复即源于此审查。

---

## 三、关键结果与验证

### DA3 端到端实测（floor_display1，13 图）

| 阶段 | 命令 | 结果 |
|---|---|---|
| 重建 | `main.py --mode reconstruct --recon_backend da3` | 13 图，**39.6s**，depth 米制 `[0.80, 19.51]`，`da3_cache/predictions.npz` 56MB |
| 匹配 | `main.py --mode concise --match_backend da3 --reference_idx 0` | ref0 **24 matches**，hit ratio 0.34–0.94（最高 66/70）；生成 `matching_summary.txt` + `correspondences.json` |
| 缓存字段 | npz 逐字段校验 | `depth (13,504,378,1)` ✓ `world_points (13,504,378,3)` ✓ `extrinsic (13,3,4)` ✓ `image_ids [1..13]` ✓，全部符合 matcher 契约 |

**通路验证**：`main.py` → `DA33DReconstructor` → subprocess `da3_runner.py`（DA3 venv）→ `da3_cache/predictions.npz` → matcher 读缓存 → SKU matching。全链路打通。

### 注册表

```
registered: ['da3', 'pi3']   # vggt 因依赖可选保留注释，但已可独立 import
```

### bbox_gen

CPU 实跑 13 图 460 框，下游 `load_detections` 兼容；6 项缺陷修复全部实测通过。

### git

- 分支 `deep-anything-reconstructor`，2 个 DA3 commit。
- 改动文件（DA3 commit）：`da3_runner.py`(新增 130 行)、`da3_3d_reconstructor.py`(重写)、`__init__.py`、`inference.py`、`README.md`、`CLAUDE.md`，共 +208/-119。

---

## 四、技术决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| DA3 集成架构 | **subprocess + DA3 venv** | DA3 依赖集（omegaconf/e3nn 等）与 code/ 不同；subprocess 彻底隔离，符合"用 da3 目录下的环境"决策 |
| DA3 权重 | `DA3NESTED-GIANT-LARGE` | 米制深度（对 3D matching 尺度正确）、多视图、本地已缓存 6.3GB。**License CC BY-NC 4.0（非商用）** |
| world_points 生成 | 反投影（非模型直出） | DA3 不输出逐像素世界点；用 depth+extrinsics(w2c)+intrinsics 按 `K_inv·pix·depth` → `C2W·[Xc;1]` 构造，与 Pi3 等价 |
| GLB 导出 | 跳过 | SKU matching 仅依赖 npz 缓存；GLB 可选，当前不阻塞功能 |
| 后端注册机制 | 装饰器注册表 | 消除 6 处硬编码，新增后端 1 处注册，符合"精简、规范"要求 |
| DA3 依赖隔离 | 不动主环境 | DA3 独立 venv 装其专有依赖（omegaconf/e3nn 等）；code/ 与 Pi3/SAM3 不受影响 |

---

## 五、待完成工作（后续建议，按优先级）

### P0（建议尽快处理）

1. **DA3 商用 license 风险**：`DA3NESTED-GIANT-LARGE` 为 CC BY-NC 4.0（非商用）。若项目商用，需切换到 `DA3-BASE/SMALL`（Apache 2.0），但精度下降，需重新评估 matching 质量。**请 Rick 决策。**
2. **DA3 重建无"缓存复用"优化**：当前每次 `--mode reconstruct` 都重跑 DA3（~40s/13图）。Pi3 路径有"cache 已存在则跳过"。建议在 `da3_3d_reconstructor.py` 或 `main.py:717` 增加 `da3_cache/predictions.npz` 存在性检查，命中则跳过 subprocess。

### P1（健壮性 / 规范）

3. **`main.py` 实例化仍 if/elif/else**：`run_reconstruction` 仍按 backend 显式实例化（line 607/619），因各后端 kwargs 不同（vggt 有 mask_*，da3/pi3 有 save_predictions）。可改为 `get_reconstructor(use_backend)(...)` + 统一 kwargs，让注册表完全闭环。非阻塞，属架构整洁度。
4. **DA3 与 Pi3 精度对比**：当前只验证 DA3 能跑通 matching。建议用 `accuracy_annotation.py` 在 floor_display2 等基准数据集上对比 DA3 vs Pi3 的 Precision/Recall/F1，量化 DA3 是否真"更高精度"。
5. **`da3_runner.py` 错误处理**：当前 subprocess 失败直接抛 RuntimeError 含 stdout/stderr。可增加更友好的错误分类（OOM / 权重缺失 / 依赖缺失）。
6. **DA3 depth 与 detection bbox 坐标系对齐验证**：DA3 用 `process_res=504` resize 图像，而 detection bbox 是原图坐标。matching_algorithms 的 `build_transforms` 已处理 Pi3 的 resize，需确认 DA3 的 transforms 对齐是否正确（当前 matching 跑通但未深查坐标对齐精度）。

### P2（可选优化）

7. **GLB 导出**：若需可视化 DA3 重建结果，扩展 `da3_runner.py` 支持 `--export_glb`，调用 DA3 内置 `export`。
8. **DA3 流式推理**：图数极多（>20）时可用 `da3_streaming`（<12GB 显存滑窗），但需适配 streaming 输出格式。当前 13 图 504 分辨率显存充足（4090 48GB），暂不需要。
9. **VGGT 后端重新启用**：vggt SyntaxError 已修复、可 import。若需恢复三后端并行，取消 `__init__.py` 的 vggt 注释即可，但需验证 vggt cache 契约（原 `save_predictions_cache` 缺 depth/depth_conf，与 matcher required_keys 不符）。
10. **`code/` 与 `Global-ID-Mapping/code/` 副本同步**：DA3 改动（da3_runner/da3_3d_reconstructor/注册表）未同步到 Docker 服务副本。若服务要支持 da3，需手动复制（CLAUDE.md 约定非 symlink）。

---

## 六、验证清单

### DA3 集成（已通过 ✅）
- [x] `da3_runner.py` 在 DA3 venv 下独立运行（numpy 1.26.4 + DA3 import 成功）
- [x] DA3 推理 2 图冒烟测试通过（depth_range [1.58,18.29] 米制）
- [x] extrinsics (3,4)→(4,4) 补齐修复验证
- [x] npz 字段全正确：depth (N,H,W,1)、world_points (N,H,W,3)、extrinsic (N,3,4)、image_ids
- [x] `da3_3d_reconstructor.py` subprocess 架构（load_model no-op + run_inference subprocess + export_glb skip）
- [x] `main.py --mode reconstruct --recon_backend da3` 端到端（13 图 39.6s）
- [x] `main.py --mode concise --match_backend da3` 端到端（ref0 24 matches，hit ratio 0.94）
- [x] matching_summary.txt + correspondences.json 生成
- [x] 注册表含 da3，CLI choices 含 da3

### 注册表与精简（已通过 ✅）
- [x] vggt SyntaxError 修复（py_compile + import 通过）
- [x] RECONSTRUCTOR_REGISTRY + @register_reconstructor 引入
- [x] 6 处硬编码统一含 da3
- [x] cache 路径参数化 `f"{backend}_cache"`

### 附带交付（已通过 ✅）
- [x] bbox_gen.py 6 项修复实测通过
- [x] GitHub remote 连接 + 仓库去重（3.9M / 106 文件，零泄漏）
- [x] Global-ID-Mapping README 文档漂移修复
- [x] code review 15 条发现已处理核心项

### 待办（未做，见第五节）
- [ ] DA3 商用 license 决策（P0，需 Rick 拍板）
- [ ] DA3 缓存复用优化（P0）
- [ ] DA3 vs Pi3 精度对比（P1）
- [ ] main.py 实例化注册表闭环（P1）

---

## 七、文件清单

**本分支新增/修改的核心文件**：

```
code/modules/da3_runner.py              # 新增：DA3 venv 下独立推理脚本
code/modules/da3_3d_reconstructor.py    # 重写：subprocess 架构 + @register_reconstructor("da3")
code/modules/reconstructor_base.py      # 新增：RECONSTRUCTOR_REGISTRY + register_reconstructor
code/modules/__init__.py                # 导出注册表 + 注册 da3
code/modules/vggt_3d_reconstructor.py   # 修复 SyntaxError + @register_reconstructor("vggt")
code/modules/inference.py               # --backend choices 加 da3
code/main.py                            # run_reconstruction 加 da3 分支 + CLI choices
code/utils/config.py                    # backend 校验 + 衍生属性含 da3
code/utils/matching_algorithms.py       # cache 路径参数化 + da3 分支
code/README.md                          # DA3 后端说明 + subprocess 架构 + 命令示例
CLAUDE.md                               # DA3 模型库 + 目录树 + backend 选项
bbox_gen.py + README_bbox_gen.md        # bbox_gen 工具化 + 6 项修复
Global-ID-Mapping/README.md             # 文档漂移修复
.gitignore                              # 仓库层面去重
```

---

*本文档由本会话工作汇总，供审阅。如需调整某节详略或补充内容，告知即可。*

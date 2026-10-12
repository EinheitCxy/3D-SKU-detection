# Viewer 清晰度改进（mapping → COS Viewer / BSON scene_glb）Codex 执行任务书

> **给 Codex coordinator：** 按本文件编排 subagents。每个工作包（WP）结束后，先由 coordinator 审查（§4.5），再进入下一波。用 `- [ ]` 跟踪进度。
> 任何超出本文件写入范围、或属于 §9 审批清单的动作，一律先停下来问 Rick。

**目标：** 让 COS ZIP Viewer 和 BSON `scene_glb` 的画面明显更清楚：去掉重影、刺状碎片、边缘锯齿和过小的取景；所有默认值由预先登记的 A/B 规则决定。

**架构：** 分三层改。
- Viewer 渲染参数和布局（只改前端，旧任务包立即受益）。
- 导出侧点质量过滤，以及"每块表面单一来源帧"（新任务生效，同时影响 ZIP 和 GLB）。
- GLB 烘焙参数。

评估用独立工具 `perf/viewer_clarity/` 做源相机视角重渲染（refview）、包级指标和固定视角截图。

**技术栈：** Python 3.11 + `uv`（numpy/scipy/open3d/opencv/scikit-image 均已在 `.venv`）；Viewer：TypeScript + Vite 8.2.1 + three 0.185.1 + vitest；浏览器评估：Playwright 1.62.1（`perf/package.json`）+ headless Chromium（SwiftShader 或 GPU2）。

---

## 0. Rick 已做的决定（不可更改）

1. **`protect_mask` 保留。** 商品点继续跳过 SOR/DBSCAN（`utils/pointcloud_filter.py` 与 `src/web_viewer_export.py` 中 `filter_scene_points(valid_points, filter_config, protect_mask=labels >= 0)` 不得修改）。新增的 viewer 过滤对商品点必须带"逐 GID 保留率"保护，不得制造新的幽灵 ID（零点 GID）。
2. **COS ZIP Viewer 和 BSON `scene_glb` 都在范围内**，客户两者都看。
3. **审计提出的 P0–P3 改动全部批准**，但具体默认值必须由 §6 的 A/B 规则决定，不得事后改阈值。
4. **推送、镜像构建、容器替换、COS 写入、Harbor 推送都需要 Rick 单独批准**（§9）。

## 1. 背景：已测事实

来源：2026-10-11 审计，video3 29 帧整货架，v3 bundle `runtime/surfel-glb-20261009/baseline/runs/source/`。

| 事实 | 数值 / 位置 | 含义 |
|---|---|---|
| 贴图无损（视频源） | 29×1080×1920 JPEG Q95 4:4:4，与原图 PSNR 62–65 dB | 视频任务的模糊不是纹理分辨率造成的 |
| 贴图被降采样（照片源） | `Output/floor_display6` 原图 5712×4284；`src/surfel_export.py` 中 `short_edge = min(texture_edge, 1080)` 把它缩到 1440×1080（约 4× 线性） | 照片任务的纹理确实有损，见可选包 WP-X4 |
| 几何分辨率 | DA3 `process_res=504` → 280×504 网格，1 cell ≈ 3.86 源像素 | 轮廓有台阶，**本次不改**（会影响匹配） |
| 多帧叠层 | 商品 1.86M 点低于 2M 预算，所有帧的观测都保留；相邻帧纹理错位中位 6.7 源像素（其中系统性约 7.4px）；只有 56% 的 surfel 在他帧有一致深度；42% 的 surfel 的 8 近邻里有他帧点 | 重影、拼块、发灰的主因 → WP-X2 |
| 曝光 | 逐帧增益 0.85–1.26，但拟合增益后残差 0.130→0.133，没有下降 | 曝光补偿**不做** |
| 渲染半径 | `docker/viewer/src/scene.ts` 中 `DEFAULT_SURFEL_SIZE = MAX_POINT_SIZE = 0.07`，经 `setRadius` 得 18.4，被夹到 2.0（`surfel-renderer.ts` `setRadius`）；导出按 1.05 设计（`surfel.json` 的 `coverage_radius_pixels: 1.05`）；26% 足迹 > 3× 近邻间距（背景 70%） | 边缘刺状拖影 → V1 |
| 纹理过滤 | `surfel-loader.ts`：`LinearFilter`，无 mipmap，anisotropy=1 | 远看闪烁、摩尔纹 → V2 |
| 取景 / 布局 | `style.css` 两侧固定 280px+300px；fit 用 `display_bounds` 外接球 ×1.25 | 商品只占画布高度约 14% → V3 |
| 无 AA | Surfel 路径的 RT 没有 MSAA；输出 quad 在 `c.a > 0` 时直接不透明 | 边缘硬切 → V4 |
| 源深度容差 | `SOURCE_DEPTH_RELATIVE_TOLERANCE = 0.015`（1.8 m 处约 27 mm），采样器断裂阈值约 9 mm | 跨深度边缘贴错表面 → V5（A/B 决定） |
| 选中 | 片元直接输出纯品红 | 纹理被抹掉 → V6 |
| points 模式 | ACES ×1.1、白雾、half-Lambert 0.6–1.0 | 发灰（只影响 `&render=points`）→ V7 |
| 低置信 / 飞点 | 导出不看 DA3 置信度；深度边缘 surfel 有 63% 低于全局 conf p40；掠射角 3.2%；斜坡飞点约 1% | 碎片和漂浮 → WP-X1 |
| 背景预算 | 边缘权重 4 → 34% 的背景 surfel 卡在最小尺度 0.5 | 背景斑点 → WP-X1 |
| 商品 mask 溢出 | 1.5% 的商品 surfel 比所属 (GID, 帧) 的中位深度靠后 >10%，且受保护点不经过 SOR/DBSCAN | 商品边缘贴着货架 / 背景 → WP-X1 |
| GLB 更糊 | `src/surfel_glb.py` `export_scene_glb` → 半径 2.0、8×8 tile（2.2× 降采样）、2×2 LOD 丢 46%、`src/surfel_glb_texture.py` 128 色调色板（PSNR 约 32 dB） | → WP-G |
| 部署无漂移 | 8011 / 5175 / 5178 容器 = 分支 tip；重建镜像本身不改画质 | — |

## 2. 全局约束（每个 WP 都隐式包含）

- Python 一律 `uv run …`，在 `/home/xingyu/3D_Recognization` 根目录执行。不新增 Python 运行时依赖。
- 前端不新增 npm 依赖。浏览器评估只用 `perf/package.json` 已声明的 `playwright@1.62.1`，以及各 viewer 的 `node_modules`。
- 遵守 `AGENTS.md`：只跑与改动直接相关的最小测试；不做 hash/checksum/provenance 比对；不做全量回归；不重复运行同一验证。
- 不用 fallback 掩盖错误；不新增兼容层或 legacy 分支。
- 代码风格、注释密度、命名都与所在文件一致。文档保持现有语言（README 中文或英文按原文件）。
- **不得修改或删除** main 工作区的未跟踪文件：`.venv`、`AGENTS.md`、`controllable-autoresearch-agent/`、`fd-videos/`、`figures/`、`findings.md`、`footprint.md`、`modules/viewer_web/masterdata.json`、`node_modules/`、`progress.md`、`sku-maindata.xlsx`、`sources/`、`task_plan.md`。
- **评估产物只写到 `runtime/viewer-clarity-20261012/`**（已被 `.gitignore` 的 `runtime/` 覆盖）。只允许删除这个目录里自己生成的内容。
- **磁盘：** 当前 `/` 剩余约 28 GB。任何一步开始前若剩余 < 15 GB，立即停下报告。每个导出臂完成 crop 和 metrics 后，删除其解包的 generation 目录，只保留 `viewer_bundle.zip`、`*.json`、PNG。需要时（如 E3）从 ZIP 重新解包。
- **GPU：** 不跑 DA3/SAM 推理。浏览器 GPU 渲染只允许用 GPU2，且必须用 `nvidia-smi` 确认 Chrome GPU 进程落在 index 2；无法确认就用 SwiftShader。GPU0/1 有他人任务，禁止使用。
- **网络：** 代理不可用。npm 用 `--registry=https://registry.npmmirror.com`；Playwright 浏览器用 `PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright`；访问 COS 用 `curl --noproxy '*'`（只读 GET）。
- **提交策略：**
  - main 工作区有多个 subagent 并行写不同文件，因此 **subagent 不在 main 提交**，由 coordinator 审查后按 WP 提交。
  - en/zn viewer 工作区各只有一个写入者，可由该 subagent 按条目提交。
  - 提交信息用 Conventional Commits 英文前缀（`feat(viewer): …`、`fix(export): …`）。
  - 一律不 push。

## 3. 仓库、分支与命令

| 检出 | 路径 | 起点 | 新分支 |
|---|---|---|---|
| main（核心导出 / GLB / 工具） | `/home/xingyu/3D_Recognization` | `main@2f2b9f8` | `feat/viewer-clarity-20261012` |
| 英文 Viewer | `/home/xingyu/3D_Recognization/docker`（viewer 在 `viewer/`） | `visualization_en@d23c166` | `feat/viewer-clarity-en` |
| 中文 Viewer | `/home/xingyu/3D_Recognization/runtime/worktrees/viewer-zn-surfel`（viewer 在 `viewer/`） | `visualization_zn@a7a07ee` | `feat/viewer-clarity-zn` |
| mapping Docker | `/home/xingyu/3D_Recognization/runtime/worktrees/mapping-docker-rebuild` | `docker@c97d94a` | `feat/viewer-clarity-docker`（仅在 GLB 默认值改变、需要更新 README 时创建） |

常用命令：

```bash
# Python 定向测试（在 main 根目录）
uv run pytest test/test_web_viewer_export.py -q

# 英文 Viewer
cd /home/xingyu/3D_Recognization/docker/viewer && npm test -- --run && npm run build

# 中文 Viewer（node_modules 缺失时先：npm ci --ignore-scripts --registry=https://registry.npmmirror.com）
cd /home/xingyu/3D_Recognization/runtime/worktrees/viewer-zn-surfel/viewer && npm test -- --run && npm run build
```

中英 Viewer 的渲染代码完全一致（`scene.ts`、`surfel-renderer.ts`、`surfel-loader.ts`、`edl.ts`、`style.css`）。两者的差异：
- 英文版的界面文案在 `src/edition.ts` 的 `viewerCopy()`；
- 中文版没有 `edition.ts`，文案直接写在 `src/main.ts`（如 `text("p", "加载中")`）；
- 中文版另有 `masterdata-filters.ts`。

## 4. 编排

### 4.1 角色

| 角色 | 职责 | 写入范围 | 读取范围 |
|---|---|---|---|
| **Coordinator**（你） | 预检、分派、审查、提交、集成、A/B 决策、最终报告 | 本计划文件的勾选状态；`runtime/viewer-clarity-20261012/{PREFLIGHT,DECISIONS,REPORT}.md`；WP-X3 与 WP-D 的集成改动 | 全部 |
| **SA-Tools** | 评估工具 WP-T | `perf/viewer_clarity/**` | `src/`、`docker/viewer/src/`、`runtime/worktrees/mapping-docker-rebuild/processor.py`、`runtime/surfel-coverage-20260924/production-smoke/{browser.mjs,focus_bundle.py}` |
| **SA-Viewer** | 英文 Viewer P0（WP-V） | `docker/viewer/src/**` | `docker/viewer/**` |
| **SA-Quality** | 导出质量过滤和背景权重（WP-X1） | `src/viewer_export_tuning.py`（新）、`src/viewer_point_quality.py`（新）、`src/surfel_sampling.py`、`test/test_viewer_point_quality.py`（新）、`test/test_surfel_sampling.py` | `src/`、`utils/`、`test/` |
| **SA-Owner** | 单一来源帧（WP-X2） | `src/surfel_ownership.py`（新）、`test/test_surfel_ownership.py`（新） | `src/`、`utils/`、`test/` |
| **SA-GLB** | GLB 参数化（WP-G） | `src/surfel_glb.py`、`src/surfel_glb_texture.py`、`test/test_surfel_glb.py`、`test/test_surfel_glb_texture.py` | `src/surfel_*.py`、`test/` |
| **SA-Integrate** | 导出集成（WP-X3），在 X1/X2 提交后开始 | `src/web_viewer_export.py`、`test/test_web_viewer_export.py` | `src/`、`utils/`、`test/` |
| **SA-Eval**（独立评估） | 跑预登记的 A/B 臂、算指标、出对比图（WP-E）；**不改任何代码** | `runtime/viewer-clarity-20261012/**` | 全部 |
| **SA-Port** | 中文 Viewer 移植（WP-Z） | `runtime/worktrees/viewer-zn-surfel/viewer/**` | 中英两个 viewer |
| **SA-Docs** | 文档（WP-Doc） | 见 WP-Doc | 全部 |
| **SA-Review** | 只读审查（WP-R） | 无 | 全部 |

### 4.2 波次与依赖

```text
Wave 0  Coordinator: WP-0 预检（建分支、基线导出 X0、浏览器就绪）
Wave 1  并行: SA-Tools(WP-T) | SA-Viewer(WP-V) | SA-Quality(WP-X1) | SA-Owner(WP-X2) | SA-GLB(WP-G)
        └ coordinator 逐个审查并提交 X1、X2、G、T
Wave 2  SA-Integrate(WP-X3)  ‖  SA-Eval(WP-E1：Viewer 臂，只需 WP-V + WP-T)
        └ Coordinator: WP-D1（Viewer 决策：半径 / 容差）
Wave 3  SA-Eval(WP-E2：导出臂，需要 WP-X3 + D1) → Coordinator: WP-D2（导出决策 + 清理开关）
Wave 4  SA-Eval(WP-E3：GLB 臂，需要 D2 的最终导出) → Coordinator: WP-D3（GLB 决策）
Wave 5  并行: SA-Port(WP-Z) | SA-Docs(WP-Doc) → SA-Review(WP-R) → Coordinator: WP-F 最终验证与报告 → 停，等 Rick
Wave 6  仅在 Rick 批准后: WP-Deploy
```

**同一文件在任一时刻只能有一个写入者。** 如果某 subagent 发现必须改写入范围以外的文件，先停下来，向 coordinator 说明原因并获批。

### 4.3 派单模板（每个 subagent 必须收到）

```text
任务: <WP 编号与标题>，按 docs/superpowers/plans/2026-10-12-viewer-clarity.md 的 <章节> 执行
检出 / 分支: <路径> @ <分支>
写入范围: <精确路径列表>（之外一律不写；需要扩展先回报）
读取范围: <路径列表>
输入: <前置 WP 的产物 / 接口签名>
完成条件: <该 WP 的验收清单>
禁止: push、部署、改 protect_mask、动未跟踪文件、新增依赖、全量回归
回报格式: 见 §4.4
```

### 4.4 回报格式（subagent → coordinator）

```text
状态: done | blocked
检出 / 分支 / 提交: <路径> <分支> <sha 列表或 "未提交（main 由 coordinator 提交）">
改动文件: <path>（新增 / 修改，简述）
接口: <新增或变更的函数签名>
测试: <命令> → <通过数 / 失败数>
偏离计划: 无 | <条目 + 理由>
未验证 / 风险: <列表>
```

失败处理：同一验收项修两次仍不过，回报 `blocked`，附错误输出，不要绕过。

### 4.5 Coordinator 审查清单（每个 WP 提交前）

- [ ] `git diff --stat` 只涉及该 WP 的写入范围
- [ ] 该 WP 列出的测试命令已运行且通过（看到输出）
- [ ] 没有新依赖、没有 fallback 吞错误、没有触碰 `protect_mask`
- [ ] 接口签名与本文件一致（后续 WP 依赖它们）
- [ ] 提交：`git add <精确文件> && git commit -m "<type(scope): …>"`

---

## 5. 工作包

### WP-0 预检（Coordinator）

- [ ] 三个检出的已跟踪文件都没有改动，否则停下来问 Rick：
  ```bash
  cd /home/xingyu/3D_Recognization
  git status --porcelain --untracked-files=no
  git -C docker status --porcelain --untracked-files=no
  git -C runtime/worktrees/viewer-zn-surfel status --porcelain --untracked-files=no
  ```
- [ ] 建分支，并先提交本计划：
  ```bash
  git switch -c feat/viewer-clarity-20261012
  git add docs/superpowers/plans/2026-10-12-viewer-clarity.md
  git commit -m "docs: add viewer clarity execution plan"
  git -C docker switch -c feat/viewer-clarity-en
  git -C runtime/worktrees/viewer-zn-surfel switch -c feat/viewer-clarity-zn
  ```
- [ ] `df -h /` 剩余 ≥ 15 GB。
- [ ] 输入齐全（`ls` 每个路径）：
  - video3：`runtime/video3-resolution-review/504/outputs/dataset/{da3_cache/predictions.npz,dedup_detections/global_mapping.json,sam3_mask_cache/v2}`、`runtime/video3-resolution-review/dataset/images/`（29 张，1080×1920）
  - fd6：`Output/floor_display6/{da3_cache/predictions.npz,dedup_detections/global_mapping.json,sam3_mask_cache_sam31/v2}`、`imdata/floor_display6/images/`（11 张，5712×4284）
- [ ] **基线导出 X0**（CPU，用当前代码；同时验证输入可用）：
  ```bash
  uv run python - <<'EOF'
  import json, time
  from pathlib import Path
  from src.web_viewer_export import export_web_viewer_bundle
  R = Path("runtime/viewer-clarity-20261012")
  S = {
    "video3": dict(cache="runtime/video3-resolution-review/504/outputs/dataset/da3_cache/predictions.npz",
                   mapping="runtime/video3-resolution-review/504/outputs/dataset/dedup_detections/global_mapping.json",
                   images="runtime/video3-resolution-review/dataset/images",
                   masks="runtime/video3-resolution-review/504/outputs/dataset/sam3_mask_cache/v2"),
    "fd6": dict(cache="Output/floor_display6/da3_cache/predictions.npz",
                mapping="Output/floor_display6/dedup_detections/global_mapping.json",
                images="imdata/floor_display6/images",
                masks="Output/floor_display6/sam3_mask_cache_sam31/v2"),
  }
  for name, s in S.items():
      out = R / name / "X0"; out.mkdir(parents=True, exist_ok=False)
      t = time.monotonic()
      r = export_web_viewer_bundle(dataset_name=f"{name}-X0", da3_cache_path=Path(s["cache"]),
          global_mapping_path=Path(s["mapping"]), output_dir=out / "viewer",
          source_images_dir=Path(s["images"]), sam3_mask_cache_root=Path(s["masks"]),
          surfel_texture_edge=1920)
      (out / "export.json").write_text(json.dumps({**r, "seconds": time.monotonic() - t}, indent=2))
      print(name, r["point_count"], round(time.monotonic() - t, 1))
  EOF
  ```
  如果 fd6 导出失败：先把 `masks` 换成 `Output/floor_display6/sam3_mask_cache/v2` 再试；仍失败就按 `floor_display4 → 8 → 10` 的顺序换第二场景。最终选用的场景和 mask 根写入 `PREFLIGHT.md`，之后所有臂固定使用。
- [ ] **生产输入分辨率核对**（只读）：
  ```bash
  mkdir -p runtime/viewer-clarity-20261012/prod
  curl --noproxy '*' -fsS -o runtime/viewer-clarity-20261012/prod/UaA0rbjmBRqaola9YDpx.zip \
    https://snapshot-video-1305849923.cos.ap-shanghai.myqcloud.com/global-id-mapping/UaA0rbjmBRqaola9YDpx/viewer_bundle.zip
  ```
  读取 ZIP 内 `surfel.json` 的 `frames[].source_size / texture_size` 和帧数，写入 `PREFLIGHT.md`。这一项决定 WP-X4 是否值得做。下载失败只需记录，不阻塞后续步骤。
- [ ] **浏览器就绪：**
  ```bash
  npm ci --prefix perf --registry=https://registry.npmmirror.com
  PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright npx --prefix perf playwright install chromium-headless-shell
  ```
  然后用参数 `--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader --no-proxy-server` 启动，确认 WebGL2 和 `EXT_color_buffer_float` 都可用。
- [ ] **GPU2 探针**（可选，限时 30 分钟）：尝试 `--use-angle=vulkan --enable-features=Vulkan --ignore-gpu-blocklist`，读取 `WEBGL_debug_renderer_info`，并用 `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv` 确认进程在 GPU2。不满足就用 SwiftShader。结论写入 `PREFLIGHT.md`。
- [ ] 产出 `runtime/viewer-clarity-20261012/PREFLIGHT.md`：场景、mask 根、X0 点数与耗时、生产分辨率、渲染后端（GPU2 或 SwiftShader）。

### WP-T 评估工具（SA-Tools）

**写入：** `perf/viewer_clarity/**`。**产出接口：**

```python
# perf/viewer_clarity/scenes.py
SCENES: dict[str, dict]  # name -> {cache, mapping, images, masks, eval_frames}
#   video3: eval_frames=(4, 14, 24)；fd6: eval_frames=(1, 5, 9)（以 PREFLIGHT 最终选定的场景为准）

# perf/viewer_clarity/pack.py
def pack_bundle(generation_dir: Path, out_zip: Path) -> int  # 返回字节数
#   语义与 processor.pack_viewer_bundle 一致：
#   ZIP_STORED，扁平成员名 manifest.json/positions.f32.bin/colors.u8.bin/normals.i8.bin/objects.json/
#   surfel.json/surfel-{u,v,depth}.f16.bin/surfel-frame.u8.bin/surfel-texture-*.jpg/(v3: surfel-scale.f16.bin)/thumbs/*.jpg

# perf/viewer_clarity/export_arm.py   CLI: uv run python -m perf.viewer_clarity.export_arm --scene video3 --arm XQ
#   输出到 runtime/viewer-clarity-20261012/<scene>/<arm>/：viewer/（generation）、viewer_bundle.zip、
#   export.json（seconds, point_count, product_points, background_points, zip_bytes, tuning, git_sha）
#   臂与 ViewerExportTuning 的映射见 §6.2。若 ViewerExportTuning 不存在（WP-X3 之前），非 X0 臂直接报错，不得静默降级。

# perf/viewer_clarity/crop.py
def crop_bundle(generation_dir: Path, out_dir: Path, *, center_fraction: float = 0.4) -> dict
#   在 world_to_view 坐标下，取商品点 1–99% 包围盒中心 center_fraction 宽 × 高、全深度的盒子，
#   保留盒内全部 surfel（商品 + 背景），重写 objects.json 的 point_ranges（零点 GID 删除并记录）、
#   surfel.json 的 point_count、manifest 的 display_bounds（保留点的 1–99%）以及逐点二进制；
#   共享资产（depth、纹理、保留 GID 的缩略图）原样复制。参考 production-smoke/focus_bundle.py 的写法。

# perf/viewer_clarity/metrics.py   CLI: --bundle-dir <generation> --out metrics.json [--baseline <metrics.json>]
#   实现 §6.3 的 M1–M8

# perf/viewer_clarity/harness/   refview 页面（见下）+ render.mjs + score.py
# perf/viewer_clarity/capture.mjs   生产界面固定交互截图
# perf/viewer_clarity/sheet.py      对比图拼版
# perf/viewer_clarity/README.md     用法、前提和局限
```

**refview harness**：在源相机位姿下重渲染，与原图比较。
- 用某个臂 viewer 自带的 vite 启动：`<arm>/viewer/node_modules/.bin/vite --config perf/viewer_clarity/harness/vite.config.ts`。
  - `root` 设为 harness 目录；
  - `resolve.alias`：`@viewer` → `$VIEWER_SRC`（该臂的 `viewer/src`），`three` → 该臂的 `viewer/node_modules/three`；
  - `server.fs.allow` 要包含这两个目录和 `runtime/viewer-clarity-20261012`。
  - 不新增依赖。
- 页面参数：`?mode=surfel|glb&zip=<url>&glb=<url>&frame=<k>&scale=2&radiusSize=<number>`。
- **Surfel 模式**：
  1. 加载：`loadDockerViewerBundle(archive: Blob, mode)` 接收的是 Blob，不是 URL。先 `const blob = await (await fetch(zip)).blob()`，再调用 `@viewer/docker-zip-loader` 的 `loadDockerViewerBundle(blob, "surfel")`，然后按 `main.ts` 的写法挂上 Surfel 数据：`bundle = { ...bundle, surfels: await loadSurfels(bundle, fetch) }`（`loadSurfels` 来自 `@viewer/surfel-loader`）；
  2. `createSurfelRenderer(renderer, camera, bundle, new Matrix4(), visibility)`：
     - `worldMatrix` 用单位阵，保持 DA3 世界坐标；
     - `visibility` 是全 1 的 `Uint8BufferAttribute`，`itemSize` 为 1；
  3. **必须调用 `setRadius(radiusSize)`**，各臂的值见 §6.2，因为渲染器初始 uniform 不等于产品默认值；
  4. `setSize(w, h)` 后 `render()` 一次，再设 `window.__done = true`。
- **GLB 模式**：用 `three/addons/loaders/GLTFLoader.js` 加载，背景设为纯品红 `#ff00ff`，`NoToneMapping`，`SRGBColorSpace`。
- **相机**：对帧 k 取 `surfel.json` 的 `frames[k].intrinsic` K（processed 网格）和 `extrinsic` E（w2c，3×4）。
  - 网格尺寸 `(W, H) = surfel.json.grid_size`，渲染尺寸 `w = s·W`、`h = s·H`，`s = scale`。
  - 记 `fx = s·K00`、`fy = s·K11`、`cx = s·(K02 + 0.5)`、`cy = s·(K12 + 0.5)`，近平面 n = 0.01、远平面 f = 100。投影矩阵：
    ```text
    P = [[2fx/w, 0,     1 - 2cx/w,      0],
         [0,     2fy/h, 2cy/h - 1,      0],
         [0,     0,     -(f+n)/(f-n),   -2fn/(f-n)],
         [0,     0,     -1,             0]]
    ```
  - 视图矩阵 `V = diag(1, -1, -1, 1) · [E; 0 0 0 1]`。GLB 模式下改为 `V · inverse(W2V)`，其中 `W2V` 是 `manifest.world_to_view`（GLB 根节点自带这个变换）。
  - 设置 `camera.matrixAutoUpdate = false`，然后赋值 `projectionMatrix`、`projectionMatrixInverse`、`matrixWorldInverse = V`、`matrixWorld = V⁻¹`。
- **render.mjs**：用 Playwright，`deviceScaleFactor` 为 1，单帧超时 900 s。对每个（臂, 帧）打开页面，等到 `window.__done`，把 canvas 截图存为 `refview/<arm>/frame-<k>.png`。
- **score.py**：
  1. 参考图：对渲染像素 (x, y)，令 `p = ((x+0.5)/s − 0.5, (y+0.5)/s − 0.5)`，`src = processed_to_source · [p, 1]`，用 `cv2.remap` 双线性采样原图（原图尺寸 = `frames[k].source_size`）。
  2. 掩码：
     - `valid` = 该帧 `surfel-depth` 网格中 > 0 的像素，按最近邻放大 s 倍；
     - `refwhite` = 参考图亮度 ≥ 245；
     - 空洞 = Surfel 模式下渲染像素恰为 (255,255,255)，GLB 模式下恰为 (255,0,255)。
  3. 指标（都在 `valid ∧ ¬refwhite` 上计算）：
     - `coverage` = 非空洞像素占比；
     - `PSNR` 在 `valid ∧ ¬refwhite ∧ ¬hole` 上计算（8-bit sRGB）；
     - `SSIM` 用 `skimage.metrics.structural_similarity(..., channel_axis=-1, data_range=255, full=True)` 得到逐像素图，再在同一掩码上取均值。
  4. 输出逐帧结果和均值到 `refview/<arm>/score.json`。
- **capture.mjs**：生产界面截图。
  - 用 `vite preview` 服务该臂的 `dist/`，并参照 `production-smoke/browser.mjs` 的 `addInitScript` 改写 `fetch`：把 `…/<taskID>/viewer_bundle.zip` 指向本地 ZIP。
  - 视口 1600×900，`deviceScaleFactor` 为 1，每步超时 600 s。
  - 交互顺序：
    1. 载入后等 canvas 宽度 > 0 并空闲 2 s，截 `overview.png`；
    2. 在 canvas 中心滚轮 −1200，截 `close.png`；
    3. 从 (60%, 50%) 拖到 (73%, 57%)，分 5 步，截 `side.png`；
    4. 切到 Global ID 模式，点第一个 `.object-list .object-item`，截 `selected.png`。
  - 统计 `overview.png` 里 canvas 区域非白像素包围盒的高度占 canvas 高度的比例，记为 `product_height_fraction`，写入 `capture.json`。
- **sheet.py**：每个场景、每个视图拼一张图，列为各臂，行为各视图，标注臂名和 PSNR/coverage。

**测试**：新建 `perf/viewer_clarity/test_tools.py`，用 5×4 网格、2 帧、3 个 GID 的合成 generation 目录覆盖：
- `pack_bundle` 成员名集合正确；
- `crop_bundle` 重写后 `point_ranges` 连续且总数等于 `point_count`；
- M2/M3/M8 在构造输入上给出预期值。

运行：

```bash
uv run pytest perf/viewer_clarity/test_tools.py -q
```

**验收：**
- [ ] 测试通过；
- [ ] 对 `video3/X0` 跑一次 `metrics.py`，M1 总点数等于 X0 的 `point_count`；
- [ ] 对 `video3/X0` 的 crop 包跑一次 Surfel refview 和 capture，单帧都在超时内完成（这是冒烟，不是评估）。

Coordinator 提交：`feat(perf): add viewer clarity evaluation tools`。

### WP-V 英文 Viewer P0（SA-Viewer，`docker/` @ `feat/viewer-clarity-en`）

按下面顺序**逐条提交**（WP-E 会用中间提交构建对照臂）。每条都要通过 `npm test -- --run`；最后再跑一次 `npm run build`。

**V1 半径与导出设计一致**（`surfel-renderer.ts`、`scene.ts`）

```ts
// surfel-renderer.ts
export const DEFAULT_SURFEL_RADIUS = 1.05; // == surfel.json coverage_radius_pixels
export function surfelRadiusForSize(size: number): number {
  return Math.min(2.0, Math.max(0.6, DEFAULT_SURFEL_RADIUS * size / 0.004));
}
// uniforms: uRadius: { value: DEFAULT_SURFEL_RADIUS }
// setRadius(size) { uniforms.uRadius.value = surfelRadiusForSize(size); }

// scene.ts
export const DEFAULT_SURFEL_SIZE = MIN_POINT_SIZE; // -> uRadius 1.05（原为 MAX_POINT_SIZE -> 2.0）
```

测试：新建 `src/surfel-renderer.test.ts`，断言 `surfelRadiusForSize(0.004) ≈ 1.05`、`surfelRadiusForSize(1) === 2.0`、`surfelRadiusForSize(0) === 0.6`；在 `scene.test.ts` 断言 `surfelRadiusForSize(DEFAULT_SURFEL_SIZE) ≈ 1.05`。
提交：`fix(viewer): render Surfels at the exporter coverage radius`。

**V2 纹理 mipmap 与各向异性**

- `surfel-loader.ts`：
  ```ts
  textures.generateMipmaps = true;
  textures.minFilter = LinearMipmapLinearFilter;
  textures.magFilter = LinearFilter;
  ```
- `createSurfelRenderer`：
  ```ts
  data.textures.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  ```
- 预期显存增加约 33%（29 帧 1080p 约 240 → 320 MB），由 WP-Doc 写入 README；WP-V 不改 README，以免中文移植时产生冲突。

测试：在 `surfel-loader.test.ts` 断言 `minFilter === LinearMipmapLinearFilter`、`generateMipmaps === true`、`magFilter === LinearFilter`。
提交：`fix(viewer): mipmap Surfel source textures`。

**V3 商品取景与可折叠面板**

1. 新建 `src/product-fit.ts`：
   ```ts
   import { Matrix4, Vector3 } from "three";
   import type { ObjectIndex } from "./contracts";
   export interface ViewBox { readonly center: Vector3; readonly size: Vector3 }
   /** 1–99% per-axis bounds of product points (all objects' point_ranges) after worldToView; null if none. */
   export function productViewBox(positions: Float32Array, objects: ObjectIndex, worldToView: Matrix4, maxSamples = 200_000): ViewBox | null;
   export function fitDistance(size: Vector3, fovDegrees: number, aspect: number, margin = 1.08): number {
     const t = Math.tan((fovDegrees * Math.PI) / 360);
     return Math.max(size.y / 2 / t, size.x / 2 / (t * aspect)) * margin + size.z / 2;
   }
   ```
   `productViewBox` 按等步长抽样，样本不超过 `maxSamples`；先逐点乘 `worldToView`，再逐轴排序取 1% 和 99% 分位。
2. `scene.ts`：
   - 创建场景时计算一次 `productBox = productViewBox(bundle.positions, bundle.objects, worldGroup.matrix)`；
   - `setViewPreset("fit")` 且没有选中对象时，`target = productBox.center`，`distance = fitDistance(productBox.size, camera.fov, camera.aspect)`，方向仍用 `viewPresetDirection("fit")`；
   - `productBox` 为 null 时（bundle 没有商品点，这是合法情况）保留原外接球逻辑；
   - `top`/`isometric` 不变；
   - 初次 `setViewPreset("fit", false)` 之前，同步调用一次 `resize()`，保证 `camera.aspect` 正确。
3. 布局（`main.ts` + `style.css`）：
   - `.viewer-shell` 改用 `grid-template: minmax(0, 1fr) / var(--left-width, 280px) minmax(0, 1fr) var(--right-width, 300px)`；
   - `[data-left="collapsed"]` 设 `--left-width: 0px`，`[data-right="collapsed"]` 设 `--right-width: 0px`；折叠的面板 `visibility: hidden; padding: 0`；
   - 在 `.scene-stage` 内加 `.scene-toolbar`（右上角绝对定位，`.scene-stage` 设 `position: relative`），放三个按钮：
     - `data-action="fit-view"`：调用 `controller.setViewPreset("fit")`；
     - `data-action="toggle-list"`：切换左面板；
     - `data-action="toggle-details"`：切换右面板；
   - 初始状态：左面板展开，右面板折叠；
   - 任何选中路径（列表点击、画布拾取、SKU facet）执行 `renderSelectedObject()` 且确有选中时，把右面板设为展开；用户可以再次手动折叠；
   - 画布尺寸由已有的 `ResizeObserver` 自动跟随。
4. 文案：`edition.ts` 的 `viewerCopy()` 新增 `fitView: "Fit view"`、`showList: "Show list"`、`hideList: "Hide list"`、`showDetails: "Show details"`、`hideDetails: "Hide details"`。

测试：
- 新建 `src/product-fit.test.ts`：
  1. `fitDistance(new Vector3(2, 1, 0), 50, 2)` 等于按公式手算的值；
  2. 背景点（不在任何 `point_ranges` 内）不影响结果；
  3. 单个离群点（坐标 1e3）不影响 1–99% 分位。
- `main.test.ts`：按现有依赖注入方式断言：
  1. 初始 `data-right="collapsed"`；
  2. 选中对象后变为 `"expanded"`；
  3. 点击 `toggle-list` 后 `data-left="collapsed"`；
  4. 点击 `fit-view` 会以 `"fit"` 调用 controller 的 `setViewPreset`。

提交：`feat(viewer): frame products by default and allow collapsing panels`。

**V7 points 模式去灰**（排在 V4 之前，因为 V4 依赖 NoToneMapping）

- `scene.ts`：
  - `renderer.toneMapping = NoToneMapping`，删除 `toneMappingExposure` 设置；
  - 删除 `scene.fog = new Fog(...)` 以及随之不再使用的 `Fog` import 和 `FOG_*` 常量；
  - 点着色器中把 `vColor * (0.6 + 0.4 * halfLambert)` 改为 `vColor * (0.85 + 0.15 * halfLambert)`。
- `edl.ts`：注释 "OutputPass (ACES tone mapping + sRGB)" 改为 "OutputPass (sRGB)"。

提交：`fix(viewer): stop tone mapping and fogging point colors`。

**V4 Surfel 边缘柔化 + SMAA**（`surfel-renderer.ts`）

- 输出着色器改为输出线性颜色，由 OutputPass 统一转换为 sRGB：
  ```glsl
  uniform sampler2D uAccumulation; uniform vec3 uBackground; varying vec2 vUv;
  void main() { vec4 c = texture2D(uAccumulation, vUv);
    vec3 rgb = c.a > 0.0 ? c.rgb / c.a : uBackground;
    gl_FragColor = vec4(mix(uBackground, rgb, smoothstep(0.03, 0.3, c.a)), 1.0);
  }
  ```
  删除 `#include <colorspace_fragment>`。`uBackground` 取场景背景色的线性值（白色即 `(1, 1, 1)`）。
- 合成：`createSurfelRenderer` 内建 `EffectComposer(renderer)`，依次加入 `RenderPass(outputScene, outputCamera)`、`new SMAAPass()`、`new OutputPass()`。
  - `render()` 中前两个 pass 不变，之后 `renderer.setRenderTarget(null); composer.render();`；
  - `setSize` 同步 `composer.setPixelRatio(renderer.getPixelRatio()); composer.setSize(width, height)`；
  - `dispose` 释放 composer。
  - import 路径与 `edl.ts` 一致（`three/addons/postprocessing/...`）。
- `pick()` 不变（不走 composer）。

提交：`fix(viewer): feather Surfel coverage edges and apply SMAA`。

**V6 选中着色保留纹理**（`surfel-renderer.ts`）

- 片元着色器新增 `uniform float uHasSelection;`，把选中那一行改为：
  ```glsl
  if (vSelected > 0.5) color = mix(color, vec3(1.0, 0.0, 1.0), 0.35);
  else if (uHasSelection > 0.5) color = mix(color, vec3(1.0), 0.5);
  ```
- `select(ranges)` 中设 `uniforms.uHasSelection.value = ranges.length > 0 ? 1 : 0;`。

提交：`fix(viewer): tint selected Surfels without hiding their texture`。

**V5 源深度容差**（单独提交，必须是最后一个；由 D1 决定保留还是 revert）

- `surfel-renderer.ts`：`export const SOURCE_DEPTH_RELATIVE_TOLERANCE = 0.005;`（原为 0.015，不加 export）。
- 测试：在 `surfel-renderer.test.ts` 断言该值。

提交：`fix(viewer): tighten Surfel source depth tolerance`。

**验收（WP-V 整体）：**
- [ ] `npm test -- --run` 和 `npm run build` 通过；
- [ ] 7 个提交顺序为 V1、V2、V3、V7、V4、V6、V5；
- [ ] 回报每个提交的 sha。

> V4/V6/V7 是着色器和管线改动，单测只覆盖导出的纯函数和常量，实际效果由 WP-E1 截图与 refview 验收。

### WP-X1 导出质量过滤与背景权重（SA-Quality，main，不提交）

1. 新建 `src/viewer_export_tuning.py`：
   ```python
   @dataclass(frozen=True)
   class ViewerExportTuning:
       """Viewer-only export options under A/B evaluation (defaults = current behavior)."""
       quality_filter: bool = False
       product_spill_trim: bool = False
       background_edge_weights: tuple[int, int, int] = (4, 2, 1)
       single_owner: bool = False
       owner_patch_cells: int = 4
   ```
   `__post_init__` 校验：权重是 3 个 1..255 的 int；`owner_patch_cells` 在 2..16；违反时抛 `ValueError`。

2. 新建 `src/viewer_point_quality.py`，所有数组都以 cache 网格 `(F, H, W)` 和 `flat = F*H*W` 为基准：
   ```python
   def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
       """w2c (F,3,4) or (F,4,4) -> camera centers (F,3) = -R^T t."""

   def pixel_quality_drop(*, depth, valid, confidence, u, v, points, centers, spacing) -> tuple[np.ndarray, dict[str, int]]:
       """(F,H,W) bool: pixels a viewer should not draw; counts per rule."""

   def viewer_quality_keep(*, indices, labels, drop, min_label_retention=0.5) -> np.ndarray:
       """Keep mask aligned with `indices` (flat grid indices) and `labels` (-1 background, >=0 product label)."""

   def product_spill_keep(*, indices, labels, depth, ratio=1.10, max_drop_fraction=0.30) -> np.ndarray:
       """Drop product points behind their (label, frame) median camera depth."""
   ```
   规则（全部写成代码常量并加注释）：
   - `conf_p20`、`conf_p40` 取 `confidence[valid]` 的 20/40 百分位；
   - `edge`：逐帧调用 `src.surfel_sampling._depth_breaks(depth[f], valid[f], spacing, axis)`（axis 1 和 0），用与 `EdgeAdaptiveSampler.__init__` 相同的方式膨胀成像素级标记（`edge[:, :-1] |= bx; edge[:, 1:] |= bx; …`）；
   - `grazing`：`n = cross(u, v)`，`ray = points - centers[f]`，`cos = |n·ray| / (|n||ray|)`；`|n| == 0` 的像素记为非掠射；`grazing = cos < 0.15`；
   - `mid_slope`（斜坡中段飞点）：沿 x 取 `dl = d - d[x-1]`、`dr = d[x+1] - d`，`tol = max(2*spacing, 0.005*d)`；满足 `|dl| > tol ∧ |dr| > tol ∧ sign(dl) == sign(dr)`、且左右邻都有效时为真；y 方向同理，取二者的或；
   - `drop = (conf < p20) | (edge & (conf < p40)) | grazing | mid_slope`；
   - `viewer_quality_keep`：背景（`labels < 0`）直接 `keep = ~drop`；每个商品 label 若 `mean(~drop) >= min_label_retention` 则 `keep = ~drop`，否则该 label **全部保留**（保护 `protect_mask` 的意图）；
   - `product_spill_keep`：按 `(label, frame = index // (H*W))` 分组，`med = median(depth)`，`spill = depth > ratio*med`；若 `spill.mean() <= max_drop_fraction` 则丢弃 spill，否则整组保留；背景不处理。

3. `src/surfel_sampling.py`：`EdgeAdaptiveSampler.__init__(self, depth, valid, spacing, regions=None, *, background_weights=(4, 2, 1))`。当 `regions is not None` 时，`regions[frame] == -1` 的像素用 `background_weights` 的（边缘, 光晕, 内部）三档，其余像素保持 4/2/1；`regions is None` 时行为不变。

4. 测试：
   - 新建 `test/test_viewer_point_quality.py`（合成网格，每个用例 ≤ 10×10 像素、≤ 3 帧）：
     - (a) 两平面之间斜坡上的插值像素被丢弃，平面像素保留；
     - (b) 某 label 全部低置信时整组保留；
     - (c) 85° 观察的背景平面被丢弃；
     - (d) 靠后 20% 的点在组内占 ≤ 30% 时被丢弃，> 30% 时整组保留；
     - (e) `camera_centers` 对已知 R、t 给出正确结果。
   - `test/test_surfel_sampling.py` 增加：背景区域使用 `background_weights`；`regions=None` 时权重与修改前完全相同。
   ```bash
   uv run pytest test/test_viewer_point_quality.py test/test_surfel_sampling.py -q
   ```

**验收：**
- [ ] 测试通过；
- [ ] 未修改 `web_viewer_export.py` 和 `utils/pointcloud_filter.py`；
- [ ] 回报接口签名。

Coordinator 提交：`feat(export): add viewer point quality filters and background sampling weights`。

### WP-X2 每块表面单一来源帧（SA-Owner，main，不提交）

新建 `src/surfel_ownership.py`：

```python
def frame_sharpness_weights(image_paths: Sequence[Path]) -> np.ndarray:
    """Per-frame weight in [0.25, 1]: Laplacian variance of the grayscale image resized
    to longest edge 960, divided by the median over frames, clipped to [0.25, 1]."""

def select_single_owner(*, points, frames, labels, normals, centers, confidence, sharpness,
                        spacing, patch_cells) -> np.ndarray:
    """Keep mask: within each (patch, normal bin, label), keep only the best-scoring frame;
    then refill fine voxels that lost all points from their best frame."""
```

算法（全部向量化，不用 Python 循环逐点处理）：
1. `cos = |n·(p - c_f)| / (|n||p - c_f|)`，零法向时 cos 取 0；`conf_q` = confidence 在输入内的百分位秩，范围 [0, 1]；`depth` = 点到 `c_f` 沿光轴的距离（用 `|p - c_f|` 近似即可，注释说明）。
2. `score = cos * (median(depth) / depth) * (0.25 + 0.75 * conf_q) * sharpness[frames]`。
3. 法向分 7 箱：世界坐标下 `argmax|n|` 的轴 × 符号得 6 箱，零法向为第 7 箱。
4. `patch = floor(p / (patch_cells * spacing))`。键 = (patch_x, patch_y, patch_z, normal_bin, label)，用 `np.unique(..., axis=0, return_inverse=True)` 得到组号。
5. 对每组按帧累加 score，用 `np.bincount(group * F + frame)`；`owner = argmax`，并列时取较小帧号。`keep = frame == owner[group]`。
6. 细体素补洞：`fine = floor(p / spacing)`，键 = (fine_xyz, normal_bin, label)。对 `keep` 中完全没有点的细键，在该细键内按 score 和选一个最佳帧，保留该帧在该细键内的全部点。
7. 返回 `keep`（bool，与输入对齐）。

**性能目标**：3.5M 点在本机 CPU ≤ 20 s（在 WP-E2 的导出耗时里记录）。

测试 `test/test_surfel_ownership.py`（合成数据）：
- (a) 两帧观测同一平面，相距 4 mm 分层：每个 patch 只保留 cos 更高的帧；
- (b) 补洞：A 帧只覆盖一半细体素、B 帧全覆盖时，缺失的一半由 B 补上；
- (c) label 保证：label 7 只被低分帧看到，仍然保留；
- (d) 同一输入运行两次结果完全相同；
- (e) 零法向点不报错；
- (f) `frame_sharpness_weights` 在一张模糊图和一张清晰图上，权重满足 模糊 < 清晰，且都在 [0.25, 1] 内。

```bash
uv run pytest test/test_surfel_ownership.py -q
```

**验收：**
- [ ] 测试通过；
- [ ] 只写了两个新文件。

Coordinator 提交：`feat(export): select a single source frame per Surfel surface patch`。

### WP-G GLB 参数化（SA-GLB，main，不提交）

- `src/surfel_glb_texture.py`：
  - `compact_png(raw: bytes, colors: int = 128) -> bytes`，要求 `2 <= colors <= 128`；透明纹素 RGB 保留的逻辑不变，调色板长度随 `colors` 变化；
  - 新增 `rgba_png(raw: bytes) -> bytes`：校验 alpha 为二值后，原样保存为 RGBA PNG（`optimize=True`）。
- `src/surfel_glb.py`：`export_scene_glb(source, output, *, radius=2.0, tile_size=8, lod=True, palette_colors: int | None = 128)`。
  - `palette_colors=None` 时用 `rgba_png`，否则用 `partial(compact_png, colors=palette_colors)`；
  - 保留 `lod and radius != 2.0 → ValueError` 的现有约束；
  - 返回字典里加入这四个参数。
  - 默认值暂不改，由 D3 决定。
- 测试：
  - `test/test_surfel_glb_texture.py`：`colors=64` 时不透明调色板项 ≤ 64；`rgba_png` 保留 alpha=0 纹素的 RGB，且 alpha 为二值。
  - `test/test_surfel_glb.py`：复用现有小夹具，`radius=1.05, tile_size=10, lod=False, palette_colors=None` 能产出合法 GLB，图集 tile 步长为 12（tile+2）。
  ```bash
  uv run pytest test/test_surfel_glb.py test/test_surfel_glb_texture.py -q
  ```

**验收：**
- [ ] 测试通过；
- [ ] `export_scene_glb()` 无参数调用时输出与修改前一致（现有测试不改即通过）。

Coordinator 提交：`feat(glb): parameterize scene GLB radius, tile size, LOD and palette`。

### WP-X3 导出集成（SA-Integrate，main，不提交；在 X1、X2 提交后开始）

- `export_web_viewer_bundle(..., surfel_texture_edge=None, tuning: ViewerExportTuning | None = None)`，函数内 `tuning = tuning or ViewerExportTuning()`。
  - 当 `surfel_texture_edge is None`（普通点云导出）而 `tuning != ViewerExportTuning()` 时，抛 `WebViewerExportError("viewer export tuning requires Surfel export")`，不得静默忽略。
- `tuning.single_owner` 为真时，在 `export_web_viewer_bundle` 中计算 `frame_sharpness_weights([images[int(i)] for i in cache["image_ids"]])`，其中 `images = _resolve_source_images(source_images_dir)`，再传给 `_sample_points(..., tuning=tuning, frame_sharpness=...)`。
- `_sample_points` 内的插入点（只在 `adaptive_surfels` 路径）：
  1. 创建 `EdgeAdaptiveSampler(..., background_weights=tuning.background_edge_weights)`；
  2. `keep_filter = filter_scene_points(...)` 之后，若 `quality_filter`：
     ```python
     drop, counts = pixel_quality_drop(depth=depth, valid=valid.reshape(depth.shape),
         confidence=cache["confidence"], u=native_u, v=native_v, points=cache["points"],
         centers=camera_centers(cache["extrinsic"]), spacing=spacing)
     keep_filter &= viewer_quality_keep(indices=valid_indices, labels=labels, drop=drop)
     ```
     若 `product_spill_trim`：`keep_filter &= product_spill_keep(indices=valid_indices, labels=labels, depth=depth)`；
  3. `filtered_points = valid_points[keep_filter]` 仍在单一来源帧筛选**之前**取值，作为朝向输入，保持 `world_to_view` 稳定；
  4. 若 `single_owner`：对 `kept = np.flatnonzero(keep_filter)` 调用 `select_single_owner`，输入 `points=valid_points[kept]`、`frames=valid_indices[kept] // (H*W)`、`labels=labels[kept]`、`normals=np.cross(native_u.reshape(-1,3)[valid_indices[kept]], native_v.reshape(-1,3)[valid_indices[kept]])`、`centers`、`confidence=flat_confidence[valid][kept]`、`sharpness`、`spacing`、`patch_cells=tuning.owner_patch_cells`，然后 `keep_filter[kept[~own]] = False`；
  5. 之后再计算 `labels`、`points`、`confidence`、`colors` 等，其余流程不变。
  6. 每个阶段用 `logger.info` 记录移除的点数。
- 测试（`test/test_web_viewer_export.py`，复用现有 `exporter_inputs` 夹具和合成缓存）：
  - (a) 现有测试不改，全部通过（默认行为不变）；
  - (b) 全开 tuning 时导出成功，默认导出中点数 > 0 的每个 GID 仍然点数 > 0，且 `point_count` ≤ 默认值；
  - (c) 普通点云导出加非默认 tuning 时抛错。
  ```bash
  uv run pytest test/test_web_viewer_export.py test/test_viewer_mask_retention.py test/test_matching_viewer_identity.py -q
  ```

**验收：**
- [ ] 测试通过；
- [ ] 对 video3 用 `export_arm.py --arm XQS4`（全开）导出一次冒烟，记录耗时。

Coordinator 提交：`feat(export): wire viewer tuning into Surfel export`。

### WP-E A/B 评估（SA-Eval，独立；只写 `runtime/viewer-clarity-20261012/**`）

**通用规则：**
- 只跑 §6.2 预登记的臂，不调参，不改代码；遇到工具 bug 回报 coordinator，由 SA-Tools 修。
- 渲染后端和 crop 设置在 E0 确定后对所有臂固定：
  - 若 PREFLIGHT 确认可用 GPU2，用全量包；
  - 否则用 `crop_bundle(center_fraction=0.4)` 的裁剪包，refview 和 capture 都用裁剪包。
- 每个臂产出 `metrics.json`、`refview/score.json`、`capture/*.png`、`capture.json`。所有命令、提交 sha、耗时写入 `RESULTS.md`。

**Viewer 臂工作树**（`docker/` 仓库的临时 worktree，用完删除）：

```bash
A=/home/xingyu/3D_Recognization/runtime/viewer-clarity-20261012/arms
git -C /home/xingyu/3D_Recognization/docker worktree add --detach "$A/V0" d23c166
ln -s /home/xingyu/3D_Recognization/docker/viewer/node_modules "$A/V0/viewer/node_modules"
(cd "$A/V0/viewer" && npm run build)
# V1a = WP-V 的 V6 提交；V1b = V1a 加未提交补丁：DEFAULT_SURFEL_SIZE = MIN_POINT_SIZE * 1.3 / 1.05；V2 = V5 提交
# 结束后：git -C /home/xingyu/3D_Recognization/docker worktree remove --force "$A/<arm>"
```

**E1 Viewer 臂**（数据 = 两个场景的 X0 包）：
- 对 V0、V1a、V1b、V2 跑 refview（Surfel 模式，`radiusSize` 见 §6.2），每个场景用 3 帧；
- 对 V0、V1a、V1b 跑 capture；
- 出对比图；
- 结果交给 coordinator 做 D1。

**E2 导出臂**（Viewer = D1 选定的提交）：
- 两个场景各导出 X0（已有）、XQ、XS4、XS8；
- 每个臂跑 metrics、refview、capture；
- D2 选出最佳单一来源帧臂后，再导出 XQS（XQ 加最佳 XS）并评估。

**E3 GLB 臂**（数据 = D2 的最终导出 `viewer_bundle.zip`；`export_scene_glb` 内部的 `_assets` 直接接受 ZIP 或 generation 目录，无需解包）：
- 用 `export_scene_glb(generation, out, **arm)` 生成 G0–G3，记录字节数和耗时；
- 对每个 GLB 跑 refview（GLB 模式）；
- 用一个最小 three GLTFLoader 页面截 overview 和 close 两张图（可放在 harness 的 `?mode=glb&view=overview|close`，相机由 `productViewBox` 加 `fitDistance` 决定）。

### WP-D 决策与定稿（Coordinator）

- **D1**（E1 之后）：按 §6.4 写 `DECISIONS.md`。
  - 若选 V1b（半径 1.3）：在 en 分支提交 `DEFAULT_SURFEL_SIZE = MIN_POINT_SIZE * 1.3 / 1.05`；同时 main 分支把 `src/surfel_sampling.py` 中 `_axis_caps` 的 `0.95 / 1.05` 和 `src/surfel_export.py` 中 `"coverage_radius_pixels": 1.05` 改为共享常量 `COVERAGE_RADIUS_PIXELS = 1.3`（放在 `src/surfel_sampling.py`，`surfel_export.py` import 它），然后只对最终臂补跑一次 E2 指标冒烟。
  - 若 V5 被否决：`git -C docker revert <V5 sha>`。
  - 若 V5 被采纳：main 分支新增共享常量 `SOURCE_DEPTH_RELATIVE_TOLERANCE = 0.005`（放在 `src/surfel_export.py`），用于 `surfel.json` 字段和 `src/surfel_mesh.py` 中原来写死 `0.015` 的那行；运行 `uv run pytest test/test_surfel_mesh.py test/test_surfel_glb.py -q`。
- **D2**（E2 之后）：
  - 被采纳的选项改为**固定行为**，删除对应开关；被否决的选项连同代码和测试一并删除；
  - `ViewerExportTuning` 若不再有字段，整个文件删除，`export_web_viewer_bundle` 的 `tuning` 参数随之删除，`perf/viewer_clarity/export_arm.py` 只保留"当前代码"臂，并在 README 注明各臂结果对应的提交 sha；
  - 背景权重若被采纳，把 `(2, 1, 1)` 作为 `_sample_points` 传给 sampler 的固定值；
  - 运行 `uv run pytest test/test_web_viewer_export.py test/test_viewer_point_quality.py test/test_surfel_ownership.py test/test_surfel_sampling.py -q`（被删除的测试文件不运行）。
  - 提交：`refactor(export): adopt evaluated viewer export defaults`。
- **D3**（E3 之后）：`export_scene_glb` 的默认参数改为胜出臂（G0 胜出则不改）；`processor.py` 不需要改，它使用默认值。提交：`feat(glb): adopt evaluated scene GLB defaults`。

### WP-Z 中文 Viewer 移植（SA-Port，`runtime/worktrees/viewer-zn-surfel` @ `feat/viewer-clarity-zn`）

- [ ] 在 D1 定稿后：
  ```bash
  git -C /home/xingyu/3D_Recognization/docker format-patch visualization_en..feat/viewer-clarity-en -o /home/xingyu/3D_Recognization/runtime/viewer-clarity-20261012/patches
  git -C /home/xingyu/3D_Recognization/runtime/worktrees/viewer-zn-surfel am --3way /home/xingyu/3D_Recognization/runtime/viewer-clarity-20261012/patches/*.patch
  ```
- [ ] 冲突处理：`edition.ts` 在中文版不存在，文案直接写进 `main.ts`：适配视图 / 显示列表 / 隐藏列表 / 显示详情 / 隐藏详情；中文版特有的 `masterdata-filters.ts` 不动。
- [ ] 中文 viewer 跑 `npm test -- --run && npm run build` 通过。
- [ ] 用 capture.mjs 对中文 dist 在 video3 上截一次 overview 和 selected 冒烟图，存入 `runtime/viewer-clarity-20261012/zn-smoke/`。

### WP-Doc 文档（SA-Docs）

按最终采纳结果更新以下文档，只写事实与用法，保持简洁：

- `docker/viewer/README.md`（en 分支）和 `runtime/worktrees/viewer-zn-surfel/viewer/README.md`（zn 分支）：默认 Surfel 半径；面板折叠和 Fit 按钮；mipmap 带来的显存增加；points 模式不再做色调映射。
- `docs/surfel_implementation.md`（main）：质量过滤、spill 裁剪、单一来源帧、背景权重的规则与实测收益（引用 `RESULTS.md` 的数字）；`protect_mask` 保留，以及逐 GID 保留率保护。
- `docs/surfel_glb.md`（main）：新的 GLB 默认值、包大小、耗时。
- `runtime/worktrees/mapping-docker-rebuild/README.md` 的"scene.glb 下载输出"一节：GLB 默认值有变化时，在 `feat/viewer-clarity-docker` 分支更新"固定使用减量方案"那句。

### WP-R 只读审查（SA-Review）

对 4 个分支的完整 diff（相对起点）检查以下各项，返回带 `file:line` 的问题清单，coordinator 逐条修复或说明理由：
- 是否超出范围；
- `protect_mask` 是否保持原样；
- 有无新依赖；
- 有无 fallback 吞错误；
- 着色器改动是否破坏 `pick()`；
- `productViewBox` 抽样是否确定；
- `select_single_owner` 的 label 保证是否成立；
- 测试是否真的覆盖了行为；
- 文档与实现是否一致。

### WP-F 最终验证与报告（Coordinator）→ 停

- [ ] main：
  ```bash
  uv run pytest test/test_web_viewer_export.py test/test_surfel_sampling.py test/test_surfel_glb.py test/test_surfel_glb_texture.py test/test_surfel_mesh.py test/test_surfel_lod.py test/test_viewer_mask_retention.py test/test_matching_viewer_identity.py perf/viewer_clarity/test_tools.py -q
  ```
  再加上 D2 后仍存在的新测试文件。
- [ ] en、zn：`npm test -- --run && npm run build`。
- [ ] 四个检出都跑 `git diff --check` 和 `git status`（确认没有遗漏文件；未跟踪文件保持原样）。
- [ ] 删除临时 viewer 臂工作树；删除 generation 解包目录。
- [ ] 写 `runtime/viewer-clarity-20261012/REPORT.md`（模板见 §7），附对比图路径。
- [ ] 向 Rick 汇报，然后**停止**，等待 §9 的批准。

### WP-Deploy（仅在 Rick 明确批准后）

- 先用 `docker inspect` 记录现有三个容器的完整运行参数（端口、GPU、环境变量、卷），写入 `runtime/viewer-clarity-20261012/DEPLOY.md`。
- **Viewer**：
  1. 在 en、zn 各自的 `viewer/` 目录 `docker build -t global-id-viewer-en:clarity-20261012 .` 和 `docker build -t global-id-viewer-zn:clarity-20261012 .`（**不要**运行会推送 Harbor 的 `build.sh`）；
  2. `docker rename global-id-viewer-en global-id-viewer-en-prev-20261012`、`docker stop global-id-viewer-en-prev-20261012`，再用相同参数（`192.168.2.6:5178->80`）运行新镜像；中文同理（`0.0.0.0:5175->80`）；
  3. 浏览器打开 `/?recognition_task_id=UaA0rbjmBRqaola9YDpx` 冒烟（旧任务包立即受益于 P0）。
- **Mapping**：
  1. 构建：
     ```bash
     CORE_REPO_ROOT=/home/xingyu/3D_Recognization BASE_IMAGE=global-id-mapping:surfel-glb-bson-20261009 \
     IMAGE_TAG=global-id-mapping:clarity-20261012 bash runtime/worktrees/mapping-docker-rebuild/build_code_update.sh
     ```
     注意：`CORE_REPO_ROOT` 必须显式设置，脚本默认值指向 `runtime/worktrees`，是错的。
  2. 按记录的参数替换 8011 容器（固定 GPU2），旧容器改名保留以便回滚；
  3. 用 Rick 指定的一个真实任务请求一次，检查：HTTP 200、`global_skus` 非空、`scene_glb` 大小落在 E3 记录范围内、COS 上新的 `viewer_bundle.zip` 能被新 Viewer 打开。
- **回滚**：停新容器，`docker rename` 改回旧容器并启动。

### WP-X4 照片源纹理上限（**默认不执行**；需 Rick 根据 PREFLIGHT 的生产分辨率确认）

- 现状：`src/surfel_export.py` 中 `short_edge = min(texture_edge, 1080)`，4:3 照片因此被限制为 1440×1080。
- 改动：最长边仍不超过 `texture_edge`（1920），取消短边 1080 的限制；同时设总纹素预算 `N × w × h ≤ 29 × 1920 × 1080`，超出时所有帧按同一比例等比缩小。
- 验收：
  - fd6 的 refview PSNR ≥ +0.5 dB；
  - `surfel_texture_edge` 校验报错文案同步更新；
  - 补测试：预算换算和 4:3 输出尺寸。
- 显存影响写进 README。

---

## 6. A/B 协议（预登记；结果出来之前锁定，不得修改）

### 6.1 场景

| 场景 | 类型 | refview 帧（cache 帧序） |
|---|---|---|
| video3 | 29 帧 1080×1920 视频 | 4、14、24 |
| fd6（或 PREFLIGHT 选定的替代场景） | 11 张 5712×4284 静态图 | 1、5、9 |

### 6.2 臂

| 臂 | 内容 | refview `radiusSize` |
|---|---|---|
| V0 | Viewer `visualization_en@d23c166` | 0.07 |
| V1a | WP-V 的 V6 提交（V1、V2、V3、V7、V4、V6，半径 1.05） | 0.004 |
| V1b | V1a + 半径 1.3 补丁 | 0.004 × 1.3 / 1.05 = 0.004952 |
| V2 | WP-V 的 V5 提交（V1 胜者 + 容差 0.005） | 同 V1 胜者 |
| X0 | 当前导出 | — |
| XQ | `quality_filter=True, product_spill_trim=True, background_edge_weights=(2,1,1)` | — |
| XS4 | `single_owner=True, owner_patch_cells=4` | — |
| XS8 | `single_owner=True, owner_patch_cells=8` | — |
| XQS | XQ + 最佳 XS | — |
| G0 | `radius=2.0, tile_size=8, lod=True, palette_colors=128`（现状） | — |
| G1 | `radius=1.05, tile_size=10, lod=False, palette_colors=128` | — |
| G2 | `radius=1.05, tile_size=10, lod=False, palette_colors=None`（RGBA） | — |
| G3 | `radius=1.3, tile_size=12, lod=False, palette_colors=128` | — |

若 D1 选了半径 1.3，G1、G2 的 `radius` 改为 1.3、`tile_size` 改为 12，G3 取消。

### 6.3 包级指标（`metrics.py`）

- **M1** 点数：总数、商品数、背景数。
- **M2** 接缝率：用 `cKDTree` 取每个 surfel 的 9 近邻（含自身）；8 个邻居中至少有一个来自其他帧的 surfel 占比。
- **M3** 多帧体素率：8 mm 体素中，含 ≥ 2 个帧 surfel 的体素占全部被占体素的比例。
- **M5** 背景最小尺度率：背景 surfel 中 `min(scale) == 0.5` 的比例。
- **M6** 孤立碎片率：以 3 × 中位最近邻距离为半径连边（`cKDTree.query_pairs` 加 `csgraph.connected_components`），点数 < 500 的连通分量所含点的占比。
- **M7** 商品溢出率：商品 surfel 在自身帧的相机深度 > 1.10 × 其 (GID, 帧) 中位深度的比例。相机参数取自 `surfel.json`。
- **M8** 新幽灵 ID 数：`objects.json` 中在本臂点数为 0、但在 X0 中点数 > 0 的 GID 个数。
- 另外记录导出秒数和 ZIP 字节数。

### 6.4 决策规则（两个场景都必须满足；pp = 百分点）

- **D1 半径：** 若 `coverage(V1a) ≥ coverage(V1b) − 1.0 pp`，选 V1a（1.05）；否则选 V1b。
- **D1 P0 整包：** 胜出的 V1x 必须同时满足以下三条，并附对比图：
  - `PSNR ≥ PSNR(V0) − 0.2 dB`；
  - `coverage ≥ coverage(V0) − 2.0 pp`；
  - `product_height_fraction ≥ 1.5 × V0`。
  
  若不满足，停下报告，不得自行放宽。
- **D1 容差：** V2 相对胜出 V1x 满足 `PSNR ≥ +0.3 dB` 且 coverage 降幅 ≤ 1.0 pp 时采纳；否则 revert V5。
- **D2 质量过滤（XQ 对 X0）：** 以下全部满足才采纳：
  - M6 相对下降 ≥ 30%；
  - M7 相对下降 ≥ 50%；
  - M5 ≤ 20%；
  - M8 = 0；
  - refview coverage 降幅 ≤ 1.0 pp；
  - PSNR ≥ −0.2 dB。
- **D2 单一来源帧：** 候选为 XS4、XS8，各自必须满足：
  - M2 相对下降 ≥ 40%；
  - M3 相对下降 ≥ 50%；
  - coverage 降幅 ≤ 1.5 pp；
  - M8 = 0；
  - 导出秒数 ≤ 1.25 × X0。
  
  在满足条件的候选中选平均 PSNR 最高者。最终采纳还需 PSNR 在至少一个场景上 ≥ X0 + 0.3 dB，且在另一个场景上 ≥ X0 − 0.2 dB。
- **D2 组合：** XQS 相对 X0 仍须满足上面两组门槛；不满足时只采纳两者中更好的单项。
- **D3 GLB：** 先筛出满足以下条件的臂：
  - video3 GLB ≤ 176 MB（当前 117 MB 的 1.5 倍）；
  - 烘焙加压缩耗时 ≤ 1.5 × G0；
  - GLB refview coverage ≥ G0 − 2.0 pp。
  
  在其中选 PSNR 最高者，且它必须比 G0 高 ≥ 1.0 dB 才改默认值；否则保持 G0。
- 任何一项"差一点没过"都按未通过处理，如实写进报告，不得事后调整阈值。

### 6.5 已知局限（写进报告）

- refview 是源视角（样本内）评估：参考深度来自模型而非真值，也没有留出视角。新视角（侧视）只能靠截图人工判断。
- 白色包装区域（参考亮度 ≥ 245）不计入覆盖率和 PSNR。
- 只有两个场景。

## 7. 最终报告模板（`runtime/viewer-clarity-20261012/REPORT.md`）

```text
# Viewer 清晰度改进报告 2026-10-12
## 结论（3 行）
## 采纳 / 否决一览（D1–D3，每项给出规则、两场景数值、结论）
## 指标表（臂 × {M1–M8, coverage, PSNR, SSIM, 导出秒, ZIP/GLB 字节}，两场景）
## 对比图路径（overview/close/side/selected，GLB overview/close）
## 分支与提交（4 个检出：分支名、提交 sha 和标题；均未 push）
## 测试与构建（命令与结果）
## 未验证项与风险（GPU 实机帧率、真实客户任务、显存增加、局限 §6.5）
## 待 Rick 批准（§9 清单中需要执行的项）
```

## 8. 不在本次范围

- DA3 `process_res`（几何分辨率）：会改变匹配基线，需单独立项。
- 位姿精化（`src/da3_geometry_refinement.py` 接入导出）：历史评估没有一致收益。
- 逐帧曝光补偿：实测无效（残差 0.130→0.133）。
- 渲染器混合容差软化、连续性容差羽化：等单一来源帧落地后再评估。
- main 仓库 `modules/viewer_web` 中的实验性 `?surfel_source=local`：已被导出侧的单一来源帧取代。
- Harbor 推送、Viewer 版本号发布。

## 9. 需要 Rick 批准的关卡（执行前必须停下来问）

1. 推送任何分支（GitHub `origin` 或 GitLab `gitlab`）。
2. 构建并替换任何容器（8011、5175、5178），或推送 Harbor。
3. 写入 COS（包括用新镜像跑真实任务时上传 `viewer_bundle.zip`）。
4. 执行 WP-X4。
5. 偏离本文件的任何决策规则或阈值。
6. 使用 GPU2 以外的 GPU，或运行 DA3/SAM 推理。
7. 修改 `protect_mask` 或 `utils/pointcloud_filter.py`。

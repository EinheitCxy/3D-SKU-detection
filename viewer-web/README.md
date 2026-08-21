# TypeScript/Three.js 3D Viewer

这是一个只读的静态审查前端。Python 负责 DA3/SAM3、匹配、去重、正式 ground footprint 面积和 provenance；TypeScript 只负责严格加载 bundle、渲染点云/正式 footprint、选择 global ID 和展示证据，不重新计算面积。

## 两步使用

先在 `code/` 导出 bundle：

```bash
cd ../code
uv run python main.py --mode viewer-web \
  --dataset imdata/my_stack --save_root ./Output
```

默认输出为 `viewer-web/public/data/`，可用 `--viewer-web-output` 指定其它目录；点云参数为 `--viewer-web-voxel-size`（默认 `0.005`）和 `--viewer-web-max-points`（默认 `1500000`）。本地 Vite 的 `/data/` 只直接对应默认 `viewer-web/public/data/`；custom output 用于部署，或必须在启动前挂载/serve 到浏览器 URL `/data/`。导出阶段只读取已有正式产物，不启动 Node、浏览器、DA3/SAM3 或 GPU。

再在本目录运行 Vite：

```bash
cd ../viewer-web
npm run dev
```

部署或本地预览若不是在站点根目录运行，可通过 URL 指定数据源，例如：

```text
http://localhost:4173/?data=./data/
http://localhost:4173/?data=https://your-host/static/data/
```

若未指定 `data`，前端用标准 `new URL(locationHref)` 把 `./data/`（当前路径）和 `/data/`（站点根）解析为绝对 URL，并按顺序去重（站点根只加载一次 `/data/`）。若 URL 有 `?data=`，前端用标准 `URLSearchParams.get('data')` **取得第一个 explicit 值**，再用 `new URL(data, pageUrl)` 解析，因此相对 `./data/` 也可作为显式 root；前端只尝试这一根，加载失败绝不回退默认 roots。空白值、非 http(s) 协议以及结果的 bare/非空 search/hash 都会显示 `.load-error`：实现以 root `href` 与清空 search/hash 后的 `cleanRoot.href` 比较，因此标准 URL 保留的裸 `?/#` 也被拒绝。解析、候选加载、空 bundle 与 mount 同处一个错误边界，因此错误不会遗留 Loading。

默认 output 导出成功时，CLI 只打印、不执行等价的 CWD-independent 命令：`npm --prefix <repo>/viewer-web run dev`；实际输出会使用当前 checkout 的绝对路径。custom output 时不会打印默认 npm 命令，而会提示必须部署或挂载/serve 到 `/data/` 后再启动前端。

## UI 操作说明

- `Fit` / `Top` / `Iso`：切换相机预设（平滑过渡）
- 点大小滑块：实时调整 RGB 点云显示粗细（world-size 语义，单位米；圆形 splat，EDL/法线光照/SMAA/ACES/fog 后处理见下文渲染管线一节）
- Footprint opacity：按比例调节 footprints 显示强度
- 左侧搜索框 + `Prev/Next/Clear`：过滤、跳转与清除选择
- 0 点 ID（所有实例 `point_index_range` 全空，即有观测但无 3D 点）：列表项灰化并附 "no geometry" 徽标（派生自实例区间，非独立契约字段）；选中时 evidence 抽屉额外显示说明文字，场景中无高亮点
- 选中 global ID 时，把该 ID 全部实例 `point_index_range` 内的点云点染成亮洋红 `RGB(255,0,255)`；**其余所有点保持原始相机颜色不动**。取消或切换选择只恢复上一次 nonempty ranges、只 tint 当前 ranges；新变化与 BufferAttribute 已排队 update ranges 合并后局部上传，避免同一 RAF 的 A→B→C 丢失脏区间。空/0-geometry 选择在没有 prior tint 时不请求 GPU upload；若需要恢复 prior tint，则只上传其 prior ranges。重复选择不请求 upload；可变 `aColor` 使用 DynamicDrawUsage。选中（列表点击 / footprint 点击）**不会自动 focus 相机**，需手动点 `Focus` 按钮触发；相机 focus 优先用 footprint focus target，**footprint rejected/缺失时回退到选中点集包围盒**（逐轴 1/99 百分位 + 对角线 2% padding，bundle 坐标经 world_to_view 变换）。该 fallback Box3 按 global ID 缓存，返回 clone 后再做场景变换，避免重复排序与缓存污染。
- 证据面板：展示 footprint 面积总览与 per-ID 证据（Global ID / Per-ID area / Observations）及 per-instance 观测缩略图网格——每个实例显示从源图按 bbox 裁出的 JPEG 小图（caption 为 `image N · object M`），用于肉眼审计跨图去重是否把同一物理商品合并对了；`removed` 实例降透明度、灰度显示并划线标注 caption（审计过滤掉了什么），v1 仅展示、不提供点击交互
- 键盘：
  - `H`：隐藏/展示证据抽屉
  - `Esc`：清空当前选中 ID

## 坐标与朝向

bundle 中的 `positions`/`normals`/`objects`/`footprints` 几何保持 DA3 原生 OpenCV 世界坐标（x-right、y-down、z-forward，首相机锚定）。`manifest.json` 必需字段 `world_to_view`（16 个 float，**行主序** 4×4）是导出侧计算的组合变换 `T_center @ R_level @ M_flip`：CV->glTF 轴翻转、过滤前有效点集上 RANSAC 地平面的最短弧摆平、以及过滤后点云逐轴 median 居中（地平面在**过滤前**点集上拟合，因为导出过滤本身会剔除地面）。viewer 在场景根 `worldGroup`（`matrixAutoUpdate = false`，用 `Matrix4.set()` 行主序装载）上应用一次该矩阵，点云、footprints、grid/axes 与选中包围框都挂在其下。缺少 `world_to_view` 的旧 bundle 直接 fail closed 并提示重新导出，不做单位阵兜底。

## 渲染管线（src/edl.ts + scene.ts 自定义点材质）

点云不再是 `PointsMaterial`，而是自定义 `ShaderMaterial` + `EffectComposer` 后处理链：

- **圆形 splat**：fragment 中 `length(gl_PointCoord - 0.5) > 0.5` 即 discard（three 官方 points_waves 做法），消除方点的稀疏感。
- **world-size 点大小**：`gl_PointSize = clamp(size * resolution.y * projMatrix[1][1] / gl_Position.w, 1, 64)`（three PR#29474 公式）。滑杆语义从"无 FOV 因子的相对值"变为**真实世界米数**（点大小 0.015 ≈ 1.5cm splat），近大远小符合物理透视，64px 截断防近景爆炸；`resolution` uniform 在 resize 时同步为 drawing buffer 尺寸。
- **法线光照**：导出侧在 voxel/protected/max-points 选择完成后的**最终代表点集合**上用 Open3D `estimate_normals(KDTreeSearchParamHybrid(radius=自适应, knn=30))` + `orient_normals_towards_camera_location(平均相机中心)` 估计法线，再按实例稳定排序与 colors 同步置换，int8 量化（×127）写入 `normals.i8.bin`（~3B/点）。fragment 用 half-Lambert：`color *= 0.6 + 0.4 * (dot(n, lightDir) * 0.5 + 0.5)`，无 specular。
- **EDL（Eye-Dome Lighting）**：点云单独（camera layer 1）渲染进带 `DepthTexture` 的 HalfFloat RT，全屏 composite 时对每个像素取 4 个对角邻居的相对深度差做 scale-invariant 响应 `1 - exp(-4·(d-dn)/d)`，`shade = exp(-strength·Σocclusion)`（strength≈0.4, radius≈1.4px），给稀疏点云提供类 AO 的局部明暗（potree 同构做法）。
- **叠加层**：grid/axes/footprints/选中包围框（layer 0）在 EDL composite 之后渲染于其上（potree 同构），不受 EDL 影响。
- **末端润色**：SMAA（three r185 要求在 OutputPass 之前的 linear-srgb 空间执行）→ `OutputPass`（ACES Filmic tone mapping + sRGB 输出）；`scene.fog`（颜色同背景，near/far 由场景包围盒半径推导 1R/8R）提供深度渐隐。

## Bundle contract

数据采用不可变 generation 与原子指针：

```text
CURRENT
runs/<run_id>/
  manifest.json
  positions.f32.bin
  colors.u8.bin
  normals.i8.bin
  objects.json
  footprints.json
  thumbs/<globalId>_<instanceIndex>.jpg
```

loader 先读取并验证 `CURRENT`，接着**单独**读取并验证 immutable generation 的 `manifest.json`；只有 manifest 通过后才并行下载 objects、footprints 与二进制数组，因此损坏 manifest 不会浪费大数组下载。`world_to_view` 必须是 row-major 刚体 affine（末行 `[0,0,0,1]`、3×3 正交单位且 determinant `+1`）。每个 instance thumbnail 必须精确等于 `thumbs/<globalId>_<instanceIndex>.jpg`，所有非空 `point_index_range` 必须全局不重叠。

运行时还要求 `objects.json` 全部实例的升序去重 `image_id` 集合，与 manifest 中实际使用的 `source.export.sam3_mask_entries[].image_id` 集合严格相等；空集、缺失或多余 entry 均 fail closed。这把 bundle 的实例几何与冻结的 SAM3 cache provenance 绑定在同一 generation，旧 bundle 需重新导出。

正式 report 绑定生成时读取的 raw `global_mapping.json` 字节快照 SHA-256（`global_mapping_sha256`）。exporter 在构建 object index 前后校验 digest；mapping 不同或在导出期间变化会 fail closed，`accepted` generation 的 object ID 集与 footprint geometry ID 集不一致也会拒绝发布。缺少该 digest 的历史 formal generation 必须先重新运行 `--mode ground-stack-area`，再执行 viewer export；不提供 fallback。

`accepted` 的数值是正式 `da3_ground_footprint_union`。`rejected` 或 `value_m2: null` 表示 unavailable，界面显示 `—`，绝不显示为 `0 m²`。实验性 front-facing area 不在 v1 中，青色保留给未来该指标，正式 ground footprint 使用琥珀色。

## 开发验证

```bash
npm test -- --run
npm run build
```

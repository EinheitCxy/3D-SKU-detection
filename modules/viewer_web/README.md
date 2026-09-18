# Web Viewer

这是静态 TypeScript/Three.js 产品 Viewer。它不运行 DA3、SAM3 或 Python pipeline，只加载 Python 已发布的 minimal schema `3.0.0` bundle。

## 导出与启动

从仓库根执行：

```bash
# 先完成 DA3 reconstruction、完整 batch-all-refs matching 和 dedup
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3 \
  --classifier-device cuda:0

# 先将大型 Excel 转为构建期窄 CSV；该文件不由浏览器加载
uv run python scripts/convert_sku_maindata.py \
  --input sku-maindata.xlsx --output runtime/sku_masterdata.csv

# 直接导出 minimal Viewer bundle；不需要先运行 ground-stack-area
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2 \
  --viewer-web-sku-masterdata-csv runtime/sku_masterdata.csv

npm --prefix modules/viewer_web run dev
```

默认 bundle 写入 `modules/viewer_web/public/data/`，Vite 的 `/data/` 直接对应这里。使用 `--viewer-web-output <dir>` 时，部署层必须把该目录挂载或 serve 到 `/data/`；Vite 不会自动发现自定义路径。

已有 bundle 可用下列命令补充主数据，不重跑点云导出：

```bash
uv run python scripts/publish_viewer_masterdata.py \
  --viewer-output modules/viewer_web/public/data \
  --sku-masterdata-csv runtime/sku_masterdata.csv
```

脚本复制当前 run 后写入 `sku_masterdata.json`，仅在新 run 完整就绪后才更新 `CURRENT`。

从视频开始的一条命令入口：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2 \
  --classifier-device cuda:0 --serve
```

该脚本在 dedup 后直接 export，成功后才启动 Vite。独立的 `main.py --mode ground-stack-area` 仍可按需运行，但不是视频到 Viewer 的前置阶段。

## 按识别任务下载

启动本地 Viewer 后直接打开：

```text
http://localhost:5173/?recognition_task_id=54ea08a0-a238-4a72-880d-662ffc65250e
```

带 `recognition_task_id` 时，从 COS 的 `<任务 ID>/viewer_bundle.zip` 下载并在浏览器内加载，优先于 `?data=`；下载或解析失败会显示错误。默认使用普通点云，Surfel 数据包需追加 `&render=surfel`。不带任务 ID 时仍使用原有本地静态数据入口。

默认 COS 基址是 `https://snapshot-video-1305849923.cos.ap-shanghai.myqcloud.com/global-id-mapping`。如需更换，在本目录 `.env.local` 中设置 `VITE_COS_VIEWER_BASE_URL`，然后重启开发服务或重新构建。COS 需要允许浏览器来源的跨域 GET。

任务 ZIP 使用 Docker 导出的扁平 ZIP_STORED 格式；SKU 主数据由本目录 `masterdata.json` 独立提供（与 Docker Viewer 同一份紧凑主数据），只在加载任务时请求。更新主数据后需重新构建；缩略图和 Surfel 资源使用 Blob URL，在加载失败或页面退出时释放。

## Minimal schema 3.0.0

发布采用不可变 `CURRENT -> runs/<run_id>/`：

- `CURRENT` 只包含非空 `run_id`。
- `manifest.json` 只包含 schema `3.0.0` 所需的 `dataset_name`、非空 `backend`、`frame_count`、`display_bounds` 和 `world_to_view`。
- 固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`；`point_count` 由 positions 长度推导。
- `objects.json` 每个 global ID 只包含 `ordered_skus`、`point_ranges` 与必填 `observations`；每个 observation 只有 `image_id`、`object_id`、`removed` 与 `thumbnail`，未知扩展字段会被忽略。
- `sku_masterdata.json` 只包含本 run 的 SKU ID。每项提供 `manufacturer`、`brand`、`category`（可为 null）和布尔 `is_posm`；导出会拒绝缺少任一 bundle SKU 的窄 CSV。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在其他品类之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只接收排序后的 SKU ID/名称，不接收或显示 confidence。

导出仍从匹配产物传播实例点标签并执行统一点云过滤；matching 的 processed-mask cache 缺失时 export 会 fail closed，但 Viewer bundle 自身不包含该 cache。

## 产品交互

- 界面保留英文，采用浅灰侧栏、白色内容卡片和橙色选中态；筛选列表名称与数量分列，固定自然行高，右侧 SKU 名称、编号、数量分级呈现。键盘焦点清晰可见，并尊重减少动画的系统设置。
- 页面不显示 Dataset、Backend 或 Points 顶部摘要；页面采用白色场景背景且不显示地面网格。
- 默认 `Select by SKU`；它与 `Select by Global ID` 互斥，切换会清除上一选择。
- 选择栏统计卡片只显示 Total（Global ID 总数）。
- 选择模式提供 Manufacturer、Brand、Category、SKU 和 Global ID。前四类按对象的主 SKU 聚合计数并高亮全部匹配 Global ID；canvas 点选自动切换到 Global ID。
- 点击 Manufacturer、Brand、Category 或 SKU 的某一项后，右栏显示该维度和值、匹配 SKU/Global ID 总数，以及每个主 SKU 的名称、独立次级编号和对应 Global ID 数量；维度、已选值、汇总和 SKU 行使用不同层级的文字与计数标记。
- `View Controls` 默认折叠；展开后只提供 Fit、Top、Iso 和 Point size，默认点大小为 `0.004`，可按 `0.001` 调整，范围为 `0.004–0.070`；折叠时不会遮挡 canvas。
- Global ID 面板的搜索、匹配计数、列表与导航控件使用较小字号；左栏统计只显示 `Total`。Selected Object 为主 SKU 显示厂商、品牌、品类，POSM 固定显示 `N/A`。
- 右栏标题为 `Selected Object`，显示 Global ID、该对象由 observations 派生的 Observations / Active / Removed、按发布顺序排列的 SKU，以及紧凑的三列优先 observation thumbnail grid；每张图片都是导出器生成的精确 `128×128` 等比补深色背景 JPEG，卡片仍显示 image/object ID，removed 卡片灰化。
- Focus 始终使用对象的 `point_ranges`，不会复制点云 geometry。

Viewer 不包含 footprint、source provenance、hash/filter metadata 或 confidence 字段；也不加载这些旧 rich-contract 产物。

## 验证

```bash
npm test -- --run
npm run build
```

Manufacturer / Brand display normalization: `冲劲`, `沖劲`, `100冲劲`, and `100沖劲` are grouped as `其他` when loading master data, consistently across filters, counts, and object details. Source spreadsheet and bundle files remain unchanged.

SKU Breakdown rows are clickable: expand a SKU to show its Global IDs in numeric ascending order within the selected Manufacturer / Brand / Category. Click an ID to highlight it in orange and select its scene points while keeping the SKU breakdown visible. Clicking the expanded SKU again collapses it.

View controls: Fit frames the scene from the front at eye level, Top looks straight down along +Y, and Iso views from a diagonal side at 45° elevation.

画布右下角显示数据包实际 backend：DA3 显示为 Depth Anything 3，Pi3X 显示为 Pi3X。

## Pi3X 对比

已有 Pi3X reconstruction 和 matching 后，从仓库根执行：

```bash
uv run python scripts/export_pi3x_viewer.py --dataset imdata/floor_display6
npm --prefix modules/viewer_web run dev -- --host 0.0.0.0 --port 5175
```

DA3 打开 `/`，Pi3X 打开 `/?data=/data-pi3x/`。导出使用 Pi3X 原生点云和 transforms.json，独立生成 `Output/floor_display6/dedup_detections_pi3x/`，发布到 `public/data-pi3x/`，保留现有 DA3 bundle。两者使用相同过滤和采样设置；Pi3X 坐标使用模型原始尺度，不承诺 DA3 的米制尺度。

Pi3X 导出脚本读取 `personalcare_classification/CURRENT` 指向的完整分类检测，保留 SKU 元数据；不会使用缺少 classification 的原始检测。

测试位于仓库根目录 `test/viewer_web/`，仅本地保存、Git 忽略。`npm test -- --run` 通过 `vitest.config.ts` 从新位置加载。

## 深度约束 Surfel

Surfel 使用已有 DA3 缓存和原图，为每个保留的点附加局部表面切向量及来源相机；浏览器在表面圆盘上逐片元采样原图，并按深度约束融合重叠圆盘。详细数据流、数组形状、shader 和交互说明见 [Surfel 集成说明](../../docs/surfel_implementation.md)。

```bash
uv run --no-sync python scripts/export_surfel_viewer.py \
  --dataset imdata/floor_display6 \
  --cache Output/floor_display6/da3_cache/predictions.npz \
  --mapping Output/floor_display6/dedup_detections/global_mapping.json \
  --mask-cache-root /path/to/matching-sam3-cache/v2 \
  --sku-masterdata runtime/sku_masterdata.csv \
  --output modules/viewer_web/public/data-surfel

npm --prefix modules/viewer_web run dev
```

`--mask-cache-root` 必须替换为与当前 DA3 processed grid 对齐的现有 v2 mask 缓存目录。导出在 CPU 上完成，不重跑 DA3/SAM3，也不修改匹配和计数结果。独立脚本默认使用 0.005 体素尺寸、最多 50 万背景点；SAM 商品点先受过滤保护，再按非空 `global_id` 均分最多 200 万商品预算，小组不足配额时回收未用配额。分辨率对照固定这两项预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

| URL | 数据与渲染方式 |
|---|---|
| `/?data=/data/&render=points` | 普通点云，使用逐点 RGB |
| `/?data=/data-surfel/&render=surfel` | 独立 Surfel 数据，逐片元原图纹理 |

普通入口默认 `surfel`；同时导出并读取 `colors.u8.bin`，供原始点云切换使用。`render=surfel-rgb` 无效。Sidecar 仅支持 v2：U/V 和来源深度为 little-endian Float16，帧编号为 Uint8，支持 1..256 帧；GPU 保持半精度属性和深度纹理。旧 Surfel 数据需重新导出，静态部署需同时发布新前端和新数据。

`--texture-edge` 默认为 1920，仅允许 256..1920；短边上限为 `min(texture-edge, 1080)`。横图最多 1920×1080、竖图最多 1080×1920，等比缩小、不放大小图，JPEG 质量 95。只调整导出纹理，不改变原图、深度网格或商品缩略图。混合横竖图的纹理阵列按最大宽高补齐，CPU/GPU 各持有 RGBA 数据，显存不等于 JPEG 文件大小。

点大小默认值为滑块最大值 0.07；Surfel 初始覆盖半径为 2 个源网格像素（现有上限），不改变中心点几何。商品选择通过独立标记显示紫色；取消选择恢复纹理颜色。GPU 拾取与颜色渲染使用相同的圆盘、来源图有效域和来源深度检查，再通过原有 `point_ranges` 找到全局id。Focus 和批量高亮保留原有接口。

面片防拉伸：渲染时按来源相机的单像素三维步长归一化 U/V，压缩异常主轴，将最大伸长限制为 2、主轴长宽比限制为 4；保留中心位置和表面所在平面，不把 U/V 强制变成等长正交向量。点大小仍乘在约束后的面片上。四邻深度与局部切向量预测相差超过 `1 mm + 半个原生像素步长` 时，该侧覆盖截在来源像素的半像素边界；颜色、前表面深度和点击均使用此边界。原图投影取色方式不变。

该约束由前端执行，已有 Surfel v2 数据包刷新页面即可使用，无需重跑 DA3 或重新导出。它减少异常面片拉丝，不能补齐缺失几何；陡斜面及遮挡处可能显露更多孔洞。

Surfel 按需重绘：相机变化、阻尼、Focus/视角动画、选择/隐藏、point size 和窗口大小变化会请求下一帧，静止时停止绘制。普通 points 保持连续绘制。该机制减少空闲绘制，不代表拖动单帧更快。

加载界面显示实际接收 MB、完成文件数和待完成文件名，区分下载等待与解析；不会添加超时、重试或自动降级。

需要 WebGL2 和 `EXT_color_buffer_float`，能力不足直接报错。该渲染不叠加普通点云的 Lambert/EDL/fog，也没有训练或完整 EWA 滤波；原图改善表面外观，不能补出新几何。薄物体、深度断层、跨视角几何误差和曝光差异仍可能产生孔洞、接缝或重影。尚无本次审查的硬件 GPU 帧率测量。

### 视频1同轨迹对比实验

本地已生成的 `/comparison/index.html` 页面可同步回放三组70视角，查看原图参考、放大和点击商品编号。此页面使用预渲染图像；DA3高斯的编号是从原表面派生的实验映射，尚未接入生产三维Viewer。

`compare-render.html` 与 `comparison-capture.mjs` 是固定相机捕获入口，复用当前surfel渲染器；`mode=consistent`仅应用独立可见性表。实验步骤和指标定义见 [对比说明](../../perf/surfel_compare/README.md)。原有主页面默认行为不变。

局域网直接访问：在服务器上执行 `npm run dev -- --host 192.168.2.6 --port 5173 --strictPort`，浏览器打开 `http://192.168.2.6:5173/comparison/index.html`。此方式监听服务器局域网地址，无需 VS Code 端口转发；浏览器所在电脑需能访问该服务器。

### 原始点云与 Surfel 切换

默认显示 Surfel，视图控制旁的“切换到原始点云 / 切换到 Surfel”按钮即时切换，不重新下载数据，保留相机、筛选和高亮。`render=points` 可选择初始点云模式。原始点云使用导出时保留的同一组点及逐点 RGB，不是未过滤的全量重建点。两种渲染共享点索引与商品归属。

新导出及 ZIP 必须同时包含 `colors.u8.bin` 和 Surfel v2 数据；之前省略颜色的数据包需要重新导出。颜色额外占每点 3 字节，切换功能也会增加点云几何与渲染目标的显存占用。修改源码不会自动更新已运行的 Docker 服务，服务需要重新构建部署。

Manufacturer 为“零眸智能”或 `lingmouzhineng` 时，统一归入“其他”；厂家筛选、计数和商品详情使用同一规范化结果。

### Pi3X 像素中心修正

Pi3X 内参按 `x+0.5/y+0.5` 拟合。Surfel 渲染先将投影坐标减去半像素，再查询整数索引的深度网格、计算边缘裁剪并映射纹理，避免取错相邻深度而隐藏本应可见的面片。DA3 的投影计算保持不变。依据 manifest 的 `backend=Pi3X` 应用，已有数据包无需重新推理或导出。该修正不补偿采样后的面片覆盖率；相同点数不代表不同源分辨率下覆盖面积相同。

### 默认点大小

Surfel 点大小默认设为滑块最大值 `0.07`，首帧即应用，对应覆盖半径上限 `2`；原始点云默认 `0.004`。两种模式在当前页面内独立保存大小，切换时恢复各自滑块值，刷新后恢复默认值。保留现有 U/V 防拉伸、邻域裁剪与深度混合限制；扩大覆盖可遮住部分采样孔洞，不校正位姿或合并货架双层。刷新页面即可生效，Docker 静态部署需重新构建发布前端。

### 商品点保留与点击遮挡

后端导出先通过 SAM mask 保护有效商品点，避免几何过滤误删；随后按非空 `global_id` 均分最多 200 万商品点，同一商品的多帧观测合并抽样，小组不足配额时回收未用配额。背景先过滤、再体素降采样，独立保留最多 50 万点；商品点不占用背景预算。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。已有 ZIP 需后端重新导出才会应用新的点数预算。

原始点云模式点击命中无商品归属的可见前景点时停止，不穿透选择后方商品；已隐藏商品仍跳过。Surfel 的面片退化、遮挡与深度裁剪仍影响画面，有效原始点不保证在任意视角产生像素。

背景点上限固定为最多 50 万点，不接受命令行或函数参数覆盖；商品预算固定为最多 200 万点，按 `global_id` 均分并回收小组未用配额，SAM 有效点先受过滤保护。Viewer 按导出点和 `point_ranges` 渲染，不在浏览器二次删点。Docker 后端构建从 `CORE_REPO_ROOT/src` 复制同一核心实现，旧容器和旧 COS ZIP 需要更新后端并重新导出才生效。

## 商品点预算更新（2026-09-10）

当前固定保留 **最多 200 万商品点 + 最多 50 万背景点**，取代此前商品 mask 全量输出规则。SAM 内有效点先受几何过滤保护；之后按非空 `global_id` 均分商品预算，将同一商品的多帧观测合并抽样。小组实际点数不足配额时回收未用配额并分配给其他组；组内使用固定随机种子、不放回均匀抽样。总商品点不足 200 万时不抽样。空 SAM 掩码不能产生商品点。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

位置、颜色、Surfel U/V、源帧和点击范围使用相同点索引。此上限由导出端控制，Viewer 不再独立截断点数组；旧包需重新导出才生效。

## 局部来源优选实验

在任务或静态数据 URL 后增加 `&surfel_source=local`，例如：

```text
http://192.168.2.6:5173/?recognition_task_id=XUa2N8dzy9AFAjKOa0wG&surfel_source=local
```

Surfel 模式下的“局部来源优选（实验）”按钮可即时开关并同步 URL；不带参数默认关闭。原始点云模式隐藏该按钮，两个模式的点大小仍独立保存。

实验在屏幕每个像素的前表面候选中选择源帧，仅混合该来源的 Surfel；不同位置可以选择不同帧。候选继续经过源深度、边缘和前表面深度检查，点击也使用同一来源筛选。它复用现有包，不改变商品/背景预算、global_id、深度或位姿，也不重新执行匹配。明显错开的前后层不会因此自动对齐。

该实验增加一次全点云来源选择渲染和屏幕大小的中间缓冲，实际帧耗时需要在目标设备检查。当前版本不保证转动时无来源切换接缝或闪烁，不能视为几何修复或匹配精度提升。

## Surfel 来源帧容量

导出和加载支持 1–256 帧，保持 Uint8 来源编号与现有 v2 文件格式。相机矩阵改用浮点纹理读取，避免超过 32 帧后碰到着色器 uniform 数量限制；矩阵列以 RGBA texel 补位存储。纹理数组超过设备层数时明确报错，不截帧。帧数上限不是内存保证，原始高分辨率纹理的显存用量仍随帧数增长。超过 256 帧需调整输入或另行设计分批格式。

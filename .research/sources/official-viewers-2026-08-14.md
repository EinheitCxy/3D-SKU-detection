# 官方 3D Viewer 资料：2026-08-14

## Viser

- Scene API：<https://viser.studio/main/api/core/scene_api/>
- Point cloud example：<https://viser.studio/main/examples/scene/point_clouds/>
- Core concepts：<https://viser.studio/main/examples/getting_started/core_concepts/>
- GUI callbacks：<https://viser.studio/main/examples/03_gui_callbacks/>
- Scene pointer example：<https://viser.studio/main/examples/interaction/scene_pointer/>
- 3D GUI in scene：<https://viser.studio/main/examples/interaction/gui_in_scene/>
- Pointer events API：<https://viser.studio/main/api/advanced/events/>

### 可迁移结论

1. `add_point_cloud()` 支持 `(N, 3)` 点和逐点颜色，场景节点名可以组织出层级路径；这适合把当前点云拆成 `/objects/<global_id>/points`、`/objects/<global_id>/footprint` 等对象层级。
2. `add_label()` 能将对象信息挂到 3D 位置，并控制屏幕/场景字号、深度测试和 anchor；适合显示短的 `ID / area / status` 标签。
3. 新的 `on_click()` / `on_rect_select()` 能把点击或框选交给 Python 回调；当前 runtime 使用的是较旧的 `on_pointer_event()` 路径，升级时要做版本 API 核验。
4. GUI API 和 scene API 可以通过 callback 更新可见性、点大小、颜色和对象状态；这与现有 ViserViewer 的 slider/dropdown 结构兼容。
5. 3D GUI container 可以把上下文控件放到场景对象附近，但首阶段应谨慎使用，以免把信息面板和相机交互耦合。

## Nerfstudio Viewer

- Viewer overview：<https://docs.nerf.studio/developer_guides/viewer/index.html>
- Python viewer control：<https://docs.nerf.studio/developer_guides/viewer/viewer_control.html>
- Viewer API elements：<https://docs.nerf.studio/reference/api/viewer.html>
- Viewer quickstart：<https://docs.nerf.studio/quickstart/viewer_quickstart.html>

### 可迁移结论

1. Nerfstudio 使用 Viser + Three.js/React viewer，并把自定义 GUI、ViewerControl、相机设置和 pointer callback 分开。
2. `set_pose(position, look_at, instant=False)` 的“平滑聚焦”适合作为点击 SKU 后的相机行为；不应在普通点云点击中阻塞相机控制。
3. `ViewerClick` 和 `ViewerRectSelect` 将点击表达成世界坐标 ray 或归一化屏幕矩形，适合后续做对象级选择。
4. pointer callback 具有生命周期和单槽限制；当前项目应集中管理注册/注销，避免 Pick Mode 与普通相机控制互相覆盖。
5. Viewer controls 适合把“数据选择”和“相机动作”拆成两个回调层，而不是让一个大型 callback 同时改所有状态。

## Potree

- Project and examples：<https://github.com/potree/potree>
- Official demo：<https://potree.org/demo/potree_2014.12.30/examples/viewer.html>

### 可迁移结论

1. Potree 将大点云 viewer 的核心操作组织为 point budget、adaptive rendering、clipping volume、measurement、profile 和 annotation 等工具。
2. 这些能力说明“点云数量控制”和“审查工具”应该是显式的 viewer 状态，而不是隐藏在一次性随机下采样里。
3. Potree 的完整格式转换和前端栈对当前 Python+Viser 项目偏重，首阶段更适合作为交互设计参考，而不是直接替换。

## Three.js

- GLTFLoader：<https://threejs.org/docs/pages/GLTFLoader.html>
- OrbitControls：<https://threejs.org/docs/pages/OrbitControls.html>
- OrbitControls example：<https://threejs.org/examples/misc_controls_orbit.html>
- CSS3DRenderer：<https://threejs.org/docs/pages/CSS3DRenderer.html>

### 可迁移结论

1. GLB 加载和 OrbitControls 能构成更强的浏览器端视觉基础，适合未来需要自定义品牌化 UI、时间线和多面板布局时使用。
2. CSS3DRenderer 可把 DOM 信息叠到 3D 场景，但它有不支持材质/几何体和显示缩放限制；若引入，必须处理 overlay 的 pointer-events 与相机事件冲突。
3. 对当前项目而言，Three.js 路线的主要成本不是渲染本身，而是把 Python 产物变成稳定的前端 schema、开发静态资源/服务、处理选择状态和回归测试。

## 访问与证据边界

- 访问日期：2026-08-14（Asia/Shanghai）。
- 资料类型：项目官方文档、官方示例和官方源码链接。
- 未下载模型、数据或示例资产；仅使用网页中可复核的 API/交互描述。

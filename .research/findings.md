# 3D 展示优化研究发现

## 1. 当前 checkout 事实

- 当前 `main` 指向 `39dc91f`，其上游为 `origin/area_prediction`。
- 原 `main` 已保存在 `main-backup-20260814`，指向 `33c5498`。
- 当前工作区有未跟踪的 `TODO.md`、`code/modules/facing_area_stage.py`、`code/utils/facing_area.py`、`code/utils/track_utils.py`、`controllable-autoresearch-agent/`、`docs/ground_stack_footprint_algorithm.md`、`figures/` 和 `knowledge/`；本研究不处理这些文件。

## 2. 项目展示目标和语义边界

- 项目核心入口是 `code/main.py`，viewer 由 `code/modules/viewer_runner.py` 编排。
- 现有 viewer 位于 `code/viewer/`，已经有点云、global ID、相机、置信度、帧过滤、拾取和 mesh 控件。
- `facing_area_m2` / `facing_share` 表示商品正对货架面的投影面积语义。
- `da3_ground_footprint_union` 表示支撑平面上的 carton OBB polygon union，不是正面面积、接触面积或包装表面积。
- viewer 应读取正式结果并呈现 provenance/status；不应重新实现面积算法。

## 3. 初步本地缺口

- 现有 viewer 的选择信息主要是 global ID、图片和 object ID，面积与证据状态还不是一等信息。
- 现有点云展示是数据点级别，缺少以 `/objects/<global_id>` 为边界的对象级场景组织。
- 需要把 `global_mapping.json`、`facing_area_report.json`、`measurement_report.json`、`footprints.geojson` 和点云缓存做显式适配，而不是在 runtime 中散落读取。
- 需要对“面积不存在 / 正式阶段 rejected / shadow evidence unavailable”做明确 UI 状态，而不是显示 `0` 或空白。

## 4. 外部研究待补充

第一轮官方资料已记录在 [official-viewers-2026-08-14.md](sources/official-viewers-2026-08-14.md)。当前可迁移的共同模式是：

- 场景对象层级与 UI 信息层分离；
- 对象选择后执行相机聚焦，但保留普通相机控制；
- 标签、面积、证据状态和源图入口作为对象上下文，而非只显示点云；
- 点云预算、采样、裁剪和可见性是显式控制；
- pointer callback 必须有生命周期管理，避免覆盖相机交互。

sol agents 将继续补充本地代码证据和项目匹配评估。

### sol architecture agent 初步结论

architecture agent 的完整收据见 [agent-019fffa2-6cf5-7872-a993-02e94db94fbe.md](receipts/agent-019fffa2-6cf5-7872-a993-02e94db94fbe.md)。它确认短期应沿用 Viser，并将 viewer 逐步拆为 scene registry、selection、camera、annotation、measurement、GUI state；还指出当前 Viser 0.2.11 已能支持 rect-select，真正的风险是 callback 生命周期、全量点云重传和证据状态与视觉投影混淆。

### sol local explorer 结论

本地审计收据见 [agent-019fffa2-6c38-7a43-84ca-b12ba2b5aff1.md](receipts/agent-019fffa2-6c38-7a43-84ca-b12ba2b5aff1.md)。最重要的不是立即加 UI，而是先建立 `ViewerInputResolver`、backend-neutral `SceneBundle` 和只读 `MeasurementOverlayAdapter`：当前 viewer 对 VGGT path/field 的假设与 DA3 schema、CLI 路径和正式 footprint generation 不一致。未跟踪 `facing_area` 草案还存在 transforms 未传入 bbox extractor 的错位风险，必须标为非正式/待审计输入。

### sol UX agent 结论

产品研究收据见 [agent-019fffa2-6b8b-7ec0-820e-a1d5bb144352.md](receipts/agent-019fffa2-6b8b-7ec0-820e-a1d5bb144352.md)。A（3D 审查驾驶舱）适合现场对象定位，B（2D/3D 联动证据检查器）最适合审查数字来源，C（双指标看板 + 按需 3D）适合管理汇报。综合本项目的“结果可审查”要求，首阶段应采用 A+B 的轻量组合：3D 对象定位作为主视图，源图/bbox/面积状态作为选中后的证据抽屉；C 的总览卡可以作为低成本首屏摘要，但不能替代证据链。

固定视觉语义：青色表示 front-facing projected area，琥珀色表示 ground footprint；两者不可共用颜色、排行榜或求和。formal status、shadow evidence status 和 provenance 必须分开显示。

## 5. 方案评价原则

1. 先保证数据语义正确，再追求视觉效果。
2. 交互必须能回答“这是哪个商品、来自哪些图、面积是多少、结果是否可信、如何回到证据”。
3. 对大点云使用可控采样、分层加载或对象可见性切换，不能以无限制全量重绘换取视觉效果。
4. 相机操作、对象选择、标签和过滤应互不干扰。

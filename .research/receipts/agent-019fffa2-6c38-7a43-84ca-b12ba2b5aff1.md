# sol agent 收据：本地 viewer 数据流审计

- agent id：`019fffa2-6c38-7a43-84ca-b12ba2b5aff1`
- model：`gpt-5.6-sol`
- role：`explorer`
- scope：只读审计 `code/main.py`、`code/viewer/`、viewer runner、area/footprint 产物和测试；禁止修改文件、GPU、DA3/SAM3、benchmark。
- result：已完成；agent 明确报告未修改任何文件。

## 关键源码事实

1. viewer 主链为 `main.py -> run_viewer() -> build_viewer_cache() -> load/downsample -> point-level global ID assignment -> pcd_gid.npz/global_object_index.json/metadata -> ViserViewer.start()`。
2. CLI 和交互入口的 cache/detection 路径假设不一致；CLI 主要搜索 `vggt_cache`，而当前重建入口会使用 backend cache。
3. `points_source=predictions` 和在线 ID 分配硬编码 `vggt_cache/predictions.npz`，DA3 不能可靠走默认链路。
4. DA3 schema-v2 使用 `world_points_conf`，viewer 在线分配强制读取 `conf`；DA3 当前也不生成 GLB，存在明确字段和默认入口缺口。
5. runtime 已有 confidence、global ID、frame、unknown ID、sampling、pick、mesh 和 camera 控件，但只消费 viewer 自有 cache，相机仍识别 VGGT `extrinsic/intrinsic/images`。
6. 正式 footprint 是 immutable generation + `CURRENT` + manifest/report/GeoJSON 的 fail-closed 只读阶段；viewer 当前完全不解析这些产物。
7. footprint report 保存 support plane `point/normal/u_axis/v_axis`，因此局部 `(u,v)` GeoJSON 可以恢复到世界坐标，但当前没有 overlay adapter。
8. 未跟踪 `facing_area` 草案写出 `facing_area_report.json` 并回写 `facing_area_m2/facing_share`，但没有 immutable publication、status/provenance/geometry，也没有注册 `area` CLI mode。
9. `facing_area_stage.py` 构造了 transforms，却没有把 transforms 传入 `extract_3d_from_bboxes()`，可能把原图 bbox 错当 processed grid，存在直接错位取点风险。
10. `facing_area` 用 bbox 内点的投影凸包，没有 SAM3 mask、支撑面或 outlier gate，不应与正式 footprint accepted 结果共享可信测量语义。

## 推荐模块边界

- `ViewerInputResolver`：统一 backend/path/cache/detection 解析。
- backend-neutral `SceneBundle`：统一 points/colors/confidence/frame/image/camera。
- `MeasurementOverlayAdapter`：只读解析 footprint `CURRENT` 和 facing scalar，保持 status/provenance。
- `ViserViewer`：只渲染 bundle/overlay，不再自行猜路径或 schema。

footprint artifact reader 应从 `da3_footprint_stage.py` 的私有 `_artifact_paths_from_current()` 提取轻量公共 reader，避免 viewer 导入 Torch、SAM3、Matplotlib 等 stage 依赖。

## 首批测试点

- CLI/交互路径矩阵；Pi3/DA3/VGGT NPZ contract；cache 失效；空提取 `gid=-1`；非连续 image IDs；accepted/rejected overlay；manifest 篡改；局部平面到世界坐标 round-trip。

## 关键文件入口

- `code/modules/viewer_runner.py:18-117`
- `code/viewer/cache.py:95-165`
- `code/viewer/id_assign.py:40-277`
- `code/viewer/runtime.py:75-360`
- `code/modules/da3_footprint_stage.py:107-304,708-940`
- `code/modules/facing_area_stage.py:132-232`（未跟踪草案，非正式 HEAD 代码）

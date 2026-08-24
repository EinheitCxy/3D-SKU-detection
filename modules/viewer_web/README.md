# Web Viewer

这是静态 TypeScript/Three.js Viewer。它不运行 DA3、SAM3 或 Python pipeline；它只严格加载 Python 已发布的 bundle。

## 导出与启动

从仓库根执行：

```bash
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2
npm --prefix modules/viewer_web run dev
```

默认 bundle 写入 `modules/viewer_web/public/data/`，Vite 的 `/data/` 直接对应这里。使用 `--viewer-web-output <dir>` 时，部署层必须把该目录挂载或 serve 到 `/data/`；Vite 不会自动发现自定义路径。

导出需要同一 dataset 已有：

- `Output/<dataset>/da3_cache/predictions.npz`
- `Output/<dataset>/dedup_detections/global_mapping.json`
- `Output/<dataset>/ground_stack_footprint/CURRENT`
- `<dataset>/images/`

bundle 使用不可变 `CURRENT -> runs/<run_id>/`。loader 对 manifest、provenance、二进制数组长度、坐标变换、缩略图与 footprint status 严格校验；历史或不完整 bundle 会明确失败而不会推测补全。

## 点云过滤

Viewer 不使用 SAM3 protection mask。SAM3 cache 仍可作为 footprint 与实例审计输入，但它不会让某些点绕过离群、地面或天空过滤；同一过滤规则适用于所有展示点。

## 验证

```bash
npm test -- --run
npm run build
```

# TypeScript/Three.js 3D Viewer

这是一个只读的静态审查前端。Python 负责 DA3/SAM3、匹配、去重、正式 ground footprint 面积和 provenance；TypeScript 只负责严格加载 bundle、渲染点云/正式 footprint、选择 global ID 和展示证据，不重新计算面积。

## 两步使用

先在 `code/` 导出 bundle：

```bash
cd ../code
uv run python main.py --mode viewer-web \
  --dataset imdata/my_stack --save_root ./Output
```

默认输出为 `viewer-web/public/data/`，可用 `--viewer-web-output` 指定其它目录；点云参数为 `--viewer-web-voxel-size`（默认 `0.01`）和 `--viewer-web-max-points`（默认 `500000`）。本地 Vite 的 `/data/` 只直接对应默认 `viewer-web/public/data/`；custom output 用于部署，或必须在启动前挂载/serve 到浏览器 URL `/data/`。导出阶段只读取已有正式产物，不启动 Node、浏览器、DA3/SAM3 或 GPU。

再在本目录运行 Vite：

```bash
cd ../viewer-web
npm run dev
```

默认 output 导出成功时，CLI 只打印、不执行等价的 CWD-independent 命令：`npm --prefix <repo>/viewer-web run dev`；实际输出会使用当前 checkout 的绝对路径。custom output 时不会打印默认 npm 命令，而会提示必须部署或挂载/serve 到 `/data/` 后再启动前端。

## Bundle contract

数据采用不可变 generation 与原子指针：

```text
CURRENT
runs/<run_id>/
  manifest.json
  positions.f32.bin
  colors.u8.bin
  confidences.f32.bin
  frame_ids.i32.bin
  objects.json
  footprints.json
```

loader 先读取 `CURRENT`，再读取其绑定的 `runs/<run_id>/`，严格检查 schema、DA3/正式 footprint provenance、TypedArray dtype/components/byte length 以及 accepted/rejected 关系；不满足 contract 时直接 fail closed。旧 generation 不会被覆盖或删除。

正式 report 绑定生成时读取的 raw `global_mapping.json` 字节快照 SHA-256（`global_mapping_sha256`）。exporter 在构建 object index 前后校验 digest；mapping 不同或在导出期间变化会 fail closed，`accepted` generation 的 object ID 集与 footprint geometry ID 集不一致也会拒绝发布。缺少该 digest 的历史 formal generation 必须先重新运行 `--mode ground-stack-area`，再执行 viewer export；不提供 fallback。

`accepted` 的数值是正式 `da3_ground_footprint_union`。`rejected` 或 `value_m2: null` 表示 unavailable，界面显示 `—`，绝不显示为 `0 m²`。实验性 front-facing area 不在 v1 中，青色保留给未来该指标，正式 ground footprint 使用琥珀色。

## 开发验证

```bash
npm test -- --run
npm run build
```

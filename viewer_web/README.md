# viewer_bundle.zip 接口

## 接收

页面提供本地文件选择框，用户选择 Docker BSON 响应保存的 `viewer_bundle.zip`。ZIP 仅在浏览器
内存中读取，不会上传、解压到磁盘或写回服务端。

## 处理

加载器只接受 Docker 当前生成的平铺、非加密 `ZIP_STORED` 归档，并要求包含：

- `manifest.json`
- `positions.f32.bin`
- `colors.u8.bin`
- `normals.i8.bin`
- `objects.json`
- 可选的 `thumbs/*.jpg`

它会拒绝缺失固定文件、重复成员、嵌套或未知路径、压缩或加密 ZIP、无效 schema 3.0.0、错误的
二进制 shape，以及 `objects.json` 指向但 ZIP 中不存在的缩略图。

## 输出

通过校验后，浏览器将二进制点云转为 typed arrays、将缩略图转为本地 object URL，并把 manifest 与
object index 交给 Viewer。页面输出交互式 3D 点云、SKU / Global ID 筛选、高亮、对象详情和对应
缩略图；加载失败时显示具体错误，不渲染不完整数据。
## 本地启动

本分支只包含 `viewer_web/`，不包含 Mapping Docker API 或推理流水线；因此需要先从原有 Mapping
Docker 的 `/api` 响应取得 `viewer_bundle.zip`，再启动本地页面选择该文件。

```bash
git clone -b visualization git@gitlab.lingmouai.com:omni/dedup-3d.git
cd dedup-3d/viewer_web

# 已有 npm cache 时使用；依赖严格按 package-lock.json 安装
npm ci --offline --ignore-scripts
npm run dev -- --host 0.0.0.0 --port 5173
```

浏览器打开 `http://127.0.0.1:5173`，然后选择 `viewer_bundle.zip`。如果本机没有 npm cache，将安装命令
改为：

```bash
npm ci --ignore-scripts
```

当前没有 `npm run build-dev` 脚本。生产构建与本地预览分别使用：

```bash
npm run build
npm run preview -- --host 0.0.0.0 --port 4173
```

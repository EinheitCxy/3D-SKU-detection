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

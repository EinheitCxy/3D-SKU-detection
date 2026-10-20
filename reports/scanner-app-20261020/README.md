# SKU Scanner 第一版交付

源码：[sources/stray-scanner](../../sources/stray-scanner/README.md)。
上游基准 ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb，独立本地分支 sku-workflow。
MIT 文件与版权声明保留。本报告记录第一版代码与验证边界；源码随主项目 main 发布。

从主项目 checkout 的 sources/stray-scanner 可直接获取完整手机 App 源码。
开发机另有临时源码 ZIP，不纳入版本库；服务端转换器位于
[prepare_bundle.py](../../scripts/3d/scanner/prepare_bundle.py)。

## 实现

- 保持 ARKit 连续 tracking，按真实 ARFrame 时间戳每个 elapsed-second bucket 选首个完整、tracking normal 的候选；保留原始回调序号，缺失秒不补造。
- 同帧原尺寸 RGB PNG、原生 LiDAR/confidence PNG、逐帧 K 和 OpenCV c2w；SDK列存储矩阵显式转 row-major，服务端只求逆。
- 最多两个待写入 ARFrame；整组临时目录 rename 提交；排空队列后发布 ready manifest；错误不能覆盖成成功或导出。
- 原图详情预览、本地 ZIP 分享、切后台/返回时结束采集；清除上游签名 Team、项目 Bundle ID 和 SKU Scanner 名称。
- 独立 prepare_bundle.py 转换为既有 prepared 输入，不再解码 MP4或采样。后续重建使用当前1008/FP16/batch2 默认，DA3 + K/E 也可消费相同输入。

## 验证与边界

| 检查 | 结果 |
|---|---|
| 服务端7项CPU单元检查 | PASS；像素、毫米转米、K/E、缺失秒、错误包和已有输出保护 |
| 原生 C++ PNG 编解码 | PASS；行 padding、毫米、无效值、固定灰度位深、原始confidence |
| C++ PNG → 服务端 OpenCV | PASS；uint16/uint8单通道与数值精确一致 |
| Python CLI / Ruff | PASS |
| 第三方 Swift syntax parser | 21/22文件无报告；唯一剩余报告为合法空tuple `()`，可在单行 `let unit = ()` 复现的 grammar 限制 |
| Xcode SDK编译、6项XCTest、真机采集/ZIP | 未运行：当前Linux没有Swift/Apple SDK或LiDAR设备 |

CPU检查使用 uv 调用现有系统 Python 3.12.5 / NumPy 1.26.2 / OpenCV 4.11.0；
服务端项目仍使用根 pyproject.toml 的 Python 3.11 环境。没有新增项目依赖。
Swift parser 安装在 /tmp 的隔离 uv cache；语法扫描不证明 API 类型正确或设备几何对齐。

发布前另在无网络、无GPU的既有 Python 3.11 容器中验证了最终发布代码：
Scanner11项、共享场景61项、bundle7项，合计79项CPU检查全部通过；
从发布目录重新编译的C++传感器PNG经OpenCV读回，单通道、位深和数值一致。

最终源码需要在 Mac 上按 App README 构建，并检查原图/深度边缘、手机转向、已知距离/相机位移、
停止后的完整帧数和后台中断状态；本次没有声称已经完成这些设备验收。
没有新增云端服务或自动上传接口。手机保留原始采集分辨率，模型缩放仅在现有服务端执行。

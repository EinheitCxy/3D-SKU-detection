# video1 / video2 清晰度复查与实验（2026-10-12）

目标是改善完整场景的白色碎片、孔洞和标签重影。前一轮最长直径上限保留；本轮不改匹配结果、不重跑 DA3/SAM，不将原 video3/fd6 D1 的未判定结果改写为通过。

## 依据与优先级

1. 英文 renderer 的 accumulation alpha 是 Gaussian 权重和，输出时已用 `rgb / alpha` 归一化，却又用 `smoothstep(0.03, 0.3, alpha)` 与白背景混合。单面片圆周的权重约为 `exp(-2)=0.135`，会被大量混入白色。这与不透明表面的颜色归一化不同；先独立测试移除这次混白。
2. 英文 renderer 对近前表面的所有来源帧累加。本地 renderer 已有按当前视角和来源分辨率选择来源帧的 pass，可作为后续候选；不能直接复制整份 renderer，因为两者还有颜色空间、AA 和深度容差等差异。
3. 现有导出 `single_owner=True, owner_patch_cells=8` 单独导出 video2，检查减少重复观测与失去覆盖之间的取舍；不与质量过滤、spill trim 或背景权重调整组合。

联网查证了以下原始资料：

- [Waechter 等：Let There Be Color!（ECCV 2014）](https://download.hrz.tu-darmstadt.de/pub/FB20/GCC/paper/Waechter-2014-LTB.pdf)：几何或位姿轻微不准时，多视角混合会产生纹理重影；单一来源选择需要同时处理可见性和接缝。它支持检验来源选择，不证明本项目的候选有效。
- [Zwicker 等：Surface Splatting（SIGGRAPH 2001）](https://dash.harvard.edu/entities/publication/73120378-88f8-6bd4-e053-0100007fdf3b)：论文区分重建滤波、遮挡处理和抗锯齿。结合本项目 shader 检查，本轮检验的是不透明表面归一化后的 Gaussian 权重是否应再次用于混白，并非复现论文完整算法。
- [Open3D：Colored point cloud registration](https://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html)：联合颜色和几何约束可改善沿表面切向的对齐。改变配准比本轮 renderer 修复涉及更多数据，因此先保留为后续方向。

豆包搜索 skill 脚本本次因 `AGENT_PLAN_URL` 未定义而退出；联网检索通过内置浏览器工具继续完成。未请求或记录 API key。

## 固定输入与实验顺序

- video1：既有完整 v2 ZIP，31 帧、1,500,000 点。
- video2：上一轮完整 v3 ZIP，12 帧、2,446,979 点；CPU ownership 导出复用同一份 DA3、GID 和 SAM3 v2 缓存。
- 同一数据包的 renderer A/B 固定相机、画布、半径和 visibility；不空间裁剪或抽稀。
- 先在 video2 默认近景对比旧 output 和直接归一化 output，并读取真实 accumulation alpha，区分颜色变深与几何覆盖变化。然后在 video1 复核。
- 全帧 / 仅 frame 0 可见只用于诊断多帧合成，不把隐藏其他帧当作可交付方案。
- ownership8 另存 ZIP，记录逐 GID 点数与固定 8 mm 体素的多来源比例；需继续检查相同相机的孔洞与商品边缘，不能只以删点比例或多来源率下降判定成功。

产物写入 `runtime/viewer-extent-20261012/video12/followup/`。浏览器使用本机 SwiftShader，仍受上一轮 LAN 审批限制；不重试 LAN 开放。每个候选的实际结果完成后记录如下。

## 实际结果

### 不透明颜色归一化

完整 video2、固定近景相机、radius=1.05、960×543 画布：旧 output 与修复后的 output 均覆盖 292,112 / 521,280 像素（56.0375%），实际 accumulation alpha 的三个计数完全相同。其中 20,513 像素处于旧混白区间，占已覆盖像素的 7.0216%。截图中的纸箱白尘减少，瓶身和标签的颜色更完整；商品顶部孔洞、价签的几何重影仍然可见。覆盖没有增加，不能把颜色变化解释为补洞。

证据：`runtime/viewer-extent-20261012/video12/followup/output-normalization/{old.png,opaque.png,alpha-mask.png,receipt.json}`。真实工作树 renderer 的 `npm run build` 通过；`production-opaque.png` / `production-receipt.json` 复核实际生产源码和固定相机，无浏览器错误。独立 WebGL 行为测试 6/6 通过：中心和三个低 Gaussian 权重内部点均保持黑色，空白和隐藏面片为白色，中心拾取命中、背景拾取为空；`probe.json` 中 GL errors 为空。

video1 也已用实际英文 renderer 打开完整 v2 ZIP（31 帧、1,500,000 / 1,500,000 点可见），保留源 frame 0 朝向的总览相机；`production-video1-opaque.png` / `production-video1-receipt.json` 无浏览器错误。该图是修复后的全场景复核，不是 video1 的成对质量评分；商品顶部仍有碎片与孔洞。

### ownership8：保留候选，不改默认

CPU 导出与打包耗时 103.7 秒，指标计算 15.5 秒。总点数 2,446,979 → 2,422,880，保留 99.0152%；非空 GID 62 → 62，最低商品保留率为 GID 3 的 93.8017%。

初次全场景 8 mm 多来源体素比例为 13.7788% → 15.3482%，但审查发现这不是纯删点比较：背景仍为 500,000 点，其中 201,434 点被重新选择。该比例受背景集合变化及商品/背景相交混淆，撤回据此判定算法变差的结论。

补充比较确认商品点为原始 DA3 world 坐标的严格子集：1,946,979 → 1,922,880。商品范围内的占据体素 204,243 → 204,172，多来源体素 26,475 → 25,783，多来源比例 **12.9625% → 12.6281%（下降 0.3344 个百分点）**。改进幅度较小，也不等价于视觉清晰度已通过。manifest 的自动居中会随点集改变，未来视觉比较需统一原始世界相机，而不能直接复用经过不同居中变换的相机坐标。

证据：`runtime/viewer-extent-20261012/video12/followup/ownership8/{report.json,comparability_audit.json}`。原报告保留，补充报告解释混淆；未重新导出。

### 按视角选择纹理来源：未采纳为默认

从当前英文 renderer 移植本地已有的 winner pass，按视角一致性（0.85）和纹理采样分辨率（0.15）打分，只在通过来源深度与前表面可见性检查的候选中选来源帧。保留英文版的颜色空间、SMAA、0.005 容差和最长直径限制。实验文件与正式 renderer 分离。

video2 OFF/ON 的实际覆盖均为 292,112 像素。截图差异较小，孔洞与价签的几何错位仍然明显；没有充分证据证明值得增加默认渲染 pass。独立 WebGL probe **9/9 通过**：关闭时红蓝叠色 `[188, 0, 188]`；开启时同分选择较小 frame index 的红色，隐藏该帧后选择蓝色；透视与正交相机的 picking 均与显示一致，GL/browser errors 为空。这证明实现行为符合预期，不等于真实场景画质改善达标。

原 capture 脚本误要求全部 alpha 统计相同，因饱和权重像素从 271,599 降为 269,888 而在截图保存后报错。选中单来源本就会减少权重和；正确的覆盖门槛是 `alpha > 0` 的像素数，而该值没有变化。保留原 `receipt.json` 的报错，不为抹去该错误重复渲染；修正之后脚本的检查条件。

证据：`runtime/viewer-extent-20261012/video12/followup/source-selection/{off.png,on.png,receipt.json,probe.json}`。实验未评估转动时的接缝稳定性与真实 GPU 开销，未加入正式 Viewer 默认。

### 来源深度容差：不是当前大片孔洞的主要解释

最后一次单因素对比把实验 renderer 的来源深度容差从 0.005 放宽到 0.015，固定完整 video2、同一近景相机、所有点可见、关闭来源选择、不透明归一化、radius=1.05 与直径 cap=6。覆盖从 292,112 增至 292,138 像素，**仅增加 26 像素**；截图中的大片孔洞和几何错位仍在，浏览器无错误。

该结果不支持把容差收紧视为这些大片孔洞的主因，也不足以采纳更宽容差。正式代码仍为 0.005。证据：`runtime/viewer-extent-20261012/video12/followup/source-depth-tolerance/{tolerance-005.png,tolerance-015.png,receipt.json}`。

### 尚未解决

- 正式源码只修复不透明颜色归一化。来源选择与 ownership 默认值均保持不变。
- video2 原始 U/V 零长度分别为 4,088 / 3,503 个（总点数 2,446,979），约 0.17% / 0.14%，不是目前大片孔洞的充分解释；未擅自给这些点补造法线或面片。
- 本轮不重跑几何重建，不代表恢复了源视频细节，也不解除原 D1 的未判定状态。

后续优先调查跨帧几何/位姿错位与表面覆盖，而不是继续扩大 Surfel。现有证据支持的下一项研究方向是：先检查源图之间的投影残差，再比较带空间连续性和接缝约束的表面块选图；论文中的该方向仍需本项目验证。像素级 winner 只能避免同一屏幕像素混色，无法消除已经位移到不同位置的重建表面。

实验结束后本机 Vite 与浏览器已停止，未部署或开放 LAN。原始 receipt 保留不改；其中非 output-normalization 模式的 `outputMode: old` 曾记录了未使用的 UI 状态，真实 renderer 为不透明归一化，已由代码、实际截图与 WebGL probe 验证。后续 harness 改为记录实际输出模式，不影响已捕获相机、点集或像素结果。

最终检查：英文 Viewer 与实验 harness 构建通过；真实 WebGL 行为检查分别为 6/6 与 9/9；既有 ownership 定向测试 7/7。CPU 实验脚本编译、已有 generation 的商品指标复算、capture 脚本语法检查均通过。未运行全量回归或重跑有效截图。

# GLB projection复用：实现与验证状态

## 变更

生产改动位于`src/surfel_mesh.py`与`src/surfel_glb.py`，没有修改LOD、compact/texture。经协调者验收，public `export_scene_glb`默认encoder_workers已改为2，4仍显式可选，内部/plain非compact入口保留1。Python导出接口增加固定选择的`projection_mode`：默认`source_world`保留旧逐texel world→camera→projection顺序；`camera_basis`仅为显式实验模式。非法模式在mkdir前直接拒绝，不是出错后fallback。mode写入顶层、bake、compact报告。

`_frame_constants`每frame建立E/K/invK/affine与texel网格，多个1024点batch共享，默认生效。仅实验`camera_basis`使用`_project_texels`：在两道footprint/radius/scale限制之后，将bu/bv先转为camera与homogeneous projection basis，复用已有center/projected；不再为每点64个texel先构造world，再逐texel做两次矩阵变换。camera只保留需要的z，而非全xyz。

depth连续性/遮挡、source边界、圆盘mask、RGB round、固定顺序RGB dilation、6native长径约束、f16/JPEG输入、frame/page顺序均保留。没有新fallback、阈值epsilon或GPU路径。

不能把线性代数等价称作float32逐bit等价：加法/矩阵乘法重排会移动少量ulp，可能改变边界alpha、完全透明Surfel删除及RGB round。少量prequant RGB差异也可能经整页MEDIANCUT改变更多解码颜色；必须同时看采样raw和完整编码输出。

## 已执行的轻量检查

- 使用既有runtime/viewer-latency-20261017/test-env、CUDA隐藏，mesh/glb/lod focused tests：**89 passed**。
- 新增独立float64几何oracle检查非轴对齐旋转、平移和倾斜basis；另验证90度精确旋转/平移fixture的RGBA与世界corners。
- 两个probe脚本syntax检查通过。
- 新增mode选择/非法值/按frame共享常量与导出report传递测试，最小选择集：**7 passed, 76 deselected**；没有重复跑此前89项。
- 默认2与提前拒绝非法mode后，仅跑新增两项和已有FIFO直接组合：**6 passed, 33 deselected**。验证非法mode不创建不存在的nested parent、默认tiny输出/queue/report为2、显式1对2/4的有界FIFO与PNG/几何一致；未重跑此前89项。
- 12个微型synthetic cases（source边界与depth阈值相邻float32、3种旋转）snapshot对新函数：0 alpha/RGB差异，结果在[bake_synthetic.json](bake_synthetic.json)。这组微型检查不构成真实视频或性能结论。
- 首轮测试发现预计算grid变量被四邻检查的局部x/y覆盖，已定位改成grid_x/grid_y并在使用处绑定；上述89项是修正后的完整owned结果。

## Probe入口与重现

真实组件入口[bake_probe.py](bake_probe.py)：读取已完成generation，执行当前保守LOD，按frame均匀选点比较snapshot与candidate，`--projection-mode both`为默认，逐行保存mode便于分组，也可以单选source_world/camera_basis。结果按完全相同corner bytes配对幸存quad，保留未配对数量、point/geometry顺序是否相同、全RGBA是否相同、alpha/含透明RGB/共同不透明RGB变化数量、最大/绝对差总和及像素/通道分母。没有放宽任何生产门槛。采样计时包含原函数各项工作，candidate常量setup置于计时外，不能用于宣称完整GLB提速。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv --cache-dir /tmp/profiling-20261018-uv-cache run --no-project \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/latency-optimization-20261018/bake_probe.py \
  --generation runtime/profiling-multivideo-20261018/video3/viewer/runs/41f1919b9cc640c9b33696ebdc834e0d \
  --out reports/latency-optimization-20261018/bake_real_probe.json \
  --frames 29 --points-per-frame 1024
```

完整入口[bake_export_compare.py](bake_export_compare.py)：同一generation顺序candidate1/2/4，`--projection-mode`默认source_world；camera_basis需显式指定。`--reference-glb`复用相同generation的既有baseline GLB，不触发额外导出；旧`--include-snapshot`仍可显式添加原baker+1 encoder。计时期间没有额外page.copy；记录完整export wall、production timing、每次export采样RSS及累计ru_maxrss、CPU差值。导出完成后才比较accessor bytes、完整document和逐页解码RGBA/alpha，比较不计入export timer。产物保留，由协调者清理。document差异可能来自PNG长度/offset变化，不自动代表几何差异；几何另看accessor bytes/数量。输出page数或shape变化会单独记未配对数量，不能把这些图像当成已比较。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv --cache-dir /tmp/profiling-20261018-uv-cache run --no-project \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/latency-optimization-20261018/bake_export_compare.py \
  --generation runtime/profiling-multivideo-20261018/video3/viewer/runs/41f1919b9cc640c9b33696ebdc834e0d \
  --out runtime/latency-optimization-20261018/bake-export-compare --include-snapshot
```

这是一轮固定顺序，不是多次配对性能验收；首次导出与后续文件页缓存、allocator历史不同，RSS绝对采样含保留allocator。2/4线程本来就是有界FIFO，不是本轮新增并行算法。不得把worker总执行、join等待和bake重叠时间相加。

## 已完成的完整source_world交付对照与默认决定

协调者已在同一baseline generation依次完成1/2/4线程，原始记录为[runtime result](../../runtime/latency-optimization-20261018/bake-export-compare/result.json)，使用既有reference GLB而非重复snapshot导出。

| encoder_workers | 完整export wall（秒） | 导出期间self RSS采样峰（bytes） |
| --- | ---: | ---: |
| 1 | 108.396069 | 957,362,176 |
| 2 | 67.627083 | 1,041,502,208 |
| 4 | 66.638055 | 1,084,465,152 |

old reference→new1、new1→new2、new1→new4的全部accessor、完整GLB document及每组128,295,600个decoded RGBA texel全部一致，alpha/RGB变化为0。它支持本generation内source_world+常量复用与现有有界编码流水的完整输出一致性；不推广camera_basis的非等价变换。

本轮2线程相对1线程约少40.769秒；4相对2仅再少0.989秒（约1.46%），self RSS采样峰多42,962,944 bytes（约40.97MiB）。这是一轮固定顺序、CPU共享、未重复的观测，不能证明4的稳定额外收益；RSS也不能当独立新增分配。因此协调者选择生产默认2，保留显式4。该组件差额不直接扣进全链路结果。

协调者随后完成独立新进程的固定video3完整pair：[结果](pair-summary.json)、[验证](pair-validation.json)。GLB自身114.041383→68.770372s，所有bake batches61.205268→58.505109s；worker encode累计wall43.639827→45.379002s与bake重叠，main-thread join43.649168→.899529s。最终GLB完整payload精确一致，1282759baked/2565518triangles/31pages。整体253.138706→186.894435s同时包括DA3和ranges优化，不归因于GLB单项。正式pair仍用source_world，无camera_basis实验收益。

## 范围与后续限制

未引入LOD共享center缓存：这需要跨函数携带原索引/有效域/精度语义的API，当前改动首先消除每texel变换，避免扩大工作域与常驻缓存。没有预烘焙未经过最终label/过滤/预算/scale/LOD的点。

旧profiling AST按`_bake_batch`固定行offset拆七块，函数主体已变化，**不得原样复跑旧AST并沿用旧区块标签**；本轮probe不使用该AST。新生产helper分别有明确职责，后续需要细分时应更新插桩位置。

PyTorch grid_sample可作为另一个实验，但coordinate normalization、padding/half-pixel和round与当前NumPy不自动等价，H2D/设备共享也有成本。固定方向/旧轮known的dilation不能直接替换成并行pool而宣称语义不变。本轮不引入GPU实现。

协调者首轮camera_basis真实sample已完成（[bake_real_probe.json](bake_real_probe.json)），29316个幸存quad几何全部配对，6个alpha texel不同、9318个RGB texel不同、4443个共同不透明RGB texel不同；all-RGB最大差41不能直接解释为可见RGB最大差。因此camera_basis不作为默认无损优化。该旧结果未覆盖新增共同不透明max/sum/分母字段，也不回写它；后续[bake_modes_real_probe.json](bake_modes_real_probe.json)明确共同不透明RGB最大差1。完整source_world输出对照、default2决定及最终pipeline见上节；本agent没有自行运行重型probe。跨视频画质泛化仍未验收。

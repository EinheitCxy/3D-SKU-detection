# 有界Torch sampling/mask/RGB延展实验

## 当前范围与接口

仅新增本目录文件，生产默认和依赖保持不变。`torch_bake.py`提供：

- `bilinear(image,x,y)`：Torch tensor输入，float64四tap RGB输出；不是grid_sample。
- `dilate_rgb(tiles)`：uint8 RGBA原地返回同对象，alpha不变。
- `FrameBaker(frame,image,depth,device=...,radius=2,tile_size=8,batch_limit=1024).bake(p,u,v,scales)`：返回NumPy corners和RGBA tiles。每instance对应一个frame，RGB/depth各一次`.to(device)`；CPU模式没有H2D。
- `bake_surfel_pages`：与生产page producer一致的主要kwargs，额外`device`和scalar `metrics`，仅允许`source_world`、tile=8、batch≤1024、atlas≤2048。frame/page顺序不变，仍交给生产LOD/atlas编码/compact writer。

这是CPU/GPU步骤整合，**不是单个融合kernel**。CPU保留原始source_world world→camera→projection、source affine及half-pixel映射的float32表达式顺序；Torch负责64texel圆盘/深度/来源/面积mask、nearest-depth gather、RGB四邻采样、round、固定方向延展。初始/最终footprint和6native长径限制仍调用生产实现。

RGB读取的是规范JPEG解码值，几何/depth来自既有f16文件按生产方式转float32；没有用processed-grid RGB或JPEG编码前原图替换。全GLB模式进程内临时替换page producer并在finally恢复，不修改生产源码。

## 数值契约

现有NumPy `_bilinear`中，float32 x/y减intp x0/y0会提升为float64，随后RGB四tap权重和求和也为float64。候选显式保留该提升与表达式分组，用Torch round的ties-to-even转uint8。nearest depth索引使用floor(px+0.5)，并非ties-to-even round。source→texture仍使用CPU NumPy f32 `(source+.5)/source_size*texture_size-.5`，最后按padded layer的图像边界clip。

RGB延展冻结每轮old known，固定左→右、右→左、上→下、下→上，目的known即时更新，先方向赢tie。使用固定shape `torch.where`而非每方向boolean advanced indexing，避免每方向CUDA nonzero及动态分配；共H+W轮，无GPU `.all().item()`早停。known满后empty fill不会改变结果。逐Surfel keep压缩仍用boolean indexing，CUDA可能发生动态shape分配/内部同步，计入实际计时。

Torch float64 eager逐步算术在CUDA是否与NumPy逐像素一致，仍须真实GPU样本及最终palette验证。未用epsilon放宽任何门槛。CPU smoke通过不等价于CUDA通过。

## Probe状态与计时

`probe.py`要求新的out目录，保留JSON；输出reference不变。GPU默认逻辑cuda:0，但必须由协调者显式用物理GPU2 UUID隔离，否则拒绝。CPU smoke必须显式`--device cpu`，不初始化CUDA。

- `smoke`：12个小型source/depth边界与旋转fixture，与生产source_world逐数组比较。
- `sample`：读取同一generation并执行现有LOD，按frame均匀选点，使用同一batch划分。JPEG decode/padding/索引准备移到两臂timer外单列。baseline timer包含一次frame constants和全部原生bake batches；candidate timer包含FrameBaker首次context、每frame H2D、CPU projection、Torch计算、keep/tiles D2H。每frame分别保存两个wall以及首frame标识；不在timer前预先同步/初始化CUDA。比较在两timer外。baseline先运行、candidate后运行，是固定顺序诊断，不是重复配对性能验收。
- `glb`：完整export timer包含生产读取/LOD、候选baker、CPU PNG/compact/writer及收尾；首次CUDA初始化、传输与下载在timer内。默认现有2个encoder worker，可显式1/4。若指定`--reference-glb`则导出后比较document/accessor/decoded RGBA；没有reference明确记`unvalidated_export_only`，不宣称验证通过。

正确性不符会写`correctness_failed`、保存精确diff并非零退出，不降低门槛；sample拒绝空comparison或任一臂baked=0。GPU/RSS峰是进程生命周期观测，不是某单kernel新增分配。计时不排除首context，也不把worker累计wall再加到重叠bake总量。

## 命令（GPU由协调者运行）

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv --cache-dir /tmp/profiling-20261018-uv-cache run --no-project \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/torch-bake-fusion-20261018/probe.py --mode smoke --device cpu \
  --out /tmp/torch-bake-smoke-new
```

```bash
CUDA_VISIBLE_DEVICES=GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
uv --cache-dir /tmp/profiling-20261018-uv-cache run --no-project \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/torch-bake-fusion-20261018/probe.py --mode sample \
  --generation runtime/latency-optimization-20261018/baseline-video3/viewer/runs/b1cfa9d40e544d4cb413343d9a91bc3b \
  --frames 29 --points-per-frame 1024 --batch-size 1024 \
  --out runtime/torch-bake-fusion-20261018/sample-new
```

完整导出将sample的mode改glb、out用新目录，再传同一generation对应的
`--reference-glb runtime/latency-optimization-20261018/baseline-video3/scene.glb`。
sample通过和预算允许后才运行完整导出，避免不必要的GPU/CPU争用。

## 已验证与限制

实现agent执行过12个微型CPU smoke，逐geometry/RGBA一致；独立reviewer的边界CPU测试覆盖half-pixel/float64/round、dilation方向/旧known、source/depth/投影和输入校验。修改后复核以reviewer报告为准。尚未由实现agent运行GPU、真实视频sample或完整GLB，也没有速度/画质promotion结论。

每frame 1920×1080 float32 RGB约23.73MiB，depth504×280 float32约0.54MiB；GPU只保留当前frame。1024点×64texel×3×float64约1.50MiB/一个RGB中间结果，eager四tap会有多个临时tensor；tile本体约0.39MiB/批。实际allocator峰必须实测，不能只加这几个数组估计显存。20轮×4方向仍产生许多GPU launch；float64吞吐、dynamic keep、NumPy projection、CPU PNG和H2D/D2H都可能抵消收益。未启用torch.compile、混合精度、grid_sample或自定义CUDA。

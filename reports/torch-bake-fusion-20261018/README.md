# 有界 Torch RGB / mask / 方向延展实验

2026-10-18，Rick批准测试。GPU2固定video3 generation的完整GLB导出 **66.589322→44.671011秒，减少21.918312秒／32.916%**。这是组件实验，每臂一次；生产默认仍使用NumPy CPU baker。上一轮源码、测试和profiling已另行推送到origin/main@37019bd。

| 同次组件指标 | 原生CPU | Torch实验 |
|---|---:|---:|
| 完整GLB export wall | 66.589322s | 44.671011s |
| Python脚本计时至文件就绪，含import | 69.928963s | 48.084271s |
| imports | 3.335594s | 3.413232s |
| producer窗口（含LOD等，扣主线程写入/join） | 63.420602s | 36.657004s |
| 主线程compact/write/join | 3.166538s | 8.012695s |
| ↳ encoding join | .877399s | 5.669511s |
| 最终Viewer输入／LOD候选 | 2,402,589／1,297,686 | 相同 |
| baked／triangles／atlas页 | 1,282,759／2,565,518／31 | 相同 |
| GLB大小 | 121,197,580bytes | 相同 |

主线程join随producer更快而增加；不能将它当成编码算法退化，也不能把重叠worker时间再加到export。脚本ready起点略有差异且都不覆盖外部uv／解释器启动，不称严格的进程启动时长。candidate whole脚本59.170519秒含文件就绪后约11.086秒验证，不作为交付耗时分母。[汇总](summary.json)、[原生结果](results/native.json)、[Torch结果](results/torch.json)。

## 实现循环与边界

候选只存在于本目录[torch_bake.py](torch_bake.py)，不修改src生产默认、不增加依赖或编译步骤。

1. 复用已完成Viewer generation；沿用原LOD、positions的f32及U/V／scale／depth的f16→f32读取边界、规范source JPEG、radius2与6native长径限制。
2. 每个source frame解码并零padding一次，RGB与depth各上传一次；只持有当前frame的GPU纹理/depth。29个frame的RGB与depth各累计上传29次，不逐batch重复上传整图。
3. 每batch最多1024surfels、固定8×8内部texels。CPU保留source_world投影、affine和half-pixel算术次序，生成processed/source坐标、depth和边界等输入；这些有界数组传给Torch。
4. Torch合并执行圆盘/source/二维depth/连续性mask、最近depth gather、四tap RGB采样、round及RGBA写入。NumPy的float32坐标减intp会提升到float64；候选显式保持float64插值，不使用默认grid_sample近似。
5. 方向延展仍每轮读取旧known，按原四方向先到先得，alpha不变。用固定shape的`torch.where`／`copy_`替代每方向布尔advanced indexing；固定H+W轮避免每轮`.all().item()`同步。每surfel keep压缩仍有动态索引，不宣称完全异步。
6. keep与tiles回CPU，现有page builder、有界两页FIFO、128色MEDIANCUT／PNG编码、geometry量化和单线程写序不变。

这是将相关工作整合到一个GPU阶段，**不是单个融合kernel**。CPU仍保留完整Viewer/LOD输入；有界声明针对GPU当前frame和batch工作集，不代表整个导出只占1024点内存。见[实现说明](implementation.md)及[独立数值契约](numerics_review.md)。

## 正确性与测试

- GPU sample：29frame×1024＝29,696输入、29,316幸存quads；2,931,600个完整raw RGBA texels（包括透明RGB和alpha）及全部corners顺序精确一致，未配对0。[sample](results/sample.json)。
- Sample匹配已解码输入，native总1.526704s、candidate1.341692s（含首次context／上传／下载）。首framecandidate.388182s，后28frame总native1.476881s、candidate.953510s。decode准备1.112241s单列；这不是完整导出数字。
- 完整GLB：完整JSON document、所有accessor对应bufferView字节、31页／128,295,600个解码RGBA texels精确相等，alpha/RGB变化0、缺页／shape差异0。
- 没有逐项对比完整导出的全部raw tiles，也没有检查整个GLB文件字节一致；不扩大上面两种证据的范围。
- 独立CPU边界测试首次30passed，方向延展改为where后受影响10passed／20deselected；12tiny smoke全部相等。测试包含f64手算、round ties、source/depth二维边界、nextafter、方向优先／旧known、空批、frame ID与上传复用，见[test_torch_bake.py](test_torch_bake.py)。未重跑无关全量测试。
- 两位medium独立最终结果复核PASS：数值／交付覆盖与GPU计时／资源口径。[状态](STATUS.md)、[事件](JOURNAL.md)。

## 测量条件与资源

固定既有video3 generation，29frame；native先、Torch后，各自新Python进程。物理GPU2 RTX4090D，完整UUID显式隔离；原6221／8550resident保留，CPU OMP／OpenBLAS／MKL各4线程，主机共享负载未隔离，OS页缓存未清空。NumPy1.26.4、Torch2.7.1+cu126。没有重跑DA3／PVS，也没有复制模型或原始数据。

Torch export计时覆盖第一次CUDA context、每frame和batch传输、CPU投影、GPU计算、D2H、LOD、palette／PNG与最终写入；比较在计时外，没有逐层同步计时或CUDA event细分。Torch allocator生命周期allocated峰 **37.119MiB**、reserved峰 **54MiB**，不含完整CUDA context／driver占用。candidate RSS含后续验证而native不含，不能比较为export内存升降。

未建立多视频均值／P95、跨GPU稳定收益、人工画质或准确率结论；没有重跑完整186.894秒流水线，不能直接减去本组件21.918秒作为新全链路实测。生产接入与设备生命周期仍未改。

## 复现

复用项目既有uv环境和本机generation。输出目录必须不存在；下面命令实际执行过。native与candidate串行；GPU2已验证可用才启动。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/torch-bake-fusion-20261018/native_export.py \
  --generation runtime/latency-optimization-20261018/baseline-video3/viewer/runs/b1cfa9d40e544d4cb413343d9a91bc3b \
  --out runtime/torch-bake-fusion-20261018/native-glb-01

CUDA_VISIBLE_DEVICES=GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/torch-bake-fusion-20261018/probe.py --mode glb --device cuda:0 \
  --generation runtime/latency-optimization-20261018/baseline-video3/viewer/runs/b1cfa9d40e544d4cb413343d9a91bc3b \
  --reference-glb runtime/torch-bake-fusion-20261018/native-glb-01/native.glb \
  --out runtime/torch-bake-fusion-20261018/torch-glb-01 --encoder-workers 2 --batch-size 1024
```

sample入口`probe.py --mode sample --frames 29 --points-per-frame 1024 --batch-size 1024`使用同一generation/device并指定新out。普通Python异常与比较失败会写result并非零退出，无reference明确标未验证；被外部SIGTERM时不保证脚本自身完成JSON，调度者需保留退出码。本轮所有样本和两个完整导出均正常退出0。

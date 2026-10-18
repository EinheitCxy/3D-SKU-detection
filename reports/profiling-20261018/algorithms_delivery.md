# 2026-10-18 delivery 算法与诊断口径

当前材料说明源码算法与独立诊断脚本；2026-10-18 delivery 实测已完成，数值统一见[当前 profiling](../../docs/profiling_breakdown.md#11-glb当前切片rgb后的详细循环与成本)和[独立原始汇总](results/delivery.json)。源码行号对应本次 checkout，后续修改应以函数名重新定位。

## 运行与产物

```bash
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python \
  python reports/profiling-20261018/diagnose_delivery.py \
  --base runtime/profiling-20261018 \
  --out runtime/profiling-20261018/diagnostics/delivery-new
```

这条命令会执行完整 CPU ZIP/GLB 导出，应由协调者安排。本轮已在独立runtime目录执行成功；示例使用新的out避免覆盖。`--out` 必须是新目录；脚本保留 `result.json`、`diagnostic.zip` 和 `diagnostic.glb`，不自动清理。输入是 `--base/viewer/CURRENT` 指向的 generation，不重跑 GPU、模型、过滤或采样。

运行显式固定 `encoder_workers=1`，其他参数使用并记录生产函数实际默认值；若生产 encoder 默认已改变则报错，避免无意更换基线。结果记录 imports、delivery 总墙钟、各函数嵌套计时、源码位置、生产配置、ZIP 成员大小、RUSAGE_SELF、异常 traceback。失败也写 result；进程被 SIGKILL 或解释器未进入 try 时无法保证写回。

## generation 与入口边界

`src/web_viewer_export.py:139 export_web_viewer_bundle` 读取或复用 cache，组织 objects/缩略图/方向，调用 `_sample_points`、`prepare_surfels`，整理 point ranges、manifest 和 masterdata，最后调用 `:1094 _publish_bundle`。

`_publish_bundle` 在临时 generation 写 Surfel 文件、positions/colors/normals、JSON、缩略图，rename 后原子更新 CURRENT。这里不生成 ZIP 或 GLB。`scripts/export_surfel_viewer.py:13 main` 也只生成 Viewer bundle，默认 texture edge 1920、voxel 0.005、DBSCAN 开启；proxy/early-background 默认 0。`scripts/export_pi3x_viewer.py:13 main` 使用 Pi3X dedup 与普通 Viewer，不可据此推断支持 DA3-only GLB。

`src/surfel_glb.py:366 export_scene_glb` 是生产 streamed compact 入口，默认 radius=2、tile_size=8、lod=True、palette_colors=128、encoder_workers=1。`:166 export_surfel_glb` 与独立 CLI 默认 lod=False，且不走 compact writer，二者应分开描述。

## 源纹理与数据装载

`src/surfel_export.py:48 prepare_surfel_textures` 按 frame 读取照片、转 RGB，按朝向把最长边限制为 texture_edge∈[256,1920]、短边限制为 min(texture_edge,1080)，使用 LANCZOS thumbnail、不放大；JPEG quality=95、subsampling=0。保存 source/texture size、inverse affine、intrinsic/extrinsic。

`:88 prepare_surfels` 根据最终 source indices 保存 U/V/scale 为 f16，frame=`index//(H*W)` 为 u8。depth 仍保存完整 `[F,H,W]`。`src/surfel_glb.py:31 _assets` 支持目录或 ZIP；`:85 _load` 读取 manifest/objects/surfel 和几何、相机、source texture bytes。每张照片在装载检查时已读一次，后续 baker 再读取并解码一次。目录与 ZIP 输入的读/解压成本不同。

## 保守 LOD

`src/surfel_lod.py:49 select_surfel_lod` 按 frame 投影点中心，只有 finite、图内、正深度且 owner≥0 的商品参与；背景 owner=-1 保留。以 `(owner,x//2,y//2)` 分组，只接受恰好四点且四个角各一个的完整 2×2 block。缺角、重复像素或更多点均原样保留。每 8192 **组**调用 `:8 _covered`。

coverage 检查四个非零法向两两 cosine≥0.95，深度极差≤平均 z 的 1%，所有 scale≥1；中心及左右上下五组 sample 必须具有正且 finite 的连续 source depth，阈值为 `0.001+0.5*native_axis`。再以左上原点作代表，用 `_bound_footprint` 后乘 `2*scale`，解其余中心在代表基底中的坐标，要求圆盘半径≤0.9、平面残差≤各自 z 的0.5%。通过后只保留左上点，最后返回原全局顺序索引。

LOD 未调用后续 `_limit_final_diameter`；四中心 coverage 不保证任意视角完整像素覆盖。即使前置有效性已失败，本批后续向量化检查仍运行。诊断分别包装 lod 模块的 `_bound_footprint` alias 与 mesh 中同名函数，避免漏计。

## frame、batch、tile、atlas

`src/surfel_mesh.py:179 bake_surfel_pages` 默认 batch_size=1024、tile_size=8、atlas_size=2048。stride=tile+2=10，columns=204，容量41616个 Surfel，满页实际为2040×2040。按 sorted unique frame 顺序读 JPEG 至 float32 RGB；较小纹理右下补零至全体 frame 最大尺寸，保持 Viewer array texture 边缘采样行为。

每 frame 的 slots 按1024点分批调用 `:95 _bake_batch`，其输出通过 while 切片装入可复用 page 缓冲，满页 yield，末尾交尾页；页面可跨 frame。`:156 _page` 将 tiles 按行列补零、reshape/transpose 成 atlas，计算内部边缘 UV、四顶点和两三角面。positions copy、UV/index 新建、atlas 来源新 padded 数组，交出页不会被 producer 后续复用缓冲覆盖。

`_bake_batch` 的连续步骤如下，诊断 AST 在函数内连续顶层语句外包 span，不修改生产文件：

1. `:96–109` 相机参数、中心投影，剔除不合法投影和非正深度。
2. `:110–121` native spacing 与第一道 footprint，检查左右上下深度连续性；不连续方向把边界收紧到中心像素±0.5。
3. `:122–124` 乘 radius/scale，再限制最终椭圆直径。
4. `:125–135` 8×8 texel 中心覆盖[-1,1]²，world→camera→processed，形成圆盘、图内、正深度及连续性边界 mask。
5. `:136–145` nearest source depth，要求 `abs(sampled-camera_z)≤0.015*sampled+0.001`；inverse affine 后检查 source 图内和非零面积，删除完全透明 Surfel。
6. `:146–148` source→texture 像素中心映射，四邻双线性 RGB。
7. `:149–153` 分配带 gutter tile、round RGB、二值 alpha、四 corners、RGB dilation。

源码结构变化后应重新检查 AST 分组 offset；当前边界为函数起点后0/15/27/30/41/51/54行，result记录实际起始行和边界。原始文件及源码行号保留。

## 两道长边限制、采样与 dilation

`src/surfel_mesh.py:16 _bound_footprint` 对 U/native_u、V/native_v 的 Gram 矩阵求 major/minor singular values；major 上限=min(2,4*minor)，minor上限=2，通过矩阵组合只缩短超限方向，之后恢复 native 单位。

`:34 _limit_final_diameter` 对已乘 radius/scale 的轴重新求世界坐标 singular values，各半轴上限=3*max(native)，最长直径≤6*max(native)。没有超限的短轴保持。它相对 source grid 定义，不是任意 model matrix 后的绝对米制上限。

`:64 _bilinear` 对 x/y clip 后取 floor/next 四邻加权。调用处使用 `(source+.5)/source_size*texture_size-.5` 像素中心映射。

`:74 _dilate_rgb` 初始 known=alpha非零，最多 tile高+宽轮（默认20）；全 batch known 满则早停。每轮保存 old known，依次右→左、左→右、下→上、上→下传播 RGB；目的 known 即时更新，先方向赢 tie，本轮新增来源只在下一轮生效。只填 RGB，始终不改 alpha。不能把方向并行或允许轮内传播后声称语义相同。

## 编码与并行

`src/surfel_glb_texture.py:27 encode_atlas` 校验 alpha 只能0/255；`:33 _compact_rgba` 把全部 RGB（含透明位置的 dilation RGB）按 MEDIANCUT、无 dither 量化到默认128色。透明像素 index 加128；两份同 RGB palette 分别设置不透明和透明，再 PNG optimize=True。palette_colors=None 时直接 RGBA PNG，不执行颜色量化。

`src/surfel_glb.py:176 _atlas_encoding` 默认不建线程；2/4 worker 时有界 FIFO 最多保留 workers 页（含运行任务），只在 worker 编码。producer 单线程 bake；主线程保持原页顺序写 geometry、等待future、写image和材质。失败会取消未启动任务、等待运行任务、关闭producer。本诊断固定串行，但 span 栈采用 thread-local，聚合用锁。

## 几何量化和写入

`src/surfel_glb_compact.py:65 _chunks` 每32766三角形初分块，unique vertices超65535时二分；生成u16本地index。`:143 geometry` 为各块求position min/span、零跨度轴设1，round至u16；position以四个u16布局存8字节，UV normalized u16，子node保存translation=min和scale=span，root保存原world_to_view。使用KHR_mesh_quantization，记录实际position/UV重建误差。

`:121 view` 对各段4字节对齐、顺序写临时BIN；`:171 image` 直接附PNG bytes，不重新解码。`:181 finish` 调用`:79 _write_glb`，补齐BIN、JSON序列化与padding，检查GLB 4GiB格式上限，exclusive create输出，写GLB header/JSON/BIN、copyfileobj临时BIN。写失败只删除自己创建的输出。

## ZIP 成员

`perf/viewer_clarity/pack.py:14 pack_bundle` 固定 manifest、positions/colors/normals、objects、surfel metadata、U/V/frame/depth；然后追加metadata顺序的source textures，v3追加scale，最后sorted thumbs/*.jpg。8个v3二进制用DEFLATE level1，JSON/JPEG用STORED；逐成员串行写入。没有CURRENT、runs外层或GLB；helper也没有追加generation可能存在的sku_masterdata.json。

诊断记录每成员wall、输入字节、压缩字节和实际compression type。zip.member分类是binary/jpeg/json，不表示重新编码内容。

## 时间与资源解释

所有span均记录calls/inclusive_s/self_s/max_s；inclusive包含子span，self只扣同线程直接子span。不得把父子inclusive相加。PIL.convert/quantize/save位于texture或baker下，属于嵌套。geometry/image内部view也是嵌套；finish包含_write_glb。

`export_scene_glb.elapsed_seconds`是完整生产导出wall。`compact.elapsed_seconds`为主线程geometry、编码调用/等待、image写、finish的离散区间之和；`encoding_join_seconds`在串行时为实际encode调用耗时，并行时只是FIFO等待；`main_thread_write_seconds`还包括几何量化和元数据，并非纯磁盘I/O。`bake.elapsed_seconds`为内部总wall减compact区间，包含load/LOD/producer及控制开销，并行时与worker执行重叠。不能再加worker时长或join。

streamed模式bake.glb_bytes、compact.source_bytes、texture_bytes_before为None，因为不存在完整中间文件。主结果中的imports、delivery_wall和elapsed_wall_s范围不同：elapsed还包含装饰、配置检查、receipt准备等；delivery_wall包含串行ZIP和GLB。

RUSAGE_SELF user/system是前后差；maxrss是Linux进程生命周期高水位KiB，不是阶段新增分配或独立arm峰值。此独立诊断不等于主链路延迟、GPU占用、HTTP/BSON、COS或浏览器渲染时间。函数包装和AST计时有开销，不能替换未插桩主基线。

## 历史 runner 适用性

`runtime/video3-full-profile-20261016/profile_delivery.py:7–19` 的串行ZIP→GLB接口仍可参考，但固定旧OUT/CURRENT，ZIP会重写、已有GLB会失败，不可直接当新运行入口。旧 `diagnostics/cpu/detail.py:22 span` 使用全局STACK，不适合多线程；LOD `_bound_footprint`预绑定alias也可能漏计。新脚本已分别处理。

`reports/viewer-latency-20261017/run-atlas-compare.py` 固定旧generation、Python3.11和4个CPU，执行B1/C1/C2/B2，并保留page.copy供完整一致性比较；raw_export_wall包含页复制，不含导出后完整读取/比较。其RSS包含保留baseline数据，不能当纯生产内存，也不能作为本轮新性能结果。

## 实测记录

已完成，status=complete；结果在[results/delivery.json](results/delivery.json)，主链路数值在[summary.json](summary.json)。包含运行命令与generation、各层inclusive/self、页数/三角数/字节数、资源口径。主链路与独立诊断分表，禁止将不同运行窗口的子项叠加。

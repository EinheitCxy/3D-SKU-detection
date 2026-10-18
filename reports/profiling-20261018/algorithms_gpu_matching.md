# DA3、SAM3.1 PVS、matching 与 dedup：实现与诊断边界

本文记录 2026-10-18 checkout 的算法事实。独立诊断尚未运行时，本文件不提供新耗时；实际数值以新主跑及独立诊断 JSON 为准。源码行号对应本次 checkout，后续编辑可能移动。

## 流水线与真实循环

`main.py:987` 的 `run_complete_pipeline()` 先验证输入；classifier 开启时异步提交其独立进程。随后重建、启动单线程 CPU Surfel prepare、matching 全 references、等待 classification、dedup、最终 Viewer export。Surfel prepare 与 matching 重叠；最终 export 等待 prepare 结果。分类关闭要求输入检测已带有效分类。诊断不重跑 detector/classifier。

`main.py:1058` 检查 DA3 schema-v3 metric cache；命中即跳过重建。`main.py:423` 遍历 detection references，经 `src/inference.py:269` 每 ref 新建 `SKUMatchingSystem`，再调用 `utils/sku_matching_system.py:111`。`main.py:270` 的全局 RNG 锁覆盖整个单 reference inference，因此多 reference 线程配置不能证明 GPU 并行。

当前 `config.yaml:35` 是 `pairing_3d: next`。`utils/sku_matching_system.py:345` 只配 `(reference+1)%N`，29 帧最多 29 个有向图对；`all` 才有最多 N(N−1) 图对。采样代码在 target 循环内部，`all` 时同一参考物体会重复采样。

## DA3 模型与预处理

`src/da3_3d_reconstructor.py:64` 默认 `DA3NESTED-GIANT-LARGE-1.1`，实际 runner Python 为根 `.venv/bin/python`，可由 `DA3_VENV_PYTHON` 覆盖。普通 CLI 每次新 subprocess；注册 standby pool 后可从 CPU 预载模型的 worker 开始，两种加载边界不同（`:103`）。runner 文件头的旧环境路径注释不是当前调用依据。

`src/da3_runner.py:208` 按 numeric stem 排序，PIL 解码所有 RGB；`da3_defaults.py:3` 默认长边 504、upper_bound_resize，再 round 到最近14倍数，不裁切；整批 processed grid 必须一致。16:9 横图一般 W×H=504×280，竖图反之；应以实际输出 shape 为准。

`Depth-Anything-3/src/depth_anything_3/api.py:196` 的顺序是：

1. `_preprocess_inputs`：InputProcessor 处理图像并生成 CPU `(N,3,H,W)`，含 ImageNet normalize。
2. `_prepare_model_inputs`（`:301`）：`.to(device, non_blocking=True)[None].float()` 得 `(1,N,3,H,W)`。本地 wall 衡量 CPU 操作/提交，不代表 H2D 已完成。
3. `_run_model_forward`（`:367`）：前后原生 `torch.cuda.synchronize(device)`，中间调用 forward。这里的原生日志是同步后的 wall，仍含 Python/CPU 开销。
4. `_convert_to_prediction`（`:391`）：OutputProcessor 将模型输出组织为 CPU Prediction，包含 D2H。
5. `_add_processed_images`（`:399`）：把 CPU 图像转 NHWC、反归一化、clip 到0..1，再转 uint8 RGB。

外层 `api.py:125` 选择 BF16（支持时）或 FP16 autocast；模型输出 depth/camera heads 内部存在禁用 autocast 的 FP32区间，不能把整个 forward 都称为 BF16。

### Nested 两分支与尺度依赖

`model/da3.py:362` 的 Nested forward 顺序执行：

- `self.da3(x, ...)`：anyview 主分支，backbone → depth head → camera/ray pose → sky/aux。
- `self.da3_metric(x)`：metric 单目分支，对同一输入形成 metric depth 和 sky。
- `_apply_metric_scaling`（`:375`）：借 anyview intrinsics 对 metric depth 做尺度校正。
- `_apply_depth_alignment`（`:386`）：sky阈值0.3筛出非天空；最多100000点估计主分支 confidence median；构造有效 alignment mask；least-squares 求单一 scale factor；缩放 anyview depth 和 extrinsic translation；写 `is_metric=1`、`scale_factor.item()`。
- `_handle_sky_regions`（`:419`）：最多100000点估计 non-sky depth q99，并受200上限约束，再填 sky depth/confidence。

`DepthAnything3Net.forward`（`:100`）backbone 在 `:132`；depth head 在 `:140`；camera/ray 选择在 `:141`。runner 默认 use_ray_pose=False、GS关闭。metric分支是否具备camera decoder由其配置决定，不应假定两分支所有head都执行。

诊断保留这些方法的顶层 AST statement spans，`source_mapping` 给实际源码行与语句；`wall_contexts` 保留调用祖先，区分同一个 `DepthAnything3Net.forward` 属于 Nested 的 anyview调用还是metric调用。聚合 `wall` 中同名head可能包含两个分支，应使用上下文表做分支归因。

独立诊断另外为 Nested forward、两个分支的 DepthAnything3Net forward、DinoV2 backbone forward、metric scaling、depth alignment及sky处理记录CUDA events。`cuda_events`的完整祖先context保留Nested中anyview/metric调用行号，因此相同class的两个分支分别归因。只积累event对象，在runner.main返回、原生forward同步与Prediction D2H已完成后读取elapsed_time，不新增synchronize。父子event区间重叠，不能相加；event creation/record和AST/wrapper自身开销也会扰动该诊断，结果不替代主跑wall。

### 反投影与缓存

`src/da3_runner.py:245` 拒绝非metric输出。`:47` 在 CPU NumPy 逐帧计算：像素齐次坐标 → `K⁻¹p × depth` → 逆外参 C2W → world points。非有限或非正depth的点置零。

缓存 shapes：depth `(N,H,W,1)`、depth/world confidence `(N,H,W)`、world points `(N,H,W,3)`、extrinsic `(N,3,4)`、intrinsic `(N,3,3)`、processed RGB uint8 `(N,H,W,3)`。包含 source尺寸、affine、image IDs及metric信息。`:296` 未压缩 `np.savez` 写 sibling partial；父进程 `src/da3_3d_reconstructor.py:219` 校验后 `os.replace` 发布。DA3重建器不生成GLB；最终GLB属于后续导出。

`np.savez` wrapper只覆盖进入函数后的工作，数组 `.astype()` 等参数求值在外面；runner AST 父span覆盖这部分。诊断临时NPZ只写诊断out，保留给主协调者清理。

## SAM3.1 PVS

`utils/sam3_utils.py:2098` 以checkpoint、device、code root缓存模型；`:2132` canonical cache强制batch4、v2目录，所有frame prompts都必须有对应mask。对象按ID绑定，不凭数组位置猜测；原matching顺序与manifest顺序可不同。

`:2179` 对齐processed image，CPU预取下一ref图像，再调用PVS；`:2007` 用RNG锁包住cache lookup和miss inference，保存/恢复Python、NumPy、torch CPU/CUDA状态。因此cache命中与未命中不会自行改变后续采样随机序列。

`utils/sam31_pvs.py:15` 构建native multiplex tracker；CUDA device context包住construction，因为position encoding caches可能分配在current device。权重只提取tracker与共享vision backbone，strict load；模型FP32、BF16 autocast；禁止CPU。

`predict`（`:54`）步骤：

1. 校验finite positive-area XYXY；空框直接返回空mask，不编码。
2. RGB resize1008×1008，scale到float32并按mean/std0.5归一化；image tensor `(1,3,1008,1008)`，boxes按1008映射。
3. 原生synchronize后起表；BF16下整图 `forward_image(...need_interactive_out=True)` 一次，interactive neck高分辨率features为全部框复用；末级embedding加 `(1,256,1,1)` no-memory embedding。
4. encoder后原生synchronize。M框按4分批，`ceil(M/4)` 次 `_forward_sam_heads`；box corner point inputs `(B,2,2)`，labels2/3；`multimask_output=False`，不带时序multiplex_state。
5. high_res logits bilinear插值回processed H×W；检查finite；阈值>0，D2H bool mask；按box round边界裁剪；取IoU/presence。保留空mask，不重试center point、不填满bbox。
6. 原生final synchronize后stack，输出 `(M,H,W)`。

原生 `encoder_s` 与 `inference_and_postprocess_s` 使用同一个起点，后者包含前者和最终stack，不能相加。每batch的finite布尔判断、`.cpu().numpy()`、score `.cpu().tolist()`本身也可能等待CUDA。

独立PVS诊断读取新主跑29个manifests及相同源图，按manifest processed grid BICUBIC resize、原manifest boxes、batch4。它不重演matching prompt reorder、RNG恢复、cache发布和next-ref prefetch，也不输出新canonical mask cache；这限制了它与matching wall的直接比较。

CUDA events包住encoder和每batch decoder；只在predict已执行原final synchronize返回后调用elapsed_time，不逐batch新增synchronize。events记录stream elapsed，和wall分别输出，不能与CPU wall相加，也不等于所有设备上独占kernel时间。

## 3D matching

`utils/sku_matching_system.py:174` 构造transforms及零存储meta RGB descriptor `(N,3,H,W)`，不把全部RGB解码/H2D。`utils/matching_algorithms.py:369` scene缓存key含NPZ路径、device、desired image IDs；首次reference解包并按ID重排geometry/H2D，后续复用。DA3 transforms使用cache中source_to_processed_affine绑定（`:447`）。

DA3几何默认值来自 `utils/config.py:462`：depth严格介于0.3..8.0m、depth/world confidence>1.5、质心距离最大0.5m、depth consistency0.3m。当前YAML命中率门槛0.3、min3D点10；max3D点默认70、Top-K默认5。

每个目标帧中逐参考物体：

1. `sam3_utils.py:1655` 从参考PVS bool mask中CPU NumPy采最多700候选pixels。按配置可Gaussian加权；round/clamp映射后GPU gather depth/conf/world points，过滤finite与confidence/depth条件；内部至少10点，超70再randperm，返回 `(P,3)`。
2. `matching_algorithms.py:591` 取mean 3D center、参考相机Z均值；Z的`.item()`形成等待。
3. `geometry_3d.py:393` 算 `K E [X,1]`，只留Z>0.1的点，再除Z得到 `(Pvalid,2)`。少于5投影点跳过。
4. target boxes>5时广播 `(P,T)` 统计2D命中，CPU stable ranking选Top5；否则全部候选验证。
5. `geometry_3d.py:428` 先筛命中率≥0.3，再从**目标bbox**取3D点，目标不使用PVS mask。bbox sampler（`:145`）做像素有效性、depth一致性、Gaussian/multinomial；单区域默认上限21点，即int(70×0.3)，最低10。
6. `_fit_plane_svd`（`:20`）把参考/目标点D2H到NumPy做SVD，得到normal与残差。满足参考法向时score=0.5命中率+0.2质心得分+0.3共面得分，否则0.6命中率+0.4质心得分；质心距离>0.5m硬拒绝。
7. `apply_uniqueness_constraint`（`:530`）将所有已验证ref/candidate对按combined score降序贪心分配，每ref/target各一次；被抢占者可取已验证的次优空闲candidate，不重采样。
8. bbox逆映射到原图，保留原object ID，写matching summary供dedup。

PVS空mask对应参考采样为空并跳过；没有mask sampling时才走原bbox/non-overlap采样路径。旧target bbox sampler已有深度重建/采样退化分支，诊断不新增此类行为。

## dedup

`main.py:854` 默认any mode调用 `src/deduplicate_detections.py:319`；读所有reference summaries，校验图像ID、过滤hit ratio。best mode才额外按图片对一对一筛边。

`:555` 给全部 `(image_id,object_idx)` 初始化union-find，包括孤立检测；按hit_ratio降序合并，两个组件如果含相同图片则拒绝，避免一帧多个物体归为一个组件。union-by-rank/path compression；按image/object顺序给组件global ID，最早observation保留，其余removed=True。`:416` 从最终mapping派生逐图survivors，逐图结果与global mapping共享同一决策。聚合classification及global SKU，再写逐图JSON和mapping/global_skus pair。这些为CPU和文件I/O。

## 计时解释与复现

`utils/profiling.py:50` 既有StageTimer仅perf_counter总和/calls，没有CUDA events和父子关系。`cache_npz_load`（matching_algorithms.py:381）只包np.load开archive，实际数组解包在后；`mask_postprocess`当前为空span。`.item()`、D2H/SVD、控制流同步使target_bbox_match等wall包含GPU等待。

新脚本thread-local stack输出inclusive/self/calls/max，并按完整祖先路径保留context。self只扣被同一仪器包住的直接children；wrapper/AST计时本身有开销；未测代码和同步等待仍属于self。返回语句的context退出、GPU异步启动、父子event排队都会影响归因。没有新增逐operator trace、CUDA同步或tensor留存；allocator峰值不reset，为进程生命周期；RSS是Linux ru_maxrss高水位。

运行示例（由协调者串行在GPU2执行，每个out必须不存在）：

```bash
CUDA_VISIBLE_DEVICES=GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python \
  python reports/profiling-20261018/diagnose_gpu.py \
  --mode da3 --base runtime/profiling-20261018 \
  --out runtime/profiling-20261018/diagnostics/da3 \
  --model-path runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
CUDA_VISIBLE_DEVICES=GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python \
  python reports/profiling-20261018/diagnose_gpu.py \
  --mode pvs --base runtime/profiling-20261018 \
  --out runtime/profiling-20261018/diagnostics/pvs \
  --model-path runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```

内部使用逻辑cuda:0；物理GPU由CUDA_VISIBLE_DEVICES限定。DA3默认29真实帧、504、seed42；PVS seed42不等同matching逐ref RNG重放。成功或失败均在新out留下mode-result.json，失败记录traceback和已完成span。out已存在等启动参数错误在输出目录建立前直接拒绝，避免覆盖旧诊断。

独立component耗时不能插入主跑闭合表；constructor⊃子模块构建、forward⊃Nested分支/head、PVS总体⊃encoder/decoder等都是嵌套；分类/Surfel prepare还可能与matching并行。新GPU2与历史GPU0的差异不能归因为某个单独优化。

## 模型内部循环与shape补充：配置推导，非新增hook测量

以下只解释当前factory/config实际建立什么计算。静态shape以B=1、S=29、输入H=504/W=280为例；来自本次输入与源码公式，不表示新增逐层hook记录。两个方向交换时patch网格随之交换。没有为这些细层追加耗时实验。

### DA3：40层anyview与24层metric

配置链为 `Depth-Anything-3/src/depth_anything_3/configs/da3nested-giant-large.yaml` → `da3-giant.yaml` + `da3metric-large.yaml`。`model/dinov2/dinov2.py:47`统一patch14；`vision_transformer.py:427,443`给出ViT-L/G的具体宽度/深度：

| 分支 | Backbone | blocks / heads / width | 取出层（0-based） | head输入宽度 |
|---|---|---|---|---|
| anyview | vitg | 40 / 24 / 1536 | 19,27,33,39 | 3072，local/global拼接 |
| metric | vitl | 24 / 16 / 1024 | 4,11,17,23 | 1024 |

`vision_transformer.py:261` patch embedding把每帧变成720个patch tokens（36×20），再加1个CLS；默认register tokens=0，因此attention前每帧721 tokens。位置编码插值适应实际patch网格。下面伪代码对应`:300–369`，省略norm/pos细节：

```text
x = patch_embed(images)                               # [B,S,720,C]
x = prepend_CLS(x) + interpolated_position             # [B,S,721,C]
for i, block in enumerate(blocks):
    if anyview and i==12 and reference_selection_gate:
        choose saddle_balanced reference; reorder S views
    if anyview and i==13:
        replace token[0] with reference/source camera token
    if anyview and i>=13 and i odd:
        x = block(reshape(x, [B,S*721,C]))              # 跨视角global attention
    else:
        x = block(reshape(x, [B*S,721,C]))              # 各图local attention
        local_x = x
    reshape back to [B,S,721,C]
    if i in out_layers:
        out = concat(local_x,x,-1) if cat_token else x
        restore original view order if reordered
        save patch tokens and separate camera token
```

anyview有14个global blocks（13,15,…,39）及26个local blocks；S=29时global序列长度为20909，而local为29条721序列。这里的“local”是单图token attention，不是SAM的局部window attention。`alt_start=qknorm_start=rope_start=13`；从该层开始按配置启用QK norm/RoPE。metric的三者均为−1，无跨图alternation，24层都按各图处理。`cat_token=True`使anyview的四组输出宽度变3072；metric保持1024。`get_intermediate_layers:371`在返回前去CLS/register tokens，head实际接收 `[1,29,720,Chead]`，camera token另存。

head执行顺序见 `model/da3.py:205`、`dualdpt.py:156,211,273`、`dpt.py:159,209,269`：

1. 两种head默认`chunk_size=8`，把B×S frame维分块；本例29帧为8/8/8/5四块。它们分块的是dense head，不是把整个anyview transformer切成四段。
2. 四个取层tokens分别LayerNorm，reshape `[chunk,Chead,36,20]`，1×1投影到 `[256,512,1024,1024]`通道。
3. 四路分别×4转置卷积、×2转置卷积、identity、stride2卷积，形成 `[144×80,72×40,36×20,18×10]`多尺度网格；这些是源码公式推导值。
4. scratch adapters统一到256通道，从粗到细refinenet4→3→2→1逐级插值并融合skip特征。
5. anyview DualDPT有**独立main与aux两套融合链**；main产生depth/conf，aux产生ray相关输出。`:238`把fused特征插值回目标图尺寸，再projection/activation。camera decoder（`da3.py:211`）读取最后层单独的camera tokens，给出pose encoding，再转intrinsics及world↔camera参数；默认camera路径会移除不再需要的ray/ray_conf。
6. metric DPT使用单套main融合链及其sky输出，再进入Nested尺度对齐。metric配置无cam_dec，因此不执行anyview同样的camera decoder。精确activation/临时buffer与kernel融合不在这些静态网格推导中测量。

### SAM3.1：每图32层encoder，每batch两层two-way decoder

实际PVS路径为 `utils/sam31_pvs.py:22` → `sam31/sam3/model_builder.py:937` → `_create_multiplex_tri_backbone:920` → `_create_vit_backbone:75`。不是更换一个普通SAM image predictor checkpoint，也不在当前predict中进行时序memory传播。

factory固定：1008×1008 RGB、patch14、ViT宽1024、32 blocks、16 heads、MLP ratio4.625；absolute position/RoPE开启，pretrain CLS不保留。patch网格72×72，共5184位置；window_size24，对应每局部window576位置，72×72可分9个window。global blocks为7/15/23/31，其余28层用window attention；`sam31/sam3/model/vitdet.py:739,969`给出window partition与block循环。`compile=False/use_fa3=False/use_rope_real=False`来自当前PVS loader。

`model/necks.py:133,218`的TriViTDetNeck以同一trunk结果生成scale4/2/1三档FPN，每档投影256通道；PVS只请求interactive输出。静态grid为288×288、144×144、72×72。`video_tracking_multiplex.py:1040`的`forward_image`在每图一次执行interactive decoder conv_s0/s1，把前两档分别投影到32/64通道，供所有box batches共享；最后档保持256通道。`utils/sam31_pvs.py:89`给最后档加no-memory embedding。

```text
F = image_encoder(image[1,3,1008,1008])                 # 每图一次，32个ViT blocks
high0 = conv_s0(FPN0)                                  # [1,32,288,288]
high1 = conv_s1(FPN1)                                  # [1,64,144,144]
embed = FPN2 + no_memory                               # [1,256,72,72]
for boxes4 in split(boxes,4):                          # ceil(M/4)次；末批可少于4
    points = two_box_corners(boxes4)                   # [b,2,2], labels2/3
    sparse,dense = interactive_prompt_encoder(points)
    masks4,iou,obj = interactive_mask_decoder(embed, high0,high1,sparse,dense)
    select_native_mask; apply_native_presence_gate
    resize_to_1008_then_processed_hw; threshold; clip_to_box
```

Prompt/decoder细节由 `video_tracking_multiplex.py:557,656,761`、`sam/prompt_encoder.py:164`、`sam/mask_decoder.py:179`、`sam/transformer.py:63,152`确定：

- point接口中`boxes=None`，PromptEncoder会在2个corner points后补一个not-a-point token，故sparse为`[b,3,256]`；没有mask prompt，dense使用learned no-mask embedding扩为`[b,256,72,72]`。
- decoder有object-score token1、IoU token1、mask tokens4（3个multimask+1个single），加3个prompt tokens，共9个sparse tokens；`repeat_image=True`把共享image embedding按batch展开，再加dense embedding。这里b≤4是prompt batch，不是重新执行b次image encoder。
- TwoWayTransformer固定2层、width256、8heads、MLP2048。每层做token self-attention、token→image cross-attention、token MLP、image→token cross-attention；结束另做一次token→image cross-attention。image序列为5184位置，token序列为9；这是配置推导而非新hook实测。
- decoder将72² image embedding两次×2转置卷积升至288²，并分别加high1/high0 skip；输出通道为32。4个mask token各过3层MLP得到32维hypernetwork向量，与flattened上采样embedding矩阵乘，生成`[b,4,288,288]`原始候选mask logits；IoU head深度3、hidden256，并有object presence score。
- `multimask_output=False`表示对外返回一个mask，不等于内部只算一个mask token。native stability/presence选择后`_forward_sam_heads`按image_size=1008插值（`:852`），PVS wrapper再插值回processed H/W，然后bool阈值与bbox裁剪。

以上fine-grained层数/shape解释“每part算什么”；CUDA events覆盖branch/backbone/scaling等聚合边界，head细分来自AST wall，没有给40/24/32个单层或每个attention kernel独立计时。不能从层数、token数直接换算本机耗时或实际attention显存峰值。

# Torch bake数值契约与独立微型验证

本评审只针对实验目录的Torch候选；不修改生产文件、不执行GPU、真实视频或重型基准。生产数值oracle是当前`src/surfel_mesh.py`的默认`source_world`路径；微型fixture复用`test/test_surfel_mesh.py`，延展另以该文件原始`np.roll`reference核对。CPU结果不代表CUDA逐位等价或完整导出提速。

## 1. 输入、工作域和dtype

- 生产入口`src/surfel_mesh.py:215–240`把positions/u/v/scales转float32；四者shape分别为(N,3)/(N,3)/(N,3)/(N,2)，全部finite；frame_indices必须一维整数且在frames范围内。Python整数list也是有效输入。入口在读取texture前拒绝非法输入。
- 每frame按frame ID升序处理，保留该frame原始点序；texture解码为RGB float32，较小图右侧/底部补0到所有frame的最大texture尺寸；depth转float32（`:242–262`）。真实实验还须使用原generation的JPEG和已经量化的f16 depth，不能换成原始高精度数据后声称保留同一输入。
- `_frame_constants`（`:95–103`）建立float32 E/K/invK/affine；q=(arange(T,float32)+0.5)*(2/T)-1，默认T=8，对应64个texel中心，不在圆盘边缘取角点。常量只在本frame的batch间共享。
- 候选`torch_bake.py:60–76`每个FrameBaker上传RGB和depth各一次，保留CPU float32的source_world投影、邻域裁剪、source affine与corner运算。每次bake只接受至多1024点，实验固定tile_size=8；page入口显式只接受source_world。每batch仍会上传processed/camera_z/bounds/source/area/texture坐标并下载keep/tiles，不能把“图上传一次”解释成零batch搬运。
- page入口本轮发现并修复的差异：原候选未将frame_indices转array，list比较会静默得到空slots；float ID还可能int截断。delivery已在`torch_bake.py:157–174`补同生产的shape/finite/ID检查，测试覆盖list和非法ID在asset I/O前拒绝。

## 2. 几何、mask与depth精确顺序

生产`src/surfel_mesh.py:125–189`，候选`torch_bake.py:79–149`：

1. CPU先求center=p@R.T+t，projected=center@K.T；center/projected均finite且各自z>0才保留。无可见点返回空float32 corners `(0,4,3)` 与uint8 tiles `(0,T+2,T+2,4)`。
2. native=max(1e-6,center_z*norm(invK[:,:2],axis=0))；先`_bound_footprint`，再radius*scales，再`_limit_final_diameter`。后者把ellipse奇异值半径限制在3*max(native)，对应最终直径6native（生产`:13,16–61`）；不能把radius缩放移到前面。
3. center像素为floor(projected_xy/projected_z+0.5)，不是ties-to-even。四邻顺序是左/右/上/下。邻居在图内、depth>0且abs(d-center_z-sign*depth_step)<=0.001+0.5*native时连续；否则在对应方向把bounds设为center像素±0.5（生产`:143–153`）。
4. 默认逐texel严格保留world=p+x*bu+y*bv，再camera=world@R.T+t，再projection=camera@K.T的float32运算顺序；projection分母为0时除法输出0。不得用线性代数重排替代并宣称逐位等价（`:158–167`）。
5. Torch mask与生产一致取交集：x²+y²<=1；camera_z>0；processed px/py各自在[0,w-1]/[0,h-1]；四邻裁剪bounds包含端点。depth gather索引floor(px+0.5)、floor(py+0.5)，只为安全读取而clip，clip不使原本越界点变有效。
6. sampled depth必须>0，abs(sampled-camera_z)<=0.015*sampled+0.001；source=processed@affine[:2,:2].T+affine[:2,2]，source的x/y分别在[0,sw-1]/[0,sh-1]且包含端点；cross(bu,bv)的norm必须>0（`:168–179`）。这些均无新增epsilon。
7. 每个surfel至少一个valid texel才保留，顺序不变；完全透明surfel删掉。输出corners按p-bu-bv、p+bu-bv、p+bu+bv、p-bu+bv排列，float32。alpha初始只有0/255，外围一圈gutter透明。

CPU微测明确区分floor(x+0.5)与torch.round：8×8内tile使用radius=4/7，使两侧投影到3.5/4.5，4.5必须读col5。另测试二维source恰好0/8及邻接越界float32、二维depth边界及内部孔、深度阈值两侧相邻float32。CUDA中乘加/比较仍需实际设备验证，不能从CPU通过推定。

## 3. RGB双线性与round

生产`src/surfel_mesh.py:64–71,182–185`，候选`torch_bake.py:21–30,137–144`：

- 首先按float32执行tx=(source_x+0.5)/sw*tw-0.5、ty同理。sw/sh是source_size；tw/th是本frame texture_size。half-pixel坐标不能直接除以padded尺寸，也不能丢掉±0.5。
- `_bilinear`内部clip到传入image的真实W/H，即全frame最大尺寸的padded图；所以较小texture的末端会与零padding混合。source mask只检查source_size，不能用texture_size替代。
- x0/y0=floor坐标转intp；x1/y1=min(x0+1,size-1)。在现有NumPy环境，float32坐标减intp得到float64，fx/fy与RGB插值均为float64。候选显式先转torch.float64再减int64，保留表达式括号顺序。直接float32 gather/`grid_sample`不满足此契约。
- RGB先np.rint（候选torch.round），再转uint8，采用nearest ties-to-even；手算fixture 0.5/1.5/2.5→0/2/2。无新RGB归一化或clamp。RGB虽然在invalid内texel上也先计算，后续已知色延展会覆盖可达unknown；必须比较透明RGB，不能只比不透明像素。
- half-pixel fixture golden是[78,78,100,255]；zero-padding fixture末端混合golden是[120,60,75,255]。微测同时比较完整page RGBA、positions/uv/indices，而非只看这一个像素。

## 4. RGB延展

生产`src/surfel_mesh.py:74–92`，候选`torch_bake.py:33–54`：

- known=alpha!=0。alpha不参与数值插值且整个延展过程不改alpha；直接helper输入允许任意非零alpha，非零值必须原样保留，不能强制变255。
- 至多H+W轮，每轮先复制old=known。方向严格为左→右、右→左、上→下、下→上；本轮source资格只看old，destination资格看实时known。因此本轮新填点不能继续作为source；同一轮多个方向争同一dest，先执行方向获胜。更新RGB并更新known，不移动其他channel，不允许边界wrap。
- 生产有batch级known.all()提前停止，候选固定H+W轮；全部known后后续fill为空，输出相同。全unknown tile保持原始RGB（并非必定全0），alpha不变。空batch、H=0、W=0也必须正确返回同一个Tensor对象。
- 微测使用现有roll reference比较完整RGBA；另有手算多seed fixture，防止in-round flood和方向优先变化。delivery随后把方向RGB填充改成固定shape的torch.where+copy_，避免逐方向boolean动态索引；每surfel keep压缩仍是动态索引，不能据此宣称全部无同步，性能由后续真实计时判断。

## 5. 独立验证结果与边界

执行一次最小CPU测试：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv CUDA_VISIBLE_DEVICES='' \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python \
python -m pytest -q reports/torch-bake-fusion-20261018/test_torch_bake.py
```

首次结果：**30 passed in 4.33s**。随后delivery将延展改成固定shape的torch.where，并将实验tile固定8；只适配受影响的nearest-depth fixture，再执行`-k 'dilation or depth_nearest'`，结果**10 passed, 20 deselected in 4.24s**，没有重跑完整30项。没有GPU运行、性能推断、安装依赖或生产文件修改。覆盖上述bilinear手算/clip/f64、round ties、depth nearest ties、双轴source端点、二维depth孔洞、depth threshold nextafter、half-pixel/零padding、延展old-known/方向优先/任意alpha/全unknown/空维度、空batch与单frame图上传复用、page frame ID契约。

这组测试直接调用当前NumPy生产reference；延展额外使用既有独立roll reference；没有把候选实现抄成测试oracle。FrameBaker复用计数是代码行为检查，不能作为设备传输profiler证据。没有重复已有生产全套测试。

## 6. 推广最小门槛

当前仅可视为CPU小fixture数值通过的实验实现，尚不能默认推广。

1. 协调者在指定CUDA设备上用同一真实generation、原source JPEG与persisted f16 depth比较，保留幸存数量、corner字节和顺序、未配对数量；raw RGBA全量比较包含alpha和透明RGB，不能只比较共同不透明区域。
2. 现有`bake_modes_real_probe.json`是旧CPU projection mode实验：source_world精确一致，camera_basis存在alpha/RGB变化；它不是本Torch实现的验收记录。Torch保留CPU source_world是减少重排风险，不能借用旧结果宣称已验收。
3. 完整导出仍需检查最终palette/PNG后的每页decoded RGBA、GLB几何与文档边界；少量raw RGB变化可能经整页量化放大。若出现差异，应先定位坐标/depth mask/round/延展来源，再决定是否接受算法变化，不新增epsilon或fallback掩盖差异。
4. 性能需包含CUDA/context启动、每frame setup与图/depth上传、全部batch上传/下载、固定轮延展、page/编码与收尾，并明确共享GPU条件；峰值内存和batch<=1024同时报告。不能以采样kernel时间直接扣减完整GLB或pipeline wall。

此次范围内没有尚未解决的CPU微型数值失败；CUDA数值与实际完整导出收益仍未验证。

# Torch bake正式接入：最小方案与生命周期边界

本文件是只读源码调查后的方案，未修改生产代码、运行GPU/测试或部署。本轮仅此文件写入。实验已有单generation数值证据，不把组件44.671011秒直接替换为全链路新耗时。

## 当前调用链与接入位置

1. `scripts/3d/pipeline/video_to_viewer.sh:104–109`设置Viewer输出、classifier device及`CUDA_VISIBLE_DEVICES=$GPU`，同步调用`modules/video_to_dedup/run.sh`；`:111–119`只检查CURRENT和启动可选Vite，**没有GLB调用**。
2. 已授权扩读的`modules/video_to_dedup/run.sh:46–49,103–114`使用uv同步运行本checkout `main.py --mode pipeline`；`:160–163`依次抽帧、检测、pipeline、提取结果，没有后台GPU交付或GLB入口。其直接Python入口就是`main.py`，不存在额外需要读取的包装入口。
3. `main.py:1587–1602`调用`run_complete_pipeline`。`:1077–1080`重建，`:1091–1099`启动CPU Surfel准备线程，`:1106–1109`逐reference matching，`:1117–1124`等待分类并dedup，`:1130–1139`发布最终Viewer，`:1160–1166`drain两个executor。
4. matching路径为`main.py:540–572`把logical device传给`src.inference.main`；后者`:269–291`在同一core Python进程中构建`SKUMatchingSystem`、执行、调用cleanup。**cleanup不是core进程退出的证据**，也不证明全局model缓存/allocator/context均已释放。
5. 当前生产库compact入口是`src/surfel_glb.py:373 export_scene_glb`，default encoder_workers=2、projection_mode=source_world；内部`:220 _export_surfel_glb`读取/LOD，`:298–301`创建page generator，后接原有有界编码与单writer。函数目前不接device，CPU baker默认固定。
6. 旧研究`profile_delivery.py`能体现“core进程退出→独立delivery进程ZIP/GLB”的时序，但它不是生产服务入口；不得把改研究runner称为正式API已接入。Docker独立checkout、服务与部署不在本次授权范围。

最小可靠生命周期是在**现有main.py Python子进程正常退出后**启动独立GLB Python进程，而不是把Torch bake塞进`:1130`之后的同一core进程。video_to_viewer现有同步shell边界天然支持这一点；失败不启动后续GLB，成功后再固定CURRENT对应的generation。

## 推荐的内部代码组织

不要把实验`FrameBaker.bake`的约100行CPU投影直接复制进生产作为第二份算法。先对当前NumPy实现做一次职责明确的拆分：

- `src/surfel_mesh.py`新增私有纯NumPy准备函数，例如`_prepare_surfel_batch(p,u,v,scales,depth,frame,constants,radius,projection_mode)`，保留当前正深度筛选、native/两道footprint、四邻连续性、source_world算术顺序。返回有界的corners、processed_xy、camera_z、bounds、source_xy、area_mask，及保持同一NumPy表达式顺序的texture_xy。空可见点返回明确空结构。
- 现有`_bake_batch`调用该准备函数，再走当前NumPy mask、bilinear、round、dilation，成为CPU端原合同。不得借拆分改变f16/JPEG边界、mask执行顺序、tile/页排序或camera_basis语义。
- 新生产模块建议`src/surfel_mesh_torch.py`只放Torch frame状态、mask/depth gather、float64 bilinear及固定方向where延展；调用上述同一准备函数。canonical image/depth每frame各上传一次，GPU batch≤1024、tile=8，CPU page组装和现有writer保持共用。
- 生产page producer只保留一份frame→batch→page循环，由明确选择的CPU/设备frame处理器提供`bake`。原图读取、padding、常量计算、page容量、_page与释放时机不要再复制到Torch模块。可以先用最小私有CPU frame adapter包装现有_bake_batch，避免大规模类体系。
- Torch模块只在明确GPU分支lazy import；CPU请求不因为导出接口增加参数就导入Torch或初始化CUDA。无需新依赖、编译器、torch.compile或自定义kernel。

共有准备函数仍在CPU，因此本方案是采样/mask/延展步骤整合，不是把完整投影移到GPU，也不是一个融合kernel。camera_basis继续CPU显式实验；本轮Torch分支只接受source_world，其他组合在创建输出目录/CUDA context前拒绝。

## 显式device合同（需root决定后实现）

推荐只增加一个关键字`bake_device: str = 'cpu'`，由`export_scene_glb`传入internal exporter、page producer与frame processor。

- `'cpu'`明确使用既有NumPy路径，保留当前CPU调用结果、支持的tile配置、encoder默认2和无GPU运行合同。
- `'cuda:N'`明确使用Torch设备路径。N为当前进程CUDA_VISIBLE_DEVICES后的logical index，不把宿主机编号直接塞进kernel。最小首版可限定`cuda:0`且在子进程显式隔离一个已授权物理GPU。
- 不设置`auto`、不在OOM/设备不可用/不支持参数时退回CPU、不隐式选择空闲卡。参数非法和不支持组合在mkdir与CUDA初始化前报错；设备失效、OOM、导出失败均传播给调用方，保留失败状态，由原有输出清理规则处理。
- 报告至少增加`bake_device`、实际backend、projection_mode、batch上限；GPU identity由调度者记录physical UUID，库报告logical device，两者不混称。
- 保留`export_surfel_glb`和原始CLI当前CPU默认；是否也给plain入口传播device不是本轮compact接入的必要条件。CPU上的Torch frame processor可作为私有测试接口，不需要再加一个公开backend枚举。

参数名称/默认及实际接入入口属于公共合同变动，待root批准。保持CPU默认不等于自动fallback；GPU路径必须由调用方显式启用。

## 正式入口与研究runner的分工

推荐先完成库接入，再为本checkout视频入口加**显式可选**交付步骤；不默认给所有旧Viewer任务多生成GLB。

- `video_to_viewer.sh`新增可选GLB输出路径与bake-device选项。只有指定输出才在`:109`成功返回、`:111`CURRENT检查之后启动独立compact导出子进程；父shell不执行模型推理、不初始化CUDA。
- delivery子进程必须再次显式设置`CUDA_VISIBLE_DEVICES="$GPU"`，因为当前`:108`的赋值只作用于前一个bash命令，不会自动成为外层shell后续命令的环境。GPU library device传`cuda:0`；不能把physical `--gpu 2`转换成隔离后不存在的`cuda:2`。
- 固定完整generation路径再交付，不让执行过程中第二次读取CURRENT转向其他任务；在exclusive新输出路径生成，保留Viewer原有发布行为。
- 最小CLI选择：给现有GLB模块增加显式compact子模式/device参数，或加一个薄compact CLI wrapper。推荐由root选择薄wrapper或明确子模式之一，禁止混淆现有plain CLI的LOD=False/atlas/global-id合同与compact默认LOD=True。这一步不需要修改main.py的模型执行逻辑。
- 研究完整runner也传同样`bake_device`并保持core→wait→delivery顺序，用于新端到端证据；它不替代正式视频入口的修改。
- 真正HTTP/API若位于Docker独立checkout，必须先单独授权读取/修改；当前设计不声称已覆盖服务的长驻model生命周期或部署。

## ≤14GB峰值预算与执行屏障

建议把用户上限定为本任务owned GPU进程的同时driver占用，而不是含其他任务的整卡memory.used；数字采用保守的14,000,000,000 bytes，若root另定14GiB需显式写清。已有resident进程只排除其显存统计，不停止、迁移或改变它们。

1. core模型进程以及其owned GPU子进程必须退出，随后才能启动GPU delivery；等待返回码0和已管理的子进程退出，失败停止后续阶段。不能用del/empty_cache/cleanup替代进程退出保证。
2. GPU delivery用独立进程，当前frame RGB/depth常驻、batch≤1024、tile=8、CPU atlas FIFO仍2页；导出完成后该进程退出释放context。每frame及时释放Tensor，不能保留整视频GPU图像或全grid texels。
3. 实验README的allocator allocated37.119MiB/reserved54MiB仅支持当前generation的Torch工作集很小，**不含CUDA context/driver占用**，不等于整个任务≤14GB的证明。任务14GB还覆盖前序DA3/PVS阶段；把GLB移到后面只避免相加，并不降低前序模型阶段本身峰值。前轮DA3阶段曾观测15,132MiB，已高于14GB，因此需root其他预算工作独立解决，不能宣称本设计单独达标。
4. 调度前检查实际设备剩余可用空间；预算用owned PID时点和，分别保留整卡峰。200ms采样可漏瞬时峰，报警/杀进程不是分配前硬上限；Torch per-process memory fraction仅限制allocator，也不涵盖所有driver/library分配。不能单用这些机制声称严格硬cap。
5. 若预算阶段无法证明有足够headroom，直接拒绝本轮GPU交付/启动，不在运行错误后换卡或退CPU。允许用户另一次显式选择CPU，但这属于新的明确选择。

上面的预算单位、owned-vs-card定义，以及是否需要全流水严格硬上限，是root必须锁定的接受标准；本设计不改变它们，也不通过省略context或resident混算来规避预算。

## 接入后的最小验证gate

- **纯重构gate：** CPU默认路径先保持同一真实generation的corners/raw RGBA和最终GLB document/accessor/decoded RGBA精确一致；有意义的边界选择集覆盖depth/source nextafter、half-pixel、float64 round、两道长边、延展方向和空批。不要全量重跑无关测试。
- **GPU gate：** 固定同一generation，对共享准备后的CPU和Torch进行全29帧有界sample，raw corners/alpha/透明RGB全部精确相等；再各一次完整CPU/GPU export，所有document/accessor、全部页面decoded RGBA相同。无缺页、shape变化、空comparisons或baked0假通过。既有实验已支持该算法，正式迁移的共用代码重构仍需最小确认。
- **生命周期gate：** 一次实际入口完整运行确认core及owned模型子进程均在delivery启动前退出；新delivery确实选择显式设备；driver时点和与allocator分别记录，前后关键结果一致。测整个新主链路，不把既有组件减少21.918秒直接扣旧186.894秒。
- **失败gate：** 无CUDA/非法mode/不支持tile在输出mkdir前明确拒绝；普通OOM/中途异常保留非零状态，原函数/文件清理与CPU可用性不受影响。测试采用小fixture或模拟失败，不新建GPU故障benchmark。

## 建议改动路径与必须确认的选择

必要生产路径为`src/surfel_mesh.py`（共用准备/唯一page循环）、新`src/surfel_mesh_torch.py`（仅Torch阶段）、`src/surfel_glb.py`（显式device传递与报告）；最小owned测试在现有surfel mesh/GLB测试中扩展。`main.py`/`src/inference.py`原则上无需算法改动；不要在matching结束时插入GPU bake。

实际视频交付另需`scripts/3d/pipeline/video_to_viewer.sh`与一个明确compact CLI入口；root须决定是否在本轮纳入，还是本轮仅正式库API＋研究runner验证并明确尚未接生产视频入口。Docker/API跨checkout与部署均另议。

root需确认四点后才能执行：①`bake_device='cpu'`兼容默认＋调用方显式cuda；②正式入口采用可选视频GLB步骤还是仅库/runner；③compact CLI用薄wrapper还是子模式；④14GB单位/owned峰及前序模型预算的接受标准。除此之外的共享准备数据结构、lazy import和frame释放细节可按以上最小方案实现，不需要额外架构扩张。

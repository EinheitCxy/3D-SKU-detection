# PromptDA / DA3 详细 profiling：d4d5b1bcd2（2026-10-20）

本次从原始 Scanner 视频开始，两条实际重建配置的连续交付时间都约 **23 秒**。DA3 warm 请求更快；PromptDA 输出深度像素为其 4 倍、任务显存明显更低。冷启动、PNG 解码和全量点反投影是明确热点，不能只看网络 forward。

2026-10-20后续：[三位medium subagents的加速路径审查](ACCELERATION_PATHS.md)，细分初始化与首次import、同source-index采样前移、有界CPU预取及服务生命周期。该页是待验证方案，未修改本次profiling结果。

**范围：原始 MP4 + depth/confidence/odometry → 1 Hz 同步输入 → 模型 → CPU dense NPZ → 彩色 POINTS GLB。** 没有运行商品 detection/PVS/matching/dedup、Viewer、纹理 Surfel、footprint、ZIP/COS 或浏览器加载。原始输入没有现成 detections；不伪造标签。当前根 SKU pipeline 的历史 profiling 仍见 [统一入口](../../docs/profiling_breakdown.md)。

## 1. 输入、模型和环境

| 项目 | 本次条件 |
| --- | --- |
| 数据 | fd-videos/d4d5b1bcd2；RGB 1920×1440、1175 帧、MP4 19.85 s、nominal 60 Hz；depth/confidence 各1175 PNG |
| 选帧 | odometry 相对首帧的每个整数秒取最近帧，共20帧，未截断场景 |
| 原始 IDs | [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 958, 1014, 1069, 1123] |
| LiDAR | 原生256×192 uint16毫米→float32米；所有有效值，包括 confidence 0/1/2；不以同一prompt做独立准确率证明 |
| DA3 | DA3NESTED-GIANT-LARGE-1.1 / snapshot b2359bdf726fb44ef62acca04d629dcf158053e7；504×378、20帧联合、原生BF16 autocast、FP32权重；输入Scanner K/E |
| PromptDA | Large/ViT-L；1008×756、FP16 autocast、batch2、FP32权重；输入RGB +原生LiDAR，K/E仅反投影使用；matmul highest，cuDNN TF32保留True |
| GPU | 物理GPU1 / GPU-2f435ae9-398e-4bc1-ded3-ddb0e360ff0c / RTX4090D / driver报49140MiB、Torch报50864390144bytes；容器cuda:0 |
| 共享负载 | GPU1原有PID7262 / 8168MiB服务保留；GPU2有其他用户任务，因此本轮不使用GPU2。CPU/磁盘共享，未设置affinity |
| 环境 | 本地global-id-mapping:cos-background-20261013镜像，无网络、无下载/安装；Python3.11.10、Torch2.7.1+cu126、CUDA12.6、cuDNN90501、NumPy1.26.4、OpenCV4.11.0、Pillow12.3.0、psutil7.2.2；Torch/OMP/OpenBLAS/MKL=8。准备进程复用本地test-env，模型和导出在容器内执行 |
| 代码 | 本地main 0fb9e0ac9979cec2b8fc02124897276585a3071e +已有dirty/untracked实现；本任务仅新增report-local仪器与产物，没有改生产算法或默认值 |

这不是等分辨率、等输入信息的架构排名。PromptDA使用LiDAR且输出4倍像素，DA3做跨20帧联合。两者都保留各自原生筛选规则。

## 2. 连续原视频到 GLB 的真实墙钟

| 区间 / 秒 | DA3 | PromptDA |
| --- | --- | --- |
| 原视频 → NPZ/GLB 完成并进程退出（连续实测） | 22.914 | 23.124 |
| 其中：准备进程（含启动/导入/抽帧/metadata） | 4.501 | 4.312 |
| 其中：重建容器启动 → 退出 | 18.412 | 18.812 |
| 重建函数体：prepared load + model load | 8.825 | 9.195 |
| 重建函数体：cold predict | 2.848 | 3.287 |
| 重建函数体：NPZ 保存 | 2.286 | 0.142 |
| 重建函数体：POINTS GLB 导出与回读 | 0.965 | 2.772 |
| 重建函数体合计 | 14.924 | 15.395 |
| 容器启动/导入/线程设置/退出/监控边界余量 | 3.488 | 3.417 |

每个backend一次新的准备目录和一次新的模型进程，起点在准备进程启动前，终点为该重建容器退出被观察到。NPZ/GLB在退出前已写入并由现有导出函数回读。此表使用同一次连续运行，没有把独立诊断的秒数拼成端到端。准备与容器阶段可相加；函数体子项与其合计不能再重复相加。外围余量含容器/uv/Python导入、线程设置、输出记录、GPU释放、退出及约200ms监控边界，不能全部叫import。未fsync，不是COS可用或浏览器首屏。

另一次 prepared-input 原生冷交付用于核对边界：DA3容器19.210906s/函数体15.686298s，PromptDA17.810743s/14.522150s。它和本表是不同新进程的观测，未混成同一个统计样本。

## 3. 常驻模型的 cold / warm 请求

| candidate | load s | cold s | warm1 s | warm2 s | warm3 s | warm mean s | warm median s | sample SD s | 20帧/median FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| da3 | 9.624584 | 2.900523 | 2.177641 | 1.987355 | 2.035445 | 2.066814 | 2.035445 | 0.098945 | 9.826 |
| promptda | 8.463265 | 3.247647 | 2.577614 | 2.602624 | 2.662421 | 2.614220 | 2.602624 | 0.043576 | 7.685 |

独立baseline各一次模型加载、一次cold与3次warm。请求从prepared PNG路径+resident模型到CPU dense depth和RGB，包含读取、预处理、H2D、网络、D2H、相机/输出打包；窗口前后CUDA同步。warm不重复落盘。load包括该候选首次依赖导入及模型上GPU，和第5节提前导入后的诊断load边界不同。cold不是文件系统冷缓存：没有清页缓存、allocator或模型缓存。3个warm样本只描述这次20帧请求，不支持P95或跨场景结论。

baseline cold请求→NPZ→GLB的连续区间：DA3 **6.158575s**，PromptDA **6.021034s**，排除模型加载/准备/进程启动。不是整条raw pipeline。

## 4. 同步准备的细节

### da3 同一次 raw→GLB 运行中的准备

| 区间 | 次数 | inclusive wall / s |
| --- | --- | --- |
| 准备脚本依赖导入 | 1 | 0.112206 |
| odometry_parse_and_1hz_selection | 1 | 0.012154 |
| ffmpeg_decode_select_png_and_rename | 1 | 3.400517 |
| png_read_decode | 60 | 0.860118 |
| camera_from_csv_row | 20 | 0.001587 |
| camera_npz_write | 1 | 0.014712 |
| metadata_total | 1 | 0.893786 |
| 准备脚本 total | 1 | 4.418684 |
| 准备进程 wall（含启动退出） | 1 | 4.501203 |

### promptda 同一次 raw→GLB 运行中的准备

| 区间 | 次数 | inclusive wall / s |
| --- | --- | --- |
| 准备脚本依赖导入 | 1 | 0.111240 |
| odometry_parse_and_1hz_selection | 1 | 0.012204 |
| ffmpeg_decode_select_png_and_rename | 1 | 3.207751 |
| png_read_decode | 60 | 0.860271 |
| camera_from_csv_row | 20 | 0.001528 |
| camera_npz_write | 1 | 0.014594 |
| metadata_total | 1 | 0.894795 |
| 准备脚本 total | 1 | 4.226006 |
| 准备进程 wall（含启动退出） | 1 | 4.311923 |


metadata_total包含60次PNG读取（每帧RGB/depth/confidence）、shape/dtype检查、毫米转米、20次CSV相机解算、stack、c2w求逆、NPZ和frames.json保存。其子项不可再加到metadata_total。首次共享准备4.620223s；上面两表来自各自完整运行。

算法：read_selected解析1175行CSV并检查连续frame IDs/递增timestamp；对t=0..19逐一扫描时间轴取argmin|time-t|。ffmpeg按原frame ordinal解码全视频，只输出20帧PNG，使用ignore_editlist=1/noautorotate，随后改回原ID。K来自逐帧fx/fy/cx/cy；qxyzw直接构造OpenCV c2w，再取逆成w2c；不额外旋转、不用末帧camera_matrix替代逐帧K。

## 5. 模型加载内部（独立诊断）

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| load.module_to | 1 | 1.972140 | 1.972140 | — |
| load.cpu_network_construct | 1 | 1.562570 | 1.562570 | — |
| load.safetensors_load_model | 1 | 0.956338 | 0.148132 | — |
| load.state_assignment | 1 | 0.682409 | 0.682409 | — |
| load.checkpoint_coverage_audit | 1 | 0.317384 | 0.317384 | — |
| load.safetensors_load_file | 1 | 0.125798 | 0.125798 | — |

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| load.cpu_network_construct_and_checkpoint | 1 | 8.542126 | 0.000919 | — |
| load.dinov2_network_construct | 1 | 7.376584 | 7.376584 | — |
| load.promptda_checkpoint | 1 | 0.894954 | 0.016545 | — |
| load.torch_load_cpu_checkpoint | 1 | 0.778435 | 0.778435 | — |
| load.module_to | 1 | 0.517447 | 0.517447 | — |
| load.dpt_head_construct | 1 | 0.269669 | 0.269669 | — |
| load.state_assignment | 1 | 0.099974 | 0.099974 | — |

DA3诊断load总4.852587s，边界在提前导入API之后；baseline load9.624584s包含首次依赖导入，不能直接相减当优化收益。DA3保留已有selective skip-init、checkpoint coverage审计与严格权重加载：CPU网络构造→覆盖检查→safetensors读取/赋值→to(cuda)。load_model包含load_file和state assignment，父子不相加。

PromptDA诊断load总9.079s。DINOv2构造7.377s、DPT head构造0.270s、checkpoint约0.895s、Module.to约0.517s。CPU cProfile中uniform_累计self **4.120s / 379次**：当前仍有大量随机填充；DINO构造也包含动态源码/import，不把整个7.38s都归为初始化或磁盘。checkpoint的torch.load约0.778s，state assignment约0.100s。后续若跳过被覆盖初始化，必须先审计checkpoint与非持久buffer覆盖；本次未修改。

## 6. DA3 请求内部（独立诊断）

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| request.da3._run_model_forward | 1 | 1.534299 | 0.001551 | 1.534379 |
| request.da3._preprocess_inputs | 1 | 0.414373 | 0.414373 | — |
| request.da3._add_processed_images | 1 | 0.156420 | 0.156420 | — |
| request.da3._align_to_input_extrinsics_intrinsics | 1 | 0.018220 | 0.018220 | — |
| request.da3._prepare_model_inputs | 1 | 0.010885 | 0.010885 | 0.010996 |
| request.da3._convert_to_prediction | 1 | 0.009141 | 0.009141 | 0.009226 |
| request.da3._apply_depth_alignment | 1 | 0.002043 | 0.002043 | 0.002108 |
| request.da3._normalize_extrinsics | 1 | 0.000846 | 0.000846 | 0.000926 |
| request.da3._handle_sky_regions | 1 | 0.000794 | 0.000794 | 0.000854 |
| request.da3._apply_metric_scaling | 1 | 0.000172 | 0.000172 | 0.000222 |

| 主要网络区间 | 次数 | host wall / s | CUDA stream / s |
| --- | --- | --- | --- |
| module.model | 1 | 1.532748 | 1.532791 |
| module.model.da3 | 1 | 1.267203 | 1.267463 |
| module.model.da3.backbone | 1 | 0.800738 | 0.829837 |
| module.model.da3_metric | 1 | 0.261384 | 0.261499 |
| module.model.da3.head | 1 | 0.142170 | 0.275285 |
| module.model.da3_metric.backbone | 1 | 0.033560 | 0.145661 |
| module.model.da3_metric.head | 1 | 0.012275 | 0.114255 |
| module.model.da3.cam_enc | 1 | 0.004172 | 0.004238 |
| module.model.da3.cam_dec | 1 | 0.000446 | 0.000178 |

输入torch形状[1,20,3,378,504]。API依次执行RGB resize/normalize/stack、H2D、外参归一化、Nested网络、CPU Prediction、轨迹对齐、RGB反归一化到uint8。Nested先anyview/giant分支（含camera encoder、40个Transformer block、depth/confidence与camera head），再metric/large分支（24个block及head），之后metric scaling、非sky/confidence mask、采样分位数、最小二乘scale与sky处理。两分支串行。已有AMP策略会让部分head在FP32运行，不能称全网BF16。

本次align_to_input_ext_scale=False仍执行官方Umeyama相机对齐；候选函数随后把E恢复为输入Scanner w2c，保留原生metric depth和processed K。没有再按轨迹比例缩放深度。详细64个block及attention/MLP的计时见CUDA_MODULES.md。

## 7. PromptDA 的逐帧、逐batch请求细节

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.candidates.predict_promptda:182 with torch.no_grad(), autocast: | 10 | 1.100150 | 0.005230 | — |
| line.candidates.predict_promptda:148 with Image.open(image_path) as source: | 20 | 1.030387 | 0.007748 | — |
| line.candidates.predict_promptda:153 rgb = np.asarray(source.convert("RGB")) | 20 | 1.022639 | 1.022639 | — |
| line.candidates.predict_promptda:191 depths.extend(depth[:, 0, :height, :width].float().cpu().numpy().copy()) | 10 | 0.548086 | 0.015319 | — |
| line.candidates.predict_promptda:183 depth = model.predict(image.to(device), depth_input.to(device)) | 10 | 0.546834 | 0.004905 | — |
| line.candidates.predict_promptda:154 pixels = rgb.astype(np.float32) / 255 | 20 | 0.189045 | 0.189045 | — |
| line.candidates.predict_promptda:159 colors.append(np.rint(pixels * 255).clip(0, 255).astype(np.uint8)) | 20 | 0.150983 | 0.150983 | — |
| line.candidates.predict_promptda:156 pixels = cv2.resize( | 20 | 0.066317 | 0.066317 | — |
| line.candidates.predict_promptda:192 return { | 1 | 0.042575 | 0.042575 | — |
| line.candidates.predict_promptda:160 padded = np.pad( | 20 | 0.039891 | 0.039891 | — |
| line.candidates.predict_promptda:175 image = torch.stack(tensors) | 10 | 0.021511 | 0.021511 | — |
| line.candidates.predict_promptda:128 prompt = np.where( | 1 | 0.003627 | 0.003627 | — |
| line.candidates.predict_promptda:163 tensors.append(torch.from_numpy(padded).permute(2, 0, 1)) | 20 | 0.001467 | 0.001467 | — |
| line.candidates.predict_promptda:176 depth_input = torch.cat(prompts) | 10 | 0.001115 | 0.001115 | — |
| line.candidates.predict_promptda:177 autocast = ( | 10 | 0.000948 | 0.000948 | — |
| line.candidates.predict_promptda:142 tensors, prompts = [], [] | 10 | 0.000251 | 0.000251 | — |
| line.candidates.predict_promptda:164 depth_input = torch.from_numpy(native_depth[None, None]) | 20 | 0.000229 | 0.000229 | — |
| line.candidates.predict_promptda:174 prompts.append(depth_input) | 20 | 0.000074 | 0.000074 | — |
| line.candidates.predict_promptda:138 width, height = promptda_grid_size(source_size, max_size) | 1 | 0.000035 | 0.000035 | — |
| line.candidates.predict_promptda:137 source_size = tuple(map(int, sensor["image_size"])) | 1 | 0.000031 | 0.000031 | — |
| line.candidates.predict_promptda:139 pad_width, pad_height = -width % 14, -height % 14 | 1 | 0.000006 | 0.000006 | — |
| line.candidates.predict_promptda:140 depths, colors = [], [] | 1 | 0.000005 | 0.000005 | — |

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| transfer.d2h.depth | 10 | 0.532768 | 0.532768 | 0.016220 |
| transfer.h2d.image | 10 | 0.046013 | 0.046013 | 0.047168 |
| transfer.h2d.depth_input | 10 | 0.001958 | 0.001958 | 0.002410 |

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| request.promptda.predict | 10 | 0.493959 | 0.005917 | 1.012598 |
| request.promptda.rgb_encoder | 10 | 0.390708 | 0.038311 | 0.719334 |
| request.promptda.lidar_minmax_normalize | 10 | 0.002343 | 0.002343 | 0.002845 |
| request.promptda.metric_denormalize | 10 | 0.000628 | 0.000628 | 0.000164 |

循环共10个batch，每batch2帧。每帧：PIL读PNG并convertRGB→float32/255→INTER_AREA到1008×756→保存uint8 RGB（rint/clip）→pad14路径（本尺寸pad=0）→CHW tensor；LiDAR保留原生[1,1,192,256]，未在CPU扩到RGB网格。batch stack成[2,3,756,1008]与[2,1,192,256]，H2D后调用predict。

网络：LiDAR逐样本aminmax归一化；RGB ImageNet normalize；DINOv2-L 24层提取4级特征（每帧54×72=3888 patch tokens，含class token后3889）；DPT projects/resize、scratch卷积、refinenet4→3→2→1，各尺度融入插值LiDAR；最后上采样到756×1008、output_conv和sigmoid，再按LiDAR min/max还原米制深度。每batch马上float().cpu().numpy().copy()；所有depth/RGB最后np.stack，K按半像素resize，E复制Scanner值。

诊断GPU stream interval：RGB encoder约0.719s、DPT约0.288s、predict约1.013s（10batch合计）。PNG解码约1.030s，float32/255约0.189s，RGB打包约0.151s，AREA resize约0.066s，pad约0.040s。**D2H调用wall约0.533s包含等待前面网络完成；CUDA复制区间仅约0.016s，不应把0.53s都叫PCIe复制。** H2D RGB stream约0.047s、LiDAR约0.0024s。

每帧PNG decode见promptda_per_frame_decode.csv；每batch predict/D2H见promptda_per_batch.csv。所有240次block/attention/MLP调用与DPT细层在CUDA_MODULES.md。

## 8. 深度 → POINTS GLB：算法、数量与热点

| candidate | depth [F,H,W] | dense像素 | 筛选后/采样前点 | GLB点 | NPZ方式 | NPZ MiB | GLB MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| da3 | 20×378×504 | 3,810,240 | 2,286,144 | 1,000,000 | deflate | 29.610 | 15.260 |
| promptda | 20×756×1008 | 15,240,960 | 15,240,960 | 1,000,000 | stored | 101.749 | 15.260 |

DA3 valid=finite(depth)>0且finite(conf)，阈值min(max(1.05,p40),p90)，本次2.1430976868；因此3,810,240深度像素保留2,286,144点。PromptDA没有原生confidence，只保留finite-positive，本次15,240,960全有效，不能把该有效性当预测确信度。

export_prediction先对每帧mask取nonzero(u,v)，拼[x,y,1]，camera=(pixel@inv(K).T)*depth，再乘Scanner c2w旋转并加平移，收集RGB；20帧concatenate并检查有限值。**之后**才用default_rng(42).choice(N,1_000_000,replace=False)选择100万点，变换到首相机显示坐标，构造trimesh.PointCloud/Scene，写GLB并一次回读检查point/color/finite。它不是mesh、纹理atlas或Surfel。

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.scanner_io.export_prediction:125 scene.export(str(output_glb)) | 1 | 0.470666 | 0.470666 | — |
| line.scanner_io.export_prediction:107 points, colors = backproject( | 1 | 0.204039 | 0.002928 | — |
| line.scanner_io.export_prediction:119 points, colors = points[indices], colors[indices] | 1 | 0.097443 | 0.097443 | — |
| line.scanner_io.export_prediction:138 return { | 1 | 0.073577 | 0.073577 | — |
| line.scanner_io.export_prediction:105 mask, threshold = prediction_mask(prediction) | 1 | 0.055759 | 0.000046 | — |
| line.scanner_io.backproject:88 colors.append(rgb[y, x]) | 20 | 0.050640 | 0.050640 | — |
| line.scanner_io.backproject:86 camera = (pixels @ np.linalg.inv(k).T) * d[y, x, None] | 20 | 0.044374 | 0.044374 | — |
| line.scanner_io.export_prediction:122 cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors) | 1 | 0.035284 | 0.035284 | — |
| line.scanner_io.export_prediction:118 indices = np.random.default_rng(seed).choice(before, max_points, replace=False) | 1 | 0.031943 | 0.031943 | — |
| line.scanner_io.backproject:85 pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32) | 20 | 0.030013 | 0.030013 | — |
| line.scanner_io.backproject:84 y, x = np.nonzero(mask & np.isfinite(d) & (d > 0)) | 20 | 0.028204 | 0.028204 | — |
| line.scanner_io.backproject:87 points.append(camera @ pose[:3, :3].T + pose[:3, 3]) | 20 | 0.025901 | 0.025901 | — |
| line.scanner_io.export_prediction:127 loaded = trimesh.load(str(output_glb), force="scene", process=False) | 1 | 0.023516 | 0.023516 | — |
| line.scanner_io.backproject:89 return np.concatenate(points).astype(np.float32), np.concatenate(colors) | 1 | 0.021974 | 0.021974 | — |
| line.scanner_io.export_prediction:121 points = points @ display[:3, :3].T + display[:3, 3] | 1 | 0.010978 | 0.010978 | — |
| line.scanner_io.export_prediction:124 scene.add_geometry(cloud, geom_name=output_glb.stem, node_name=output_glb.stem) | 1 | 0.000229 | 0.000229 | — |
| line.scanner_io.export_prediction:123 scene = trimesh.Scene() | 1 | 0.000105 | 0.000105 | — |
| line.scanner_io.export_prediction:120 display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0] | 1 | 0.000098 | 0.000098 | — |
| line.scanner_io.export_prediction:128 geometries = list(loaded.geometry.values()) | 1 | 0.000008 | 0.000008 | — |
| line.scanner_io.export_prediction:114 before = len(points) | 1 | 0.000008 | 0.000008 | — |
| line.scanner_io.backproject:80 points, colors = [], [] | 1 | 0.000006 | 0.000006 | — |
| line.scanner_io.export_prediction:131 restored = geometries[0] | 1 | 0.000003 | 0.000003 | — |

| 区间 | 次数 | inclusive wall / s | exclusive wall / s | CUDA stream / s |
| --- | --- | --- | --- | --- |
| line.scanner_io.export_prediction:107 points, colors = backproject( | 1 | 1.311909 | 0.004522 | — |
| line.scanner_io.export_prediction:125 scene.export(str(output_glb)) | 1 | 0.415129 | 0.415129 | — |
| line.scanner_io.backproject:88 colors.append(rgb[y, x]) | 20 | 0.337411 | 0.337411 | — |
| line.scanner_io.backproject:86 camera = (pixels @ np.linalg.inv(k).T) * d[y, x, None] | 20 | 0.274712 | 0.274712 | — |
| line.scanner_io.backproject:85 pixels = np.column_stack((x, y, np.ones(len(x)))).astype(np.float32) | 20 | 0.264032 | 0.264032 | — |
| line.scanner_io.backproject:87 points.append(camera @ pose[:3, :3].T + pose[:3, 3]) | 20 | 0.158395 | 0.158395 | — |
| line.scanner_io.backproject:89 return np.concatenate(points).astype(np.float32), np.concatenate(colors) | 1 | 0.156758 | 0.156758 | — |
| line.scanner_io.export_prediction:119 points, colors = points[indices], colors[indices] | 1 | 0.125446 | 0.125446 | — |
| line.scanner_io.backproject:84 y, x = np.nonzero(mask & np.isfinite(d) & (d > 0)) | 20 | 0.116073 | 0.116073 | — |
| line.scanner_io.export_prediction:138 return { | 1 | 0.063489 | 0.063489 | — |
| line.scanner_io.export_prediction:118 indices = np.random.default_rng(seed).choice(before, max_points, replace=False) | 1 | 0.053941 | 0.053941 | — |
| line.scanner_io.export_prediction:122 cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors) | 1 | 0.029507 | 0.029507 | — |
| line.scanner_io.export_prediction:105 mask, threshold = prediction_mask(prediction) | 1 | 0.019597 | 0.000029 | — |
| line.scanner_io.export_prediction:127 loaded = trimesh.load(str(output_glb), force="scene", process=False) | 1 | 0.014738 | 0.014738 | — |
| line.scanner_io.export_prediction:121 points = points @ display[:3, :3].T + display[:3, 3] | 1 | 0.011956 | 0.011956 | — |
| line.scanner_io.export_prediction:124 scene.add_geometry(cloud, geom_name=output_glb.stem, node_name=output_glb.stem) | 1 | 0.000169 | 0.000169 | — |
| line.scanner_io.export_prediction:120 display = np.diag([1, -1, -1, 1]).astype(np.float32) @ sensor["w2c"][0] | 1 | 0.000110 | 0.000110 | — |
| line.scanner_io.export_prediction:123 scene = trimesh.Scene() | 1 | 0.000046 | 0.000046 | — |
| line.scanner_io.export_prediction:128 geometries = list(loaded.geometry.values()) | 1 | 0.000007 | 0.000007 | — |
| line.scanner_io.export_prediction:114 before = len(points) | 1 | 0.000006 | 0.000006 | — |
| line.scanner_io.backproject:80 points, colors = [], [] | 1 | 0.000004 | 0.000004 | — |
| line.scanner_io.export_prediction:131 restored = geometries[0] | 1 | 0.000003 | 0.000003 | — |

PromptDA反投影父区间1.311909s；其20帧相机点0.274712s、世界变换0.158395s、RGB gather0.337411s、homogeneous pixels构造0.264032s、concatenate0.156758s都包含在其中。GLB只需100万点，却先算1524万点，这是后续可优先验证的工作量削减方向。若前移采样，须保持同一valid集合、扁平顺序与seed对应的source indices；不能直接改成逐帧定额而宣称显示点集不变。

DA3当前DEFLATE落盘在raw完整运行用2.286015s；PromptDA stored用0.141522s。数组内容没有因压缩改变，但时间和磁盘体积不同；这解释了PromptDA网络与导出更慢而全程仍接近DA3。DA3改stored是待测优化，不在本次悄悄更改。

## 9. 显存与主机内存

| candidate | 加载后allocated GiB | request allocated峰 GiB | request reserved峰 GiB | driver任务采样峰 MiB | 进程RSS采样峰 GiB | 外部树RSS采样峰 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| da3 | 6.358 | 11.909 | 15.838 | 16642 | 13.454 | 12.795 |
| promptda | 1.268 | 3.863 | 4.467 | 4996 | 3.169 | 3.212 |

allocated/reserved是各baseline请求前记录baseline并reset后的绝对峰，含常驻模型；两者不能相加。driver是约200ms外部nvidia-smi采样，只计本容器后代PID，不把原有8168MiB服务算给本任务；频率会漏短峰。进程RSS由100ms内部采样，启动在Torch imports之后；树RSS由外部采样且共享页可能重复计入。两种RSS峰可能不同，不强行当同一窗口。当前请求峰可在requests.csv逐cold/warm检查。

GPU1每次完成后回到原服务占用，未终止其他用户任务。相对历史14帧/GPU2的峰值，本次DA3有20帧且设备显存容量不同，不能把新16,642MiB归因于一次源码回归。没有实施显存压缩或减少帧。

## 10. 插桩影响与算子视角

| candidate | baseline warm median s | source/hooks诊断请求 s | 另一次torch.profiler请求 s |
| --- | --- | --- | --- |
| da3 | 2.035445 | 2.151631 | 2.484758 |
| promptda | 2.602624 | 2.675653 | 3.323739 |

这三个区间属于独立请求，差异含插桩和运行波动，不能把它们混为同一time line。hooks只保留shape/dtype/scalar与CUDA events，不保留tensor，也不每层synchronize。GPU时间为current-stream事件间隔，包含host launch/stream idle，非kernel活跃时间。operator profile另测CPU/CUDA活动和shape，不写巨大trace。算子与底层kernel归因视图有重叠，不能相加。

DA3可见64次FlashAttention kernel与giant MLP/SwiGLU、BF16 GEMM、部分FP32卷积；PromptDA可见240次FlashAttention、FP16 GEMM、DPT卷积和插值，attention已经使用高效实现。详细列表见OPERATORS.md与operator_timings.csv。CPU cudaStreamSynchronize/cudaMemcpyAsync墙钟往往是等待先前GPU工作，不应单列为额外、可无损消除的GPU成本。

## 11. 根据这次证据排序的后续方案

| 优先级 | 方向 | 证据与必须检查的边界 |
| --- | --- | --- |
| 1 | 常驻模型/复用worker | raw冷启动模型/输入加载约9s；先确认服务是否允许resident GPU内存与串行调度，不把本次cold数当warm服务延迟 |
| 2 | PromptDA selective skip-init | load期uniform_ self4.120s；先覆盖审计，保留未被checkpoint覆盖的参数/buffer初始化，再对真实输出做对照 |
| 3 | POINTS采样前移到反投影前 | PromptDA反投影1524万，交付100万；可先从相同valid扁平域选相同indices，仅投影入选点。需证明point/color/source顺序与数值影响；不适用于下游仍需要dense几何的部分 |
| 4 | DA3 NPZ stored/避免同步持久化阻塞反馈 | 当前DEFLATE约2.29s，PromptDA stored0.14s；格式/磁盘体积不同，需同一arrays实测存储方式再决定 |
| 5 | RGB预解码/有界prefetch并和GPU计算重叠 | PromptDA PNG decode约1.03s、串行10batch；可复用prepared RGB或有界预取，但需要计入准备/IPC/RAM，不能把decode移出请求后只报变快 |
| 6 | PromptDA预处理减少重复copy/归一化打包 | float32/255约0.189s、uint8打包0.151s、pad=0仍走np.pad约0.040s；保持网络float32 resize与保存RGB量化边界 |
| 7 | 进一步调batch/compile | 本轮未扫描batch；历史另一段视频不能证明本段最优。encoder已有FlashAttention；不能把开启相同attention再算一次收益 |

以上是待验证方向，本任务没有实施或声称对应加速已经实现；没有独立几何真值、视觉验收或matching/footprint结果。

## 12. 产物、复现和异常记录

[汇总JSON](summary.json) · [全部区间CSV](all_stage_timings.csv) · [逐请求](requests.csv) · [CUDA全部层](CUDA_MODULES.md) · [源码语句明细](SOURCE_SPANS.md) · [算子](OPERATORS.md) · [CPU profiles](cpu_profiles.csv) · [定向核对](validation.json)

[DA3 GLB](../../Output/promptda-da3-d4d5b1bcd2-20261020/da3-full/artifacts/cloud.glb) · [PromptDA GLB](../../Output/promptda-da3-d4d5b1bcd2-20261020/promptda-full-ready/artifacts/cloud.glb)。两者各100万彩色点，保留当前Scanner显示坐标。

原始profile.json、diagnostic.json、run.log、host_samples.json和各次launch-*.json均保留在Output/promptda-da3-d4d5b1bcd2-20261020/。每个launch receipt含实际Docker argv、环境、模型路径、前后GPU占用、PID和exit code。运行需新目录/显式新tag，拒绝覆盖。

源码入口：[同步准备](../../scripts/3d/scanner/prepare_scanner.py) · [模型加载与predict循环](../../scripts/3d/scanner/candidates.py) · [mask/反投影/采样/GLB](../../scripts/3d/scanner/scanner_io.py) · [DA3 API](../../Depth-Anything-3/src/depth_anything_3/api.py) · [DA3 Nested](../../Depth-Anything-3/src/depth_anything_3/model/da3.py) · [本次动态baseline入口](baseline.py) · [诊断仪器](diagnostic.py) · [串行运行与采样](launch.py)。

```bash
# 仓库根目录；复用现有本地test-env和Docker模型环境
UV_CACHE_DIR=/tmp/3d-profile-uv PYTHONDONTWRITEBYTECODE=1 \
uv run --no-project --offline \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/promptda-da3-d4d5b1bcd2-20261020/launch.py \
  --mode baseline --candidate both --tag new
# --mode diagnostic：独立细分诊断；--mode full：连续原视频到GLB；--mode delivery：prepared输入冷交付
# 完成后 summarize.py 从本次已登记目录生成报告；新tag需相应选择新结果，不会自动覆盖基线。
```

共8个实际GPU模型进程成功：两baseline、两diagnostic、两prepared冷交付、两raw完整交付。一次full编排在DA3成功后看到驱动利用率窗口仍为100%，于PromptDA模型启动前中止；当时显存已恢复、后续检查只剩原服务且利用率0%。记录保留；增加最多10s readiness等待后，以独立promptda-full-ready新目录完成该项。它不是PromptDA OOM或推理失败。

没有修改生产参数、原视频/深度/相机数据、Git分支或已有dirty文件；没有安装依赖、联网下载、发布、部署或清理他人文件。仅运行profiling脚本和原生导出已有回读检查，未新增无关全量测试或hash验证。

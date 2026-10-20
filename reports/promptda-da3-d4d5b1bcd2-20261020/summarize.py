"""Build report tables from the retained measurements, without rerunning models."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "Output/promptda-da3-d4d5b1bcd2-20261020"


def read(path):
    return json.loads(path.read_text())


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(map(cell, row)) + " |" for row in rows)])


def event_summary(detail):
    events = detail["events"]
    groups = defaultdict(list)
    for row in events:
        children = sorted((max(child["start"], row["start"]), min(child["end"], row["end"]))
                          for child in events if child["phase"] == row["phase"]
                          and child["parent"] == row["name"]
                          and child["start"] >= row["start"] and child["end"] <= row["end"])
        covered, previous = 0.0, row["start"]
        for start, end in children:
            covered += max(0.0, end - max(previous, start))
            previous = max(previous, end)
        row["wall_self_s"] = max(0.0, row["wall_s"] - covered)
        groups[(row["phase"], row["name"])].append(row)
    result = []
    for (phase, name), rows in groups.items():
        times = [x["wall_s"] for x in rows]
        cuda = [x["cuda_stream_ms"] for x in rows if "cuda_stream_ms" in x]
        result.append(dict(phase=phase, name=name, count=len(rows), wall_sum_s=sum(times),
                           wall_self_sum_s=sum(x["wall_self_s"] for x in rows),
                           wall_min_s=min(times), wall_max_s=max(times),
                           cuda_stream_sum_ms=sum(cuda) if cuda else "",
                           cuda_stream_mean_ms=sum(cuda) / len(cuda) if cuda else ""))
    return sorted(result, key=lambda x: (x["phase"], -x["wall_sum_s"]))


def selected_table(rows, phase, prefix):
    items = [x for x in rows if x["phase"] == phase and x["name"].startswith(prefix)]
    return table(["区间", "次数", "inclusive wall / s", "exclusive wall / s", "CUDA stream / s"],
                 [[x["name"], x["count"], f"{x['wall_sum_s']:.6f}", f"{x['wall_self_sum_s']:.6f}",
                   "—" if x["cuda_stream_sum_ms"] == "" else f"{x['cuda_stream_sum_ms']/1000:.6f}"] for x in items])


def main():
    baselines = {name: read(OUT / f"{name}-baseline/profile.json") for name in ("da3", "promptda")}
    details = {name: read(OUT / f"{name}-diagnostic/diagnostic.json") for name in baselines}
    native = {name: read(OUT / f"{name}-delivery/artifacts/run.json") for name in baselines}
    full_dirs = {"da3": "da3-full", "promptda": "promptda-full-ready"}
    full_runs = {"da3": read(OUT / "launch-full-both.json")["runs"][0],
                 "promptda": read(OUT / "launch-full-promptda-ready.json")["runs"][0]}
    full = {name: read(OUT / directory / "artifacts/run.json") for name, directory in full_dirs.items()}
    prepares = {name: read(OUT / directory / "prepared/preparation_profile.json") for name, directory in full_dirs.items()}
    baseline_launch = {x["candidate"]: x for x in read(OUT / "launch-baseline-both.json")["runs"]}
    summaries = {name: event_summary(detail) for name, detail in details.items()}
    combined_rows = [dict(candidate=name, **row) for name, rows in summaries.items() for row in rows]
    csv_write(HERE / "all_stage_timings.csv", combined_rows)
    csv_write(HERE / "module_timings.csv", [x for x in combined_rows if x["name"].startswith("module.")])
    csv_write(HERE / "operator_timings.csv", [dict(candidate=name, **row) for name, detail in details.items() for row in detail["operator_rows"]])
    csv_write(HERE / "cpu_profiles.csv", [dict(candidate=name, phase=key, **row) for name, detail in details.items()
                                         for key in ("cpu_model_load_top", "cpu_request_top", "cpu_export_top") for row in detail[key]])
    request_rows = []
    for name, baseline in baselines.items():
        for request in baseline["requests"]:
            request_rows.append(dict(candidate=name, label=request["label"], wall_s=request["predict_wall_s"],
                                     **request["gpu_memory"], host_rss_sampled_peak_bytes=request["host_rss"]["observed_peak_bytes"]))
    csv_write(HERE / "requests.csv", request_rows)
    frames = baselines["promptda"]["frame_ids"]
    prompt_events = details["promptda"]["events"]
    decode = [x for x in prompt_events if x["phase"] == "instrumented_request" and "with Image.open(image_path)" in x["name"]]
    csv_write(HERE / "promptda_per_frame_decode.csv", [dict(frame_id=frame_id, wall_s=event["wall_s"])
                                                      for frame_id, event in zip(frames, decode, strict=True)])
    forward = [x for x in prompt_events if x["phase"] == "instrumented_request" and x["name"] == "request.promptda.predict"]
    to_cpu = [x for x in prompt_events if x["phase"] == "instrumented_request" and x["name"] == "transfer.d2h.depth"]
    csv_write(HERE / "promptda_per_batch.csv", [dict(batch=index, frame_ids=frames[index*2:index*2+2],
                 predict_host_wall_s=event["wall_s"], predict_cuda_stream_ms=event["cuda_stream_ms"],
                 depth_cpu_wall_s=copy["wall_s"], depth_copy_cuda_stream_ms=copy["cuda_stream_ms"])
                 for index, (event, copy) in enumerate(zip(forward, to_cpu, strict=True))])
    summary = {"date": "2026-10-20", "input": "fd-videos/d4d5b1bcd2", "source_frames": 1175,
               "selected_frames": 20, "source_rgb_size": [1920, 1440], "frame_ids": frames,
               "scope": "Scanner raw video to native dense predictions and colored POINTS GLB; no SKU matching, Viewer, Surfel, COS or browser milestone",
               "physical_gpu": 1, "git_head": "0fb9e0ac9979cec2b8fc02124897276585a3071e",
               "code_state": "main plus pre-existing dirty/untracked Scanner and PromptDA integration; no production-source edits by this task",
               "low_overhead_baselines": baselines, "native_prepared_delivery": native,
               "continuous_raw_delivery": {name: dict(measurement=full_runs[name], native=full[name], preparation=prepares[name]) for name in full},
               "diagnostic": {name: {key: detail[key] for key in ("status", "instrumented_model_load_wall_s", "instrumented_request_wall_s", "operator_profile_wall_s", "instrumented_points_export_wall_s", "parameter_count", "parameter_bytes")} for name, detail in details.items()},
               "limitations": ["Current production configurations are not equal-resolution or equal-input comparisons.",
                               "Cold is a fresh model/process; filesystem page cache is not flushed.",
                               "One full raw run and three resident warm requests per candidate; no tail-latency or multi-scene inference.",
                               "Driver/RSS peaks are sampled; CPU tree RSS sums can double-count shared pages.",
                               "Fine-grained diagnostic and operator-profile requests are distinct from the baseline.",
                               "No independent geometric accuracy, product matching, footprint or visual-quality acceptance.",
                               "One orchestration stopped at a transient post-exit GPU utilization preflight; retained and completed via explicit new PromptDA attempt."]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    overview = []
    for label, getter, digits in [
        ("原视频 → NPZ/GLB 完成并进程退出（连续实测）", lambda n: full_runs[n]["raw_scan_to_process_exit_wall_s"], 3),
        ("其中：准备进程（含启动/导入/抽帧/metadata）", lambda n: full_runs[n]["preparation_process_wall_s"], 3),
        ("其中：重建容器启动 → 退出", lambda n: full_runs[n]["container_wall_s"], 3),
        ("重建函数体：prepared load + model load", lambda n: full[n]["input_and_model_load_s"], 3),
        ("重建函数体：cold predict", lambda n: full[n]["request_s"], 3),
        ("重建函数体：NPZ 保存", lambda n: full[n]["npz_save_s"], 3),
        ("重建函数体：POINTS GLB 导出与回读", lambda n: full[n]["cloud_export_s"], 3),
        ("重建函数体合计", lambda n: full[n]["total_after_imports_s"], 3),
        ("容器启动/导入/线程设置/退出/监控边界余量", lambda n: full_runs[n]["container_wall_s"] - full[n]["total_after_imports_s"], 3),
    ]:
        overview.append([label, *[f"{getter(name):.{digits}f}" for name in baselines]])
    warm_rows = [[name, f"{b['model_load']['combined_wall_s']:.6f}", f"{b['requests'][0]['predict_wall_s']:.6f}",
                  *[f"{x['predict_wall_s']:.6f}" for x in b["requests"][1:]],
                  f"{b['warm_summary']['mean_s']:.6f}", f"{b['warm_summary']['median_s']:.6f}",
                  f"{b['warm_summary']['sample_stdev_s']:.6f}", f"{20 / b['warm_summary']['median_s']:.3f}"] for name,b in baselines.items()]
    memory_rows = []
    for name, b in baselines.items():
        reqs = b["requests"]
        memory_rows.append([name, f"{b['model_load']['resident_allocated_bytes']/1024**3:.3f}",
                            f"{max(x['gpu_memory']['peak_allocated_bytes'] for x in reqs)/1024**3:.3f}",
                            f"{max(x['gpu_memory']['peak_reserved_bytes'] for x in reqs)/1024**3:.3f}",
                            baseline_launch[name]["task_gpu_sampled_peak_mib"],
                            f"{b['host_rss']['observed_peak_bytes']/1024**3:.3f}",
                            f"{baseline_launch[name]['tree_rss_sampled_peak_bytes']/1024**3:.3f}"])
    points_rows = []
    for name,b in baselines.items():
        export = b["cold_artifacts"]["export"]
        points_rows.append([name, "×".join(map(str,b["processed_shape"])),
                            f"{20*b['processed_shape'][1]*b['processed_shape'][2]:,}",
                            f"{export['points_before_limit']:,}", f"{export['points']:,}",
                            b["npz_compression"], f"{(OUT/f'{name}-baseline/predictions.npz').stat().st_size/1024**2:.3f}",
                            f"{export['bytes']/1024**2:.3f}"])
    prep_tables = []
    for name, prep in prepares.items():
        groups = defaultdict(list)
        for event in prep["events"]:
            groups[event["name"]].append(event["end"]-event["start"])
        prep_tables.append(f"### {name} 同一次 raw→GLB 运行中的准备\n\n" + table(["区间", "次数", "inclusive wall / s"],
                            [["准备脚本依赖导入",1,f"{prep['imports_s']:.6f}"],
                             *[[key,len(values),f"{sum(values):.6f}"] for key,values in groups.items()],
                             ["准备脚本 total",1,f"{prep['total_after_script_imports_s']:.6f}"],
                             ["准备进程 wall（含启动退出）",1,f"{full_runs[name]['preparation_process_wall_s']:.6f}"]])+"\n")
    module_doc = ["# CUDA module detail", "", "独立诊断请求中的 inclusive host wall 与 CUDA stream interval。父/子行有包含关系，不能相加。CPU wall 是主机调用区间；异步模块不能把它解释为 GPU 完成时间。CUDA event 没有逐层 synchronize，最后统一解析。", ""]
    source_doc = ["# Source statement detail", "", "来自当前 native 函数的 AST 包装，源码文件和原行号保留；不改算法语句。exclusive wall 仅扣除已插桩的直接子区间，不等于纯算法 CPU 指令时间。", ""]
    op_doc = ["# PyTorch operator detail", "", "每个模型另执行一次完整请求，用 PyTorch profiler 记录 CPU/CUDA 活动与 input shapes，不记录 tensor、stack 或 memory trace。下面的算子归因与底层 kernel 是重叠视图，不相加；self device 指 profiler 的归因统计，不等同于请求 wall。完整未截断名称见 operator_timings.csv。", ""]
    for name, rows in summaries.items():
        module_doc += ["## " + name, "", selected_table(rows,"instrumented_request","module."), ""]
        for phase in ("model_load","instrumented_request","instrumented_points_export"):
            source_doc += [f"## {name}: {phase}", "", selected_table(rows,phase,""), ""]
        top_ops = sorted(details[name]["operator_rows"],key=lambda x:x["self_device_us"],reverse=True)[:35]
        op_doc += ["## " + name, "", table(["算子 / kernel", "次数", "self CPU / ms", "self device / ms", "input shapes"],
                    [[x["name"][:130],x["calls"],f"{x['self_cpu_us']/1000:.3f}",f"{x['self_device_us']/1000:.3f}",x["input_shapes"]] for x in top_ops]), ""]
    (HERE/"CUDA_MODULES.md").write_text("\n".join(module_doc).rstrip()+"\n")
    (HERE/"SOURCE_SPANS.md").write_text("\n".join(source_doc).rstrip()+"\n")
    (HERE/"OPERATORS.md").write_text("\n".join(op_doc).rstrip()+"\n")

    lines = ["# PromptDA / DA3 详细 profiling：d4d5b1bcd2（2026-10-20）", "",
        "本次从原始 Scanner 视频开始，两条实际重建配置的连续交付时间都约 **23 秒**。DA3 warm 请求更快；PromptDA 输出深度像素为其 4 倍、任务显存明显更低。冷启动、PNG 解码和全量点反投影是明确热点，不能只看网络 forward。", "",
        "2026-10-20后续：[三位medium subagents的加速路径审查](ACCELERATION_PATHS.md)，细分初始化与首次import、同source-index采样前移、有界CPU预取及服务生命周期。该页是待验证方案，未修改本次profiling结果。", "",
        "**范围：原始 MP4 + depth/confidence/odometry → 1 Hz 同步输入 → 模型 → CPU dense NPZ → 彩色 POINTS GLB。** 没有运行商品 detection/PVS/matching/dedup、Viewer、纹理 Surfel、footprint、ZIP/COS 或浏览器加载。原始输入没有现成 detections；不伪造标签。当前根 SKU pipeline 的历史 profiling 仍见 [统一入口](../../docs/profiling_breakdown.md)。", "",
        "## 1. 输入、模型和环境", "",
        table(["项目", "本次条件"], [
            ["数据", "fd-videos/d4d5b1bcd2；RGB 1920×1440、1175 帧、MP4 19.85 s、nominal 60 Hz；depth/confidence 各1175 PNG"],
            ["选帧", "odometry 相对首帧的每个整数秒取最近帧，共20帧，未截断场景"],
            ["原始 IDs", str(frames)],
            ["LiDAR", "原生256×192 uint16毫米→float32米；所有有效值，包括 confidence 0/1/2；不以同一prompt做独立准确率证明"],
            ["DA3", "DA3NESTED-GIANT-LARGE-1.1 / snapshot b2359bdf726fb44ef62acca04d629dcf158053e7；504×378、20帧联合、原生BF16 autocast、FP32权重；输入Scanner K/E"],
            ["PromptDA", "Large/ViT-L；1008×756、FP16 autocast、batch2、FP32权重；输入RGB +原生LiDAR，K/E仅反投影使用；matmul highest，cuDNN TF32保留True"],
            ["GPU", "物理GPU1 / GPU-2f435ae9-398e-4bc1-ded3-ddb0e360ff0c / RTX4090D / driver报49140MiB、Torch报50864390144bytes；容器cuda:0"],
            ["共享负载", "GPU1原有PID7262 / 8168MiB服务保留；GPU2有其他用户任务，因此本轮不使用GPU2。CPU/磁盘共享，未设置affinity"],
            ["环境", "本地global-id-mapping:cos-background-20261013镜像，无网络、无下载/安装；Python3.11.10、Torch2.7.1+cu126、CUDA12.6、cuDNN90501、NumPy1.26.4、OpenCV4.11.0、Pillow12.3.0、psutil7.2.2；Torch/OMP/OpenBLAS/MKL=8。准备进程复用本地test-env，模型和导出在容器内执行"],
            ["代码", "本地main 0fb9e0ac9979cec2b8fc02124897276585a3071e +已有dirty/untracked实现；本任务仅新增report-local仪器与产物，没有改生产算法或默认值"],
        ]), "",
        "这不是等分辨率、等输入信息的架构排名。PromptDA使用LiDAR且输出4倍像素，DA3做跨20帧联合。两者都保留各自原生筛选规则。", "",
        "## 2. 连续原视频到 GLB 的真实墙钟", "",
        table(["区间 / 秒", "DA3", "PromptDA"], overview), "",
        "每个backend一次新的准备目录和一次新的模型进程，起点在准备进程启动前，终点为该重建容器退出被观察到。NPZ/GLB在退出前已写入并由现有导出函数回读。此表使用同一次连续运行，没有把独立诊断的秒数拼成端到端。准备与容器阶段可相加；函数体子项与其合计不能再重复相加。外围余量含容器/uv/Python导入、线程设置、输出记录、GPU释放、退出及约200ms监控边界，不能全部叫import。未fsync，不是COS可用或浏览器首屏。", "",
        "另一次 prepared-input 原生冷交付用于核对边界：DA3容器19.210906s/函数体15.686298s，PromptDA17.810743s/14.522150s。它和本表是不同新进程的观测，未混成同一个统计样本。", "",
        "## 3. 常驻模型的 cold / warm 请求", "",
        table(["candidate","load s","cold s","warm1 s","warm2 s","warm3 s","warm mean s","warm median s","sample SD s","20帧/median FPS"],warm_rows), "",
        "独立baseline各一次模型加载、一次cold与3次warm。请求从prepared PNG路径+resident模型到CPU dense depth和RGB，包含读取、预处理、H2D、网络、D2H、相机/输出打包；窗口前后CUDA同步。warm不重复落盘。load包括该候选首次依赖导入及模型上GPU，和第5节提前导入后的诊断load边界不同。cold不是文件系统冷缓存：没有清页缓存、allocator或模型缓存。3个warm样本只描述这次20帧请求，不支持P95或跨场景结论。", "",
        "baseline cold请求→NPZ→GLB的连续区间：DA3 **6.158575s**，PromptDA **6.021034s**，排除模型加载/准备/进程启动。不是整条raw pipeline。", "",
        "## 4. 同步准备的细节", "",
        *prep_tables, "",
        "metadata_total包含60次PNG读取（每帧RGB/depth/confidence）、shape/dtype检查、毫米转米、20次CSV相机解算、stack、c2w求逆、NPZ和frames.json保存。其子项不可再加到metadata_total。首次共享准备4.620223s；上面两表来自各自完整运行。", "",
        "算法：read_selected解析1175行CSV并检查连续frame IDs/递增timestamp；对t=0..19逐一扫描时间轴取argmin|time-t|。ffmpeg按原frame ordinal解码全视频，只输出20帧PNG，使用ignore_editlist=1/noautorotate，随后改回原ID。K来自逐帧fx/fy/cx/cy；qxyzw直接构造OpenCV c2w，再取逆成w2c；不额外旋转、不用末帧camera_matrix替代逐帧K。", "",
        "## 5. 模型加载内部（独立诊断）", "",
        selected_table(summaries["da3"],"model_load","load."), "",
        selected_table(summaries["promptda"],"model_load","load."), "",
        "DA3诊断load总4.852587s，边界在提前导入API之后；baseline load9.624584s包含首次依赖导入，不能直接相减当优化收益。DA3保留已有selective skip-init、checkpoint coverage审计与严格权重加载：CPU网络构造→覆盖检查→safetensors读取/赋值→to(cuda)。load_model包含load_file和state assignment，父子不相加。", "",
        "PromptDA诊断load总9.079s。DINOv2构造7.377s、DPT head构造0.270s、checkpoint约0.895s、Module.to约0.517s。CPU cProfile中uniform_累计self **4.120s / 379次**：当前仍有大量随机填充；DINO构造也包含动态源码/import，不把整个7.38s都归为初始化或磁盘。checkpoint的torch.load约0.778s，state assignment约0.100s。后续若跳过被覆盖初始化，必须先审计checkpoint与非持久buffer覆盖；本次未修改。", "",
        "## 6. DA3 请求内部（独立诊断）", "",
        selected_table(summaries["da3"],"instrumented_request","request.da3."), "",
        table(["主要网络区间", "次数", "host wall / s", "CUDA stream / s"],
              [[x["name"],x["count"],f"{x['wall_sum_s']:.6f}",f"{x['cuda_stream_sum_ms']/1000:.6f}"] for x in summaries["da3"]
               if x["phase"]=="instrumented_request" and x["name"] in {"module.model","module.model.da3","module.model.da3_metric","module.model.da3.backbone","module.model.da3.head","module.model.da3_metric.backbone","module.model.da3_metric.head","module.model.da3.cam_enc","module.model.da3.cam_dec"}]), "",
        "输入torch形状[1,20,3,378,504]。API依次执行RGB resize/normalize/stack、H2D、外参归一化、Nested网络、CPU Prediction、轨迹对齐、RGB反归一化到uint8。Nested先anyview/giant分支（含camera encoder、40个Transformer block、depth/confidence与camera head），再metric/large分支（24个block及head），之后metric scaling、非sky/confidence mask、采样分位数、最小二乘scale与sky处理。两分支串行。已有AMP策略会让部分head在FP32运行，不能称全网BF16。", "",
        "本次align_to_input_ext_scale=False仍执行官方Umeyama相机对齐；候选函数随后把E恢复为输入Scanner w2c，保留原生metric depth和processed K。没有再按轨迹比例缩放深度。详细64个block及attention/MLP的计时见CUDA_MODULES.md。", "",
        "## 7. PromptDA 的逐帧、逐batch请求细节", "",
        selected_table(summaries["promptda"],"instrumented_request","line.candidates."), "",
        selected_table(summaries["promptda"],"instrumented_request","transfer."), "",
        selected_table(summaries["promptda"],"instrumented_request","request.promptda."), "",
        "循环共10个batch，每batch2帧。每帧：PIL读PNG并convertRGB→float32/255→INTER_AREA到1008×756→保存uint8 RGB（rint/clip）→pad14路径（本尺寸pad=0）→CHW tensor；LiDAR保留原生[1,1,192,256]，未在CPU扩到RGB网格。batch stack成[2,3,756,1008]与[2,1,192,256]，H2D后调用predict。", "",
        "网络：LiDAR逐样本aminmax归一化；RGB ImageNet normalize；DINOv2-L 24层提取4级特征（每帧54×72=3888 patch tokens，含class token后3889）；DPT projects/resize、scratch卷积、refinenet4→3→2→1，各尺度融入插值LiDAR；最后上采样到756×1008、output_conv和sigmoid，再按LiDAR min/max还原米制深度。每batch马上float().cpu().numpy().copy()；所有depth/RGB最后np.stack，K按半像素resize，E复制Scanner值。", "",
        "诊断GPU stream interval：RGB encoder约0.719s、DPT约0.288s、predict约1.013s（10batch合计）。PNG解码约1.030s，float32/255约0.189s，RGB打包约0.151s，AREA resize约0.066s，pad约0.040s。**D2H调用wall约0.533s包含等待前面网络完成；CUDA复制区间仅约0.016s，不应把0.53s都叫PCIe复制。** H2D RGB stream约0.047s、LiDAR约0.0024s。", "",
        "每帧PNG decode见promptda_per_frame_decode.csv；每batch predict/D2H见promptda_per_batch.csv。所有240次block/attention/MLP调用与DPT细层在CUDA_MODULES.md。", "",
        "## 8. 深度 → POINTS GLB：算法、数量与热点", "",
        table(["candidate","depth [F,H,W]","dense像素","筛选后/采样前点","GLB点","NPZ方式","NPZ MiB","GLB MiB"],points_rows), "",
        "DA3 valid=finite(depth)>0且finite(conf)，阈值min(max(1.05,p40),p90)，本次2.1430976868；因此3,810,240深度像素保留2,286,144点。PromptDA没有原生confidence，只保留finite-positive，本次15,240,960全有效，不能把该有效性当预测确信度。", "",
        "export_prediction先对每帧mask取nonzero(u,v)，拼[x,y,1]，camera=(pixel@inv(K).T)*depth，再乘Scanner c2w旋转并加平移，收集RGB；20帧concatenate并检查有限值。**之后**才用default_rng(42).choice(N,1_000_000,replace=False)选择100万点，变换到首相机显示坐标，构造trimesh.PointCloud/Scene，写GLB并一次回读检查point/color/finite。它不是mesh、纹理atlas或Surfel。", "",
        selected_table(summaries["da3"],"instrumented_points_export","line.scanner_io."), "",
        selected_table(summaries["promptda"],"instrumented_points_export","line.scanner_io."), "",
        "PromptDA反投影父区间1.311909s；其20帧相机点0.274712s、世界变换0.158395s、RGB gather0.337411s、homogeneous pixels构造0.264032s、concatenate0.156758s都包含在其中。GLB只需100万点，却先算1524万点，这是后续可优先验证的工作量削减方向。若前移采样，须保持同一valid集合、扁平顺序与seed对应的source indices；不能直接改成逐帧定额而宣称显示点集不变。", "",
        "DA3当前DEFLATE落盘在raw完整运行用2.286015s；PromptDA stored用0.141522s。数组内容没有因压缩改变，但时间和磁盘体积不同；这解释了PromptDA网络与导出更慢而全程仍接近DA3。DA3改stored是待测优化，不在本次悄悄更改。", "",
        "## 9. 显存与主机内存", "",
        table(["candidate","加载后allocated GiB","request allocated峰 GiB","request reserved峰 GiB","driver任务采样峰 MiB","进程RSS采样峰 GiB","外部树RSS采样峰 GiB"],memory_rows), "",
        "allocated/reserved是各baseline请求前记录baseline并reset后的绝对峰，含常驻模型；两者不能相加。driver是约200ms外部nvidia-smi采样，只计本容器后代PID，不把原有8168MiB服务算给本任务；频率会漏短峰。进程RSS由100ms内部采样，启动在Torch imports之后；树RSS由外部采样且共享页可能重复计入。两种RSS峰可能不同，不强行当同一窗口。当前请求峰可在requests.csv逐cold/warm检查。", "",
        "GPU1每次完成后回到原服务占用，未终止其他用户任务。相对历史14帧/GPU2的峰值，本次DA3有20帧且设备显存容量不同，不能把新16,642MiB归因于一次源码回归。没有实施显存压缩或减少帧。", "",
        "## 10. 插桩影响与算子视角", "",
        table(["candidate","baseline warm median s","source/hooks诊断请求 s","另一次torch.profiler请求 s"],
              [[name,f"{baselines[name]['warm_summary']['median_s']:.6f}",f"{details[name]['instrumented_request_wall_s']:.6f}",f"{details[name]['operator_profile_wall_s']:.6f}"] for name in baselines]), "",
        "这三个区间属于独立请求，差异含插桩和运行波动，不能把它们混为同一time line。hooks只保留shape/dtype/scalar与CUDA events，不保留tensor，也不每层synchronize。GPU时间为current-stream事件间隔，包含host launch/stream idle，非kernel活跃时间。operator profile另测CPU/CUDA活动和shape，不写巨大trace。算子与底层kernel归因视图有重叠，不能相加。", "",
        "DA3可见64次FlashAttention kernel与giant MLP/SwiGLU、BF16 GEMM、部分FP32卷积；PromptDA可见240次FlashAttention、FP16 GEMM、DPT卷积和插值，attention已经使用高效实现。详细列表见OPERATORS.md与operator_timings.csv。CPU cudaStreamSynchronize/cudaMemcpyAsync墙钟往往是等待先前GPU工作，不应单列为额外、可无损消除的GPU成本。", "",
        "## 11. 根据这次证据排序的后续方案", "",
        table(["优先级","方向","证据与必须检查的边界"],[
            [1,"常驻模型/复用worker","raw冷启动模型/输入加载约9s；先确认服务是否允许resident GPU内存与串行调度，不把本次cold数当warm服务延迟"],
            [2,"PromptDA selective skip-init","load期uniform_ self4.120s；先覆盖审计，保留未被checkpoint覆盖的参数/buffer初始化，再对真实输出做对照"],
            [3,"POINTS采样前移到反投影前","PromptDA反投影1524万，交付100万；可先从相同valid扁平域选相同indices，仅投影入选点。需证明point/color/source顺序与数值影响；不适用于下游仍需要dense几何的部分"],
            [4,"DA3 NPZ stored/避免同步持久化阻塞反馈","当前DEFLATE约2.29s，PromptDA stored0.14s；格式/磁盘体积不同，需同一arrays实测存储方式再决定"],
            [5,"RGB预解码/有界prefetch并和GPU计算重叠","PromptDA PNG decode约1.03s、串行10batch；可复用prepared RGB或有界预取，但需要计入准备/IPC/RAM，不能把decode移出请求后只报变快"],
            [6,"PromptDA预处理减少重复copy/归一化打包","float32/255约0.189s、uint8打包0.151s、pad=0仍走np.pad约0.040s；保持网络float32 resize与保存RGB量化边界"],
            [7,"进一步调batch/compile","本轮未扫描batch；历史另一段视频不能证明本段最优。encoder已有FlashAttention；不能把开启相同attention再算一次收益"],
        ]), "",
        "以上是待验证方向，本任务没有实施或声称对应加速已经实现；没有独立几何真值、视觉验收或matching/footprint结果。", "",
        "## 12. 产物、复现和异常记录", "",
        "[汇总JSON](summary.json) · [全部区间CSV](all_stage_timings.csv) · [逐请求](requests.csv) · [CUDA全部层](CUDA_MODULES.md) · [源码语句明细](SOURCE_SPANS.md) · [算子](OPERATORS.md) · [CPU profiles](cpu_profiles.csv) · [定向核对](validation.json)", "",
        "[DA3 GLB](../../Output/promptda-da3-d4d5b1bcd2-20261020/da3-full/artifacts/cloud.glb) · [PromptDA GLB](../../Output/promptda-da3-d4d5b1bcd2-20261020/promptda-full-ready/artifacts/cloud.glb)。两者各100万彩色点，保留当前Scanner显示坐标。", "",
        "原始profile.json、diagnostic.json、run.log、host_samples.json和各次launch-*.json均保留在Output/promptda-da3-d4d5b1bcd2-20261020/。每个launch receipt含实际Docker argv、环境、模型路径、前后GPU占用、PID和exit code。运行需新目录/显式新tag，拒绝覆盖。", "",
        "源码入口：[同步准备](../../scripts/3d/scanner/prepare_scanner.py) · [模型加载与predict循环](../../scripts/3d/scanner/candidates.py) · [mask/反投影/采样/GLB](../../scripts/3d/scanner/scanner_io.py) · [DA3 API](../../Depth-Anything-3/src/depth_anything_3/api.py) · [DA3 Nested](../../Depth-Anything-3/src/depth_anything_3/model/da3.py) · [本次动态baseline入口](baseline.py) · [诊断仪器](diagnostic.py) · [串行运行与采样](launch.py)。", "",
        "```bash\n# 仓库根目录；复用现有本地test-env和Docker模型环境\nUV_CACHE_DIR=/tmp/3d-profile-uv PYTHONDONTWRITEBYTECODE=1 \\\nuv run --no-project --offline \\\n  --python runtime/viewer-latency-20261017/test-env/bin/python python \\\n  reports/promptda-da3-d4d5b1bcd2-20261020/launch.py \\\n  --mode baseline --candidate both --tag new\n# --mode diagnostic：独立细分诊断；--mode full：连续原视频到GLB；--mode delivery：prepared输入冷交付\n# 完成后 summarize.py 从本次已登记目录生成报告；新tag需相应选择新结果，不会自动覆盖基线。\n```", "",
        "共8个实际GPU模型进程成功：两baseline、两diagnostic、两prepared冷交付、两raw完整交付。一次full编排在DA3成功后看到驱动利用率窗口仍为100%，于PromptDA模型启动前中止；当时显存已恢复、后续检查只剩原服务且利用率0%。记录保留；增加最多10s readiness等待后，以独立promptda-full-ready新目录完成该项。它不是PromptDA OOM或推理失败。", "",
        "没有修改生产参数、原视频/深度/相机数据、Git分支或已有dirty文件；没有安装依赖、联网下载、发布、部署或清理他人文件。仅运行profiling脚本和原生导出已有回读检查，未新增无关全量测试或hash验证。", "",
    ]
    (HERE / "README.md").write_text("\n".join(lines).rstrip()+"\n")
    print(json.dumps({"report": str(HERE / 'README.md'), "raw_da3_s": full_runs['da3']['raw_scan_to_process_exit_wall_s'],
                      "raw_promptda_s": full_runs['promptda']['raw_scan_to_process_exit_wall_s'],
                      "csv_timing_rows": len(combined_rows)},indent=2))


if __name__ == "__main__":
    main()

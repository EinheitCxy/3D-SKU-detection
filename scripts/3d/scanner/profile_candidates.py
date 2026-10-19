"""Profile one Scanner candidate in a new process: one cold and three warm requests.

The request starts with prepared PNG paths and a resident model, and finishes
with dense predictions and processed RGB on CPU. It includes image loading,
preprocessing, transfers and output packing performed by predict_candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone


SCRIPT_STARTED = time.perf_counter()
SEED = 42
WARM_REQUESTS = 3
RSS_INTERVAL_S = 0.1


class RSSSampler:
    """Observed RSS peaks for this process only, sampled every 100 ms."""

    def __init__(self, process):
        self.process = process
        self.phase = "setup"
        self.phases = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        with self.lock:
            rss = self.process.memory_info().rss
            values = self.phases.setdefault(
                self.phase,
                {"baseline_bytes": rss, "observed_peak_bytes": rss, "samples": 0},
            )
            values["observed_peak_bytes"] = max(values["observed_peak_bytes"], rss)
            values["samples"] += 1

    def set_phase(self, name):
        self.sample()
        with self.lock:
            self.phase = name
        self.sample()

    def _run(self):
        while not self.stop_event.wait(RSS_INTERVAL_S):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def phase_stats(self, name):
        with self.lock:
            return dict(self.phases[name])

    def summary(self):
        with self.lock:
            phases = {name: dict(values) for name, values in self.phases.items()}
        return {
            "scope": "This PID only, after sampler startup; no child-process RSS or pre-sampler import peak",
            "sampling_interval_s": RSS_INTERVAL_S,
            "boundary_samples": True,
            "observed_peak_bytes": max(
                values["observed_peak_bytes"] for values in phases.values()
            ),
            "phases": phases,
            "limitation": "Observed sampled peaks can miss brief allocation spikes",
        }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Prepared camera_inputs.npz and frames/ directory",
    )
    parser.add_argument("--candidate", required=True, choices=("da3", "promptda"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--promptda-repo", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="Per-candidate profiling directory"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def render_report(report):
    requests = report["requests"]
    artifacts = report["cold_artifacts"]
    lines = [
        f"# Scanner candidate profiling: {report['candidate']}",
        "",
        "模型已常驻、准备好的 PNG 路径 → CPU dense predictions，包含候选实现内部的 RGB 读取、"
        "预处理、H2D、推理、D2H 和 RGB 打包；不等同于 RAM 中 RGB 输入或纯模型前向。",
        "",
        f"输入：{report['frame_count']} 帧，{report['processed_shape'][2]}×{report['processed_shape'][1]}，"
        f"seed={SEED}。执行精度：{report['native_precision']}。",
        "",
        f"模型加载（load_candidate 至 GPU 同步完成）：{report['model_load']['combined_wall_s']:.4f} s，"
        "只计一次，未拆估计 CPU 与 H2D。",
        "",
        "| 请求 | 完整 predict / s | GPU allocated baseline / GiB | GPU allocated peak / GiB | GPU reserved peak / GiB | sampled RSS peak / GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for request in requests:
        gpu = request["gpu_memory"]
        lines.append(
            f"| {request['label']} | {request['predict_wall_s']:.4f} | "
            f"{gpu['baseline_allocated_bytes'] / 1024**3:.3f} | {gpu['peak_allocated_bytes'] / 1024**3:.3f} | "
            f"{gpu['peak_reserved_bytes'] / 1024**3:.3f} | {request['host_rss']['observed_peak_bytes'] / 1024**3:.3f} |"
        )
    warm = report["warm_summary"]
    lines += [
        "",
        f"Warm 完整请求 n={warm['n']}：mean={warm['mean_s']:.4f} s，median={warm['median_s']:.4f} s，"
        f"min/max={warm['min_s']:.4f}/{warm['max_s']:.4f} s，sample SD={warm['sample_stdev_s']:.4f} s。"
        "三个样本不用于推断尾延迟。",
        "",
        f"Cold NPZ 压缩保存：{artifacts['npz_save_s']:.4f} s；"
        f"GLB 反投影、候选筛选、预算采样、写入及一次回读：{artifacts['glb_export_and_readback_s']:.4f} s。",
        "",
        f"一次 cold predict → NPZ 保存 → GLB 就绪的实际连续墙钟："
        f"**{artifacts['cold_request_through_artifacts_wall_s']:.4f} s**，包含检查与控制开销，"
        "不含模型加载、准备数据读取和进程启动。warm 请求不重复写 NPZ 或 GLB。",
        "",
        "GPU peaks 为各 predict 窗口前记录基线并 reset 后的绝对峰值，包含模型；"
        "allocated 与 reserved 独立报告，不能相加。RSS 每 100 ms 采样当前 PID，可能漏短峰值。",
        "",
        "新进程不表示文件冷缓存；未清空系统页缓存、CUDA allocator 或模型缓存。"
        "cold 保存与 CPU 导出完成后再执行三个 warm 请求；模型全程复用，不添加内部诊断 hook。",
        "",
        "两候选保留各自原生计算精度与批处理方式，以及 DA3 native confidence 筛选 / "
        "PromptDA finite-positive 筛选。NPZ 字段和入选点数可能不同，IO 时间不代表纯网络成本。",
        "",
        "详见 [profile.json](profile.json)，预测仅保存为 predictions.npz，点云仅导出为 scene.glb。",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    import numpy as np
    import psutil
    import torch

    sys.path.insert(1, str(Path(__file__).resolve().parents[3]))
    from candidates import load_candidate, predict_candidate
    from scanner_io import export_prediction, load_prepared

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This profiling protocol requires CUDA")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.candidate == "promptda" and args.promptda_repo is None:
        raise ValueError("--promptda-repo is required for PromptDA")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("profile.json", "profiling.md", "predictions.npz", "scene.glb"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    torch.set_num_threads(8)
    sampler = RSSSampler(psutil.Process())
    sampler.start()
    report = {
        "status": "running",
        "candidate": args.candidate,
        "checkpoint": str(args.checkpoint),
        "promptda_repo": None
        if args.promptda_repo is None
        else str(args.promptda_repo),
        "base": str(args.base),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "seed": SEED,
        "native_precision": "DA3 native BF16 autocast; FP32 parameters; joint multi-view request"
        if args.candidate == "da3"
        else "PromptDA official FP32; sequential per-frame predict; full raw valid LiDAR",
        "cache_state": "One model load per CLI process; no external warmup; filesystem page cache unknown/unflushed; allocator/model caches retained between requests",
        "predict_scope": "Resident model + prepared PNG paths and sensor arrays through dense predictions and processed RGB on CPU; caller synchronizes GPU before and after predict_candidate",
        "gpu_memory_scope": "Per-predict absolute allocated/reserved peaks include model; baseline sampled then peak counters reset before request; no cache emptying",
        "requests": [],
    }
    report_path = args.output / "profile.json"
    try:
        prepared_started = time.perf_counter()
        images, sensor = load_prepared(args.base)
        if len(images) != 14 or len(sensor["frame_ids"]) != 14:
            raise ValueError("This comparison requires the same prepared 14-frame scan")
        report["prepared_input_load_s"] = time.perf_counter() - prepared_started
        report["frame_count"] = len(images)
        report["frame_ids"] = np.asarray(sensor["frame_ids"]).tolist()
        report["cpu_threads"] = torch.get_num_threads()

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        sampler.set_phase("model_load")
        load_started = time.perf_counter()
        model = load_candidate(
            args.candidate,
            str(args.checkpoint),
            str(device),
            promptda_repo=None
            if args.promptda_repo is None
            else str(args.promptda_repo),
        )
        torch.cuda.synchronize(device)
        loaded_at = time.perf_counter()
        sampler.sample()
        report["model_load"] = {
            "combined_wall_s": loaded_at - load_started,
            "resident_allocated_bytes": torch.cuda.memory_allocated(device),
            "resident_reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes_since_cuda_init": torch.cuda.max_memory_allocated(
                device
            ),
            "peak_reserved_bytes_since_cuda_init": torch.cuda.max_memory_reserved(
                device
            ),
            "host_rss": sampler.phase_stats("model_load"),
        }
        properties = torch.cuda.get_device_properties(device)
        report["environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "logical_device": str(device),
            "device_name": properties.name,
            "device_total_memory_bytes": properties.total_memory,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        }
        print(
            f"{args.candidate}: model loaded in {report['model_load']['combined_wall_s']:.3f}s",
            flush=True,
        )

        for index in range(1 + WARM_REQUESTS):
            label = "cold" if index == 0 else f"warm_{index}"
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.synchronize(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            sampler.set_phase(label)
            started = time.perf_counter()
            prediction = predict_candidate(
                args.candidate, model, images, sensor, str(device)
            )
            torch.cuda.synchronize(device)
            predicted_at = time.perf_counter()
            sampler.sample()
            gpu = {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            gpu["peak_allocated_increase_bytes"] = (
                gpu["peak_allocated_bytes"] - baseline_allocated
            )
            gpu["peak_reserved_increase_bytes"] = (
                gpu["peak_reserved_bytes"] - baseline_reserved
            )
            request = {
                "label": label,
                "predict_wall_s": predicted_at - started,
                "gpu_memory": gpu,
                "host_rss": sampler.phase_stats(label),
            }
            report["requests"].append(request)

            if index == 0:
                sampler.set_phase("cold_output_check")
                depth = prediction["depth"]
                if (
                    not isinstance(depth, np.ndarray)
                    or depth.dtype != np.float32
                    or depth.shape != (14, 378, 504)
                ):
                    raise ValueError(
                        "Expected CPU float32 depth with shape (14,378,504)"
                    )
                if not np.all(np.isfinite(depth) & (depth > 0)):
                    raise ValueError("Cold prediction depth is not finite and positive")
                if not np.array_equal(prediction["frame_ids"], sensor["frame_ids"]):
                    raise ValueError("Cold prediction frame order changed")
                report["processed_shape"] = list(depth.shape)
                report["npz_field_shapes"] = {
                    key: list(value.shape)
                    for key, value in prediction.items()
                    if isinstance(value, np.ndarray)
                }
                sampler.set_phase("cold_npz_save")
                save_started = time.perf_counter()
                np.savez_compressed(args.output / "predictions.npz", **prediction)
                npz_save_s = time.perf_counter() - save_started
                sampler.set_phase("cold_glb_export")
                export_started = time.perf_counter()
                exported = export_prediction(
                    prediction,
                    sensor,
                    args.output / "scene.glb",
                    max_points=1_000_000,
                    seed=SEED,
                )
                exported_at = time.perf_counter()
                report["cold_artifacts"] = {
                    "npz_save_s": npz_save_s,
                    "glb_export_and_readback_s": exported_at - export_started,
                    "cold_request_through_artifacts_wall_s": exported_at - started,
                    "scope": "One continuous cold predict, output check, NPZ save, GLB export/readback; excludes model load and prepared metadata load",
                    "export": exported,
                }
                del depth
            del prediction
            sampler.set_phase("between_requests")
            print(
                f"{args.candidate} {label}: full predict={request['predict_wall_s']:.4f}s; "
                f"allocated peak={gpu['peak_allocated_bytes'] / 1024**3:.3f}GiB; "
                f"reserved peak={gpu['peak_reserved_bytes'] / 1024**3:.3f}GiB",
                flush=True,
            )
            write_json(report_path, report)

        durations = [request["predict_wall_s"] for request in report["requests"][1:]]
        report["warm_summary"] = {
            "n": len(durations),
            "mean_s": statistics.mean(durations),
            "median_s": statistics.median(durations),
            "min_s": min(durations),
            "max_s": max(durations),
            "sample_stdev_s": statistics.stdev(durations),
            "scope": "Three complete resident-model requests; no duplicate NPZ or GLB writes; no tail percentile claims",
        }
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        sampler.stop()
        report["host_rss"] = sampler.summary()
        report["script_wall_before_final_report_s"] = (
            time.perf_counter() - SCRIPT_STARTED
        )
        write_json(report_path, report)
    (args.output / "profiling.md").write_text(render_report(report))
    print(args.output / "profiling.md", flush=True)


if __name__ == "__main__":
    main()

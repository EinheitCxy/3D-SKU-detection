"""Coordinator-only serial experiment launcher, existing local image, no network."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Output/promptda-acceleration-validation-20261020"
REPORT = "/workspace/reports/promptda-acceleration-validation-20261020"
SOURCE = "/workspace/Output/promptda-da3-d4d5b1bcd2-20261020"
IMAGE = "global-id-mapping:cos-background-20261013"
GPU_UUID = "GPU-2f435ae9-398e-4bc1-ded3-ddb0e360ff0c"


def snapshot():
    result = {}
    for key, query in (("devices", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu"),
                       ("apps", "--query-compute-apps=gpu_uuid,pid,used_memory")):
        result[key] = subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True).strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("load", "input", "fault", "sampling", "requests", "evaluate", "delivery-native", "delivery-optimized"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--raw", action="store_true", help="For delivery tasks, include new raw-video preparation in the continuous window")
    args = parser.parse_args()
    import psutil

    if args.raw and not args.task.startswith("delivery-"):
        raise ValueError("--raw is only valid for delivery experiments")
    label = args.task + ("-raw" if args.raw else "") + ("-" + args.tag if args.tag else "")
    target = OUT / label
    target.mkdir(parents=True, exist_ok=False)
    gpu = args.task in ("requests", "delivery-native", "delivery-optimized")
    before = snapshot()
    if gpu:
        row = next(row for row in csv.reader(before["devices"].splitlines(), skipinitialspace=True) if row[1] == GPU_UUID)
        if int(row[3]) < 10000:
            raise RuntimeError(f"Selected GPU has insufficient free memory: {row}")
    if psutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError("Root disk below3GiB")
    name = f"codex-promptda-ab-{label}-{os.getpid()}"
    pipeline_started = time.perf_counter()
    base = f"{SOURCE}/prepared"
    preparation_wall_s = None
    if args.raw:
        prepared = target / "prepared"
        prepare_argv = ["/home/xingyu/.local/bin/uv", "run", "--no-project", "--offline",
                        "--python", str(ROOT / "runtime/viewer-latency-20261017/test-env/bin/python"), "python",
                        str(ROOT / "reports/promptda-da3-d4d5b1bcd2-20261020/prepare_profile.py"),
                        str(ROOT / "fd-videos/d4d5b1bcd2"), str(prepared)]
        with (target / "preparation.log").open("w") as log:
            subprocess.run(prepare_argv, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8"))
        preparation_wall_s = time.perf_counter() - pipeline_started
        base = "/workspace/" + str(prepared.relative_to(ROOT))
    argv = ["docker", "run", "--rm", "--name", name, "--network", "none", "--shm-size", "1g",
            "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "24g",
            "-v", "/home/xingyu/.local/bin/uv:/usr/local/bin/uv:ro", "-v", f"{ROOT}:/workspace:ro",
            "-v", f"{target}:/profile:rw", "-w", "/workspace"]
    if gpu:
        argv.extend(["--gpus", "device=1"])
    for key, value in {"UV_CACHE_DIR": "/tmp/uv", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                       "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                       "MPLCONFIGDIR": "/tmp/mpl"}.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["--entrypoint", "/usr/local/bin/uv", IMAGE,
                 "run", "--no-project", "--offline", "--python", "/app/.venv/bin/python", "python"])
    checkpoint = "/workspace/runtime/models/promptda/vitl/model.ckpt"
    repo = "/workspace/runtime/promptda-counterpart-20261019/PromptDA"
    if args.task == "load":
        argv.extend([f"{REPORT}/load_probe.py", "--repo", repo, "--checkpoint", checkpoint, "--output", "/profile/result.json"])
    elif args.task == "sampling":
        argv.extend([f"{REPORT}/sampling_probe.py", "--prediction", f"{SOURCE}/promptda-full-ready/artifacts/predictions.npz",
                     "--sensor", f"{SOURCE}/promptda-full-ready/prepared/camera_inputs.npz", "--include-float32-pixels", "--output", "/profile/result.json"])
    elif args.task in ("input", "fault"):
        script = "input_check.py" if args.task == "input" else "fault_check.py"
        argv.extend([f"{REPORT}/{script}", "--base", base, "--output", "/profile/result.json"])
    elif args.task == "requests":
        argv.extend([f"{REPORT}/request_ab.py", "--base", f"{SOURCE}/prepared", "--repo", repo,
                     "--checkpoint", checkpoint, "--output", "/profile/result.json"])
    elif args.task == "evaluate":
        produced = "/workspace/Output/promptda-acceleration-validation-20261020"
        argv.extend([f"{REPORT}/evaluate_outputs.py", "--native", f"{produced}/delivery-native-raw",
                     "--candidate", f"{produced}/delivery-optimized-raw", "--output", "/profile/result.json"])
    else:
        argv.extend([f"{REPORT}/delivery_ab.py", "--base", base, "--repo", repo,
                     "--checkpoint", checkpoint, "--mode", args.task.removeprefix("delivery-"), "--output", "/profile"])
    receipt = {"task": args.task, "gpu": gpu, "argv": argv, "before": before,
               "raw": args.raw, "preparation_process_wall_s": preparation_wall_s,
               "sample_target_interval_s": 0.2, "samples": [], "sampling_errors": [],
               "scope": "One owned container; task GPU memory sums its PIDs only; tree RSS can double-count shared pages"}
    path = target / "launch.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Starting " + label, flush=True)
    started = time.perf_counter()
    receipt["started_monotonic"] = started
    root_pid = None
    owned_pids = set()
    with (target / "run.log").open("w") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                tick = time.perf_counter()
                if tick - started > 900:
                    raise TimeoutError("Owned experiment exceeded900s")
                try:
                    if root_pid is None:
                        found = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", name], capture_output=True, text=True)
                        if found.returncode == 0 and int(found.stdout.strip()) > 0:
                            root_pid = int(found.stdout.strip())
                    rss = 0
                    if root_pid is not None and psutil.pid_exists(root_pid):
                        root = psutil.Process(root_pid)
                        for proc in [root, *root.children(recursive=True)]:
                            owned_pids.add(proc.pid)
                            try:
                                rss += proc.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    state = snapshot() if gpu else None
                    memory = 0
                    if state:
                        for row in csv.reader(state["apps"].splitlines(), skipinitialspace=True):
                            if len(row) == 3 and row[0] == GPU_UUID and int(row[1]) in owned_pids:
                                memory += int(row[2])
                    receipt["samples"].append({"elapsed_s": time.perf_counter() - started, "tree_rss_bytes": rss,
                                                "task_gpu_mib": memory, "gpu": state})
                    if memory > 16642:
                        raise RuntimeError("Task sampled GPU memory exceeded original16642MiB reference")
                except (ValueError, subprocess.CalledProcessError, psutil.Error) as error:
                    receipt["sampling_errors"].append(repr(error))
                time.sleep(max(0.0, 0.2 - (time.perf_counter() - tick)))
            receipt["exit_code"] = process.wait()
        except BaseException as error:
            receipt["error"] = repr(error)
            subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True)
            process.wait(timeout=30)
            raise
        finally:
            receipt["container_wall_s"] = time.perf_counter() - started
            if args.raw:
                receipt["raw_to_process_exit_wall_s"] = time.perf_counter() - pipeline_started
            receipt["owned_pids"] = sorted(owned_pids)
            receipt["sampled_task_gpu_peak_mib"] = max((x["task_gpu_mib"] for x in receipt["samples"]), default=0)
            receipt["sampled_tree_rss_peak_bytes"] = max((x["tree_rss_bytes"] for x in receipt["samples"]), default=0)
            receipt["after"] = snapshot()
            path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("task", "exit_code", "container_wall_s", "sampled_task_gpu_peak_mib", "sampled_tree_rss_peak_bytes")}), flush=True)
    raise SystemExit(receipt["exit_code"])


if __name__ == "__main__":
    main()

"""Serial isolated local profiling; only this script's containers are managed."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Output/promptda-da3-d4d5b1bcd2-20261020"
IMAGE = "global-id-mapping:cos-background-20261013"
GPU_UUID = "GPU-2f435ae9-398e-4bc1-ded3-ddb0e360ff0c"


def query(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def gpu_snapshot():
    devices = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    apps = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    return {"devices_csv": devices, "apps_csv": apps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "diagnostic", "delivery", "full"), default="baseline")
    parser.add_argument("--candidate", choices=("both", "da3", "promptda"), default="both")
    parser.add_argument("--tag", default="", help="Explicit new output suffix for a retained recovery attempt")
    args = parser.parse_args()
    import psutil

    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = "-" + args.tag if args.tag else ""
    receipt_path = OUTPUT / f"launch-{args.mode}-{args.candidate}{suffix}.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    receipt = {"mode": args.mode, "physical_gpu": 1, "gpu_uuid": GPU_UUID,
               "image": IMAGE, "runs": [], "sample_target_interval_s": 0.2,
               "memory_scope": "GPU sums only matching container descendant PIDs; host RSS sum may double-count shared pages; sampled values can miss short peaks"}
    candidates = ("da3", "promptda") if args.candidate == "both" else (args.candidate,)
    for candidate in candidates:
        output = OUTPUT / f"{candidate}-{args.mode}{suffix}"
        output.mkdir(exist_ok=False)
        before = gpu_snapshot()
        gpu_row = next(csv.reader([row], skipinitialspace=True).__next__()
                       for row in before["devices_csv"].splitlines() if GPU_UUID in row)
        if int(gpu_row[3]) < 22000:
            raise RuntimeError(f"Insufficient free GPU memory before {candidate}: {gpu_row}")
        readiness = []
        for _ in range(10):
            if int(gpu_row[4]) <= 10:
                break
            readiness.append({"gpu_row": gpu_row, "apps_csv": before["apps_csv"]})
            time.sleep(1)
            before = gpu_snapshot()
            gpu_row = next(csv.reader([row], skipinitialspace=True).__next__()
                           for row in before["devices_csv"].splitlines() if GPU_UUID in row)
        if int(gpu_row[4]) > 10:
            raise RuntimeError(f"Selected GPU remains busy before {candidate}: {gpu_row}")
        if int(gpu_row[3]) < 22000:
            raise RuntimeError(f"GPU free memory changed before {candidate}: {gpu_row}")
        if psutil.disk_usage(ROOT).free < 3 * 1024**3:
            raise RuntimeError("Root filesystem below 3 GiB free before profiling")
        name = f"codex-d4d5-profile-{candidate}-{args.mode}-{os.getpid()}"
        pipeline_started = time.perf_counter()
        prepared = OUTPUT / "prepared"
        prepare_wall = None
        if args.mode == "full":
            prepared = output / "prepared"
            prepare_argv = ["/home/xingyu/.local/bin/uv", "run", "--no-project", "--offline",
                            "--python", str(ROOT / "runtime/viewer-latency-20261017/test-env/bin/python"),
                            "python", str(Path(__file__).with_name("prepare_profile.py")),
                            str(ROOT / "fd-videos/d4d5b1bcd2"), str(prepared)]
            with (output / "preparation.log").open("w") as prepare_log:
                subprocess.run(prepare_argv, stdout=prepare_log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8"))
            prepare_wall = time.perf_counter() - pipeline_started
        script = ("/workspace/scripts/3d/scanner/reconstruct.py" if args.mode in ("delivery", "full")
                  else f"/workspace/reports/promptda-da3-d4d5b1bcd2-20261020/{args.mode if args.mode == 'diagnostic' else 'baseline'}.py")
        argv = ["docker", "run", "--rm", "--name", name, "--network", "none",
                "--gpus", "device=1", "--shm-size", "1g", "--user", f"{os.getuid()}:{os.getgid()}",
                "-v", "/home/xingyu/.local/bin/uv:/usr/local/bin/uv:ro",
                "-v", f"{ROOT}:/workspace:ro", "-v", f"{prepared}:/input:ro",
                "-v", f"{output}:/profile:rw", "-w", "/workspace"]
        for key, value in {"UV_CACHE_DIR": "/tmp/uv", "PYTHONDONTWRITEBYTECODE": "1",
                           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8",
                           "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                           "MPLCONFIGDIR": "/tmp/mpl"}.items():
            argv.extend(["-e", f"{key}={value}"])
        argv.extend(["--entrypoint", "/usr/local/bin/uv", IMAGE,
                     "run", "--no-project", "--offline", "--python", "/app/.venv/bin/python", "python",
                     script,
                     "--base", "/input", "--candidate", candidate,
                     "--output", "/profile/artifacts" if args.mode in ("delivery", "full") else "/profile",
                     "--device", "cuda:0"])
        if candidate == "da3":
            argv.extend(["--checkpoint", "/opt/models/da3/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7"])
        else:
            argv.extend(["--checkpoint", "/workspace/runtime/models/promptda/vitl/model.ckpt",
                         "--promptda-repo", "/workspace/runtime/promptda-counterpart-20261019/PromptDA",
                         "--promptda-batch-size", "2", "--promptda-max-size", "1008",
                         "--promptda-precision", "fp16"])
        entry = {"candidate": candidate, "argv": argv, "before": before,
                 "sampling_errors": [], "container_name": name, "preparation_process_wall_s": prepare_wall}
        entry["readiness_wait_observations"] = readiness
        receipt["runs"].append(entry)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Starting {candidate} {args.mode}", flush=True)
        started = time.perf_counter()
        owned_pids = set()
        root_pid = None
        samples = []
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    now = time.perf_counter()
                    if now - started > 1200:
                        raise TimeoutError("Owned profiling container exceeded 1200 seconds")
                    try:
                        if root_pid is None:
                            found = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", name], capture_output=True, text=True)
                            if found.returncode == 0 and int(found.stdout.strip()) > 0:
                                root_pid = int(found.stdout.strip())
                        rss = 0
                        if root_pid is not None and psutil.pid_exists(root_pid):
                            procs = [psutil.Process(root_pid)]
                            procs.extend(procs[0].children(recursive=True))
                            for proc in procs:
                                owned_pids.add(proc.pid)
                                try:
                                    rss += proc.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                        snapshot = gpu_snapshot()
                        task_mib = 0
                        for row in csv.reader(snapshot["apps_csv"].splitlines(), skipinitialspace=True):
                            if len(row) == 3 and row[0] == GPU_UUID and int(row[1]) in owned_pids:
                                task_mib += int(row[2])
                        samples.append({"seconds_since_launch": time.perf_counter() - started,
                                        "task_gpu_mib": task_mib, "tree_rss_bytes": rss,
                                        "gpu": snapshot})
                    except (subprocess.CalledProcessError, ValueError, psutil.Error) as error:
                        entry["sampling_errors"].append(repr(error))
                    time.sleep(max(0, 0.2 - (time.perf_counter() - now)))
                entry["exit_code"] = process.wait()
            except BaseException:
                subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True)
                process.wait(timeout=30)
                raise
            finally:
                entry["container_wall_s"] = time.perf_counter() - started
                if args.mode == "full":
                    entry["raw_scan_to_process_exit_wall_s"] = time.perf_counter() - pipeline_started
                entry["owned_pids"] = sorted(owned_pids)
                entry["task_gpu_sampled_peak_mib"] = max((x["task_gpu_mib"] for x in samples), default=None)
                entry["tree_rss_sampled_peak_bytes"] = max((x["tree_rss_bytes"] for x in samples), default=None)
                (output / "host_samples.json").write_text(json.dumps(samples, indent=2) + "\n")
                entry["after"] = gpu_snapshot()
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Finished {candidate}: exit={entry['exit_code']} wall={entry['container_wall_s']:.3f}s sampled_gpu={entry['task_gpu_sampled_peak_mib']}MiB", flush=True)
    if any(row.get("exit_code") != 0 for row in receipt["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

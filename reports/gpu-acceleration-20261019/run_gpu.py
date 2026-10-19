"""Run one authorized GPU2 experiment with task-PID memory sampling."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=900)
    parser.add_argument("--peak-mib", type=float, default=15132)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="running", command=command, gpu_uuid=GPU,
                  peak_limit_mib=args.peak_mib, sampled_task_peak_mib=0,
                  task_peak_members_mib={}, observed_task_pids=[], invalid_rows=0,
                  other_gpu_pids={}, unattributed_gpu_pids=[],
                  samples=0, sampling_ms=100,
                  scope="Owned GPU PIDs summed per timestamp; sampled peak, not an instantaneous hard cap")
    child = monitor = None
    known, births, group, stamp = set(), {}, {}, None
    start = time.perf_counter()

    def remember(process):
        known.add(process.pid)
        births.setdefault(process.pid, process.create_time())

    def live_owned():
        live = []
        for pid, birth in births.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        return live

    def group_members():
        if child is None:
            return []
        members = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if (os.getpgid(process.pid) == child.pid
                        and process.info["status"] != psutil.STATUS_ZOMBIE):
                    members.append(process.pid)
            except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
                pass
        return members

    def interrupted(signum, _frame):
        raise InterruptedError(f"Supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def flush():
        if stamp is None:
            return
        result["samples"] += 1
        members = {pid: mib for pid, mib in group.items() if pid in known}
        total = sum(members.values())
        if total > result["sampled_task_peak_mib"]:
            result.update(sampled_task_peak_mib=total, task_peak_members_mib=members,
                          task_peak_timestamp=stamp)
        if total > args.peak_mib:
            raise RuntimeError(f"Task sampled GPU peak {total}MiB exceeds original {args.peak_mib}MiB")

    def consume(reader):
        nonlocal stamp, group
        for line in reader:
            try:
                row = next(csv.reader([line]))
                current, pid, mib = row[0].strip(), int(row[1]), float(row[2])
            except (ValueError, IndexError):
                result["invalid_rows"] += 1
                continue
            if pid not in known and str(pid) not in result["other_gpu_pids"]:
                try:
                    pgid = os.getpgid(pid)
                except ProcessLookupError:
                    result["unattributed_gpu_pids"].append(pid)
                    raise RuntimeError(f"GPU PID{pid} exited before task ownership could be determined")
                if child is not None and pgid == child.pid:
                    remember(psutil.Process(pid))
                else:
                    result["other_gpu_pids"][str(pid)] = {"pgid": pgid}
            if current != stamp:
                flush()
                stamp, group = current, {}
            group[pid] = mib

    try:
        if shutil.disk_usage(out).free < 4 * 1024**3:
            raise RuntimeError("Less than4GiB disk free before experiment")
        result["preflight_free_mib"] = float(subprocess.check_output([
            "nvidia-smi", "-i", GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"
        ], text=True, timeout=10).strip())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   UV_CACHE_DIR="/tmp/3d-profile-20261018-uv", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        # No allocator fraction or new DA3 memory cap is imposed.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        with (out / "gpu.csv").open("w") as samples, (out / "monitor.log").open("w") as errors, (out / "worker.log").open("w") as log:
            monitor = subprocess.Popen([
                "nvidia-smi", "-i", GPU, "--query-compute-apps=timestamp,pid,used_memory",
                "--format=csv,noheader,nounits", "--loop-ms=100",
            ], stdout=samples, stderr=errors)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            remember(psutil.Process(child.pid))
            with (out / "gpu.csv").open() as reader:
                while child.poll() is None:
                    try:
                        for process in psutil.Process(child.pid).children(recursive=True):
                            remember(process)
                    except psutil.NoSuchProcess:
                        pass
                    consume(reader)
                    if monitor.poll() is not None:
                        raise RuntimeError("GPU sample process exited unexpectedly")
                    if time.perf_counter() - start > args.deadline:
                        raise TimeoutError("Experiment deadline exceeded")
                    if shutil.disk_usage(out).free < 4 * 1024**3:
                        raise RuntimeError("Experiment disk floor4GiB reached")
                    time.sleep(.1)
                consume(reader)
                flush()
                remaining = sorted(set(group_members()) | {p.pid for p in live_owned()})
                if remaining:
                    result["residual_processes"] = remaining
                    raise RuntimeError("Root exited while experiment process group still has live descendants")
            result["worker_exitcode"] = child.wait()
            if child.returncode:
                raise RuntimeError(f"Worker exited{child.returncode}; see worker.log")
            if result["invalid_rows"] or not result["sampled_task_peak_mib"]:
                raise RuntimeError("Missing/invalid task GPU memory samples")
            result["status"] = "complete"
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        if child is not None:
            # Clean the owned session even if its root has already exited.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # DA3 may own a separate session. Kill only remembered identities.
            remaining_owned = live_owned()
            for process in remaining_owned:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(remaining_owned, timeout=5)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            if group_members():
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        result.update(observed_task_pids=sorted(known), wall_s=time.perf_counter()-start)
        (out / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({key: result.get(key) for key in ("status", "sampled_task_peak_mib", "worker_exitcode", "wall_s")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

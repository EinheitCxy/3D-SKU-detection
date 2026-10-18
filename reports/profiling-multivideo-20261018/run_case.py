"""Run one prepared video serially on GPU2, with in-run fine instrumentation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GPU = "GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7"
PYTHON = ROOT / "runtime/viewer-latency-20261017/test-env/bin/python"
MODEL = ROOT / "runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7"


def run(case: Path):
    case = case.resolve()
    metadata = json.loads((case / "input.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Input preparation is not complete")
    if (case / "receipt.json").exists() or (case / "outputs").exists():
        raise FileExistsError("A measured case cannot be overwritten")
    if not (MODEL / "model.safetensors").is_file():
        raise FileNotFoundError(MODEL)
    frames = list((case / "dataset/images").glob("*.jpg"))
    detections = list((case / "dataset/detections_results").glob("*.json"))
    if len(frames) != metadata["frames"] or len(detections) != metadata["frames"]:
        raise ValueError("Prepared frame count mismatch")
    if shutil.disk_usage(case).free < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB disk free")
    free_mib = int(subprocess.check_output([
        "nvidia-smi", "-i", GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True).strip())
    if free_mib < 17 * 1024:
        raise RuntimeError("GPU2 has less than 17 GiB free before profiling")
    shutil.copyfile(ROOT / "config.yaml", case / "config.yaml")
    wrapper = case / "da3_python.sh"
    wrapper.write_text(f'#!/bin/sh\nexec uv run --no-project --python "{PYTHON}" python "{HERE / "profile_da3.py"}" "$@"\n')
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER="PCI_BUS_ID",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1", UV_CACHE_DIR="/tmp/3d-profile-20261018-uv",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PROFILE_OUT=str(case), PROFILE_MODEL_PATH=str(MODEL), DA3_VENV_PYTHON=str(wrapper),
    )
    keys = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PROFILE_MODEL_PATH", "DA3_VENV_PYTHON"]
    (case / "settings.json").write_text(json.dumps({k: env[k] for k in keys}, indent=2))
    receipt = dict(status="running", video_id=metadata["video_id"], frames=metadata["frames"],
                   objects=metadata["objects"], gpu_uuid=GPU, gpu_free_before_mib=free_mib,
                   start_unix=time.time(), stages=[], mode="same-run fine instrumentation; overhead unquantified")
    receipt_path = case / "receipt.json"
    monitors, streams = [], []
    try:
        for label, query in [
            ("gpu-device.csv", "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw"),
            ("gpu-processes.csv", "--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory"),
        ]:
            stream = (case / label).open("w")
            streams.append(stream)
            monitors.append(subprocess.Popen([
                "nvidia-smi", "-i", GPU, query, "--format=csv,noheader,nounits", "--loop-ms=200"
            ], stdout=stream, stderr=subprocess.PIPE))
        anchor_before = time.perf_counter()
        anchor_epoch = time.time()
        anchor_after = time.perf_counter()
        anchor = dict(epoch_s=anchor_epoch, monotonic_s=(anchor_before + anchor_after) / 2,
                      uncertainty_s=(anchor_after-anchor_before) / 2,
                      purpose="Map nvidia-smi local timestamps to monotonic stage intervals")
        (case / "clock-anchor.json").write_text(json.dumps(anchor, indent=2))
        receipt["start_monotonic"] = time.perf_counter()
        receipt_path.write_text(json.dumps(receipt, indent=2))
        with (case / "host-processes.jsonl").open("w") as host:
            for label, script in [("core", "profile_core.py"), ("delivery", "profile_delivery.py")]:
                command = ["uv", "run", "--no-project", "--python", str(PYTHON), "python", str(HERE / script)]
                start = time.perf_counter()
                print(f"START video{metadata['video_id']} {label}", flush=True)
                seen = {}
                stage = dict(name=label, command=command, start=start, returncode=None, processes=[])
                receipt["stages"].append(stage)
                receipt_path.write_text(json.dumps(receipt, indent=2))
                with (case / f"{label}.log").open("w") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    parent = psutil.Process(proc.pid)
                    try:
                        while proc.poll() is None:
                            try:
                                members = [parent, *parent.children(recursive=True)]
                            except psutil.NoSuchProcess:
                                members = []
                            rows = []
                            for member in members:
                                try:
                                    if member.pid not in seen:
                                        seen[member.pid] = dict(pid=member.pid, ppid=member.ppid(), argv=member.cmdline())
                                    rows.append(dict(pid=member.pid, rss=member.memory_info().rss,
                                                     cpu=list(member.cpu_times())[:2]))
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
                            host.write(json.dumps(dict(t=time.perf_counter(), stage=label, processes=rows,
                                                       rss_sum=sum(row["rss"] for row in rows))) + "\n")
                            host.flush()
                            if time.perf_counter() - start > 1800 or shutil.disk_usage(case).free < 4 * 1024**3:
                                raise RuntimeError("Stage exceeded 30 min or disk free below 4 GiB")
                            time.sleep(.2)
                    except BaseException:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                                proc.wait()
                        raise
                    finally:
                        stage.update(end=time.perf_counter(), returncode=proc.poll(), processes=list(seen.values()))
                        stage["seconds"] = stage["end"] - start
                        receipt_path.write_text(json.dumps(receipt, indent=2))
                print(f"END video{metadata['video_id']} {label} code={proc.returncode} seconds={stage['seconds']:.3f}", flush=True)
                if proc.returncode:
                    raise RuntimeError(f"{label} failed; see {case / (label + '.log')}")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["end_monotonic"] = time.perf_counter()
        if "start_monotonic" in receipt:
            receipt["elapsed_s"] = receipt["end_monotonic"] - receipt["start_monotonic"]
        receipt_path.write_text(json.dumps(receipt, indent=2))
        for monitor in monitors:
            monitor.terminate()
            monitor.wait(timeout=5)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    run(parser.parse_args().case)

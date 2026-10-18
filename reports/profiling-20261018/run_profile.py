"""One fresh-result-cache GPU2/video3 run, preserving all existing processes."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil
from profile_common import OUT, ROOT, record

GPU = "GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7"
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "runtime/viewer-latency-20261017/test-env/bin/python"
MODEL = OUT / "da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7"


def prepare():
    if (OUT / "receipt.json").exists() or (OUT / "outputs").exists():
        raise FileExistsError("Refusing to reuse a measured output directory")
    if not (MODEL / "model.safetensors").is_file():
        raise FileNotFoundError(MODEL)
    source = ROOT / "runtime/video3-full-profile-20261016/dataset"
    (OUT / "dataset").mkdir(exist_ok=True)
    for directory in ["images", "detections_results"]:
        link = OUT / "dataset" / directory
        if not link.exists():
            link.symlink_to(source / directory, target_is_directory=True)
        elif not link.is_symlink() or link.resolve() != (source / directory).resolve():
            raise ValueError(f"Unexpected prepared input: {link}")
    count = len(list((OUT / "dataset/images").glob("*.jpg")))
    objects = sum(len(sku["objects"]) for p in (OUT / "dataset/detections_results").glob("*.json") for sku in json.loads(p.read_text())["skus"])
    if count != 29 or objects != 711:
        raise ValueError((count, objects))
    (OUT / "input.json").write_text(json.dumps(dict(frames=count, objects=objects,
        prepared_source=str(source), source_description="same previously restored JPEG quality95 and canonical detections; no detector/classifier run",
        frame_times_s=[i*1.5 for i in range(count)]), indent=2))
    shutil.copyfile(ROOT / "config.yaml", OUT / "config.yaml")
    wrapper = OUT / "da3_python.sh"
    wrapper.write_text(f'#!/bin/sh\nexec uv run --no-project --python "{PYTHON}" python "{HERE / "profile_da3.py"}" "$@"\n')
    wrapper.chmod(0o755)


def run():
    prepare()
    if shutil.disk_usage(OUT).free < 4*1024**3:
        raise RuntimeError("Less than 4 GiB disk free")
    gpu = subprocess.check_output(["nvidia-smi", "-i", GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if int(gpu.strip()) < 17*1024:
        raise RuntimeError("GPU2 has less than 17 GiB free before profiling")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER="PCI_BUS_ID",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1", UV_CACHE_DIR="/tmp/3d-profile-20261018-uv",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
        PROFILE_OUT=str(OUT), PROFILE_MODEL_PATH=str(MODEL), DA3_VENV_PYTHON=str(OUT / "da3_python.sh"))
    keys = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PROFILE_MODEL_PATH", "DA3_VENV_PYTHON"]
    (OUT / "settings.json").write_text(json.dumps({k:env[k] for k in keys}, indent=2))
    streams, monitors = [], []
    receipt = dict(status="running", gpu_uuid=GPU, start_unix=time.time(), stages=[])
    receipt_path = OUT / "receipt.json"
    try:
        for label,query in [
            ("gpu-device.csv", "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw"),
            ("gpu-processes.csv", "--query-compute-apps=timestamp,gpu_uuid,pid,used_gpu_memory")]:
            stream = (OUT / label).open("w")
            streams.append(stream)
            monitors.append(subprocess.Popen(["nvidia-smi", "-i", GPU, query, "--format=csv,noheader,nounits", "--loop-ms=200"], stdout=stream, stderr=subprocess.PIPE))
        receipt["start_monotonic"] = time.perf_counter()
        receipt_path.write_text(json.dumps(receipt, indent=2))
        with (OUT / "host-processes.jsonl").open("w") as host:
            for label,script in [("core", "profile_core.py"), ("delivery", "profile_delivery.py")]:
                command = ["uv", "run", "--no-project", "--python", str(PYTHON), "python", str(HERE / script)]
                start = time.perf_counter()
                print("START", label, flush=True)
                with (OUT / f"{label}.log").open("w") as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    parent = psutil.Process(proc.pid)
                    seen = {}
                    try:
                        while proc.poll() is None:
                            try:
                                members = [parent, *parent.children(recursive=True)]
                            except psutil.NoSuchProcess:
                                members = []
                            rows = []
                            for member in members:
                                try:
                                    info = dict(pid=member.pid, rss=member.memory_info().rss, cpu=list(member.cpu_times())[:2])
                                    if member.pid not in seen:
                                        seen[member.pid] = dict(pid=member.pid, ppid=member.ppid(), argv=member.cmdline())
                                    rows.append(info)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue
                            host.write(json.dumps(dict(t=time.perf_counter(), stage=label, processes=rows, rss_sum=sum(x["rss"] for x in rows)))+"\n")
                            host.flush()
                            if time.perf_counter()-start > 1800 or shutil.disk_usage(OUT).free < 4*1024**3:
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
                end = time.perf_counter()
                record(f"{label}_process_wall", start, end, exit_code=proc.returncode)
                receipt["stages"].append(dict(name=label, command=command, start=start, end=end, seconds=end-start, returncode=proc.returncode, processes=list(seen.values())))
                receipt_path.write_text(json.dumps(receipt, indent=2))
                print("END", label, proc.returncode, round(end-start,3), flush=True)
                if proc.returncode:
                    raise RuntimeError(f"{label} failed; see {OUT / (label+'.log')}")
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["end_monotonic"] = time.perf_counter()
        if "start_monotonic" in receipt:
            receipt["elapsed_s"] = receipt["end_monotonic"]-receipt["start_monotonic"]
        receipt_path.write_text(json.dumps(receipt, indent=2))
        for monitor in monitors:
            monitor.terminate()
            monitor.wait(timeout=5)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    run()

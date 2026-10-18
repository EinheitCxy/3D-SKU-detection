"""Prepare explicit 1.5s experimental recipes and execute new videos serially."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "runtime/profiling-multivideo-20261018"


def run(videos):
    records = HERE / "sampling-records"
    records.mkdir(exist_ok=True)
    status_path = BASE / "batch-status.json"
    if status_path.exists():
        raise FileExistsError("Batch receipt already exists")
    status = dict(status="running", requested=videos, cases=[], start_unix=time.time())
    status_path.write_text(json.dumps(status, indent=2))
    try:
        for video in videos:
            if video not in {1, 4, 5, 6, 7}:
                raise ValueError("This batch recipe is only for the unambiguous-count 1.5s cohort; video2 is pending")
            count = len(json.loads((ROOT / f"fd-videos/{video}.json").read_text()))
            record = records / f"video{video}.json"
            if record.exists():
                raise FileExistsError(record)
            record.write_text(json.dumps(dict(video_id=video, video_frame_indices=[i*45 for i in range(count)],
                sampling_basis=dict(source="Coordinator-selected explicit experimental recipe for the user's 2026-10-18 multivideo latency request; JSON count matches 0-start/1.5s full-video sampling at 30fps",
                    type="experimental_recipe_not_original_timestamp_evidence", interval_s=1.5,
                    detection_alignment_verified=False,
                    limitation="No original timestamps available; performance applies to this declared reconstruction recipe, not validated matching accuracy")), indent=2))
            case = BASE / f"video{video}"
            row = dict(video_id=video, status="preparing", start_unix=time.time())
            status["cases"].append(row)
            status_path.write_text(json.dumps(status, indent=2))
            command = [sys.executable, str(HERE / "prepare_inputs.py"), "--video", str(video), "--out", str(case), "--sampling-record", str(record)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            print(f"PREPARE video{video}", flush=True)
            with (BASE / f"prepare-video{video}.log").open("w") as log:
                prep = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row.update(preparation_command=command, preparation_returncode=prep.returncode)
            if prep.returncode:
                row.update(status="preparation_failed", end_unix=time.time())
                status_path.write_text(json.dumps(status, indent=2))
                print(f"FAILED video{video} preparation", flush=True)
                continue
            row["status"] = "profiling"
            status_path.write_text(json.dumps(status, indent=2))
            command = [sys.executable, str(HERE / "run_case.py"), "--case", str(case)]
            measured = subprocess.run(command, cwd=ROOT)
            row.update(status="complete" if measured.returncode == 0 else "profiling_failed",
                       profile_command=command, profile_returncode=measured.returncode, end_unix=time.time())
            status_path.write_text(json.dumps(status, indent=2))
            print(f"CASE video{video} {row['status']}", flush=True)
        status["status"] = "complete" if all(row["status"] == "complete" for row in status["cases"]) else "completed_with_failures"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["end_unix"] = time.time()
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, nargs="+", required=True)
    run(parser.parse_args().videos)

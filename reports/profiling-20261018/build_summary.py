"""Summarize one completed local run, preserving nesting and sampled peak scope."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text())


def build(base):
    receipt = read_json(base / "receipt.json")
    if receipt["status"] != "complete":
        raise ValueError("A completed core+delivery run is required")
    events = [json.loads(line) for path in sorted(base.glob("events-*.jsonl"))
              for line in path.read_text().splitlines() if line.strip()]
    start, end = receipt["start_monotonic"], receipt["end_monotonic"]
    events = [e for e in events if start <= e["start"] <= e["end"] <= end]
    grouped = defaultdict(list)
    for event in events:
        grouped[event["name"]].append(event)
    for rows in grouped.values():
        rows.sort(key=lambda e:e["start"])

    def one(name):
        rows = grouped[name]
        if len(rows) != 1:
            raise ValueError((name, len(rows)))
        return rows[0]

    def timing(name):
        return one(name)["seconds"]

    names = ["run_reconstruction", "run_sku_matching", "run_dedup_sequence",
             "export_web_viewer_bundle", "zip_pack", "glb_export"]
    phases = [{"name":name, "seconds":timing(name),
               "start_offset_s":one(name)["start"]-start,
               "end_offset_s":one(name)["end"]-start} for name in names]
    phases.append(dict(name="imports_process_lifecycle_and_unattributed",
                       seconds=receipt["elapsed_s"]-sum(p["seconds"] for p in phases)))
    prep, match = one("prepare_web_viewer_geometry"), one("run_sku_matching")
    overlap = max(0., min(prep["end"],match["end"])-max(prep["start"],match["start"]))
    task_pids = {process["pid"] for phase in receipt["stages"] for process in phase["processes"]}
    host_rows = [json.loads(line) for line in (base / "host-processes.jsonl").read_text().splitlines()]
    tree_peak = max(host_rows, key=lambda row:row["rss_sum"])
    tree_by_stage = {phase["name"]:max((row["rss_sum"] for row in host_rows if row["stage"] == phase["name"]), default=None)
                     for phase in receipt["stages"]}
    gpu_task_samples = defaultdict(dict)
    gpu_foreign_samples = defaultdict(dict)
    gpu_valid_rows = 0
    for line in (base / "gpu-processes.csv").read_text().splitlines():
        columns = [s.strip() for s in line.split(",")]
        if len(columns) != 4 or columns[1] != receipt["gpu_uuid"]:
            continue
        try:
            pid, mib = int(columns[2]), float(columns[3])
        except ValueError:
            continue
        gpu_valid_rows += 1
        target = gpu_task_samples if pid in task_pids else gpu_foreign_samples
        target[columns[0]][pid] = mib
    if not gpu_task_samples:
        raise ValueError("No driver samples for actual task PIDs")
    peak_timestamp, peak_members = max(gpu_task_samples.items(), key=lambda pair:sum(pair[1].values()))
    device_rows = []
    for line in (base / "gpu-device.csv").read_text().splitlines():
        columns = [s.strip() for s in line.split(",")]
        if len(columns) == 6 and columns[2] == receipt["gpu_uuid"]:
            device_rows.append(dict(timestamp=columns[0], index=int(columns[1]),
                used_mib=float(columns[3]), utilization_percent=float(columns[4]), power_w=float(columns[5])))
    if not device_rows or {row["index"] for row in device_rows} != {2}:
        raise ValueError("No complete GPU2 device sampling")
    driver_start = datetime.strptime(device_rows[0]["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
    driver_end = datetime.strptime(device_rows[-1]["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
    processes = []
    for path in sorted(base.glob("process-*.json")):
        process = read_json(path)
        if process["pid"] in task_pids:
            processes.append(process)
    generation = base / "viewer/runs" / read_json(base / "viewer/CURRENT")["run_id"]
    objects = read_json(generation / "objects.json")
    surfel = read_json(generation / "surfel.json")
    manifest = read_json(generation / "manifest.json")
    result = dict(status="complete", total_wall_s=receipt["elapsed_s"], phases=phases,
        input=read_json(base/"input.json"), settings=read_json(base/"settings.json"),
        milestones_s=dict(viewer_publish=one("_publish_bundle")["end"]-start,
                          viewer_export_return=one("export_web_viewer_bundle")["end"]-start,
                          zip_ready=one("zip_pack")["end"]-start,
                          glb_ready=one("glb_export")["end"]-start),
        preparation=dict(seconds=prep["seconds"], start_offset_s=prep["start"]-start,
                         end_offset_s=prep["end"]-start, matching_overlap_s=overlap,
                         seconds_outside_matching=prep["seconds"]-overlap),
        spans={name:dict(calls=len(rows), total_s=sum(e["seconds"] for e in rows),
                        min_s=min(e["seconds"] for e in rows), max_s=max(e["seconds"] for e in rows),
                        median_s=statistics.median(e["seconds"] for e in rows)) for name,rows in grouped.items()},
        pvs_native=dict(frames=len(grouped["pvs_predict"]),
            boxes=sum(row["box_count"] for row in grouped["pvs_predict"]),
            encoder_s=sum(row["metrics"]["encoder_s"] for row in grouped["pvs_predict"]),
            inference_and_postprocess_including_encoder_s=sum(row["metrics"]["inference_and_postprocess_s"] for row in grouped["pvs_predict"])),
        matching_stages=read_json(base/"matching-stages.json"),
        references=[dict(reference_slot=index, target_slot=(index+1)%29,
            wall_s=row["seconds"], pvs_wall_s=grouped["pvs_predict"][index]["seconds"],
            boxes=grouped["pvs_predict"][index]["box_count"],
            native_encoder_s=grouped["pvs_predict"][index]["metrics"]["encoder_s"],
            native_predict_including_encoder_s=grouped["pvs_predict"][index]["metrics"]["inference_and_postprocess_s"])
            for index,row in enumerate(grouped["reference_matching"])],
        memory=dict(sample_interval_ms=200, target_pids=sorted(task_pids),
            gpu_task_driver_peak_mib=sum(peak_members.values()), gpu_task_peak_timestamp=peak_timestamp,
            gpu_task_peak_members_mib=peak_members,
            gpu_foreign_min_sum_mib=min(sum(row.values()) for row in gpu_foreign_samples.values()),
            gpu_foreign_max_sum_mib=max(sum(row.values()) for row in gpu_foreign_samples.values()),
            gpu_device_peak_mib=max(row["used_mib"] for row in device_rows),
            gpu_process_valid_rows=gpu_valid_rows, gpu_device_rows=len(device_rows),
            gpu_device_coverage_seconds=(driver_end-driver_start).total_seconds(),
            gpu_device_first_timestamp=device_rows[0]["timestamp"], gpu_device_last_timestamp=device_rows[-1]["timestamp"],
            host_tree_peak_bytes=tree_peak["rss_sum"], host_tree_peak_stage=tree_peak["stage"],
            host_tree_peak_offset_s=tree_peak["t"]-start, host_tree_by_stage_bytes=tree_by_stage,
            definition="GPU target process samples aligned by timestamp and actual process-tree PIDs; existing processes excluded. RSS is simultaneous tree sum with possible shared-page duplication. All sampled peaks can miss sub-200ms transients."),
        process_lifetimes=processes,
        outputs=dict(generation=str(generation), points=surfel["point_count"],
            frame_count=manifest["frame_count"], global_ids=len(objects),
            thumbnail_count=len(list((generation/"thumbs").glob("*.jpg"))),
            viewer_generation_bytes=sum(path.stat().st_size for path in generation.rglob("*") if path.is_file()),
            zip_bytes=(base/"viewer_bundle.zip").stat().st_size, glb_bytes=(base/"scene.glb").stat().st_size,
            surfel_metadata={key:value for key,value in surfel.items() if key != "frames"}, manifest=manifest),
        glb_report=read_json(base/"glb-report.json"),
        interpretation="One current-default GPU2 cold-process/fresh-result-cache run; no end-to-end warm estimate, no detector/classifier/COS/HTTP/browser. Independent component diagnostics are not additive to this run.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("runtime/profiling-20261018"))
    parser.add_argument("--output", type=Path, default=Path("reports/profiling-20261018/summary.json"))
    args = parser.parse_args()
    result = build(args.base.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(total_wall_s=result["total_wall_s"], phases=result["phases"],
                         milestones=result["milestones_s"], preparation=result["preparation"],
                         memory=result["memory"], points=result["outputs"]["points"]), indent=2))

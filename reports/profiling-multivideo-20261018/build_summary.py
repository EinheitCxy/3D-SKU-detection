"""Summarize complete cases and equal-video means without adding nested timers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

from summarize_core import summarize as summarize_core
from summarize_delivery import summarize as summarize_delivery
from detail_metrics import extract

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOP = ("core_imports_and_instrumentation", "run_reconstruction", "run_sku_matching",
       "run_dedup_sequence", "export_web_viewer_bundle", "delivery_imports",
       "delivery_detail_setup", "zip_pack", "glb_export")


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = [value for value in values if value is not None]
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values)) if values else dict(n=0, mean=None, median=None, min=None, max=None)


def resource_summary(case, receipt, events):
    start = receipt["start_monotonic"]
    anchor = read(case / "clock-anchor.json")

    def sample_mono(stamp):
        # nvidia-smi and this launcher use the same host local timezone.
        epoch = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        return anchor["monotonic_s"] + epoch - anchor["epoch_s"]
    pids = {process["pid"] for stage in receipt["stages"] for process in stage["processes"]}
    processes = [read(path) for path in sorted(case.glob("process-*.json"))]
    pids.update(process["pid"] for process in processes)
    samples, foreign = defaultdict(dict), defaultdict(dict)
    valid_process_timestamps = set()
    for line in (case / "gpu-processes.csv").read_text().splitlines():
        cols = [field.strip() for field in line.split(",")]
        if len(cols) != 4 or cols[1] != receipt["gpu_uuid"]:
            continue
        if not start <= sample_mono(cols[0]) <= receipt["end_monotonic"]:
            continue
        try:
            pid, mib = int(cols[2]), float(cols[3])
        except ValueError:
            continue
        valid_process_timestamps.add(cols[0])
        (samples if pid in pids else foreign)[cols[0]][pid] = mib
    device = []
    for line in (case / "gpu-device.csv").read_text().splitlines():
        cols = [field.strip() for field in line.split(",")]
        if len(cols) == 6 and cols[2] == receipt["gpu_uuid"]:
            if not start <= sample_mono(cols[0]) <= receipt["end_monotonic"]:
                continue
            device.append(dict(timestamp=cols[0], index=int(cols[1]), used_mib=float(cols[3]),
                               utilization_percent=float(cols[4]), power_w=float(cols[5])))
    if not samples or not device or {row["index"] for row in device} != {2}:
        raise ValueError("Missing task PID/GPU2 driver samples")
    peak_time, peak_members = max(samples.items(), key=lambda pair: sum(pair[1].values()))

    phase_peaks = []
    selected = set(TOP) | {"prepare_web_viewer_geometry", "pvs_model_load", "pvs_predict", "da3_runner_main"}
    for event in events:
        if event["name"] not in selected:
            continue
        rows = [(stamp, samples.get(stamp, {})) for stamp in valid_process_timestamps
                if event["start"] <= sample_mono(stamp) <= event["end"]]
        if rows:
            stamp, members = max(rows, key=lambda pair: sum(pair[1].values()))
            phase_peaks.append(dict(name=event["name"], start_offset_s=event["start"]-start,
                                    end_offset_s=event["end"]-start, task_peak_mib=sum(members.values()),
                                    members_mib=members, timestamp=stamp, valid_query_timestamps=len(rows),
                                    observed_no_task_gpu_process=not members))
        else:
            phase_peaks.append(dict(name=event["name"], start_offset_s=event["start"]-start,
                                    end_offset_s=event["end"]-start, task_peak_mib=None))
    host = [json.loads(line) for line in (case / "host-processes.jsonl").read_text().splitlines()]
    tree_peak = max(host, key=lambda row: row["rss_sum"])
    return dict(sample_interval_ms=200, target_pids=sorted(pids), clock_anchor=anchor,
                gpu_task_driver_peak_mib=sum(peak_members.values()), gpu_task_peak_timestamp=peak_time,
                gpu_task_peak_members_mib=peak_members, gpu_task_peak_offset_s=sample_mono(peak_time)-start,
                gpu_device_peak_mib=max(row["used_mib"] for row in device),
                gpu_foreign_min_sum_mib=min((sum(row.values()) for row in foreign.values()), default=None),
                gpu_foreign_max_sum_mib=max((sum(row.values()) for row in foreign.values()), default=None),
                gpu_device_rows=len(device), gpu_process_task_timestamps=len(samples), phase_peaks=phase_peaks,
                gpu_device_coverage_s=sample_mono(device[-1]["timestamp"])-sample_mono(device[0]["timestamp"]),
                host_tree_peak_bytes=tree_peak["rss_sum"], host_tree_peak_offset_s=tree_peak["t"]-start,
                host_tree_peak_stage=tree_peak["stage"],
                host_tree_by_stage_bytes={stage["name"]: max((row["rss_sum"] for row in host if row["stage"] == stage["name"]), default=None)
                                          for stage in receipt["stages"]},
                process_lifetimes=processes,
                definition="200ms sampled simultaneous task PID sums; existing residents excluded. RSS can double-count shared pages. Phase peaks and allocator/process peaks overlap and must not be added.")


def build(case):
    receipt = read(case / "receipt.json")
    if receipt["status"] != "complete":
        return dict(status=receipt["status"], video_id=receipt["video_id"], receipt=receipt, input=read(case / "input.json"))
    start, end = receipt["start_monotonic"], receipt["end_monotonic"]
    events = sorted([json.loads(line) for path in case.glob("events-*.jsonl") for line in path.read_text().splitlines() if line.strip()],
                    key=lambda event: event["start"])
    grouped = defaultdict(list)
    for event in events:
        if start <= event["start"] <= event["end"] <= end:
            grouped[event["name"]].append(event)

    def one(name):
        rows = grouped[name]
        if len(rows) != 1:
            raise ValueError(f"{case.name}: expected one {name}, found {len(rows)}")
        return rows[0]

    # These are sequential outer scopes; geometry preparation is intentionally excluded.
    intervals = sorted([one(name) for name in TOP], key=lambda event: event["start"])
    exclusive, cursor = [], start
    for event in intervals:
        if event["start"] < cursor - .001:
            raise ValueError(f"Top-level overlap at {event['name']}")
        if event["start"] > cursor:
            exclusive.append(dict(name="process_setup_teardown_and_other_gaps", start_offset_s=cursor-start,
                                  end_offset_s=event["start"]-start, seconds=event["start"]-cursor))
        exclusive.append(dict(name=event["name"], start_offset_s=event["start"]-start,
                              end_offset_s=event["end"]-start, seconds=event["seconds"]))
        cursor = event["end"]
    if end > cursor:
        exclusive.append(dict(name="process_setup_teardown_and_other_gaps", start_offset_s=cursor-start,
                              end_offset_s=end-start, seconds=end-cursor))
    milestones = dict(viewer_publish=one("_publish_bundle")["end"]-start,
                      viewer_export_return=one("export_web_viewer_bundle")["end"]-start,
                      zip_ready=one("zip_pack")["end"]-start, glb_ready=one("glb_export")["end"]-start)
    prefixes = {}
    for name, milestone in dict(milestones, full_pipeline=receipt["elapsed_s"]).items():
        contributions = defaultdict(float)
        for interval in exclusive:
            contributions[interval["name"]] += max(0., min(milestone, interval["end_offset_s"]) - interval["start_offset_s"])
        prefixes[name] = dict(contributions_s=dict(contributions), total_s=sum(contributions.values()),
                              closure_error_s=sum(contributions.values())-milestone)
    prep, match = one("prepare_web_viewer_geometry"), one("run_sku_matching")
    overlap = max(0., min(prep["end"], match["end"]) - max(prep["start"], match["start"]))
    run_id = read(case / "viewer/CURRENT")["run_id"]
    generation = case / "viewer/runs" / run_id
    surfel, manifest = read(generation / "surfel.json"), read(generation / "manifest.json")
    objects = read(generation / "objects.json")
    outputs = dict(generation=str(generation), points=surfel["point_count"], frame_count=manifest["frame_count"],
                   global_ids=len(objects), thumbnail_count=len(list((generation / "thumbs").glob("*.jpg"))),
                   viewer_generation_bytes=sum(path.stat().st_size for path in generation.rglob("*") if path.is_file()),
                   zip_bytes=(case / "viewer_bundle.zip").stat().st_size, glb_bytes=(case / "scene.glb").stat().st_size,
                   surfel_metadata={key: value for key, value in surfel.items() if key != "frames"}, manifest=manifest)
    return dict(status="complete", video_id=receipt["video_id"], case=str(case), total_wall_s=receipt["elapsed_s"],
                input=read(case / "input.json"), settings=read(case / "settings.json"),
                exclusive_timeline=exclusive, milestones_s=milestones, milestone_prefixes=prefixes,
                spans={name: dict(calls=len(rows), total_s=sum(event["seconds"] for event in rows),
                                 min_s=min(event["seconds"] for event in rows), max_s=max(event["seconds"] for event in rows),
                                 median_s=statistics.median(event["seconds"] for event in rows)) for name, rows in grouped.items()},
                preparation=dict(seconds=prep["seconds"], start_offset_s=prep["start"]-start,
                                 end_offset_s=prep["end"]-start, matching_overlap_s=overlap,
                                 seconds_outside_matching=prep["seconds"]-overlap),
                core_detail=summarize_core(case), delivery_detail=summarize_delivery(case),
                memory=resource_summary(case, receipt, events), outputs=outputs,
                interpretation="One full video, one cold model process/fresh result cache, same-run fine instrumentation; no detector/classifier/COS/HTTP/browser. Host/filesystem caches are not flushed. Nested/overlapping times are not additive.")


def aggregate(cases):
    complete = [case for case in cases if case["status"] == "complete"]
    pending_states = {"not_prepared", "awaiting_sampling", "running", "preflight_or_not_launched"}
    metrics = {}

    def metric(name, values):
        metrics[name] = distribution(values)

    metric("total_wall_s", [case["total_wall_s"] for case in complete])
    for name in ("viewer_publish", "viewer_export_return", "zip_ready", "glb_ready"):
        metric("milestones_s."+name, [case["milestones_s"][name] for case in complete])
    for name in TOP:
        metric("phase_s."+name, [case["spans"][name]["total_s"] for case in complete])
    for milestone in ("viewer_publish", "viewer_export_return", "zip_ready", "glb_ready", "full_pipeline"):
        keys = {key for case in complete for key in case["milestone_prefixes"][milestone]["contributions_s"]}
        for key in keys:
            metric("prefix_s."+milestone+"."+key, [case["milestone_prefixes"][milestone]["contributions_s"].get(key, 0.) for case in complete])
    for key in ("seconds", "matching_overlap_s", "seconds_outside_matching"):
        metric("preparation."+key, [case["preparation"][key] for case in complete])
    for key in ("gpu_task_driver_peak_mib", "gpu_device_peak_mib", "host_tree_peak_bytes"):
        metric("memory."+key, [case["memory"][key] for case in complete])
    for key in ("points", "frame_count", "global_ids", "thumbnail_count", "viewer_generation_bytes", "zip_bytes", "glb_bytes"):
        metric("outputs."+key, [case["outputs"][key] for case in complete])
    for key in ("preprocessing_wall_s", "frames", "objects"):
        metric("input."+key, [case["input"].get(key) for case in complete])
    input_steps = {key for case in complete for key in case["input"].get("timings_s", {})}
    for key in input_steps:
        metric("input.timings_s."+key, [case["input"].get("timings_s", {}).get(key) for case in complete])
    for key in {key for case in complete for key in case["core_detail"]["matching_stages"]}:
        for field in ("total", "calls"):
            metric("matching."+key+"."+field, [case["core_detail"]["matching_stages"].get(key, {}).get(field) for case in complete])
    for key in ("encoder_s", "inference_and_postprocess_including_encoder_s", "boxes", "predict_calls"):
        metric("pvs_native."+key, [case["core_detail"]["pvs_native"].get(key) for case in complete])
    for category in ("binary", "jpeg", "json"):
        for field in ("calls", "input_bytes", "compressed_bytes", "member_wall_s"):
            metric("zip."+category+"."+field, [case["delivery_detail"]["zip"]["categories"][category].get(field) for case in complete])
    for key in {key for case in complete for key in case["delivery_detail"]["counts"]}:
        values = [case["delivery_detail"]["counts"].get(key) for case in complete]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            metric("glb_counts."+key, values)
    for key in {key for case in complete for key in case["delivery_detail"]["production_timing"]}:
        metric("glb_timing."+key, [case["delivery_detail"]["production_timing"].get(key) for case in complete])
    phase_names = {phase["name"] for case in complete for phase in case["memory"]["phase_peaks"]}
    for name in phase_names:
        # One maximum within this named phase per video; repeated predicts are
        # pooled within a video first, rather than overweighting long videos.
        per_video = []
        for case in complete:
            values = [phase["task_peak_mib"] for phase in case["memory"]["phase_peaks"]
                      if phase["name"] == name and phase["task_peak_mib"] is not None]
            per_video.append(max(values) if values else None)
        metric("phase_gpu_peak_mib."+name, per_video)
    for role in ("da3", "core", "delivery"):
        for field in ("rss_hwm_kib", "cpu_user_s", "cpu_system_s", "cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes"):
            values = []
            for case in complete:
                rows = [process for process in case["memory"]["process_lifetimes"]
                        if any(str(arg).endswith("profile_"+role+".py") for arg in process.get("argv", []))]
                if len(rows) > 1:
                    raise ValueError("Duplicate process role")
                values.append(rows[0].get(field) if rows else None)
            metric("process."+role+"."+field, values)
    detail_rows = {case["video_id"]: extract(case) for case in complete}
    detail_means = {}
    for key in sorted({key for rows in detail_rows.values() for key in rows}):
        present = [(video, rows[key]) for video, rows in detail_rows.items() if key in rows]
        exemplar = present[0][1]
        values = [rows.get(key, {}).get("value") for rows in detail_rows.values()]
        detail_means[key] = dict(distribution(values), **{field: exemplar[field] for field in ("unit", "section", "semantics", "process_role", "label", "field")},
                                missing_video_ids=[video for video, rows in detail_rows.items() if rows.get(key, {}).get("value") is None],
                                per_video={str(video): rows.get(key, {}).get("value") for video, rows in detail_rows.items()},
                                sources={str(video): row["source"] for video, row in present})
    return dict(protocol="Equal video arithmetic means; one run per video; instrumentation overhead unquantified. No P95 or repeated-scene variance estimate.",
                requested=len(cases), attempted=sum(case["status"] not in {"not_prepared", "awaiting_sampling", "preflight_or_not_launched"} for case in cases),
                succeeded=len(complete), failed=[dict(video_id=case["video_id"], status=case["status"]) for case in cases if case["status"] not in pending_states | {"complete"}],
                pending=[dict(video_id=case["video_id"], status=case["status"]) for case in cases if case["status"] in pending_states],
                video_ids=[case["video_id"] for case in complete], metrics=metrics, detail_metrics=detail_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "runtime/profiling-multivideo-20261018")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for video_id in range(1, 8):
        case = args.base / f"video{video_id}"
        if (case / "receipt.json").exists():
            result = build(case.resolve())
        elif (case / "input.json").exists():
            metadata = read(case / "input.json")
            result = dict(video_id=video_id, status="preparation_failed" if metadata.get("status") != "complete" else "preflight_or_not_launched", input=metadata)
        else:
            result = dict(video_id=video_id, status="awaiting_sampling" if video_id == 2 else "not_prepared")
        results.append(result)
        (args.output / f"{case.name}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = aggregate(results)
    (args.output / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "detail-averages.csv").open("w", newline="") as stream:
        fields = ("key", "section", "process_role", "label", "field", "unit", "n", "mean", "median", "min", "max", "missing_video_ids", "semantics")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, row in summary["detail_metrics"].items():
            writer.writerow(dict(key=key, **{field: row[field] for field in fields if field != "key"}))
    with (args.output / "per-video-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("video_id", "key", "value", "unit", "source"))
        for result in results:
            if result["status"] == "complete":
                for key, row in extract(result).items():
                    writer.writerow((result["video_id"], key, row["value"], row["unit"], row["source"]))
    with (args.output / "references.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("video_id", "reference_slot", "target_slot", "reference_wall_s", "pvs_predict_wall_s", "pvs_model_load_wall_s", "boxes", "native_encoder_s", "native_predict_including_encoder_s"))
        for result in results:
            if result["status"] == "complete":
                for ref in result["core_detail"]["references"]:
                    predicts = ref["pvs_predict_events"]
                    native = [event.get("metrics", {}) for event in predicts]
                    writer.writerow((result["video_id"], ref["reference_slot"], ref["target_slot"], ref["reference_event"]["seconds"],
                                     sum(event["seconds"] for event in predicts), sum(event["seconds"] for event in ref["pvs_model_load_events"]),
                                     sum(event.get("box_count", 0) for event in predicts), sum(row.get("encoder_s", 0.) for row in native),
                                     sum(row.get("inference_and_postprocess_s", 0.) for row in native)))
    print(json.dumps(dict(attempted=summary["attempted"], succeeded=summary["succeeded"], failed=summary["failed"],
                          headline={key: summary["metrics"][key] for key in ("total_wall_s", "milestones_s.viewer_export_return", "milestones_s.zip_ready", "phase_s.glb_export", "memory.gpu_task_driver_peak_mib", "outputs.points")}), indent=2))

"""Summarize only this fixed-input pair; parent and child spans remain separate."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(case):
    receipt = json.loads((case / "receipt.json").read_text())
    if receipt["status"] != "complete":
        return {"status": receipt["status"], "case": str(case)}
    events = [json.loads(line) for path in case.glob("events-*.jsonl")
              for line in path.read_text().splitlines() if line.strip()]
    start = receipt["start_monotonic"]
    def duration(name):
        rows = [event for event in events if event["name"] == name]
        return sum(event["seconds"] for event in rows) if rows else None
    def milestone(name):
        rows = [event for event in events if event["name"] == name]
        return rows[0]["end"] - start if len(rows) == 1 else None
    details = [json.loads(path.read_text()) for path in case.glob("detail-*.json")]
    wall = {}
    for detail in details:
        for name, values in detail["wall"].items():
            if name in wall:
                raise ValueError("Duplicate cross-process span: " + name)
            wall[name] = values
    run_id = json.loads((case / "viewer/CURRENT").read_text())["run_id"]
    generation = case / "viewer/runs" / run_id
    surfel = json.loads((generation / "surfel.json").read_text())
    objects = json.loads((generation / "objects.json").read_text())
    return dict(status="complete", case=str(case), full_wall_s=receipt["elapsed_s"],
                viewer_ready_s=milestone("export_web_viewer_bundle"),
                zip_ready_s=milestone("zip_pack"),
                stages_s={name: duration(name) for name in
                    ("run_reconstruction", "run_sku_matching", "run_dedup_sequence",
                     "export_web_viewer_bundle", "zip_pack", "glb_export",
                     "diagnostic_ground_capture", "diagnostic_ranges_capture", "diagnostic_domain_save")},
                function_spans=wall, viewer_points=surfel["point_count"], global_ids=len(objects),
                generation=str(generation), glb_report=json.loads((case / "glb-report.json").read_text()),
                interpretation="Paired video3 function-level instrumentation, n=1 per arm, seed42; imports/lifecycle and diagnostic data capture included. Host caches not flushed. Nested spans are not additive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = {"baseline": summarize(args.baseline)}
    if args.candidate is not None:
        output["candidate"] = summarize(args.candidate)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items()
                            if key in ("status", "full_wall_s", "viewer_ready_s", "zip_ready_s", "stages_s", "viewer_points")}
                      for name, row in output.items()}, indent=2))


if __name__ == "__main__":
    main()

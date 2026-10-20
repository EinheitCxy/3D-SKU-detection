"""Time the existing Scanner preparation implementation without changing it."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))
import cv2
import numpy as np
import prepare_scanner as prepare

events = []


def measure(owner, name, label):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        events.append({"name": label, "start": start, "end": time.perf_counter()})
        return result

    setattr(owner, name, wrapped)


def main():
    source, output = map(Path, sys.argv[1:3])
    output.mkdir(parents=True, exist_ok=False)
    imports_done = time.perf_counter()
    measure(prepare, "read_selected", "odometry_parse_and_1hz_selection")
    measure(prepare, "extract_frames", "ffmpeg_decode_select_png_and_rename")
    measure(prepare, "prepare_metadata", "metadata_total")
    measure(prepare, "camera_from_row", "camera_from_csv_row")
    measure(cv2, "imread", "png_read_decode")
    measure(np, "savez", "camera_npz_write")
    rows, selected = prepare.read_selected(source)
    prepare.extract_frames(source, output, selected)
    prepare.prepare_metadata(source, output, rows, selected)
    end = time.perf_counter()
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_frame_count": len(rows),
        "selected_frame_count": len(selected),
        "frame_ids": [int(row["frame"]) for _, _, row in selected],
        "imports_s": imports_done - START,
        "total_after_script_imports_s": end - START,
        "events": events,
        "output_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "scope": "Existing prepare_scanner functions; full decode to original-ID PNGs and camera_inputs.npz; no cache clearing or fsync",
    }
    (output / "preparation_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()

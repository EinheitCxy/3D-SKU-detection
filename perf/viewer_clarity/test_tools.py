import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np

from perf.viewer_clarity.crop import crop_bundle
from perf.viewer_clarity.metrics import bundle_metrics
from perf.viewer_clarity.pack import pack_bundle
from perf.viewer_clarity import export_arm
from src.viewer_export_tuning import ViewerExportTuning


def make_generation(root: Path) -> Path:
    root.mkdir()
    count = 40
    coords = [[x * 0.001, y * 0.001, 1.0] for y in range(4) for x in range(5)]
    order = [0, 4, 15, 19, 2, 17, 7, 12, 10, 9, 1, 3, 5, 6, 8, 11, 13, 14, 16, 18]
    base = np.asarray([coords[i] for i in order], dtype=np.float32)
    points = np.concatenate([base, base + np.array([0, 0, 0.0001], dtype=np.float32)])
    objects = {
        "1": {"point_ranges": [[0, 6]], "observations": [{"thumbnail": "thumbs/1.jpg"}]},
        "2": {"point_ranges": [[6, 10]], "observations": [{"thumbnail": "thumbs/2.jpg"}]},
        "3": {"point_ranges": [], "observations": []},
    }
    frames = [{"texture": "surfel-texture-0.jpg", "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]},
              {"texture": "surfel-texture-1.jpg", "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}]
    (root / "manifest.json").write_text(json.dumps({"world_to_view": np.eye(4).reshape(-1).tolist(), "display_bounds": [0] * 6}))
    (root / "objects.json").write_text(json.dumps(objects))
    (root / "surfel.json").write_text(json.dumps({"version": 3, "point_count": count, "grid_size": [5, 4], "frames": frames}))
    points.astype("<f4").tofile(root / "positions.f32.bin")
    np.zeros((count, 3), dtype="u1").tofile(root / "colors.u8.bin")
    np.zeros((count, 3), dtype="i1").tofile(root / "normals.i8.bin")
    np.zeros((count, 3), dtype="<f2").tofile(root / "surfel-u.f16.bin")
    np.zeros((count, 3), dtype="<f2").tofile(root / "surfel-v.f16.bin")
    np.tile([0.5, 0.8], (count, 1)).astype("<f2").tofile(root / "surfel-scale.f16.bin")
    np.repeat(np.arange(2, dtype="u1"), 20).tofile(root / "surfel-frame.u8.bin")
    np.ones((2, 4, 5), dtype="<f2").tofile(root / "surfel-depth.f16.bin")
    (root / "surfel-texture-0.jpg").write_bytes(b"texture0")
    (root / "surfel-texture-1.jpg").write_bytes(b"texture1")
    (root / "thumbs").mkdir()
    (root / "thumbs/1.jpg").write_bytes(b"thumb1")
    (root / "thumbs/2.jpg").write_bytes(b"thumb2")
    return root


def test_pack_members_are_flat_and_complete(tmp_path: Path) -> None:
    generation = make_generation(tmp_path / "generation")
    archive_path = tmp_path / "bundle.zip"
    size = pack_bundle(generation, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json", "positions.f32.bin", "colors.u8.bin", "normals.i8.bin", "objects.json",
            "surfel.json", "surfel-u.f16.bin", "surfel-v.f16.bin", "surfel-frame.u8.bin",
            "surfel-depth.f16.bin", "surfel-scale.f16.bin", "surfel-texture-0.jpg", "surfel-texture-1.jpg",
            "thumbs/1.jpg", "thumbs/2.jpg",
        }
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    assert size == archive_path.stat().st_size


def test_crop_rebuilds_ranges_and_counts(tmp_path: Path) -> None:
    source = make_generation(tmp_path / "generation")
    target = tmp_path / "cropped"
    result = crop_bundle(source, target, center_fraction=0.4)
    objects = json.loads((target / "objects.json").read_text())
    surfel = json.loads((target / "surfel.json").read_text())
    ranges = [pair for item in objects.values() for pair in item["point_ranges"]]
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert sum(end - start for start, end in ranges) == result["product_points"]
    assert surfel["point_count"] == result["point_count"]
    referenced = {obs["thumbnail"] for entry in objects.values() for obs in entry.get("observations", [])}
    remaining = {path.relative_to(target).as_posix() for path in (target / "thumbs").glob("*.jpg")}
    assert remaining == referenced
    try:
        crop_bundle(source, target)
    except FileExistsError:
        pass
    else:
        raise AssertionError("crop must not overwrite an existing output directory")


def test_m2_m3_m5_and_m8_expected_values(tmp_path: Path) -> None:
    generation = make_generation(tmp_path / "generation")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"gid_point_counts": {"1": 6, "2": 4, "3": 7}}))
    values = bundle_metrics(generation, baseline)
    assert values["M2"] == 1.0
    assert values["M3"] == 1.0
    assert values["M5"] == 1.0
    assert values["M8"] == 1


def test_export_arm_builds_only_selected_tuning(tmp_path: Path, monkeypatch) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "manifest.json").write_text("{}")
    (generation / "surfel.json").write_text('{"point_count":4}')
    (generation / "objects.json").write_text('{"gid":{"point_ranges":[[0,2]]}}')
    captured = []

    def fake_export(**kwargs):
        captured.append(kwargs["tuning"])
        return {"manifest_path": str(generation / "manifest.json")}

    export_module = types.ModuleType("src.web_viewer_export")
    export_module.export_web_viewer_bundle = fake_export
    monkeypatch.setitem(sys.modules, "src.web_viewer_export", export_module)
    monkeypatch.setattr(export_arm, "ROOT", tmp_path)
    monkeypatch.setattr(export_arm, "SCENES", {"video3": {"cache": "cache", "mapping": "mapping", "images": "images", "masks": "masks"}})
    monkeypatch.setattr(export_arm, "pack_bundle", lambda *_: 1)
    monkeypatch.setattr(export_arm.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="sha\n"))

    expected = {
        "XQ": ViewerExportTuning(True, True, (2, 1, 1), False, 4),
        "XS4": ViewerExportTuning(False, False, (4, 2, 1), True, 4),
        "XS8": ViewerExportTuning(False, False, (4, 2, 1), True, 8),
        "XQS4": ViewerExportTuning(True, True, (2, 1, 1), True, 4),
        "XQS8": ViewerExportTuning(True, True, (2, 1, 1), True, 8),
    }
    for case, wanted in expected.items():
        arm, cells = (case, None) if case != "XQS4" and case != "XQS8" else ("XQS", int(case[-1]))
        monkeypatch.setattr(export_arm, "ROOT", tmp_path / case)
        export_arm.run("video3", arm, cells)
        assert captured[-1] == wanted

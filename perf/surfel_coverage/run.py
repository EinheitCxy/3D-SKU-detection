"""Build one fixed-input candidate; coordinator launches official runs sequentially."""
import argparse
import json
import time

import numpy as np

from perf.roi_fusion.fusion import coherent_surfels, disc_mesh
from perf.roi_fusion.texture import choose_sources, export_textured_mesh
from perf.surfel_coverage.common import OLD, OUT, load_dense, load_sparse, selected_data


def run(method):
    started = time.monotonic()
    folder = OUT / method
    folder.mkdir(parents=True, exist_ok=False)
    data = load_sparse()
    timings = {"load_seconds": time.monotonic() - started}
    selection_stats = {}
    tick = time.monotonic()
    if method in ("fused", "texture"):
        with np.load(OLD / "fused.npz", allow_pickle=False) as archive:
            surfels = dict(archive)
        u, v = surfels["u"] * 4, surfels["v"] * 4
        reference = "fused"
    elif method == "uniform":
        surfels = coherent_surfels(data)
        u, v = surfels["u"] * 4, surfels["v"] * 4
        reference = "uniform"
    else:
        dense = load_dense()
        timings["dense_load_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        if method == "adaptive":
            from perf.surfel_coverage.adaptive import select
        elif method == "coverage":
            from perf.surfel_coverage.coverage import select
        elif method == "coverage_bounded":
            from perf.surfel_coverage.coverage_bounded import select
        else:
            raise ValueError(method)
        selection = select(dense, len(data["points"]))
        timings["selection_seconds"] = time.monotonic() - tick
        selection_stats = selection["stats"]
        np.savez_compressed(folder / "selection.npz", indices=selection["indices"], scales=selection["scales"])
        data = selected_data(dense, selection)
        if method in ("coverage", "coverage_bounded") and not np.all(np.asarray(selection["scales"]) == 4):
            raise ValueError("Coverage candidate must keep the baseline footprint")
        surfels = coherent_surfels(data)
        u, v = surfels["u"], surfels["v"]
        reference = "uniform"
        del dense
    tick = time.monotonic()
    vertices, faces = disc_mesh(surfels["points"], u, v)
    preferred = np.repeat(surfels["frame"], 8)
    timings["mesh_seconds"] = time.monotonic() - tick
    tick = time.monotonic()
    print(f"{method}: texture selection for {len(faces):,} faces", flush=True)
    if method == "texture":
        from perf.surfel_coverage.texture import choose_sources as candidate_sources
        sources, texture_stats = candidate_sources(vertices, faces, data, preferred_sources=preferred)
    else:
        sources, texture_stats = choose_sources(vertices, faces, data,
            preferred_sources=None if method == "fused" else preferred)
    timings["texture_seconds"] = time.monotonic() - tick
    tick = time.monotonic()
    export_stats = export_textured_mesh(vertices, faces, sources, data, folder / "model.glb")
    timings["export_seconds"] = time.monotonic() - tick
    tick = time.monotonic()
    np.savez_compressed(folder / "mesh.npz", vertices=vertices.astype(np.float32),
                        faces=faces.astype(np.int32), sources=sources.astype(np.int32))
    timings["mesh_save_seconds"] = time.monotonic() - tick
    timings["total_seconds"] = time.monotonic() - started
    stats = dict(method=method, reference=reference, points=len(surfels["points"]),
                 faces=len(faces), selection=selection_stats, texture=texture_stats,
                 export=export_stats, timings=timings,
                 scope="CPU export from fixed 29-frame cache; no model inference; single warm/cold-uncontrolled run")
    if method in ("uniform", "fused"):
        old_name = "baseline" if method == "uniform" else "fused"
        old = json.loads((OLD / f"{old_name}-stats.json").read_text())
        expected = old["texture"]["untextured_area_fraction"]
        actual = texture_stats["untextured_area_fraction"]
        if abs(expected - actual) > 1e-7 or len(faces) != old["export"]["faces"]:
            raise ValueError(f"Baseline reproduction differs: {expected} vs {actual}")
        stats["baseline_reproduced"] = True
    (folder / "stats.json").write_text(json.dumps(stats, indent=2, default=lambda value: value.item()) + "\n")
    print(json.dumps({"method": method, "points": stats["points"],
        "untextured_area_fraction": texture_stats["untextured_area_fraction"], "timings": timings}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("uniform", "fused", "texture", "adaptive", "coverage", "coverage_bounded"))
    run(parser.parse_args().method)

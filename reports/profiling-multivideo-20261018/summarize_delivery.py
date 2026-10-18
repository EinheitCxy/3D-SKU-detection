"""Pure JSON projection of one case's same-run delivery instrumentation.

API: summarize(case: Path) -> JSON-serializable dict. No production imports,
exports, file writes, hashes, or timing aggregation across nested scopes.
Absent data is None with its dotted path listed in `missing`; recorded zeroes
and intentional production nulls remain distinguishable from absent fields.
"""
from __future__ import annotations

import json
from pathlib import Path


BAKE_BLOCKS = (
    "camera_visibility", "footprint_and_continuity", "radius_final_diameter",
    "texel_projection_mask", "depth_and_source_mask", "bilinear_rgb",
    "tile_corners_dilation",
)
LABELS = (
    "delivery.imports", "delivery.detail_setup", "delivery.resolve_generation",
    "delivery.total", "zip.total", "zip.member.binary", "zip.member.jpeg",
    "zip.member.json", "glb.total", "glb._load", "lod.select_surfel_lod",
    "lod._covered", "lod._bound_footprint", "bake._bake_batch",
    *("bake.block." + block for block in BAKE_BLOCKS),
    "bake._bound_footprint", "bake._limit_final_diameter", "bake._bilinear",
    "bake._dilate_rgb", "bake._page", "texture.encode_atlas",
    "texture._binary_alpha", "texture._compact_rgba", "texture._rgba_png",
    "PIL.convert", "PIL.quantize", "PIL.save", "compact.geometry",
    "compact.image", "compact.view", "compact.finish", "compact._write_glb",
)


def summarize(case: Path) -> dict:
    """Read existing case JSON only; reject ambiguous delivery process receipts.

    Schema: sources identifies input files and production source mappings;
    stats preserves common.span per-label inclusive/self/calls/max/thread CPU;
    contexts preserves complete same-thread parent paths; zip has category
    member metadata; counts/configuration/production_report expose production
    outputs. `missing` lists absent observations without inventing zero values.
    In a serial indexed-PNG export texture._rgba_png is normally uncalled and
    therefore missing, even when the overall export completed successfully.
    """
    case = Path(case)
    missing = []

    def read(path):
        if not path.is_file():
            missing.append("files." + path.name)
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return value

    def field(mapping, name, path):
        if not isinstance(mapping, dict) or name not in mapping:
            missing.append(path)
            return None
        return mapping[name]

    # A case has several process details; only the delivery process registers
    # delivery_extra. Never combine another process's wall or CUDA records.
    candidates = []
    for path in sorted(case.glob("detail-*.json")):
        value = read(path)
        if value is not None and "delivery_extra" in value:
            candidates.append((path, value))
    if len(candidates) > 1:
        raise ValueError(f"Multiple delivery details in {case}: "
                         + ", ".join(path.name for path, _ in candidates))
    if candidates:
        detail_path, detail = candidates[0]
    else:
        detail_path, detail = None, None
        missing.append("files.delivery_detail")
    extra_path = case / "delivery-detail-extra.json"
    report_path = case / "glb-report.json"
    extra, report = read(extra_path), read(report_path)
    wall = field(detail, "wall", "detail.wall")
    contexts = field(detail, "wall_contexts", "detail.wall_contexts")
    stats = {label: field(wall, label, "stats." + label) for label in LABELS}
    bake = field(report, "bake", "production_report.bake")
    compact = field(report, "compact", "production_report.compact")
    inputs = field(extra, "input_counts", "extra.input_counts")
    categories = field(extra, "zip_categories", "extra.zip_categories")
    zip_categories = {}
    for category in ("binary", "jpeg", "json"):
        data = field(categories, category, "zip.categories." + category)
        zip_categories[category] = {
            key: field(data, key, "zip.categories." + category + "." + key)
            for key in ("calls", "input_bytes", "compressed_bytes", "member_wall_s")}
        zip_categories[category]["timer"] = stats["zip.member." + category]

    counts = {key: field(bake, key, "counts." + key) for key in (
        "input_point_count", "selected_point_count", "lod_removed_point_count",
        "baked_surfel_count", "skipped_surfel_count", "triangle_count", "atlas_pages")}
    counts.update(
        frames=field(inputs, "frames", "counts.frames"),
        loaded_points=field(inputs, "points", "counts.loaded_points"),
        depth_shape=field(inputs, "depth_shape", "counts.depth_shape"),
        lod=field(bake, "lod", "counts.lod"),
        observed_page_count=field(extra, "page_count", "counts.observed_page_count"),
        observed_baked_surfel_count=field(extra, "baked_surfel_count", "counts.observed_baked_surfel_count"),
        zip_bytes=field(extra, "zip_bytes", "counts.zip_bytes"),
        glb_bytes=field(report, "glb_bytes", "counts.glb_bytes"))
    production_timing = {
        "total_wall_s": field(report, "elapsed_seconds", "production_timing.total_wall_s"),
        "bake_load_lod_wall_s": field(bake, "elapsed_seconds", "production_timing.bake_load_lod_wall_s"),
        "compact_encode_write_join_wall_s": field(compact, "elapsed_seconds", "production_timing.compact_encode_write_join_wall_s"),
        "encoding_join_s": field(compact, "encoding_join_seconds", "production_timing.encoding_join_s"),
        "main_thread_write_s": field(compact, "main_thread_write_seconds", "production_timing.main_thread_write_s"),
    }
    result = dict(
        schema_version="delivery-summary-v1", case=str(case),
        status=field(extra, "status", "status"),
        exception=extra.get("exception") if isinstance(extra, dict) else None,
        sources=dict(detail_file=str(detail_path) if detail_path else None,
                     extra_file=str(extra_path) if extra is not None else None,
                     production_report_file=str(report_path) if report is not None else None,
                     detail_pid=field(detail, "pid", "sources.detail_pid"),
                     production_source_locations=field(extra, "source_locations", "sources.production_source_locations"),
                     batch_instrumentation=field(extra, "batch_instrumentation", "sources.batch_instrumentation")),
        timing_semantics=("Same instrumented export. Never add parent/child inclusive_s or add worker/join to parent wall. "
                          "Contexts are same-thread nested paths; self_s excludes measured direct children only. "
                          "Serial encoding_join_s is encode call wall; with parallel encoding it would be FIFO wait only. "
                          "ZIP member_wall_s includes per-member metadata wrapper overhead; timer is its inner measured write span. "
                          "No nested totals, multi-case means, or resource peaks are computed here."),
        stats=stats, contexts=contexts,
        zip=dict(categories=zip_categories,
                 members=field(extra, "zip_members", "zip.members")),
        counts=counts, production_timing=production_timing,
        configuration={key: field(extra, key, "configuration." + key) for key in (
            "configuration", "export_defaults", "baker_defaults", "final_diameter_max_native")},
        generation=field(extra, "generation", "generation"),
        run_id=field(extra, "run_id", "run_id"),
        export_count=field(extra, "export_count", "export_count"),
        production_report=report,
    )
    result["missing"] = list(dict.fromkeys(missing))
    result["availability"] = ("missing" if detail is None or extra is None or report is None else "present")
    return result

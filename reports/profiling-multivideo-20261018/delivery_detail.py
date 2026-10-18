"""In-process CPU delivery instrumentation, sharing detail_common.span."""
from __future__ import annotations

import ast
import functools
import inspect
import time


def instrument_batch(function, span):
    """Time seven contiguous blocks without changing numerical statements."""
    source, first_line = inspect.getsourcelines(function)
    tree = ast.parse("".join(source))
    ast.increment_lineno(tree, first_line - 1)
    definition = tree.body[0]
    boundaries = [(0, "camera_visibility"), (15, "footprint_and_continuity"),
                  (27, "radius_final_diameter"), (30, "texel_projection_mask"),
                  (41, "depth_and_source_mask"), (51, "bilinear_rgb"),
                  (54, "tile_corners_dilation")]
    groups = []
    for statement in definition.body:
        offset = statement.lineno - first_line
        label = next(label for start, label in reversed(boundaries) if offset >= start)
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(statement)
    definition.body = [ast.copy_location(ast.With(
        items=[ast.withitem(context_expr=ast.Call(
            func=ast.Name(id="_delivery_span", ctx=ast.Load()),
            args=[ast.Constant("bake.block." + label)], keywords=[]))],
        body=statements), statements[0]) for label, statements in groups]
    scope = dict(function.__globals__, _delivery_span=span)
    ast.fix_missing_locations(tree)
    exec(compile(tree, inspect.getfile(function), "exec"), scope)
    return scope[function.__name__], dict(
        file=inspect.getfile(function), first_line=first_line,
        boundaries=[dict(offset=start, label=label) for start, label in boundaries])


class DeliveryDetail:
    """Own reversible patches and scalar metadata; never retain page arrays."""

    def __init__(self, span):
        self.span = span
        self.patches = []
        self.extra = dict(source_locations={}, zip_members=[], page_count=0,
                          baked_surfel_count=0,
                          timing_definition="Same actual ZIP/GLB export as coarse spans; nested inclusive timings are not additive; self excludes instrumented direct children only")

    def patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def wrap(self, owner, name, label, fields=None):
        original = getattr(owner, name)
        self.extra["source_locations"][label] = dict(
            file=inspect.getfile(original), line=inspect.getsourcelines(original)[1])
        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.span(label):
                result = original(*args, **kwargs)
            if fields is not None:
                fields(result)
            return result
        self.patch(owner, name, measured)

    def install(self):
        import zipfile
        from PIL import Image
        from src import surfel_glb as glb, surfel_mesh as mesh, surfel_lod as lod
        from src import surfel_glb_texture as texture, surfel_glb_compact as compact

        def defaults(function):
            return {key: p.default for key, p in inspect.signature(function).parameters.items()
                    if p.default is not inspect.Parameter.empty}
        self.extra.update(export_defaults=defaults(glb.export_scene_glb),
                          baker_defaults=defaults(mesh.bake_surfel_pages),
                          final_diameter_max_native=mesh.FINAL_DIAMETER_MAX_NATIVE,
                          configuration=dict(encoder_workers=1, other_overrides=None))
        if self.extra["export_defaults"]["encoder_workers"] != 1:
            raise RuntimeError("Production encoder default changed; review instrumentation")

        def loaded(result):
            inputs = result[1]
            self.extra["input_counts"] = dict(
                points=len(inputs["positions"]), frames=len(inputs["frames"]),
                depth_shape=list(inputs["depths"].shape))

        def page_ready(page):
            self.extra["page_count"] += 1
            self.extra["baked_surfel_count"] += int(page["surfel_count"])

        self.wrap(glb, "_load", "glb._load", loaded)
        for owner, names, prefix in [
            (lod, ["select_surfel_lod", "_covered", "_bound_footprint"], "lod."),
            (mesh, ["_bound_footprint", "_limit_final_diameter", "_bilinear", "_dilate_rgb"], "bake."),
            (texture, ["encode_atlas", "_binary_alpha", "_compact_rgba", "_rgba_png"], "texture."),
            (compact, ["_write_glb"], "compact."),
            (compact._CompactWriter, ["geometry", "image", "finish", "view"], "compact."),
            (Image.Image, ["convert", "quantize", "save"], "PIL.")]:
            for name in names:
                self.wrap(owner, name, prefix + name)
        self.wrap(mesh, "_page", "bake._page", page_ready)
        batch, mapping = instrument_batch(mesh._bake_batch, self.span)
        self.extra["batch_instrumentation"] = mapping
        self.patch(mesh, "_bake_batch", batch)
        self.wrap(mesh, "_bake_batch", "bake._bake_batch")
        original_write = zipfile.ZipFile.write

        def zip_write(archive, filename, arcname=None, compress_type=None, compresslevel=None):
            category = "binary" if str(filename).endswith(".bin") else (
                "jpeg" if str(filename).endswith(".jpg") else "json")
            tick = time.perf_counter()
            entry = dict(name=str(arcname or filename), category=category, status="failed")
            try:
                with self.span("zip.member." + category):
                    result = original_write(archive, filename, arcname, compress_type, compresslevel)
                info = archive.filelist[-1]
                entry.update(name=info.filename, status="complete", input_bytes=info.file_size,
                             compressed_bytes=info.compress_size, compression=info.compress_type)
                return result
            finally:
                entry["wall_s"] = time.perf_counter() - tick
                self.extra["zip_members"].append(entry)
        self.patch(zipfile.ZipFile, "write", zip_write)

    def restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()
        members = self.extra["zip_members"]
        self.extra["zip_categories"] = {
            category: dict(calls=sum(row["category"] == category for row in members),
                           input_bytes=sum(row.get("input_bytes", 0) for row in members if row["category"] == category),
                           compressed_bytes=sum(row.get("compressed_bytes", 0) for row in members if row["category"] == category),
                           member_wall_s=sum(row["wall_s"] for row in members if row["category"] == category))
            for category in ("binary", "jpeg", "json")}

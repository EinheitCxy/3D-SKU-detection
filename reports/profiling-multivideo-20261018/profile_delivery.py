"""One real ZIP+GLB export with coarse events and same-run CPU detail."""
import json
import sys
import time
import traceback

from detail_common import OUT, ROOT, DATA, record, span


def main():
    sys.path.insert(0, str(ROOT))
    extra = dict(status="failed", export_count=0,
                 scope="Instrumented same-run delivery; no independent second export")
    DATA["delivery_extra"] = extra
    detail = None
    started = time.perf_counter()
    try:
        before = time.perf_counter()
        with span("delivery.imports"):
            from src.surfel_glb import export_scene_glb
            from perf.viewer_clarity.pack import pack_bundle
            from delivery_detail import DeliveryDetail
        record("delivery_imports", before, time.perf_counter())
        before = time.perf_counter()
        with span("delivery.detail_setup"):
            detail = DeliveryDetail(span)
            # Share the same object so partial metadata survives installation failures.
            extra.update(detail.extra)
            detail.extra = extra
            detail.install()
        record("delivery_detail_setup", before, time.perf_counter())
        with span("delivery.resolve_generation"):
            run_id = json.loads((OUT / "viewer/CURRENT").read_text())["run_id"]
            generation = OUT / "viewer/runs" / run_id
            extra.update(run_id=run_id, generation=str(generation))
        with span("delivery.total"):
            before = time.perf_counter()
            with span("zip.total"):
                size = pack_bundle(generation, OUT / "viewer_bundle.zip")
            record("zip_pack", before, time.perf_counter(), bytes=size)
            extra["zip_bytes"] = size
            before = time.perf_counter()
            extra["export_count"] += 1
            with span("glb.total"):
                report = export_scene_glb(generation, OUT / "scene.glb", encoder_workers=1)
            record("glb_export", before, time.perf_counter())
            (OUT / "glb-report.json").write_text(json.dumps(report, indent=2) + "\n")
            extra["production_report"] = report
        extra["status"] = "complete"
        print(json.dumps(report), flush=True)
    except BaseException as error:
        extra["exception"] = dict(type=type(error).__name__, message=str(error),
                                  traceback=traceback.format_exc())
        record("delivery_failed", started, time.perf_counter(), error=repr(error))
        raise
    finally:
        if detail is not None:
            detail.restore()
        extra["elapsed_wall_s"] = time.perf_counter() - started
        extra["artifacts"] = {name: (OUT / name).stat().st_size
                              for name in ("viewer_bundle.zip", "scene.glb", "glb-report.json")
                              if (OUT / name).exists()}
        (OUT / "delivery-detail-extra.json").write_text(
            json.dumps(extra, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

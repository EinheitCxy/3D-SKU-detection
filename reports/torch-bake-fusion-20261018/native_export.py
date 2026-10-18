"""Fresh-process native CPU export control for the bounded Torch experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(status='failed', generation=str(args.generation), command=sys.argv,
                  encoder_workers=2, projection_mode='source_world')
    try:
        tick = time.perf_counter()
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.surfel_glb import export_scene_glb
        result['imports_wall_s'] = time.perf_counter() - tick
        tick = time.perf_counter()
        result['production_report'] = export_scene_glb(
            args.generation, args.out / 'native.glb', encoder_workers=2,
            projection_mode='source_world')
        result['export_wall_s'] = time.perf_counter() - tick
        result['process_to_export_ready_s'] = time.perf_counter() - started
        result['status'] = 'complete'
    except BaseException:
        result['exception'] = traceback.format_exc()
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['resources'] = dict(cpu_user_s=usage.ru_utime, cpu_system_s=usage.ru_stime,
                                   lifetime_maxrss_kib=usage.ru_maxrss)
        (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'], output=str(args.out))), flush=True)


if __name__ == '__main__':
    main()

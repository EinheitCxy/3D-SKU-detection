"""CPU-only real-input arithmetic check, separate from GPU timing processes."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from scanner_io import load_prepared
    from request_probe import cpu_input_check

    torch.set_num_threads(8)
    images, sensor = load_prepared(args.base)
    started = time.perf_counter()
    report = cpu_input_check(images, sensor, batch_size=2, max_size=1008)
    report["check_wall_s"] = time.perf_counter() - started
    report["cuda_initialized"] = torch.cuda.is_initialized()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

"""Meaningful CPU producer/consumer failure and thread-shutdown checks."""
import argparse
import json
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/3d/scanner"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from scanner_io import load_prepared
    from request_probe import predict_prefetch
    import torch
    torch.set_num_threads(8)
    images, sensor = load_prepared(args.base)
    count = len(images)
    sensor = {key: value[:3] if value.ndim and value.shape[0] == count else value for key, value in sensor.items()}

    class FailingConsumer:
        def predict(self, *args):
            raise RuntimeError("consumer failure probe")

    checks = []
    for label, paths in (("consumer", images[:3]), ("producer", ["/missing-probe.png", *images[1:3]])):
        try:
            predict_prefetch(FailingConsumer(), paths, sensor, "cpu", precision="fp32", max_size=1008)
        except (RuntimeError, FileNotFoundError) as error:
            if label == "consumer":
                assert str(error) == "consumer failure probe"
            else:
                assert isinstance(error, FileNotFoundError)
            assert not any(thread.name.startswith("promptda-rgb") for thread in threading.enumerate())
            checks.append({"case": label, "exception": type(error).__name__, "no_producer_thread_remaining": True})
        else:
            raise AssertionError(label + " error did not propagate")
    args.output.write_text(json.dumps({"status": "PASS", "checks": checks, "cuda_initialized": torch.cuda.is_initialized()}, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()

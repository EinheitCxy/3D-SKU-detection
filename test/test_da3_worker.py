"""Preloaded one-shot DA3 worker and standby pool (CPU, fake runner)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from src import da3_worker
from src.da3_3d_reconstructor import DA33DReconstructor

FAKE_RUNNER = '''
import argparse, os, sys

LOADS = []


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--mode", default="ok")
    return ap.parse_known_args(argv)[0]


def load_model(model_path, device):
    if model_path == "broken":
        raise OSError("weights missing")
    LOADS.append(model_path)
    import torch

    return torch.nn.Linear(2, 1)


def main(argv=None, model=None):
    args = _parse_args(argv)
    print("runner stdout noise must not reach the protocol")
    if args.mode == "raise":
        raise ValueError("bad request payload")
    if args.mode == "crash":
        print("fatal native error", file=sys.stderr, flush=True)
        os._exit(7)
    with open(args.output_npz, "w") as out:
        out.write(f"pid={os.getpid()} loads={len(LOADS)} model={model is not None}")
'''


@pytest.fixture(autouse=True)
def fake_worker_script(tmp_path, monkeypatch):
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    script = worker_dir / "da3_worker.py"
    shutil.copy(da3_worker.WORKER_SCRIPT, script)
    (worker_dir / "da3_runner.py").write_text(FAKE_RUNNER, encoding="utf-8")
    monkeypatch.setattr(da3_worker, "WORKER_SCRIPT", script)
    yield
    da3_worker.shutdown_standby_pools()


def _argv(out: Path, mode: str = "ok") -> list[str]:
    return [
        "--output_npz", str(out), "--model_path", "m", "--device", "cpu",
        "--mode", mode,
    ]


def test_worker_serves_one_request_then_exits(tmp_path):
    worker = da3_worker.DA3Worker(sys.executable, "m", "cpu")
    assert worker.wait_ready()["ready"] is True
    response = worker.run(_argv(tmp_path / "a"))
    assert response["ok"] is True and "to_device_s" in response
    assert not worker.alive and worker._proc.returncode == 0
    assert (tmp_path / "a").read_text() == f"pid={worker.pid} loads=1 model=True"
    with pytest.raises(da3_worker.DA3WorkerError, match="只服务一次"):
        worker.run(_argv(tmp_path / "b"))


def test_worker_request_error_raises_with_traceback(tmp_path):
    worker = da3_worker.DA3Worker(sys.executable, "m", "cpu")
    with pytest.raises(da3_worker.DA3WorkerError, match="bad request payload"):
        worker.run(_argv(tmp_path / "a", "raise"))
    assert not worker.alive and worker._proc.returncode == 1


def test_worker_rejects_mismatched_model(tmp_path):
    worker = da3_worker.DA3Worker(sys.executable, "m", "cpu")
    with pytest.raises(da3_worker.DA3WorkerError, match="model/device"):
        worker.run(["--output_npz", str(tmp_path / "a"),
                    "--model_path", "other", "--device", "cpu"])


def test_worker_crash_reports_exit_code_and_stderr_tail(tmp_path):
    worker = da3_worker.DA3Worker(sys.executable, "m", "cpu")
    with pytest.raises(da3_worker.DA3WorkerError) as error:
        worker.run(_argv(tmp_path / "a", "crash"))
    assert "exit=7" in str(error.value)
    assert "fatal native error" in str(error.value)


def test_pool_acquire_spawns_next_standby(tmp_path):
    pool = da3_worker.keep_standby(sys.executable, "m", "cpu")
    assert da3_worker.keep_standby(sys.executable, "m", "cpu") is pool
    first = pool.wait_ready()
    assert pool.acquire() is first
    second = pool._standby
    assert second is not first and second.alive
    first.run(_argv(tmp_path / "a"))
    assert pool.acquire() is second
    second.run(_argv(tmp_path / "b"))
    assert (tmp_path / "a").read_text() != (tmp_path / "b").read_text()
    da3_worker.shutdown_standby_pools()
    assert da3_worker.standby_pool(pool.key) is None


def test_pool_raises_load_failure_and_replaces_standby():
    pool = da3_worker.keep_standby(sys.executable, "broken", "cpu")
    with pytest.raises(da3_worker.DA3WorkerError, match="weights missing"):
        pool.acquire()
    assert pool._standby is not None
    with pytest.raises(da3_worker.DA3WorkerError, match="weights missing"):
        pool.acquire()


def test_reconstructor_uses_standby_only_when_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("DA3_VENV_PYTHON", sys.executable)
    reconstructor = DA33DReconstructor(device="cpu", model_path="m")
    calls = []
    monkeypatch.setattr(
        "subprocess.run", lambda cmd, **_: calls.append(cmd) or "one-shot"
    )
    cmd = [sys.executable, "runner.py", *_argv(tmp_path / "a")]
    assert reconstructor._run_runner(cmd) == "one-shot" and calls == [cmd]

    pool = reconstructor.keep_standby_worker()
    standby = pool.wait_ready()
    proc = reconstructor._run_runner(cmd)
    assert proc.returncode == 0 and calls == [cmd]
    assert (tmp_path / "a").read_text().startswith(f"pid={standby.pid} ")
    assert not standby.alive and pool._standby is not standby

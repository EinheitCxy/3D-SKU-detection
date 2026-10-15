#!/usr/bin/env python3
"""Preloaded one-shot DA3 worker ("standby") and its parent-side pool.

Server (spawned by ``DA3Worker``, in the DA3 interpreter)::

    python src/da3_worker.py --model_path <id> --device cuda

loads the model on CPU without touching CUDA (a standby holds 0 MiB of GPU
memory), answers ``{"ready": true}`` on the original stdout, then waits for
exactly one JSON request line ``{"argv": [<da3_runner argv>]}``.  It moves the
model to the device, runs ``da3_runner.main(argv, model=model)``, answers one
JSON line and exits; process exit frees all of its GPU memory.  Everything
else the worker or its libraries print goes to stderr.

``DA3StandbyPool`` (size 1) keeps one standby ready per
``(python, model_path, device)``: ``acquire()`` hands out the ready standby and
immediately spawns the next one, so the 15-20 s model load overlaps the rest
of the request instead of preceding the next DA3 run.
"""
from __future__ import annotations

import argparse
import atexit
import collections
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve()
STDERR_TAIL_LINES = 80
WorkerKey = Tuple[str, str, str]


class DA3WorkerError(RuntimeError):
    """The DA3 worker died or its request failed."""


class DA3Worker:
    """One DA3 process: loads on CPU, serves a single request, then exits."""

    def __init__(self, python: str | Path, model_path: str, device: str) -> None:
        self.key: WorkerKey = (str(python), model_path, device)
        self._lock = threading.Lock()
        self._ready: Optional[Dict[str, Any]] = None
        self._used = False
        self._stderr_tail: collections.deque[str] = collections.deque(
            maxlen=STDERR_TAIL_LINES
        )
        cmd = [
            str(python), str(WORKER_SCRIPT),
            "--model_path", model_path, "--device", device,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info(f"DA3 standby worker spawned pid={self._proc.pid}: {' '.join(cmd)}")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="da3-worker-stderr", daemon=True
        )
        self._stderr_thread.start()

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def _drain_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            line = line.rstrip("\n")
            self._stderr_tail.append(line)
            logger.info(f"[da3_worker {self._proc.pid}] {line}")

    def _tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _read_message(self) -> Dict[str, Any]:
        assert self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.wait()
            self._stderr_thread.join(timeout=5)
            raise DA3WorkerError(
                f"DA3 worker 意外退出 (pid={self._proc.pid}, exit={code})\n"
                f"STDERR tail:\n{self._tail()}"
            )
        return json.loads(line)

    def wait_ready(self) -> Dict[str, Any]:
        """Block until the model is loaded; raise with stderr tail if it died."""
        with self._lock:
            if self._ready is None:
                message = self._read_message()
                if not message.get("ready"):
                    self._proc.wait()
                    self._stderr_thread.join(timeout=5)
                    raise DA3WorkerError(
                        f"DA3 worker 加载失败: {message.get('error')}\n"
                        f"STDERR tail:\n{self._tail()}"
                    )
                self._ready = message
            return self._ready

    def run(self, argv: Sequence[str]) -> Dict[str, Any]:
        """Serve the single request and wait for the process to exit."""
        self.wait_ready()
        with self._lock:
            if self._used:
                raise DA3WorkerError("DA3 worker 只服务一次请求")
            self._used = True
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps({"argv": list(argv)}) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                pass  # _read_message reports the exit code and stderr tail
            response = self._read_message()
            code = self._proc.wait()
            self._stderr_thread.join(timeout=5)
        if not response.get("ok"):
            raise DA3WorkerError(
                f"DA3 worker 请求失败 (exit={code}): {response.get('error')}\n"
                f"{response.get('traceback', '')}\nSTDERR tail:\n{self._tail()}"
            )
        if code != 0:
            raise DA3WorkerError(
                f"DA3 worker 退出码 {code}\nSTDERR tail:\n{self._tail()}"
            )
        logger.info(
            f"DA3 worker pid={self._proc.pid} done: load={self._ready['load_seconds']:.1f}s "
            f"(standby) h2d={response['to_device_s']:.2f}s request={response['elapsed']:.2f}s"
        )
        return response

    def close(self, timeout: float = 5.0) -> None:
        """Discard the worker: EOF on stdin ends a standby; kill if it lingers."""
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._stderr_thread.join(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


class DA3StandbyPool:
    """Keep exactly one preloaded standby worker for one key."""

    def __init__(self, key: WorkerKey) -> None:
        self.key = key
        self._lock = threading.Lock()
        self._standby: Optional[DA3Worker] = DA3Worker(*key)

    def wait_ready(self) -> DA3Worker:
        """Wait for the current standby without consuming it (startup warm-up)."""
        with self._lock:
            standby = self._require_standby()
        standby.wait_ready()
        return standby

    def acquire(self) -> DA3Worker:
        """Return the ready standby and start loading the next one."""
        with self._lock:
            worker = self._require_standby()
            try:
                worker.wait_ready()
            finally:
                # Replace it whether it loaded or died, so the next request
                # finds a fresh standby; a dead one raises its tail here.
                self._standby = DA3Worker(*self.key)
        return worker

    def _require_standby(self) -> DA3Worker:
        if self._standby is None:
            raise DA3WorkerError("DA3 standby pool 已关闭")
        return self._standby

    def close(self) -> None:
        with self._lock:
            standby, self._standby = self._standby, None
        if standby is not None:
            standby.close()


_POOLS: Dict[WorkerKey, DA3StandbyPool] = {}
_POOLS_LOCK = threading.Lock()


def keep_standby(python: str | Path, model_path: str, device: str) -> DA3StandbyPool:
    """Register (once) and return the standby pool for this key."""
    key: WorkerKey = (str(python), model_path, device)
    with _POOLS_LOCK:
        if key not in _POOLS:
            _POOLS[key] = DA3StandbyPool(key)
        return _POOLS[key]


def standby_pool(key: WorkerKey) -> Optional[DA3StandbyPool]:
    with _POOLS_LOCK:
        return _POOLS.get(key)


def shutdown_standby_pools() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(shutdown_standby_pools)


# ---- server side (runs inside the DA3 interpreter) ----


def _serve(model_path: str, device: str) -> int:
    # Keep the protocol on a private copy of stdout; route fd 1 (runner logging,
    # library prints, C extensions) to stderr so nothing can corrupt it.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout.reconfigure(line_buffering=True)

    import traceback

    sys.path.insert(0, str(WORKER_SCRIPT.parent))
    import da3_runner
    import torch

    def send(message: Dict[str, Any]) -> None:
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    t0 = time.time()
    try:
        model = da3_runner.load_model(model_path, device)  # CPU; no CUDA context
    except BaseException as exc:
        traceback.print_exc()
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    model.eval()
    send({"ready": True, "load_seconds": time.time() - t0})

    line = sys.stdin.readline()
    if not line.strip():
        return 0  # standby discarded
    started = time.time()
    response: Dict[str, Any] = {}
    try:
        argv = [str(item) for item in json.loads(line)["argv"]]
        args = da3_runner._parse_args(argv)
        if (args.model_path, args.device) != (model_path, device):
            raise ValueError(
                f"request model/device {args.model_path}/{args.device} != "
                f"worker {model_path}/{device}"
            )
        moved = time.time()
        model.to(torch.device(device))
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        response["to_device_s"] = time.time() - moved
        da3_runner.main(argv, model=model)
        response["ok"] = True
    except (Exception, SystemExit) as exc:
        response.update(
            ok=False, error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        print(response["traceback"], file=sys.stderr, flush=True)
    response["elapsed"] = time.time() - started
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        response["cuda_max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
    send(response)
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="preloaded one-shot DA3 worker")
    _ap.add_argument("--model_path", required=True)
    _ap.add_argument("--device", required=True)
    _args = _ap.parse_args()
    sys.exit(_serve(_args.model_path, _args.device))

"""Synchronous BSON API for isolated global-ID mapping requests."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Callable

import bson
from fastapi import FastAPI, Request
from fastapi.responses import Response

from docker.processor import build_success_response, prepare_request
from docker.request_runner import RequestRunnerError, run_mapping_request

app = FastAPI()
_REQUEST_LOCK = threading.Lock()


class RequestExecutionError(RuntimeError):
    """A spawned request worker did not produce a successful result."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _child_mapping_target(
    sender: Connection,
    runner: Callable[[Path, Path, Path, str], dict[str, object]],
    dataset_dir: str,
    output_root: str,
    viewer_root: str,
    model_path: str,
) -> None:
    """Picklable spawn target; it must not retain CUDA after returning."""
    try:
        result = runner(
            Path(dataset_dir), Path(output_root), Path(viewer_root), model_path
        )
        sender.send({"ok": True, "result": result})
    except RequestRunnerError as error:
        sender.send({"ok": False, "stage": error.stage, "message": str(error)})
    except BaseException as error:
        sender.send({"ok": False, "stage": "pipeline", "message": str(error)})
    finally:
        sender.close()


def execute_mapping_child(
    dataset_dir: Path,
    output_root: Path,
    viewer_root: Path,
    model_path: str,
    *,
    context_factory: Callable[[str], Any] = multiprocessing.get_context,
    runner: Callable[[Path, Path, Path, str], dict[str, object]] = run_mapping_request,
    waiter: Callable[[list[Any]], list[Any]] = wait,
) -> dict[str, object]:
    """Run a request in a ``spawn`` process and read its small result pre-join."""
    context = context_factory("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_mapping_target,
        args=(
            sender,
            runner,
            str(dataset_dir),
            str(output_root),
            str(viewer_root),
            model_path,
        ),
    )
    started = False
    try:
        process.start()
        started = True
        sender.close()
        ready = waiter([receiver, process.sentinel])
        if receiver in ready:
            try:
                payload = receiver.recv()
            except EOFError as error:
                process.join()
                raise RequestExecutionError(
                    "child", f"mapping child exited with code {process.exitcode}"
                ) from error
        else:
            process.join()
            if receiver.poll():
                try:
                    payload = receiver.recv()
                except EOFError as error:
                    raise RequestExecutionError(
                        "child", f"mapping child exited with code {process.exitcode}"
                    ) from error
            else:
                raise RequestExecutionError(
                    "child", f"mapping child exited with code {process.exitcode}"
                )
        process.join()
        if process.exitcode != 0:
            raise RequestExecutionError(
                "child", f"mapping child exited with code {process.exitcode}"
            )
    finally:
        receiver.close()
        sender.close()
        if started and process.is_alive():
            process.terminate()
        if started:
            process.join()

    if not isinstance(payload, dict):
        raise RequestExecutionError("child", "mapping child returned an invalid result")
    if payload.get("ok") is not True:
        stage = payload.get("stage")
        message = payload.get("message")
        if not isinstance(stage, str) or not isinstance(message, str):
            raise RequestExecutionError("child", "mapping child returned an invalid error")
        raise RequestExecutionError(stage, message)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RequestExecutionError("child", "mapping child returned an invalid result")
    return result


def _bson_response(status_code: int, payload: dict[str, Any]) -> Response:
    return Response(
        content=bson.dumps(payload), status_code=status_code, media_type="application/bson"
    )


def _handle_bson_request(body: bytes) -> Response:
    """Process one BSON request while serializing all CUDA work in this worker."""
    with _REQUEST_LOCK:
        try:
            inputs = bson.loads(body)
        except Exception as error:
            return _bson_response(400, {"stage": "contract", "message": str(error)})
        try:
            with tempfile.TemporaryDirectory(prefix="global-id-mapping-") as work_dir:
                work_root = Path(work_dir)
                prepared = prepare_request(inputs, work_root)
                try:
                    model_path = os.environ["DA3_MODEL_PATH"]
                except KeyError as error:
                    raise RequestExecutionError(
                        "configuration", "DA3_MODEL_PATH is required"
                    ) from error
                output_root = work_root / "outputs"
                viewer_root = work_root / "viewer"
                result = execute_mapping_child(
                    prepared.dataset_dir, output_root, viewer_root, model_path
                )
                try:
                    response = build_success_response(
                        Path(str(result["global_skus_path"])),
                        Path(str(result["viewer_root"])),
                    )
                except (KeyError, TypeError, ValueError, OSError) as error:
                    raise RequestExecutionError("response", str(error)) from error
                return _bson_response(200, response)
        except (TypeError, ValueError) as error:
            return _bson_response(400, {"stage": "contract", "message": str(error)})
        except OSError as error:
            return _bson_response(500, {"stage": "request", "message": str(error)})
        except RequestExecutionError as error:
            return _bson_response(500, {"stage": error.stage, "message": str(error)})


@app.post("/api")
async def mapping_api(request: Request) -> Response:
    """Receive BSON then synchronously run the one-worker serialized request."""
    return _handle_bson_request(await request.body())

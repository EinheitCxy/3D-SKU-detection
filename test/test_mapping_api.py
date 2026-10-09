"""API concurrency tests without importing the model-backed processor."""

import asyncio
import importlib.util
import sys
import threading
import types
import uuid
from pathlib import Path

import bson
import pytest
from httpx import ASGITransport, AsyncClient


API_PATH = Path(__file__).resolve().parents[1] / "api.py"


class ObservedLock:
    """A process lock that exposes when a second request attempts acquisition."""

    def __init__(self):
        self._lock = threading.Lock()
        self._attempts_lock = threading.Lock()
        self._attempts = 0
        self.second_acquire_started = threading.Event()

    def __enter__(self):
        with self._attempts_lock:
            self._attempts += 1
            if self._attempts == 2:
                self.second_acquire_started.set()
        self._lock.acquire()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._lock.release()


@pytest.fixture
def api_module(monkeypatch):
    processor_stub = types.ModuleType("processor")
    processor_stub.process = lambda payload: payload
    monkeypatch.setitem(sys.modules, "processor", processor_stub)

    module_name = f"_mapping_api_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, API_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_mapping_runs_off_event_loop_and_remains_serial(api_module):
    first_process_started = threading.Event()
    release_first_process = threading.Event()
    request_lock = ObservedLock()
    api_module._REQUEST_LOCK = request_lock
    state_lock = threading.Lock()
    active_processes = 0
    maximum_active_processes = 0
    process_calls = 0

    def process(payload):
        nonlocal active_processes, maximum_active_processes, process_calls
        with state_lock:
            active_processes += 1
            maximum_active_processes = max(
                maximum_active_processes, active_processes
            )
            process_calls += 1
        try:
            if payload["block"]:
                first_process_started.set()
                if not release_first_process.wait(timeout=10):
                    raise TimeoutError("test did not release the first mapping")
            return {
                "global_skus": [payload["name"]],
                "scene_glb": b"glTF\x00binary-scene",
            }
        finally:
            with state_lock:
                active_processes -= 1

    api_module.process = process

    async def exercise_requests():
        async with AsyncClient(
            transport=ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/api", content=bson.dumps({"block": True, "name": "first"})
                )
            )
            try:
                assert await asyncio.wait_for(
                    asyncio.to_thread(first_process_started.wait, 5), timeout=6
                )
                second = asyncio.create_task(
                    client.post(
                        "/api",
                        content=bson.dumps({"block": False, "name": "second"}),
                    )
                )
                assert await asyncio.wait_for(
                    asyncio.to_thread(request_lock.second_acquire_started.wait, 5),
                    timeout=6,
                )
                with state_lock:
                    assert process_calls == 1
                # The event loop must serve this while process() is blocked in a worker.
                openapi = await asyncio.wait_for(
                    client.get("/openapi.json"), timeout=5
                )
                assert openapi.status_code == 200
            finally:
                release_first_process.set()

            return await asyncio.wait_for(
                asyncio.gather(first, second), timeout=10
            )

    first_response, second_response = asyncio.run(exercise_requests())

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert bson.loads(first_response.content)["global_skus"] == ["first"]
    assert bson.loads(second_response.content)["global_skus"] == ["second"]
    assert maximum_active_processes == 1


def test_cancelled_request_keeps_lock_until_worker_finishes(api_module):
    first_process_started = threading.Event()
    release_first_process = threading.Event()
    request_lock = ObservedLock()
    api_module._REQUEST_LOCK = request_lock
    state_lock = threading.Lock()
    active_processes = 0
    maximum_active_processes = 0
    process_calls = 0

    def process(payload):
        nonlocal active_processes, maximum_active_processes, process_calls
        with state_lock:
            active_processes += 1
            maximum_active_processes = max(
                maximum_active_processes, active_processes
            )
            process_calls += 1
        try:
            if payload["name"] == "cancelled":
                first_process_started.set()
                if not release_first_process.wait(timeout=10):
                    raise TimeoutError("test did not release the cancelled mapping")
            return {"name": payload["name"]}
        finally:
            with state_lock:
                active_processes -= 1

    api_module.process = process

    async def exercise_cancellation():
        async with AsyncClient(
            transport=ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as client:
            first = asyncio.create_task(
                client.post("/api", content=bson.dumps({"name": "cancelled"}))
            )
            second = None
            try:
                assert await asyncio.wait_for(
                    asyncio.to_thread(first_process_started.wait, 5), timeout=6
                )
                first.cancel()
                second = asyncio.create_task(
                    client.post("/api", content=bson.dumps({"name": "second"}))
                )
                assert await asyncio.wait_for(
                    asyncio.to_thread(request_lock.second_acquire_started.wait, 5),
                    timeout=6,
                )
                with state_lock:
                    assert process_calls == 1
                    assert active_processes == 1
            finally:
                release_first_process.set()

            second_response = await asyncio.wait_for(second, timeout=10)
            try:
                await asyncio.wait_for(first, timeout=10)
            except asyncio.CancelledError:
                pass
            return second_response

    second_response = asyncio.run(exercise_cancellation())
    assert second_response.status_code == 200
    assert bson.loads(second_response.content) == {"name": "second"}
    assert maximum_active_processes == 1


def test_bson_binary_roundtrip_and_http_500_errors_release_lock(api_module):
    scene_glb = b"glTF\x00\xffbinary-scene"
    api_module.process = lambda payload: {
        "global_skus": payload["global_skus"],
        "scene_glb": scene_glb,
    }

    async def exercise_errors():
        async with AsyncClient(
            transport=ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api",
                content=bson.dumps({"global_skus": ["item-7"]}),
            )
            assert response.status_code == 200
            decoded = bson.loads(response.content)
            assert decoded["global_skus"] == ["item-7"]
            assert bytes(decoded["scene_glb"]) == scene_glb

            malformed = await client.post("/api", content=b"not valid BSON")
            assert malformed.status_code == 500
            assert malformed.text

            def fail_once(_payload):
                raise RuntimeError("processor failed")

            api_module.process = fail_once
            failed = await client.post(
                "/api", content=bson.dumps({"request": 1})
            )
            assert failed.status_code == 500
            assert "processor failed" in failed.text

            api_module.process = lambda _payload: {"ok": True}
            recovered = await client.post(
                "/api", content=bson.dumps({"request": 2})
            )
            assert recovered.status_code == 200
            assert bson.loads(recovered.content) == {"ok": True}

    asyncio.run(exercise_errors())

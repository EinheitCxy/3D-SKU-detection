"""Synchronous BSON API for global-ID mapping."""

import logging
import threading
import traceback

import bson
from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from processor import process

app = FastAPI()
_REQUEST_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def _process_mapping(payload: bytes) -> bytes:
    """Decode, process, and encode one request while holding the process lock."""
    with _REQUEST_LOCK:
        result = process(bson.loads(payload))
        return bson.dumps(result)


@app.post("/api")
async def mapping_api(request: Request) -> Response:
    try:
        payload = await request.body()
        encoded_result = await run_in_threadpool(_process_mapping, payload)
        return Response(content=encoded_result, media_type="application/bson")
    except Exception:
        _LOGGER.exception("mapping request failed")
        return Response(status_code=500, content=traceback.format_exc())

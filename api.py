"""Synchronous BSON API for global-ID mapping."""

import threading
import traceback

import bson
from fastapi import FastAPI, Request
from fastapi.responses import Response

from docker.processor import process

app = FastAPI()
_REQUEST_LOCK = threading.Lock()


@app.post("/api")
async def mapping_api(request: Request) -> Response:
    try:
        payload = await request.body()
        with _REQUEST_LOCK:
            result = process(bson.loads(payload))
        return Response(content=bson.dumps(result), media_type="application/bson")
    except Exception:
        return Response(status_code=500, content=traceback.format_exc())

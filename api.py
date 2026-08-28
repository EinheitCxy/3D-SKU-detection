"""Synchronous BSON API for global-ID mapping."""

import threading
import traceback

import bson
from fastapi import Body, FastAPI
from fastapi.responses import Response

from docker.processor import process

app = FastAPI()
_REQUEST_LOCK = threading.Lock()


@app.post("/api")
def mapping_api(payload: bytes = Body(...)) -> Response:
    with _REQUEST_LOCK:
        try:
            result = process(bson.loads(payload))
            return Response(content=bson.dumps(result), media_type="application/bson")
        except Exception:
            return Response(status_code=500, content=traceback.format_exc())

import bson
import traceback
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from processor import process

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/api')
def api(payload: bytes = Body(...)):
    try:
        return Response(content=bson.dumps(process(bson.loads(payload))))
    except Exception as e:
        return Response(status_code=500, content=traceback.format_exc())


# 注意：StaticFiles必须放在最后，否则会拦截所有路由
app.mount('/', StaticFiles(directory='static', html=True), name='static')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8010)
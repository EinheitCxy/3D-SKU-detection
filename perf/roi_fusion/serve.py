"""Serve only this experiment's artifacts with CORS for the local Vite page."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('runtime/roi-fusion-video3-gid3'))
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    if not a.root.is_dir():
        raise FileNotFoundError(a.root)
    server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(a.root.resolve())))
    print(f'Artifacts: http://{a.host}:{a.port}/', flush=True)
    server.serve_forever()

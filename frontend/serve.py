"""Serves the SmartDoc UI and forwards its API calls to the backend.

The backend has no CORS middleware, so a page loaded from any other origin
can't call it — the browser refuses the request before it's even sent. Serving
the HTML and proxying /api/* from this one origin avoids that entirely, which
means the backend stays exactly as it is.

Run it with:  python frontend/serve.py
"""

import json
import mimetypes
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND_URL = "http://127.0.0.1:8001"
PORT = 5173
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    # Without this, Python announces HTTP/1.0 and some browsers close the
    # connection after every single request.
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path.startswith("/api/"):
            self.proxy("GET")
        else:
            self.serve_static()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.proxy("POST")
        else:
            self.send_error(404)

    def proxy(self, method):
        """Replay this request against the backend and hand back its answer."""
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        request = urllib.request.Request(
            BACKEND_URL + self.path[len("/api") :],
            data=body,
            method=method,
        )
        # Uploads are multipart and the boundary lives in this header, so it
        # has to travel with the body or the backend can't parse the file.
        content_type = self.headers.get("Content-Type")
        if content_type:
            request.add_header("Content-Type", content_type)

        try:
            with urllib.request.urlopen(request) as response:
                self.relay(
                    response.status,
                    response.headers.get("Content-Type"),
                    response.read(),
                )
        except urllib.error.HTTPError as error:
            # A 4xx/5xx is still a real answer from the backend — pass it
            # through so the UI can show the actual message instead of a
            # generic failure.
            self.relay(
                error.code,
                error.headers.get("Content-Type"),
                error.read(),
            )
        except urllib.error.URLError as error:
            self.relay(
                502,
                "application/json",
                json.dumps({"detail": f"Backend not reachable: {error.reason}"}).encode(),
            )

    def serve_static(self):
        name = "index.html" if self.path == "/" else self.path.lstrip("/").split("?")[0]
        path = os.path.normpath(os.path.join(STATIC_DIR, name))

        # normpath collapses any ".." in the URL, so this check is what stops a
        # request from reaching files outside the frontend folder.
        if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
            self.send_error(404)
            return

        with open(path, "rb") as f:
            body = f.read()

        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        # This is a dev server: never cache, so a plain refresh always shows
        # the edit you just made.
        self.relay(200, content_type, body, cache=False)

    def relay(self, status, content_type, body, cache=True):
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"SmartDoc UI  ->  http://127.0.0.1:{PORT}")
    print(f"proxying /api/*  ->  {BACKEND_URL}\n")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        # Ctrl+C is how you're meant to stop this — no need to print a
        # traceback as if something went wrong.
        print("\nstopped")
        server.server_close()

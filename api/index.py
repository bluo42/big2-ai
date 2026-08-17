"""Vercel entrypoint: the stateless Flask app as a serverless function.

Static pages are served by Vercel's edge from public/; vercel.json
rewrites /api/* here.  The middleware below restores the original
request path if the platform ever hands us the rewrite destination
(/api/index) instead of the caller's path — belt and suspenders across
proxy behaviors."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from big2.server import app as _flask_app  # noqa: E402

_PATH_HEADERS = (
    "HTTP_X_VERCEL_ORIGINAL_PATH",
    "HTTP_X_ORIGINAL_PATH",
    "HTTP_X_REWRITE_URL",
    "HTTP_X_FORWARDED_URI",
)


def app(environ, start_response):
    if environ.get("PATH_INFO") in ("/api/index", "/api/index/"):
        for header in _PATH_HEADERS:
            original = environ.get(header)
            if original:
                environ["PATH_INFO"] = original.split("?", 1)[0]
                break
    return _flask_app(environ, start_response)

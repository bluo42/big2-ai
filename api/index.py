"""Vercel entrypoint: exposes the stateless Flask app as a serverless
function.  vercel.json rewrites every path here, so the same app serves
the static pages and the JSON API in both environments."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from big2.server import app  # noqa: E402,F401  (Vercel picks up `app`)

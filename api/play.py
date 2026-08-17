"""Vercel function: play cards"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request  # noqa: E402

from big2 import webapi  # noqa: E402

app = Flask(__name__)


# Catch-all: correct behavior regardless of the PATH_INFO the platform
# hands the WSGI app (original path, destination path, or bare "/").
@app.route("/", defaults={"_p": ""}, methods=["GET", "POST"])
@app.route("/<path:_p>", methods=["GET", "POST"])
def handler(_p):
    try:
        body = request.get_json(force=True, silent=True) or {}
        return jsonify(webapi.apply_action(body))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

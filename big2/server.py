"""Web server for the Big 2 UI: a thin, stateless Flask wrapper.

All game logic and state serialization live in big2/webapi.py — every
request carries the full game state, so this app holds nothing in
memory and runs identically on a laptop and on serverless hosting
(Vercel routes all paths to this same app via api/index.py).

    python -m big2.server            # http://127.0.0.1:8080
    /                                # play against the AIs
    /admin                           # replay viewer: simulate games with
                                     # every hand exposed and step through
"""

from __future__ import annotations

import argparse

from flask import Flask, jsonify, request

from big2 import webapi

# Pages live in the repo-level public/ dir: Vercel serves them from its
# edge (cleanUrls maps /admin -> admin.html); locally Flask serves them.
app = Flask(__name__, static_folder="../public", static_url_path="")


def _handle(fn):
    try:
        return jsonify(fn(request.get_json(force=True, silent=True) or {}))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/admin")
def admin():
    return app.send_static_file("admin.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    return _handle(webapi.new_game)


@app.route("/api/play", methods=["POST"])
def play():
    return _handle(webapi.apply_action)


@app.route("/api/pass", methods=["POST"])
def pass_turn():
    def as_pass(body):
        body["pass"] = True
        return webapi.apply_action(body)

    return _handle(as_pass)


@app.route("/api/hint", methods=["POST"])
def hint():
    return _handle(webapi.hint)


@app.route("/api/beliefs", methods=["POST"])
def beliefs():
    return _handle(webapi.beliefs)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    return _handle(webapi.simulate)


@app.route("/api/progress")
def progress():
    return jsonify(webapi.progress())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Big 2 UI: http://{args.host}:{args.port}  (admin: /admin)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

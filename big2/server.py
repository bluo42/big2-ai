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
import os

from flask import Flask, abort, jsonify, request, send_file

from big2 import webapi

# Pages live in the repo-level public/ dir: Vercel serves them from its
# edge; locally Flask serves them.
app = Flask(__name__, static_folder="../public", static_url_path="")


def _admin_enabled() -> bool:
    """Assist/analysis surface (explorer, hints, beliefs, simulate) is
    dev-only: on for `python -m big2.server` and when BIG2_ADMIN is set,
    off on public deployments so players get no unfair help."""
    return bool(app.config.get("BIG2_ADMIN") or os.environ.get("BIG2_ADMIN"))


def _admin_request_ok() -> bool:
    """The dev switch, or a signed-in admin account (e.g. the project
    owner on the public deploy): token via X-Big2-Token header, ?token=
    query, or the JSON body."""
    if _admin_enabled():
        return True
    token = (
        request.headers.get("X-Big2-Token")
        or request.args.get("token")
        or (request.get_json(force=True, silent=True) or {}).get("token")
    )
    return webapi.is_admin_token(token)


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
    if not _admin_request_ok():
        abort(404)
    return send_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "admin_page.html")
    )


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
    if not _admin_request_ok():
        abort(404)
    return _handle(webapi.hint)


@app.route("/api/beliefs", methods=["POST"])
def beliefs():
    if not _admin_request_ok():
        abort(404)
    return _handle(webapi.beliefs)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    if not _admin_request_ok():
        abort(404)
    return _handle(webapi.simulate)


@app.route("/api/progress")
def progress():
    if not _admin_request_ok():
        abort(404)
    return jsonify(webapi.progress())


@app.route("/api/leaderboard", methods=["POST"])
def leaderboard():
    # Public by design: every player sees the testers leaderboard.
    return _handle(webapi.leaderboard)


@app.route("/api/register", methods=["POST"])
def register():
    return _handle(webapi.register_user)


@app.route("/api/login", methods=["POST"])
def login():
    return _handle(webapi.login_user)


@app.route("/api/record-game", methods=["POST"])
def record_game():
    return _handle(webapi.record_game)


@app.route("/api/stats", methods=["POST"])
def stats():
    return _handle(webapi.user_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.config["BIG2_ADMIN"] = True  # local dev: explorer + assists on
    print(f"Big 2 UI: http://{args.host}:{args.port}  (admin: /admin)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

"""Local web UI for playing Big 2 against the AIs.

    python -m big2.server            # serves http://127.0.0.1:8080
    python -m big2.server --port 5000

Single-player, single-game server: one game lives in memory at a time.
The human always sits at seat 0; 1-3 AI opponents fill the other seats.
The frontend (big2/static/index.html) is a self-contained page that
talks to the small JSON API below.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional

from flask import Flask, jsonify, request

from big2.cards import card_name
from big2.combos import Combo
from big2.game import Big2Game, ScoringConfig
from big2.rules import RuleConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    RandomPolicy,
    SmartHeuristic,
    Strategy,
)

app = Flask(__name__, static_folder="static", static_url_path="")

POLICY_FILE = os.path.join(os.path.dirname(__file__), "policies", "linear_cem.npz")

# One active game per server process.
SESSION: Dict = {"game": None}

HUMAN = 0  # the human always sits at seat 0


DMC_FILE = os.path.join(os.path.dirname(__file__), "policies", "dmc_linear.npz")


def make_ai(name: str, seed: Optional[int] = None) -> Strategy:
    name = (name or "smart").lower()
    if name == "linear":
        try:
            from big2.rl import LinearPolicy

            return LinearPolicy.load(POLICY_FILE)
        except Exception:
            return SmartHeuristic()  # weights missing: closest scripted AI
    if name == "dmc":
        try:
            from big2.dmc import DMCPolicy

            return DMCPolicy.load(DMC_FILE)
        except Exception:
            return SmartHeuristic()
    if name == "evo":
        try:
            from big2.nn import NNPolicy

            return NNPolicy.load(
                os.path.join(os.path.dirname(__file__), "policies", "evo_mlp.npz")
            )
        except Exception:
            return SmartHeuristic()
    if name == "ismcts":
        from big2.ismcts import ISMCTSStrategy

        return ISMCTSStrategy(n_sims=250, seed=seed)
    if name == "decomp":
        from big2.decomposition import DecompositionStrategy

        return DecompositionStrategy()
    return {
        "random": lambda: RandomPolicy(seed),
        "lowest": PlayLowest,
        "highest": PlayHighest,
        "dumper": FiveCardDumper,
        "smart": SmartHeuristic,
    }[name]()


def advisor() -> Strategy:
    return make_ai("linear")


def combo_payload(combo: Optional[Combo]) -> Optional[Dict]:
    if combo is None:
        return None
    return {"cards": list(combo.cards), "type": combo.type.name}


def state_payload() -> Dict:
    game: Optional[Big2Game] = SESSION.get("game")
    if game is None:
        return {"phase": "idle"}
    human_turn = not game.game_over and game.turn == HUMAN
    payload = {
        "phase": "over" if game.game_over else "playing",
        "num_players": game.num_players,
        "human_seat": HUMAN,
        "turn": None if game.game_over else game.turn,
        "leading": game.leading,
        "start_card": game.start_card,
        "first_play": game.first_play,
        "hand": list(game.hands[HUMAN]),
        "table": combo_payload(game.table_combo),
        "table_player": game.table_player,
        "players": [
            {
                "seat": p,
                "name": SESSION["names"][p],
                "ai": SESSION["ai_kinds"][p],
                "cards": len(game.hands[p]),
                "passed": game.passed[p],
            }
            for p in range(game.num_players)
        ],
        "history": [
            {
                "player": rec.player,
                "name": SESSION["names"][rec.player],
                "cards": list(rec.combo.cards) if rec.combo else None,
                "type": rec.combo.type.name if rec.combo else None,
                "trick_end": rec.trick_end,
            }
            for rec in game.history
        ],
        "legal_moves": [combo_payload(m) for m in game.legal_moves()] if human_turn else [],
        "can_pass": human_turn and game.can_pass(),
        "rules": {
            "allow_triples": game.rules.allow_triples,
            "pass_locks": game.rules.pass_locks,
            "flush_rank_first": game.rules.flush_rank_first,
        },
        "scoring": game.scoring.label(),
        "winner": game.winner,
        "scores": (
            {str(p): s for p, s in game.scores.items()} if game.scores else None
        ),
    }
    return payload


def run_ai_turns() -> None:
    game: Big2Game = SESSION["game"]
    while not game.game_over and game.turn != HUMAN:
        policy = SESSION["policies"][game.turn]
        game.step(policy.select(game, game.turn))


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    body = request.get_json(force=True) or {}
    num_ai = int(body.get("num_ai", 3))
    if not 1 <= num_ai <= 3:
        return jsonify({"error": "num_ai must be 1, 2, or 3"}), 400
    ai_kinds: List[str] = body.get("ai") or ["smart"] * num_ai
    ai_kinds = (ai_kinds + ["smart"] * num_ai)[:num_ai]

    rules_body = body.get("rules") or {}
    rules = RuleConfig(
        allow_triples=bool(rules_body.get("allow_triples", False)),
        pass_locks=bool(rules_body.get("pass_locks", True)),
        flush_rank_first=bool(rules_body.get("flush_rank_first", False)),
    )
    scoring_body = body.get("scoring") or {}
    scoring = ScoringConfig(
        big_hand_double=bool(scoring_body.get("tiered", True)),
        full_hand_triple=bool(scoring_body.get("tiered", True)),
        two_modifier=bool(scoring_body.get("two", False)),
    )

    game = Big2Game(
        scoring=scoring,
        rules=rules,
        num_players=num_ai + 1,
        rng=random.Random(),
    )
    SESSION["game"] = game
    SESSION["policies"] = {
        seat: make_ai(kind, seed=seat) for seat, kind in enumerate(ai_kinds, start=1)
    }
    SESSION["ai_kinds"] = {HUMAN: None, **{s: k for s, k in enumerate(ai_kinds, start=1)}}
    SESSION["names"] = {
        HUMAN: "You",
        **{s: f"AI {s} ({k})" for s, k in enumerate(ai_kinds, start=1)},
    }
    run_ai_turns()
    return jsonify(state_payload())


@app.route("/api/state")
def state():
    return jsonify(state_payload())


@app.route("/api/play", methods=["POST"])
def play():
    game: Optional[Big2Game] = SESSION.get("game")
    if game is None or game.game_over or game.turn != HUMAN:
        return jsonify({"error": "not your turn"}), 400
    body = request.get_json(force=True) or {}
    selected = frozenset(int(c) for c in body.get("cards", []))
    move = next(
        (m for m in game.legal_moves() if frozenset(m.cards) == selected), None
    )
    if move is None:
        return jsonify({"error": "not a legal play"}), 400
    game.step(move)
    run_ai_turns()
    return jsonify(state_payload())


@app.route("/api/pass", methods=["POST"])
def pass_turn():
    game: Optional[Big2Game] = SESSION.get("game")
    if game is None or game.game_over or game.turn != HUMAN:
        return jsonify({"error": "not your turn"}), 400
    if not game.can_pass():
        return jsonify({"error": "cannot pass when leading"}), 400
    game.step(None)
    run_ai_turns()
    return jsonify(state_payload())


@app.route("/api/beliefs")
def beliefs():
    """Probability panel: what the human can infer about each opponent."""
    game: Optional[Big2Game] = SESSION.get("game")
    if game is None or game.game_over:
        return jsonify({"error": "no game"}), 400
    from big2.beliefs import BeliefState

    b = BeliefState(game, HUMAN, pass_honesty=0.35, rng=random.Random(0))
    classes = b.class_probabilities(k=150)
    beat = b.prob_can_beat(game.table_combo, k=150) if game.table_combo else None
    rank_map = b.rank_probability_map()
    out = {}
    for p in b.opponents:
        known = b.known_hand(p)
        out[str(p)] = {
            "name": SESSION["names"][p],
            "two": b.prob_holds_two(p),
            "ace": b.prob_holds_ace(p),
            "pair": classes[p]["pair"],
            "triple": classes[p]["triple"],
            "beat_table": None if beat is None else beat[p],
            "rank_map": rank_map[p],
            "known_hand": known,
        }
    return jsonify({"unseen": b.n_unseen, "opponents": out})


@app.route("/api/hint")
def hint():
    game: Optional[Big2Game] = SESSION.get("game")
    if game is None or game.game_over or game.turn != HUMAN:
        return jsonify({"error": "not your turn"}), 400
    move = advisor().select(game, HUMAN)
    return jsonify({"cards": list(move.cards) if move else None,
                    "type": move.type.name if move else "PASS"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Big 2 UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

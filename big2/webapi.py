"""Stateless web API core: serializable games, pure request handlers.

Serverless platforms (Vercel) give no in-memory session between
requests, so the full game state travels with every request: responses
carry a ``full_state`` blob and the client echoes it back on each
action.  The blob includes every hand — the client is trusted, which is
fine for a demo/analysis tool (and exactly what the admin replay viewer
wants), but it means a determined player can peek; don't use this for
money games.

Handlers (all pure: dict in -> dict out):
    new_game(body)      start a game, run AI turns, return view+state
    apply_action(body)  play cards or pass for the human, run AI turns
    hint(body)          advisor's suggested move
    beliefs(body)       the human-viewpoint belief panel
    simulate(body)      agent-vs-agent games with full replays exposed
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from big2.cards import Card
from big2.combos import Combo, classify
from big2.game import Big2Game, PlayRecord, ScoringConfig
from big2.rules import RuleConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    RandomPolicy,
    SmartHeuristic,
    Strategy,
)

HUMAN = 0
MAX_SIMULATE_PER_CALL = 20

_POLICY_DIR = None


def _policy_path(name: str) -> str:
    import os

    global _POLICY_DIR
    if _POLICY_DIR is None:
        _POLICY_DIR = os.path.join(os.path.dirname(__file__), "policies")
    return os.path.join(_POLICY_DIR, name)


def make_ai(name: str, seed: Optional[int] = None) -> Strategy:
    name = (name or "smart").lower()
    if name == "linear":
        try:
            from big2.rl import LinearPolicy

            return LinearPolicy.load(_policy_path("linear_cem.npz"))
        except Exception:
            return SmartHeuristic()
    if name == "dmc":
        try:
            from big2.dmc import DMCPolicy

            return DMCPolicy.load(_policy_path("dmc_linear.npz"))
        except Exception:
            return SmartHeuristic()
    if name == "evo":
        try:
            from big2.nn import NNPolicy

            return NNPolicy.load(_policy_path("evo_mlp.npz"))
        except Exception:
            return SmartHeuristic()
    if name == "ppo":
        # torch is training-only; deploys use the numpy inference port
        try:
            from big2.neural import PPOPolicy

            return PPOPolicy.load(_policy_path("ppo_attn.pt"))
        except Exception:
            pass
        try:
            from big2.ppo_numpy import NumpyPPOPolicy

            return NumpyPPOPolicy.load(_policy_path("ppo_attn_np.npz"))
        except Exception:
            return SmartHeuristic()
    if name == "ismcts":
        from big2.ismcts import ISMCTSStrategy

        return ISMCTSStrategy(n_sims=150, seed=seed)
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


AI_KINDS = [
    "smart", "ppo", "evo", "dmc", "ismcts", "decomp", "linear",
    "dumper", "lowest", "highest", "random",
]

_POLICY_FILES = {
    "ppo": "ppo_attn.pt",
    "evo": "evo_mlp.npz",
    "linear": "linear_cem.npz",
    "dmc": "dmc_linear.npz",
}


def model_stamp(kind: Optional[str]) -> str:
    """Time-stamped model identity, e.g. 'ppo@20260817-1755', so scores
    stay attributable to the exact model version that was playing."""
    if kind is None:
        return "you"
    fname = _POLICY_FILES.get(kind)
    if fname is None:
        return f"{kind}@builtin"
    import time as _time

    try:
        mtime = os.path.getmtime(_policy_path(fname))
        return f"{kind}@{_time.strftime('%Y%m%d-%H%M', _time.gmtime(mtime))}"
    except OSError:
        return f"{kind}@builtin"


import os  # noqa: E402  (used by model_stamp/_policy_path)


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def rules_to_dict(rules: RuleConfig) -> Dict:
    return {
        "allow_triples": rules.allow_triples,
        "pass_locks": rules.pass_locks,
        "flush_rank_first": rules.flush_rank_first,
    }


def rules_from_dict(d: Dict) -> RuleConfig:
    return RuleConfig(
        allow_triples=bool(d.get("allow_triples", False)),
        pass_locks=bool(d.get("pass_locks", False)),
        flush_rank_first=bool(d.get("flush_rank_first", False)),
    )


def scoring_from_dict(d: Dict) -> ScoringConfig:
    tiered = bool(d.get("tiered", True))
    return ScoringConfig(
        big_hand_double=tiered,
        full_hand_triple=tiered,
        two_modifier=bool(d.get("two", False)),
    )


def scoring_to_dict(s: ScoringConfig) -> Dict:
    return {"tiered": s.big_hand_double, "two": s.two_modifier}


def serialize_game(game: Big2Game, ai_kinds: Dict[int, Optional[str]]) -> Dict:
    return {
        "num_players": game.num_players,
        "rules": rules_to_dict(game.rules),
        "scoring": scoring_to_dict(game.scoring),
        "ai_kinds": {str(k): v for k, v in ai_kinds.items()},
        "hands": [list(h) for h in game.hands],
        "start_card": game.start_card,
        "turn": game.turn,
        "first_play": game.first_play,
        "table": list(game.table_combo.cards) if game.table_combo else None,
        "table_player": game.table_player,
        "passed": list(game.passed),
        "played": list(game.played_cards),
        "history": [
            [r.player, list(r.combo.cards) if r.combo else None, r.trick_end]
            for r in game.history
        ],
        "winner": game.winner,
        "scores": (
            {str(p): s for p, s in game.scores.items()} if game.scores else None
        ),
    }


def deserialize_game(d: Dict) -> Big2Game:
    rules = rules_from_dict(d["rules"])
    game = object.__new__(Big2Game)
    game.scoring = scoring_from_dict(d["scoring"])
    game.rules = rules
    game.num_players = int(d["num_players"])
    game.rng = random.Random()
    game.hands = [sorted(int(c) for c in h) for h in d["hands"]]
    game.start_card = int(d["start_card"])
    game.turn = int(d["turn"])
    game.first_play = bool(d["first_play"])
    game.table_combo = (
        classify([int(c) for c in d["table"]], rules) if d["table"] else None
    )
    game.table_player = (
        int(d["table_player"]) if d["table_player"] is not None else None
    )
    game.passed = [bool(x) for x in d["passed"]]
    game.played_cards = [int(c) for c in d["played"]]
    game.history = [
        PlayRecord(
            int(p),
            classify([int(c) for c in cards], rules) if cards else None,
            bool(te),
        )
        for p, cards, te in d["history"]
    ]
    game.winner = int(d["winner"]) if d["winner"] is not None else None
    game.scores = (
        {int(p): int(s) for p, s in d["scores"].items()} if d["scores"] else None
    )
    return game


# ----------------------------------------------------------------------
# View payload (shape the frontend renders)
# ----------------------------------------------------------------------


def _names(ai_kinds: Dict[int, Optional[str]]) -> Dict[int, str]:
    return {
        seat: "You" if kind is None else f"AI {seat} ({kind})"
        for seat, kind in ai_kinds.items()
    }


def _combo_payload(combo: Optional[Combo]) -> Optional[Dict]:
    if combo is None:
        return None
    return {"cards": list(combo.cards), "type": combo.type.name}


def view_payload(game: Big2Game, ai_kinds: Dict[int, Optional[str]]) -> Dict:
    names = _names(ai_kinds)
    human_turn = not game.game_over and game.turn == HUMAN
    return {
        "phase": "over" if game.game_over else "playing",
        "num_players": game.num_players,
        "human_seat": HUMAN,
        "turn": None if game.game_over else game.turn,
        "leading": game.leading,
        "start_card": game.start_card,
        "first_play": game.first_play,
        "hand": list(game.hands[HUMAN]),
        "table": _combo_payload(game.table_combo),
        "table_player": game.table_player,
        "players": [
            {
                "seat": p,
                "name": names[p],
                "ai": ai_kinds[p],
                "stamp": model_stamp(ai_kinds[p]),
                "cards": len(game.hands[p]),
                "passed": game.passed[p],
            }
            for p in range(game.num_players)
        ],
        "history": [
            {
                "player": r.player,
                "name": names[r.player],
                "cards": list(r.combo.cards) if r.combo else None,
                "type": r.combo.type.name if r.combo else None,
                "trick_end": r.trick_end,
            }
            for r in game.history
        ],
        "legal_moves": (
            [_combo_payload(m) for m in game.legal_moves()] if human_turn else []
        ),
        "can_pass": human_turn and game.can_pass(),
        "rules": rules_to_dict(game.rules),
        "scoring": game.scoring.label(),
        "winner": game.winner,
        "scores": (
            {str(p): s for p, s in game.scores.items()} if game.scores else None
        ),
        "full_state": serialize_game(game, ai_kinds),
    }


def _run_ai_turns(game: Big2Game, ai_kinds: Dict[int, Optional[str]]) -> None:
    policies = {
        seat: make_ai(kind, seed=seat)
        for seat, kind in ai_kinds.items()
        if kind is not None
    }
    while not game.game_over and ai_kinds.get(game.turn) is not None:
        game.step(policies[game.turn].select(game, game.turn))


def _load(body: Dict) -> (Big2Game, Dict):
    state = body.get("state")
    if not state:
        raise ValueError("missing state")
    game = deserialize_game(state)
    ai_kinds = {
        int(k): v for k, v in state.get("ai_kinds", {}).items()
    }
    return game, ai_kinds


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------


def new_game(body: Dict) -> Dict:
    num_ai = int(body.get("num_ai", 3))
    if not 1 <= num_ai <= 3:
        raise ValueError("num_ai must be 1, 2, or 3")
    kinds: List[str] = list(body.get("ai") or [])
    kinds = ([k or "smart" for k in kinds] + ["smart"] * num_ai)[:num_ai]
    rules = rules_from_dict(body.get("rules") or {})
    scoring = scoring_from_dict(body.get("scoring") or {})
    game = Big2Game(
        scoring=scoring, rules=rules, num_players=num_ai + 1,
        rng=random.Random(body.get("seed")),
    )
    ai_kinds: Dict[int, Optional[str]] = {HUMAN: None}
    for seat, kind in enumerate(kinds, start=1):
        ai_kinds[seat] = kind
    _run_ai_turns(game, ai_kinds)
    return view_payload(game, ai_kinds)


def apply_action(body: Dict) -> Dict:
    game, ai_kinds = _load(body)
    if game.game_over or game.turn != HUMAN:
        raise ValueError("not your turn")
    if body.get("pass"):
        if not game.can_pass():
            raise ValueError("cannot pass when leading")
        game.step(None)
    else:
        selected = frozenset(int(c) for c in body.get("cards", []))
        move = next(
            (m for m in game.legal_moves() if frozenset(m.cards) == selected),
            None,
        )
        if move is None:
            raise ValueError("not a legal play")
        game.step(move)
    _run_ai_turns(game, ai_kinds)
    return view_payload(game, ai_kinds)


def hint(body: Dict) -> Dict:
    game, _ = _load(body)
    if game.game_over or game.turn != HUMAN:
        raise ValueError("not your turn")
    move = make_ai("evo").select(game, HUMAN)
    return {
        "cards": list(move.cards) if move else None,
        "type": move.type.name if move else "PASS",
    }


def beliefs(body: Dict) -> Dict:
    game, ai_kinds = _load(body)
    if game.game_over:
        raise ValueError("game over")
    from big2.beliefs import BeliefState
    from big2.opponents import OpponentModel

    names = _names(ai_kinds)
    # Per-opponent pass evidence, adapted to each opponent's observed style.
    honesty = OpponentModel(game, HUMAN).honesty_map(game)
    b = BeliefState(game, HUMAN, pass_honesty=honesty, rng=random.Random(0))
    classes = b.class_probabilities(k=150)
    beat = b.prob_can_beat(game.table_combo, k=150) if game.table_combo else None
    rank_map = b.rank_probability_map()
    out = {}
    for p in b.opponents:
        out[str(p)] = {
            "name": names[p],
            "two": b.prob_holds_two(p),
            "ace": b.prob_holds_ace(p),
            "pair": classes[p]["pair"],
            "triple": classes[p]["triple"],
            "beat_table": None if beat is None else beat[p],
            "rank_map": rank_map[p],
            "known_hand": b.known_hand(p),
        }
    return {"unseen": b.n_unseen, "opponents": out}


def register_user(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    token = store.register(
        str(body.get("username") or ""), str(body.get("password") or "")
    )
    return {"token": token, "username": str(body.get("username")).strip(),
            "persistent": store.persistent}


def login_user(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    token = store.login(
        str(body.get("username") or ""), str(body.get("password") or "")
    )
    return {"token": token, "username": str(body.get("username")).strip(),
            "persistent": store.persistent}


def record_game(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    auth = store.auth(body.get("token"))
    if auth is None:
        raise ValueError("not signed in")
    store.record_game(auth[0], body.get("game") or {})
    return {"ok": True, "persistent": store.persistent}


def user_stats(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    auth = store.auth(body.get("token"))
    if auth is None:
        raise ValueError("not signed in")
    return {"username": auth[1], **store.stats(auth[0])}


def progress(_body: Optional[Dict] = None) -> Dict:
    """Plateau-probe rows written by big2/evolve.py during training."""
    import os

    path = os.path.join(
        os.path.dirname(__file__), "policies", "evolve", "progress.csv"
    )
    rows = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 8:
                    continue
                isl, games, phase, tid, layers, lr, vs_base, vs_anchor = parts
                rows.append(
                    {
                        "island": int(isl),
                        "games": int(games),
                        "phase": int(phase),
                        "tid": tid,
                        "layers": int(layers),
                        "lr": float(lr),
                        "vs_baselines": float(vs_base),
                        "vs_anchors": (
                            None if vs_anchor == "nan" else float(vs_anchor)
                        ),
                    }
                )
    except FileNotFoundError:
        pass
    return {"rows": rows}


def simulate(body: Dict) -> Dict:
    """Agent-vs-agent games with everything exposed, for the admin viewer."""
    kinds: List[str] = [k for k in (body.get("agents") or []) if k]
    if not 2 <= len(kinds) <= 4:
        raise ValueError("need 2-4 agents")
    n_games = max(1, min(MAX_SIMULATE_PER_CALL, int(body.get("games", 1))))
    rules = rules_from_dict(body.get("rules") or {})
    scoring = scoring_from_dict(body.get("scoring") or {})
    seed = body.get("seed")
    rng = random.Random(seed)

    replays = []
    for _ in range(n_games):
        policies = [make_ai(k, seed=rng.randrange(2**31)) for k in kinds]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=len(kinds),
            rng=random.Random(rng.randrange(2**31)),
        )
        initial = [list(h) for h in game.hands]
        scores = game.play_out(policies)
        replays.append(
            {
                "players": [f"{i}:{k}" for i, k in enumerate(kinds)],
                "num_players": len(kinds),
                "rules": rules_to_dict(rules),
                "scoring": game.scoring.label(),
                "start_card": game.start_card,
                "initial_hands": initial,
                "actions": [
                    {
                        "p": r.player,
                        "cards": list(r.combo.cards) if r.combo else None,
                        "type": r.combo.type.name if r.combo else None,
                        "te": r.trick_end,
                    }
                    for r in game.history
                ],
                "scores": {str(p): s for p, s in scores.items()},
                "winner": game.winner,
            }
        )
    return {"replays": replays}

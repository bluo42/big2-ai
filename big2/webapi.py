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


def admin_users() -> set:
    """Accounts with the full analysis surface (explorer, beliefs, hints,
    leaderboard, unlocked settings) on public deploys.  Comma-separated
    override via BIG2_ADMIN_USERS; defaults to the project owner."""
    raw = os.environ.get("BIG2_ADMIN_USERS", "brandonluo")
    return {u.strip().lower() for u in raw.split(",") if u.strip()}


def is_admin_token(token: Optional[str]) -> bool:
    """True when the bearer token belongs to an admin account."""
    if not token:
        return False
    try:
        from big2.store import get_store

        auth = get_store().auth(token)
    except Exception:
        return False
    return auth is not None and auth[1].lower() in admin_users()

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
    if name in ("wangbot2", "sicario", "leonidas", "khabib",
                "v2_patient", "v2_adversarial", "v2_self_trained",
                "v2_human_trained"):
        # The chain finals (2026-08-18): the locked default table.
        # torch locally, numpy port on serverless; either way the bot
        # plays in its deployed shape — the tree (exact solver +
        # IS-MCTS below 26 cards) is ON in production.
        # The house table runs two Khabib seats and one v2: the
        # "sicario" kind loads Khabib's weights.  Its label, stamp and
        # policy-file entry are deliberately unchanged, so records,
        # leaderboard filters and saved games keep resolving exactly as
        # before -- only the network behind the seat differs.
        # The v2_* kinds are the depth-program canonical names
        # (docs/DEPTH_PROGRAM.md): admin-visible aliases of the same
        # lineage, plus the new AWR-trained patient model.
        stems = {"wangbot2": "wangbot_v2", "sicario": "khabib_v1",
                 "leonidas": "leonidas_v1", "khabib": "khabib_v1",
                 "v2_patient": "v2_patient",
                 "v2_adversarial": "sicario_v1",
                 "v2_self_trained": "leonidas_v1",
                 "v2_human_trained": "khabib_v1"}
        policy = None
        try:
            from big2.neural import PPOPolicy

            policy = PPOPolicy.load(_policy_path(f"{stems[name]}.pt"))
        except Exception:
            try:
                from big2.ppo_numpy import NumpyPPOPolicy

                policy = NumpyPPOPolicy.load(
                    _policy_path(f"{stems[name]}_np.npz"))
            except Exception:
                policy = None
        if policy is None:
            return make_ai("ppo11", seed)
        try:
            from big2.neural import SearchAssist

            # Deployment strength: the tree runs from the opening deal
            # (search_from=53, i.e. every decision), up to 64 simulations
            # spread evenly over the prior's shortlist, 500ms per move --
            # a hard cap chosen for feel, since three bots think in
            # sequence and the wait compounds across a turn.
            return SearchAssist(policy, seed=seed,
                                simulations=64, time_budget=0.5,
                                search_from=53)
        except Exception:
            return policy
    if name in ("ppo", "ppo11"):
        # torch is training-only; deploys use the numpy inference port.
        # 'ppo' is the shipped v1 champion; 'ppo11' the endgame-aware
        # v1.1 line (kept side by side — v1.1 never overrides v1).
        stem = "ppo_attn_v11" if name == "ppo11" else "ppo_attn"
        try:
            from big2.neural import PPOPolicy

            return PPOPolicy.load(_policy_path(f"{stem}.pt"))
        except Exception:
            pass
        try:
            from big2.ppo_numpy import NumpyPPOPolicy

            return NumpyPPOPolicy.load(_policy_path(f"{stem}_np.npz"))
        except Exception:
            if name == "ppo11":  # not exported yet: play the champion
                return make_ai("ppo", seed)
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
    "wangbot2", "sicario", "leonidas", "khabib",
    "v2_patient", "v2_adversarial", "v2_self_trained", "v2_human_trained",
    "smart", "ppo", "ppo11", "evo", "dmc", "ismcts", "decomp", "linear",
    "dumper", "lowest", "highest", "random",
]

# Public display names where they differ from the internal kind (the
# kind string stays stable so serialized games keep deserializing).
KIND_LABEL = {"ppo11": "WangBot_v1", "wangbot2": "v2",
              "sicario": "Sicario", "leonidas": "Leonidas",
              "khabib": "Khabib"}

_POLICY_FILES = {
    "ppo": "ppo_attn.pt",
    "ppo11": "ppo_attn_v11.pt",
    "wangbot2": "wangbot_v2.pt",
    "sicario": "sicario_v1.pt",
    "leonidas": "leonidas_v1.pt",
    "khabib": "khabib_v1.pt",
    "evo": "evo_mlp.npz",
    "linear": "linear_cem.npz",
    "dmc": "dmc_linear.npz",
}


# Generation tag baked into the stamp's version part, so the
# leaderboard can separate eras cleanly.
#   v4t (2026-08-18) -- the tree went live, but the serverless export
#       had no value head, so its IS-MCTS scored every leaf with a
#       card-count heuristic.
#   v5  (2026-08-19) -- value head exported and used at search leaves,
#       plus trick-level potential in the leaf evaluation.  This is the
#       first deployment where the tree thinks with the trained net.
_GENERATION = {"wangbot2": "v5", "sicario": "v5", "khabib": "v5",
               "leonidas": "v5", "v2_patient": "v5",
               "v2_adversarial": "v5", "v2_self_trained": "v5",
               "v2_human_trained": "v5"}


def model_stamp(kind: Optional[str]) -> str:
    """Time-stamped model identity, e.g. 'WangBot_v1@20260818-0114', so
    scores stay attributable to the exact model version that was playing."""
    if kind is None:
        return "you"
    label = KIND_LABEL.get(kind, kind)
    gen = _GENERATION.get(kind)
    prefix = f"{gen}-" if gen else ""
    fname = _POLICY_FILES.get(kind)
    if fname is None:
        return f"{label}@{prefix}builtin"
    import time as _time

    try:
        mtime = os.path.getmtime(_policy_path(fname))
        stamp = _time.strftime('%Y%m%d-%H%M', _time.gmtime(mtime))
        return f"{label}@{prefix}{stamp}"
    except OSError:
        return f"{label}@{prefix}builtin"


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
        seat: (
            "You" if kind is None
            else f"AI {seat} ({KIND_LABEL.get(kind, kind)})"
        )
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


# Built agents, reused across requests.  Constructing a chain bot means
# reading its weights off disk and standing up a search agent, and on
# the first call the belief machinery imports lazily -- all of which was
# being paid INSIDE the move's time budget, starving the search (~8
# simulations instead of ~30).  Policies are stateless with respect to
# the position, so a warm one is safe to share.
_AI_CACHE: Dict[Tuple[str, Optional[int]], Strategy] = {}
_AI_CACHE_CAP = 24


def get_ai(name: str, seed: Optional[int] = None) -> Strategy:
    key = ((name or "smart").lower(), seed)
    hit = _AI_CACHE.get(key)
    if hit is None:
        if len(_AI_CACHE) >= _AI_CACHE_CAP:
            _AI_CACHE.clear()
        hit = _AI_CACHE[key] = make_ai(name, seed=seed)
    return hit


def _run_ai_turns(game: Big2Game, ai_kinds: Dict[int, Optional[str]]) -> None:
    policies = {
        seat: get_ai(kind, seed=seat)
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

    def _view(seat: int) -> Dict:
        """What ``seat`` believes about everyone else."""
        honesty = OpponentModel(game, seat).honesty_map(game)
        b = BeliefState(game, seat, pass_honesty=honesty,
                        rng=random.Random(0))
        classes = b.class_probabilities(k=150)
        beat = (b.prob_can_beat(game.table_combo, k=150)
                if game.table_combo else None)
        rank_map = b.rank_probability_map()
        return {
            "unseen": b.n_unseen,
            "opponents": {
                str(p): {
                    "name": names[p],
                    "two": b.prob_holds_two(p),
                    "ace": b.prob_holds_ace(p),
                    "pair": classes[p]["pair"],
                    "triple": classes[p]["triple"],
                    "beat_table": None if beat is None else beat[p],
                    "rank_map": rank_map[p],
                    "known_hand": b.known_hand(p),
                }
                for p in b.opponents
            },
        }

    mine = _view(HUMAN)
    # Admin surface: every seat's beliefs, including what each bot
    # thinks the human and the other bots are holding.
    mine["seats"] = {
        str(seat): {"name": names[seat], **_view(seat)}
        for seat in range(game.num_players)
    }
    return mine


def register_user(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    username = str(body.get("username") or "").strip()
    token = store.register(username, str(body.get("password") or ""))
    return {"token": token, "username": username,
            "admin": username.lower() in admin_users(),
            "persistent": store.persistent}


def login_user(body: Dict) -> Dict:
    from big2.store import get_store

    store = get_store()
    username = str(body.get("username") or "").strip()
    token = store.login(username, str(body.get("password") or ""))
    return {"token": token, "username": username,
            "admin": username.lower() in admin_users(),
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
    return {"username": auth[1],
            "admin": auth[1].lower() in admin_users(),
            **store.stats(auth[0])}


def leaderboard(_body: Optional[Dict] = None) -> Dict:
    """Public testers leaderboard: usernames, games, wins, scores.

    Admin accounts are filtered out: they can face-up the table and
    read the bots' beliefs, so their scores are not comparable.
    """
    from big2.store import get_store

    board = get_store().leaderboard()
    admins = admin_users()
    for key in ("rows", "testers"):
        if key in board:
            board[key] = [r for r in board[key]
                          if str(r.get("username", "")).lower() not in admins]
    return board


def bot_records(_body: Optional[Dict] = None) -> Dict:
    """Per-bot running record against human players (public)."""
    from big2.store import get_store

    return get_store().bot_records()


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


def recorded_games(body: Dict) -> Dict:
    """Stored human games, newest first, for the admin explorer."""
    from big2.store import get_store

    limit = max(1, min(200, int(body.get("limit", 50))))
    rows = get_store().export_rows(limit) if hasattr(
        get_store(), "export_rows") else []
    return {"games": rows}


def delete_recorded(body: Dict) -> Dict:
    """Admin: remove recorded games by id, or every game for one user.

    Two-step by design: without ``confirm`` it reports exactly what
    would go, so the caller sees the damage before authorising it.
    Deletion is permanent -- the replays are not archived elsewhere.
    """
    from big2.store import get_store

    ids = body.get("ids") or []
    username = (body.get("username") or "").strip() or None
    if not ids and not username:
        raise ValueError("give ids or a username")
    store = get_store()
    doomed = store.preview_delete(ids, username)
    if not body.get("confirm"):
        return {"confirmed": False, "would_delete": len(doomed),
                "games": doomed[:50]}
    n = store.delete_games(ids, username)
    return {"confirmed": True, "deleted": n}


def analyze(body: Dict) -> Dict:
    """Full analysis of one position in a recorded game.

    Rebuilds the hand at action index ``k``, then reports, from the
    acting seat's point of view: what each candidate move is worth to
    the chosen model (policy probability, search visits and values,
    exact solver EV where the position is solvable), which move the
    agent would actually play and why, and the seat's beliefs about
    every opponent -- the 52-card posterior and the combo classes.
    """
    import numpy as np

    from big2.beliefs import BeliefState
    from big2.endgame import move_key, remaining_cards
    from big2.offline import rebuild_game
    from big2.opponents import OpponentModel

    replay = body.get("replay") or {}
    k = int(body.get("k", 0))
    kind = body.get("model") or "wangbot2"
    game = rebuild_game(replay)
    actions = replay.get("actions") or []
    for act in actions[:k]:
        if game.game_over:
            break
        cards = act.get("cards")
        try:
            game.step(None if not cards
                      else classify([int(c) for c in cards], game.rules))
        except (ValueError, RuntimeError):
            break
    if game.game_over:
        return {"over": True}

    seat = game.turn
    options: List[Optional[Combo]] = list(game.legal_moves(seat))
    if game.can_pass():
        options.append(None)

    agent = get_ai(kind, seed=0)
    policy = getattr(agent, "policy", agent)

    # Policy preferences over the candidate set.
    probs: Dict[str, float] = {}
    if hasattr(policy, "option_scores"):
        opts, logits = policy.option_scores(game, seat)
        arr = np.asarray(logits, dtype=np.float64)
        arr = np.exp(arr - arr.max())
        arr = arr / max(arr.sum(), 1e-9)
        for m, pr in zip(opts, arr):
            probs[_move_id(m)] = float(pr)

    # What the deployed agent decides here, and its internals.
    decision: Dict = {}
    inner = getattr(agent, "agent", None)
    if inner is not None and hasattr(inner, "explain"):
        d = inner.explain(game, seat)
        decision = {
            "source": d.source,
            "move": list(d.move) if d.move else None,
            "margin": float(getattr(d, "margin", 0.0)),
            "exact": bool(getattr(d, "exact", False)),
            "elapsed": float(getattr(d, "elapsed", 0.0)),
            "cards_left": int(getattr(d, "cards_left", 0)),
            # Pass has move_key None -- it must keep its row, since it is
            # frequently the move the agent actually chooses.
            "visits": {_ev_key(mk): int(v)
                       for mk, v in (getattr(d, "visits", {}) or {}).items()},
            "values": {_ev_key(mk): float(v)
                       for mk, v in (getattr(d, "values", {}) or {}).items()},
        }

    # Exact per-move EV when the position is inside solver range.
    exact_ev: Dict[str, float] = {}
    if remaining_cards(game) <= 16:
        from big2.endgame import solve_move_values

        values, solved = solve_move_values(game, seat, budget=20000)
        if solved:
            exact_ev = {_ev_key(mk): float(v) for mk, v in values.items()}

    rows = []
    for m in options:
        mid = _move_id(m)
        key = _ev_key(move_key(m) if m else None)
        visits = (decision.get("visits") or {}).get(key)
        value = (decision.get("values") or {}).get(key)
        rows.append({
            "cards": list(m.cards) if m else None,
            "type": (m.type.name if m and hasattr(m.type, "name")
                     else (m.type if m else "pass")),
            "policy": probs.get(mid),
            "visits": visits,
            # An unvisited move carries a placeholder 0.0 internally.
            # Reporting that as a Q reads as "measured, scored zero" and
            # flatters moves nobody simulated -- show nothing instead.
            "value": value if visits else None,
            "exact_ev": exact_ev.get(key),
        })

    # POV beliefs: the deep 52-card posterior plus combo classes.
    honesty = OpponentModel(game, seat).honesty_map(game)
    b = BeliefState(game, seat, pass_honesty=honesty, rng=random.Random(0))
    cmap = b.card_probability_map()
    classes = b.class_probabilities(k=150)
    beat = (b.prob_can_beat(game.table_combo, k=150)
            if game.table_combo else None)
    names = {p: (replay.get("players") or [f"seat {p}"] * 4)[p]
             for p in range(game.num_players)}
    bel = {}
    for p in b.opponents:
        bel[str(p)] = {
            "name": names.get(p, f"seat {p}"),
            "cards": len(game.hands[p]),
            "card_probs": {str(c): float(v) for c, v in cmap[p].items()},
            "classes": {kk: float(vv) for kk, vv in classes[p].items()},
            "beat_table": None if beat is None else float(beat[p]),
            "known_hand": b.known_hand(p),
        }

    return {
        "seat": seat,
        "seat_name": names.get(seat, f"seat {seat}"),
        "hand": sorted(game.hands[seat]),
        "cards_left": remaining_cards(game),
        "table": (list(game.table_combo.cards)
                  if game.table_combo else None),
        "played_move": (actions[k].get("cards") if k < len(actions)
                        else None),
        "options": rows,
        "decision": decision,
        "beliefs": bel,
        "unseen": b.n_unseen,
    }


def _move_id(move: Optional[Combo]) -> str:
    return "pass" if move is None else ",".join(
        str(c) for c in sorted(move.cards))


def _ev_key(mk) -> str:
    """Stable row key for a move key, pass included."""
    return str(list(mk)) if mk else "[]"


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

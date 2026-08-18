"""Midgame lookahead over a coarse opponent model.

The endgame solver enumerates concrete deals, which is only affordable
once most cards are accounted for.  In the middle of a hand there are
far too many deals to enumerate and, more to the point, the individual
cards are the wrong grain: what decides a midgame trick is not *which*
diamond an opponent holds but whether they hold **a flush at all**,
**a pair above yours**, **anything above your king**.

So the midgame tree runs on the chunky distribution instead — the
hand-shape profile (big2/handshape.py) — and asks a much smaller
question at each node:

    if I play this, what is the chance it survives the table,
    and what is the position worth to me either way?

A ply is therefore two branches, not a fan-out over every legal reply:
the move stands (nobody answers, I keep the lead) or it is taken (I
lose the lead and the tempo).  Recursing a few plies deep over those
two outcomes captures the thing that actually matters in the midgame —
whether spending this card buys tempo or throws it away — at a cost
that does not depend on the number of hidden deals.

Two exits keep it honest.  When the position shrinks into solver range
the branch is handed to the exact endgame search, so the coarse tree
terminates in real values rather than guesses.  And when the policy's
value head is available it evaluates any leaf the recursion cannot
finish, so depth is a budget knob rather than a correctness one.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.combos import Combo, ComboType
from big2.game import Big2Game
from big2.handshape import SHAPE_DIM, beat_probability, profile_from_worlds

# Cards remaining at which we stop abstracting and solve exactly.
SOLVE_BELOW = 14


def opponent_profiles(
    game: Big2Game, player: int, k: int = 40,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """(3, SHAPE_DIM) coarse read of what each opponent can do."""
    from big2.inference import InferenceState

    inf = InferenceState(game, player, rng=rng or random.Random(0))
    others = [p for p in range(game.num_players) if p != player]
    return profile_from_worlds(inf.sample_worlds(k), others, game.rules)


def survival_probability(
    game: Big2Game, player: int, move: Combo, profiles: np.ndarray
) -> float:
    """P(nobody answers this move), from the chunky distribution.

    Each opponent is asked the shape question — do you hold a higher
    pair, any five-card hand, a card above this one — and the answers
    are combined as independent events.  That independence is an
    approximation, and a deliberate one: it costs a fraction of the
    exact computation and preserves the ordering that matters.
    """
    from big2.cards import rank

    top = rank(max(move.cards))
    survive = 1.0
    others = [p for p in range(game.num_players) if p != player]
    for j, p in enumerate(others[:3]):
        if not game.hands[p]:
            continue
        beat = beat_probability(profiles[j], move.type, top)
        survive *= (1.0 - float(np.clip(beat, 0.0, 1.0)))
    return float(np.clip(survive, 0.0, 1.0))


def _tempo_value(game: Big2Game, player: int, keeps_lead: bool) -> float:
    """Cheap positional value: how close we are to out, versus the field.

    Used only where no learned value head is supplied.
    """
    mine = len(game.hands[player])
    others = [len(game.hands[p]) for p in range(game.num_players)
              if p != player]
    if not others:
        return 0.0
    edge = (sum(others) / len(others)) - mine
    return edge * 0.6 + (0.8 if keeps_lead else 0.0)


def lookahead_value(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    profiles: np.ndarray,
    depth: int = 2,
    value_fn=None,
    solver_budget: int = 6000,
) -> float:
    """Value of playing ``move``, ``depth`` tempo-plies deep."""
    from big2.endgame import remaining_cards, search_clone, solve_move_values

    sim = search_clone(game)
    try:
        sim.step(move)
    except (ValueError, RuntimeError):
        return -99.0
    if sim.game_over:
        return float(sim.scores[player])

    # Collapsed into solver range: finish exactly rather than guessing.
    if remaining_cards(sim) <= SOLVE_BELOW:
        values, exact = solve_move_values(sim, sim.turn, budget=solver_budget)
        if exact and values:
            best = max(values.values())
            # maxn: the mover takes their best line; read our own share
            # from a fresh solve of the resulting position
            from big2.endgame import solve

            vec = solve(sim, budget=None)
            return float(vec[player]) if vec else best

    if move is None:
        return (value_fn(sim, player) if value_fn
                else _tempo_value(sim, player, keeps_lead=False))

    survive = survival_probability(game, player, move, profiles)
    if depth <= 1:
        held = (value_fn(sim, player) if value_fn
                else _tempo_value(sim, player, keeps_lead=True))
        lost = (value_fn(sim, player) if value_fn
                else _tempo_value(sim, player, keeps_lead=False))
        return survive * held + (1.0 - survive) * lost

    # It stands: we lead again and take our best follow-up.
    lead = search_clone(sim)
    lead.table_combo = None
    lead.table_player = None
    lead.passed = [False] * lead.num_players
    lead.turn = player
    follow = list(lead.legal_moves(player))
    if follow:
        held = max(
            lookahead_value(lead, player, m, profiles, depth - 1, value_fn,
                            solver_budget)
            for m in follow[: 6]
        )
    else:
        held = (value_fn(lead, player) if value_fn
                else _tempo_value(lead, player, keeps_lead=True))

    lost = (value_fn(sim, player) if value_fn
            else _tempo_value(sim, player, keeps_lead=False))
    return survive * held + (1.0 - survive) * lost


def lookahead_values(
    game: Big2Game,
    player: int,
    options: Sequence[Optional[Combo]],
    depth: int = 2,
    k_worlds: int = 40,
    value_fn=None,
    rng: Optional[random.Random] = None,
    profiles: Optional[np.ndarray] = None,
) -> Dict[Optional[Tuple[int, ...]], float]:
    """Coarse EV of every option, keyed like the endgame solver's."""
    from big2.endgame import move_key

    if profiles is None:
        profiles = opponent_profiles(game, player, k=k_worlds, rng=rng)
    return {
        move_key(m): lookahead_value(game, player, m, profiles, depth, value_fn)
        for m in options
    }

"""Endgame tree search: exact backward induction over determinizations.

Late in a Big 2 hand the state is small and *nearly deterministic* — few
cards are unseen, and many of ours are provably unbeatable.  That is
exactly the regime where search beats a policy network, and where the
champion's mistakes lived: it would dribble out a cheap single while
holding the boss card, handing the trick (and the game) to a player who
was one card from out.

Three layers, cheapest first:

1. ``boss_move`` / ``unbeatable_probability`` — *no search at all*.  If
   every card that could beat a move is already played or in our own
   hand, the move is unbeatable as a matter of fact, not probability.
   Combined with ``boss_chain`` (how much of our hand is unbeatable) this
   answers "can I just run out from here?" exactly.

2. ``solve`` — **maxn backward induction** on a perfect-information
   state.  Big 2 with card-count payments is not two-player zero-sum, so
   minimax does not apply: every player maximizes *their own* final
   score, which is the maxn algorithm (Luckhardt & Irani).  Memoized on
   the full state key, with a node budget and a static fallback so a
   too-large state degrades instead of hanging.

3. ``pimc_move_values`` — **Perfect Information Monte Carlo** (Ginsberg's
   GIB for bridge): sample determinizations of the hidden hands from the
   belief posterior, solve each exactly, average per move.  PIMC is known
   to suffer strategy fusion (it assumes hidden state is revealed), which
   is precisely why it is used *only* near the end and blended with the
   learned value (see big2/search.py) rather than trusted outright.

The solver returns a score *vector* (one entry per seat) so the caller
sees the whole payout structure, not just its own EV.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from big2.cards import Card
from big2.combos import Combo, generate_moves
from big2.game import Big2Game

# A move's identity for dict keys: None (pass) or its card tuple.
MoveKey = Optional[Tuple[Card, ...]]

DEFAULT_BUDGET = 40_000


def move_key(move: Optional[Combo]) -> MoveKey:
    return None if move is None else tuple(move.cards)


# ----------------------------------------------------------------------
# Layer 1: exact "can anyone beat this?" — no sampling, no search
# ----------------------------------------------------------------------


def outstanding_cards(game: Big2Game, player: int) -> List[Card]:
    """Cards neither played nor in ``player``'s hand: what opponents may
    hold (a superset in 2-3 player games, where some cards are undealt)."""
    seen = set(game.played_cards) | set(game.hands[player])
    return [c for c in range(52) if c not in seen]


def boss_move(game: Big2Game, player: int, move: Combo,
              pool: Optional[Sequence[Card]] = None) -> bool:
    """True if *no* arrangement of the outstanding cards beats ``move``.

    Checked against the union of everything we cannot see: if even a
    player holding every outstanding card could not answer, then no
    actual opponent can.  Exact, and one ``generate_moves`` call.
    """
    if pool is None:
        pool = outstanding_cards(game, player)
    return not generate_moves(list(pool), move, game.rules)


def boss_chain(game: Big2Game, player: int) -> Tuple[int, int]:
    """(unbeatable units, total units) for our hand's greedy partition.

    If every unit is unbeatable and we hold the lead, we can simply run
    the hand out — the "play the trump first" plan, made explicit.
    """
    from big2.strategies import SmartHeuristic

    pool = outstanding_cards(game, player)
    units = SmartHeuristic._partition(game.hands[player])
    boss = sum(1 for u in units if boss_move(game, player, u, pool))
    return boss, len(units)


def unbeatable_probability(
    game: Big2Game,
    player: int,
    move: Combo,
    worlds: Optional[Sequence[Tuple[Dict[int, List[Card]], float]]] = None,
) -> float:
    """P(no opponent can answer ``move``).

    Exactly 1.0 when the move is boss (no sampling needed).  Otherwise
    the belief-weighted fraction of sampled worlds in which no opponent
    holds an answer.
    """
    if boss_move(game, player, move):
        return 1.0
    if not worlds:
        return 0.0
    num = den = 0.0
    for world, w in worlds:
        den += w
        if not any(
            generate_moves(hand, move, game.rules) for hand in world.values()
        ):
            num += w
    return num / den if den else 0.0


# ----------------------------------------------------------------------
# Layer 2: exact maxn backward induction
# ----------------------------------------------------------------------


def _state_key(game: Big2Game) -> Tuple:
    return (
        tuple(tuple(h) for h in game.hands),
        game.turn,
        None if game.table_combo is None else tuple(game.table_combo.cards),
        game.table_player,
        tuple(game.passed),
        game.first_play,
    )


def _static_eval(game: Big2Game) -> Tuple[float, ...]:
    """Budget-exhausted fallback: assume the shortest hand goes out and
    everyone pays their current holding."""
    pays = [game.scoring.payment(h) for h in game.hands]
    lead = min(range(game.num_players), key=lambda p: len(game.hands[p]))
    out = [-float(pays[p]) for p in range(game.num_players)]
    out[lead] = float(sum(pays) - pays[lead])
    return tuple(out)


class _Budget:
    __slots__ = ("left",)

    def __init__(self, n: int):
        self.left = n

    def spend(self) -> bool:
        self.left -= 1
        return self.left >= 0


def search_clone(game: Big2Game) -> Big2Game:
    """Clone for search: drops the history and played-cards bookkeeping,
    which the solver never reads (legality comes from hands + table, and
    scoring from hands).  Copying a 40-entry played list at every node
    otherwise dominates the search."""
    g = object.__new__(Big2Game)
    g.scoring = game.scoring
    g.rules = game.rules
    g.num_players = game.num_players
    g.rng = game.rng
    g.hands = [list(h) for h in game.hands]
    g.start_card = game.start_card
    g.turn = game.turn
    g.first_play = game.first_play
    g.table_combo = game.table_combo
    g.table_player = game.table_player
    g.passed = list(game.passed)
    g.history = []
    g.played_cards = []
    g.winner = game.winner
    g.scores = game.scores
    return g


# Legal-move cache: the same (hand, table) pairs recur constantly across
# branches of the tree, and move generation is the solver's hot spot.
_MOVE_CACHE: Dict[Tuple, Tuple[Combo, ...]] = {}
_MOVE_CACHE_CAP = 200_000


def cached_moves(game: Big2Game) -> Tuple[Combo, ...]:
    hand = game.hands[game.turn]
    key = (
        tuple(hand),
        None if game.table_combo is None else game.table_combo.cards,
        game.first_play,
        id(game.rules),
    )
    hit = _MOVE_CACHE.get(key)
    if hit is None:
        if len(_MOVE_CACHE) > _MOVE_CACHE_CAP:
            _MOVE_CACHE.clear()
        hit = tuple(game.legal_moves())
        _MOVE_CACHE[key] = hit
    return hit


def solve(
    game: Big2Game,
    memo: Optional[Dict[Tuple, Tuple[float, ...]]] = None,
    budget: Optional[_Budget] = None,
) -> Tuple[float, ...]:
    """Exact maxn value vector of a perfect-information position.

    Every player maximizes their own component; ties keep the first
    (move-generation) order, which is stable.  ``game`` is not mutated.
    """
    if game.game_over:
        return tuple(
            float(game.scores[p]) for p in range(game.num_players)
        )
    memo = {} if memo is None else memo
    budget = _Budget(DEFAULT_BUDGET) if budget is None else budget
    key = _state_key(game)
    hit = memo.get(key)
    if hit is not None:
        return hit
    if not budget.spend():
        return _static_eval(game)

    player = game.turn
    options: List[Optional[Combo]] = list(cached_moves(game))
    if game.can_pass():
        options.append(None)
    best: Optional[Tuple[float, ...]] = None
    for m in options:
        child = search_clone(game)
        child.step(m)
        val = solve(child, memo, budget)
        if best is None or val[player] > best[player]:
            best = val
        # Nothing beats going out with the whole pot: stop looking.
        if best is not None and not child.hands[player] and child.winner == player:
            break
    if best is None:  # no options at all (cannot happen: leading always has moves)
        best = _static_eval(game)
    memo[key] = best
    return best


def solve_move_values(
    game: Big2Game,
    player: int,
    budget: int = DEFAULT_BUDGET,
) -> Tuple[Dict[MoveKey, float], bool]:
    """(exact value of each option for ``player``, solved_exactly).

    The flag matters: once the node budget runs out the solver starts
    guessing with a static evaluation, and a guess dressed as an exact
    value is worse than no search at all.  Callers drop unsolved
    positions rather than trusting them (see big2/search.py).
    """
    memo: Dict[Tuple, Tuple[float, ...]] = {}
    b = _Budget(budget)
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    out: Dict[MoveKey, float] = {}
    for m in options:
        child = search_clone(game)
        child.step(m)
        out[move_key(m)] = solve(child, memo, b)[player]
    return out, b.left >= 0


# ----------------------------------------------------------------------
# Layer 3: PIMC over belief-sampled determinizations
# ----------------------------------------------------------------------


def remaining_cards(game: Big2Game) -> int:
    return sum(len(h) for h in game.hands)


def pimc_move_values(
    game: Big2Game,
    player: int,
    worlds: Sequence[Tuple[Dict[int, List[Card]], float]],
    budget: int = DEFAULT_BUDGET,
    with_agreement: bool = False,
):
    """Belief-weighted average exact value of every option.

    Each world is a full assignment of the hidden hands; the position is
    then perfect information and ``solve`` gives the exact continuation
    value under maxn play.

    With ``with_agreement`` the result also carries how *settled* the
    position is: the weighted share of worlds whose own best move is the
    move that wins on average.  Agreement near 1 means the unseen cards
    no longer change the answer — the hand has become effectively
    deterministic and the tree can be trusted.  Low agreement means the
    right move depends on cards we cannot see, which is exactly where
    perfect-information search deceives itself (strategy fusion) and the
    learned policy should keep the decision.
    """
    totals: Dict[MoveKey, float] = {}
    per_world: List[Tuple[MoveKey, float]] = []
    weight = 0.0
    for world, w in worlds:
        if w <= 0.0:
            continue
        det = search_clone(game)
        for p, hand in world.items():
            det.hands[p] = sorted(hand)
        vals, exact = solve_move_values(det, player, budget)
        if not exact:
            continue  # budget ran out: a guess, not a solve — drop it
        weight += w
        for k, v in vals.items():
            totals[k] = totals.get(k, 0.0) + w * v
        if vals:
            per_world.append((max(vals, key=vals.get), w))
    if not weight:
        return ({}, 0.0) if with_agreement else {}
    mean = {k: v / weight for k, v in totals.items()}
    if not with_agreement:
        return mean
    best = max(mean, key=mean.get) if mean else None
    agree = sum(w for k, w in per_world if k == best) / weight
    return mean, agree

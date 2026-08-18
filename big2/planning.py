"""Planning features: "is this move answerable, and what happens after?"

The v1.1 danger block told the net *how many cards* opponents held.  It
still had no notion of **control** — whether a move can be answered at
all, and what our hand looks like as a sequence of plays rather than a
bag of cards.  That is the gap behind the endgame misplay: holding the
boss card and dribbling out a cheap single, handing the lead (and the
game) to someone about to go out.

Two exact ideas, both cheap:

* **Boss detection.**  A move is *boss* when nothing outside our hand
  can answer it — computed against the union of all unseen cards, so it
  is a fact, not an estimate.  Singles and pairs resolve in a rank scan;
  5-card hands fall back to move generation only once the unseen pool is
  small (late game), and report "unknown" before that rather than
  burning time mid-game.

* **Run-out planning.**  Our hand partitions into units; counting how
  many are boss says whether we can simply run the hand out from the
  lead.  ``turns to empty`` against the opponents' card counts turns the
  endgame into an explicit race, and ``wastes_boss`` flags precisely the
  losing pattern: playing a non-boss card while holding a boss unit.

Everything here is arithmetic over sets — no sampling, no search — so it
runs inside a training rollout at every decision.  The expensive layers
(belief-weighted determinizations, exact solving) live in
``big2.inference`` and ``big2.endgame`` and are used near the end of the
game by ``big2.search``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from big2.cards import Card, rank
from big2.combos import Combo, ComboType, classify, generate_moves
from big2.game import Big2Game

PLAN_DIM = 16          # per candidate move
PLAN_STATE_DIM = 6     # per decision (hand-level plan summary)

# Above this many unseen cards, 5-card boss checks are skipped as too
# costly (and rarely decisive that early).
FIVE_CARD_POOL_CAP = 16


def _pair_key(cards: Sequence[Card]) -> tuple:
    c = sorted(cards)
    return (rank(c[1]), c[1] % 4)


def beatable(
    game: Big2Game, move: Combo, pool: Sequence[Card]
) -> Optional[bool]:
    """Can any arrangement of ``pool`` beat ``move``?

    Exact for singles and pairs; for 5-card hands, exact when the pool is
    small and ``None`` (unknown) when checking would be expensive.
    """
    if move.type == ComboType.SINGLE:
        top = move.cards[0]
        return any(c > top for c in pool)
    if move.type == ComboType.PAIR:
        by_rank: Dict[int, List[Card]] = {}
        for c in pool:
            by_rank.setdefault(rank(c), []).append(c)
        mine = _pair_key(move.cards)
        for r, cs in by_rank.items():
            if len(cs) >= 2 and (r, max(cs) % 4) > mine:
                return True
        return False
    if len(pool) > FIVE_CARD_POOL_CAP:
        return None
    return bool(generate_moves(list(pool), move, game.rules))


class PlanContext:
    """Per-decision scratch shared by every candidate move."""

    def __init__(self, game: Big2Game, player: int):
        from big2.strategies import SmartHeuristic

        self.game = game
        self.player = player
        seen = set(game.played_cards) | set(game.hands[player])
        self.pool: List[Card] = [c for c in range(52) if c not in seen]
        self.hand: List[Card] = list(game.hands[player])
        self.units: List[Combo] = SmartHeuristic._partition(self.hand)
        self.boss_units = [
            u for u in self.units if beatable(game, u, self.pool) is False
        ]
        self.n_units = max(1, len(self.units))
        others = [p for p in range(game.num_players) if p != player]
        self.opp_counts = [len(game.hands[p]) for p in others]
        self.next_count = len(game.hands[(player + 1) % game.num_players])
        self.min_opp = min(self.opp_counts) if self.opp_counts else 13
        self._boss_cache: Dict[tuple, Optional[bool]] = {}

    def is_boss(self, move: Combo) -> Optional[bool]:
        key = tuple(move.cards)
        if key not in self._boss_cache:
            b = beatable(self.game, move, self.pool)
            self._boss_cache[key] = None if b is None else (not b)
        return self._boss_cache[key]

    def units_after(self, move: Optional[Combo]) -> int:
        """Turns still needed once ``move`` is played.

        Re-partitioning per candidate move doubled the cost of encoding a
        decision, so this walks the existing partition instead: units the
        move consumes whole disappear, units it breaks into leave their
        remainder as loose cards.  Exact when the move is a unit (the
        common case) and a tight upper bound otherwise.
        """
        if move is None:
            return len(self.units)
        played = set(move.cards)
        left = 0
        for u in self.units:
            rest = [c for c in u.cards if c not in played]
            if not rest:
                continue                    # unit spent exactly
            left += 1 if len(rest) == len(u) else len(rest)
        return left


def plan_state_features(ctx: PlanContext) -> np.ndarray:
    """Hand-level plan summary (same for every candidate move)."""
    f = np.zeros(PLAN_STATE_DIM, dtype=np.float32)
    n_boss = len(ctx.boss_units)
    f[0] = n_boss / ctx.n_units                    # share of hand that is boss
    f[1] = min(1.0, n_boss / 4.0)                  # absolute control
    f[2] = min(1.0, len(ctx.units) / 8.0)          # turns needed to empty
    # Can we simply run the hand out from the lead?
    f[3] = 1.0 if (n_boss == len(ctx.units) and ctx.game.leading) else 0.0
    # Race: our turns to empty vs the shortest opponent hand.
    f[4] = float(np.clip((ctx.min_opp - len(ctx.units)) / 6.0, -1.0, 1.0))
    f[5] = 1.0 if len(ctx.units) <= 2 else 0.0     # two plays from out
    return f


def plan_features(ctx: PlanContext, move: Optional[Combo]) -> np.ndarray:
    """Per-move planning block."""
    f = np.zeros(PLAN_DIM, dtype=np.float32)
    if move is None:                                # pass
        f[0] = 1.0
        f[1] = len(ctx.boss_units) / ctx.n_units
        return f

    boss = ctx.is_boss(move)
    empties = len(move) == len(ctx.hand)
    after = ctx.units_after(move)

    f[2] = 1.0 if boss else 0.0
    f[3] = 1.0 if boss is None else 0.0            # unknown (mid-game 5-card)
    f[4] = 1.0 if empties else 0.0                 # this play ends the game
    f[5] = 1.0 if (empties and boss) else 0.0      # ...and cannot be stopped
    # Unbeatable and it keeps the lead: we get to play again.
    f[6] = 1.0 if (boss and not empties) else 0.0
    # The champion's bug: spending a non-boss card while holding control.
    f[7] = 1.0 if (boss is False and ctx.boss_units) else 0.0
    f[8] = min(1.0, after / 8.0)                   # turns left after this
    f[9] = 1.0 if after == 0 else 0.0
    f[10] = 1.0 if after == 1 else 0.0             # one play from out
    # Race margin after the move: our remaining turns vs shortest hand.
    f[11] = float(np.clip((ctx.min_opp - after) / 6.0, -1.0, 1.0))
    # Denial: an unanswerable move while someone is nearly out.
    f[12] = 1.0 if (boss and ctx.min_opp <= 2) else 0.0
    # Gifting: an answerable move while the next player is nearly out.
    f[13] = 1.0 if (boss is False and ctx.next_count <= 2) else 0.0
    # How much of the unseen pool outranks this move's top card.
    top = max(move.cards)
    f[14] = (
        sum(1 for c in ctx.pool if c > top) / len(ctx.pool)
        if ctx.pool else 0.0
    )
    f[15] = len(move) / 5.0
    return f


def boss_singles(game: Big2Game, player: int) -> List[Card]:
    """Cards in hand that win outright as singles right now."""
    seen = set(game.played_cards) | set(game.hands[player])
    pool = [c for c in range(52) if c not in seen]
    top_unseen = max(pool) if pool else -1
    return [c for c in game.hands[player] if c > top_unseen]

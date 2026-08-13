"""Baseline strategies for Big 2.

A strategy implements ``select(game, player) -> Optional[Combo]`` where
``None`` means pass.  Strategies must only return legal moves (a combo
from ``game.legal_moves()`` or None when ``game.can_pass()``).

Baselines:
- RandomPolicy:    uniform over legal moves (+ pass when allowed).
- PlayLowest:      always the weakest feasible play; leads its lowest single.
- PlayHighest:     always the strongest feasible play; leads its biggest class.
- FiveCardDumper:  leads 5-card hands (lowest first), else pairs, else
                   singles; follows with the weakest feasible play.
- SmartHeuristic:  partitions its hand into units (5-card hands, pairs,
                   singles), avoids breaking units, passes to protect
                   strong holdings, and plays to deny near-finished
                   opponents.
"""

from __future__ import annotations

import random
from typing import List, Optional

from big2.cards import Card, rank
from big2.combos import Combo, classify, generate_moves
from big2.game import NUM_PLAYERS, Big2Game


class Strategy:
    name = "base"

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


def _weakest_of_len(moves: List[Combo], n: int) -> Optional[Combo]:
    """First (weakest) move of size n; moves are sorted weakest-first."""
    for m in moves:
        if len(m) == n:
            return m
    return None


class RandomPolicy(Strategy):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        return self.rng.choice(options)


class PlayLowest(Strategy):
    """Play the weakest feasible combo; never passes voluntarily."""

    name = "lowest"

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves = game.legal_moves(player)
        if not moves:
            return None
        if game.leading:
            return moves[0]  # singles sort first -> lowest single
        return moves[0]


class PlayHighest(Strategy):
    """Play the strongest feasible combo; leads with its biggest class."""

    name = "highest"

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves = game.legal_moves(player)
        if not moves:
            return None
        if game.leading:
            biggest = max(len(m) for m in moves)
            return [m for m in moves if len(m) == biggest][-1]
        return moves[-1]


class FiveCardDumper(Strategy):
    """Shed the largest classes early: lead lowest 5-card hand, then pairs,
    then singles; follow with the weakest feasible play."""

    name = "dumper"

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves = game.legal_moves(player)
        if not moves:
            return None
        if game.leading:
            for size in (5, 2, 1):
                m = _weakest_of_len(moves, size)
                if m is not None:
                    return m
        return moves[0]


class SmartHeuristic(Strategy):
    """Unit-preserving heuristic.

    - Partitions the hand greedily into 5-card hands, pairs, and loose
      singles (5-card units are chosen to use the lowest cards so high
      singles stay flexible).
    - Follows with the weakest move that does not break a unit; passes
      to protect units while the hand is still large.
    - Endgame (<= ``endgame_size`` cards): plays the weakest feasible
      move unconditionally, since shedding beats preserving.
    - Denies opponents who are nearly out by playing high singles.
    """

    name = "smart"

    def __init__(self, endgame_size: int = 6, deny_threshold: int = 2):
        self.endgame_size = endgame_size
        self.deny_threshold = deny_threshold

    # -- hand partitioning -------------------------------------------------

    @staticmethod
    def _partition(hand: List[Card]) -> List[Combo]:
        remaining = list(hand)
        units: List[Combo] = []
        while True:
            five_cards = [m for m in generate_moves(remaining) if len(m) == 5]
            if not five_cards:
                break
            best = min(five_cards, key=lambda m: sum(m.cards))
            units.append(best)
            for c in best.cards:
                remaining.remove(c)
        pairs = [m for m in generate_moves(remaining) if len(m) == 2]
        used = set()
        for p in sorted(pairs, key=lambda m: m.sort_key):
            if not used.intersection(p.cards):
                units.append(p)
                used.update(p.cards)
        for c in remaining:
            if c not in used:
                units.append(classify((c,)))
        return units

    @staticmethod
    def _non_breaking(moves: List[Combo], units: List[Combo]) -> List[Combo]:
        unit_sets = [frozenset(u.cards) for u in units]
        keep = []
        for m in moves:
            ms = frozenset(m.cards)
            if any(ms == u for u in unit_sets):
                keep.append(m)
        return keep

    # -- decision ----------------------------------------------------------

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves = game.legal_moves(player)
        if not moves:
            return None

        hand = game.hands[player]
        opp_counts = [
            len(game.hands[p]) for p in range(NUM_PLAYERS) if p != player
        ]
        danger = min(opp_counts) <= self.deny_threshold
        endgame = len(hand) <= self.endgame_size

        units = self._partition(hand)
        safe = self._non_breaking(moves, units)

        if game.leading:
            if danger and not endgame:
                # Don't hand a near-finished opponent a cheap ride: lead a
                # multi-card unit if possible, else our highest single.
                for size in (5, 2):
                    m = _weakest_of_len(safe, size)
                    if m is not None:
                        return m
                singles = [m for m in moves if len(m) == 1]
                return singles[-1] if singles else moves[0]
            for size in (1, 2, 5):
                m = _weakest_of_len(safe, size)
                if m is not None:
                    return m
            return moves[0]

        # Following.
        if endgame:
            return moves[0]
        if danger and len(moves[0]) == 1:
            # Deny: beat with our strongest single (prefer non-breaking).
            pool = safe if safe else moves
            return pool[-1]
        if safe:
            return safe[0]
        # Only unit-breaking moves available: protect the hand and pass.
        return None

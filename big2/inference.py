"""Opponent-hand inference from public play: what their moves give away.

``beliefs.BeliefState`` already weights sampled worlds by *pass*
evidence — a player who passed probably could not beat the table.  This
module adds the other half of the signal, the one that shows up when a
player **plays higher than they had to**:

    the table is a 5D, and they answer with the K-spades.

A rational player answers with the cheapest card that wins.  Playing far
above the requirement is evidence about the cards in between:

* **they don't hold them** — the plain reading, so worlds where they held
  a cheap loose winner are down-weighted; or
* **they hold them locked in a unit** — a 6-6 pair or a straight they
  won't break.  So the same evidence *raises* the odds that any
  intermediate card they do hold sits inside a pair or a 5-card hand.

Both readings are handled by scoring a world against what the player
*could* have done instead: a world is penalized only when it gives them
a cheaper winner that was **loose** (breaking nothing).  Worlds where the
cheap card was tied up in a unit survive at full weight — so the
posterior naturally shifts probability toward "their low cards are in
combinations", which is exactly the read a strong human makes.

Two consumers:

* ``InferenceState.worlds_for_search`` feeds determinizations to the
  endgame solver (big2/endgame.py) — better worlds, better search.
* ``overplay_features`` is a cheap per-opponent summary (no sampling)
  for the policy network's inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from big2.beliefs import BeliefState
from big2.cards import Card, NUM_CARDS, rank
from big2.combos import Combo, generate_moves
from big2.game import Big2Game

OVERPLAY_DIM = 6


@dataclass(frozen=True)
class _SkipEvidence:
    """They beat ``table`` with ``played`` — did a cheaper answer exist?"""

    player: int
    table: Combo
    played: Combo
    later_cards: Tuple[Card, ...]  # cards they played after this moment


def _loose_cards(hand: Sequence[Card], cards: Sequence[Card]) -> bool:
    """True if none of ``cards`` is part of a pair or a 5-card unit in
    ``hand`` — i.e. playing them breaks nothing worth keeping."""
    from big2.strategies import SmartHeuristic

    ranks = [rank(c) for c in hand]
    for c in cards:
        if ranks.count(rank(c)) > 1:
            return False
    keep = {
        frozenset(u.cards)
        for u in SmartHeuristic._partition(list(hand))
        if len(u) >= 5
    }
    for unit in keep:
        if any(c in unit for c in cards):
            return False
    return True


class InferenceState(BeliefState):
    """Belief state that also reads *overplay* evidence.

    ``skip_honesty`` is how strongly to trust that a player would have
    taken a cheaper winner when one was loose in hand: 1.0 ignores the
    signal entirely, 0.0 rules such worlds out.
    """

    def __init__(
        self,
        game: Big2Game,
        viewpoint: int,
        pass_honesty=0.85,
        skip_honesty: float = 0.45,
        rng: Optional[random.Random] = None,
    ):
        self.skip_honesty = float(skip_honesty)
        self._skips: List[_SkipEvidence] = []  # before super(): sampling is
        super().__init__(game, viewpoint, pass_honesty=pass_honesty, rng=rng)
        self._skips = self._collect_skip_evidence()  # lazy, so this is in time

    # ------------------------------------------------------------------

    def _collect_skip_evidence(self) -> List[_SkipEvidence]:
        table: Optional[Combo] = None
        out: List[Tuple[int, Combo, Combo, List[Card]]] = []
        for rec in self.game.history:
            if rec.combo is None:
                continue
            if table is not None and rec.player != self.viewpoint:
                out.append((rec.player, table, rec.combo, []))
            for entry in out:
                if entry[0] == rec.player:
                    entry[3].extend(rec.combo.cards)
            table = rec.combo
            if rec.trick_end:
                table = None
        return [
            _SkipEvidence(p, t, m, tuple(later)) for p, t, m, later in out
        ]

    def _skip_weight(self, world: Dict[int, List[Card]]) -> float:
        if not self._skips or self.skip_honesty >= 1.0:
            return 1.0
        w = 1.0
        for ev in self._skips:
            if ev.player not in world:
                continue
            hand_then = list(world[ev.player]) + list(ev.later_cards)
            cheaper = [
                m
                for m in generate_moves(hand_then, ev.table, self.game.rules)
                if m.sort_key < ev.played.sort_key and len(m) == len(ev.played)
            ]
            # Only a *loose* cheaper winner is evidence against the world:
            # skipping one that would break a pair or a 5-card unit is
            # ordinary good play, not information.
            if any(_loose_cards(hand_then, m.cards) for m in cheaper):
                w *= self.skip_honesty
        return w

    def _world_weight(self, world: Dict[int, List[Card]]) -> float:
        return super()._world_weight(world) * self._skip_weight(world)

    # ------------------------------------------------------------------

    def card_marginals(self, k: int = 150) -> Dict[int, List[float]]:
        """P(opponent p holds card c), evidence-weighted, per opponent."""
        worlds = self.sample_worlds(k)
        out = {p: [0.0] * NUM_CARDS for p in self.opponents}
        total = sum(w for _, w in worlds) or 1.0
        for world, w in worlds:
            if w <= 0.0:
                continue
            for p, hand in world.items():
                row = out[p]
                for c in hand:
                    row[c] += w
        for p in out:
            out[p] = [v / total for v in out[p]]
        return out

    def worlds_for_search(
        self, k: int = 24, top: Optional[int] = None
    ) -> List[Tuple[Dict[int, List[Card]], float]]:
        """Determinizations for the endgame solver, best-weighted first.

        Zero-weight worlds are dropped; ``top`` keeps only the most
        plausible ones so the (expensive) exact search is spent where the
        posterior actually lives.
        """
        worlds = [(w, wt) for w, wt in self.sample_worlds(k) if wt > 0.0]
        worlds.sort(key=lambda pair: -pair[1])
        return worlds[:top] if top else worlds


# ----------------------------------------------------------------------
# Cheap (sampling-free) summary features for the policy network
# ----------------------------------------------------------------------


def overplay_features(game: Big2Game, player: int) -> Dict[int, List[float]]:
    """Per-opponent read of how they have been answering the table.

    All counting, no sampling — cheap enough to compute inside a
    training rollout at every decision.  Per opponent:

    0. overplay rate: answers made far above the requirement
    1. mean size of the jump (rank gap / 12), how far above they went
    2. rate of answering with the *minimum* winner (tight, card-poor play)
    3. lead-high rate: leading with a high card instead of a low one
    4. pass-then-play rate: passed a trick, later played in it (soft pass)
    5. sample weight: how many answers we have actually seen (confidence)
    """
    n = game.num_players
    out = {p: [0.0] * OVERPLAY_DIM for p in range(n) if p != player}
    answers = {p: 0 for p in out}
    jumps = {p: 0.0 for p in out}
    over = {p: 0 for p in out}
    tight = {p: 0 for p in out}
    leads = {p: 0 for p in out}
    high_leads = {p: 0.0 for p in out}
    passed_trick = {p: False for p in out}
    pass_then_play = {p: 0 for p in out}

    table: Optional[Combo] = None
    for rec in game.history:
        p = rec.player
        if rec.combo is None:
            if p in passed_trick:
                passed_trick[p] = True
        else:
            if p in out:
                if table is None:
                    leads[p] += 1
                    high_leads[p] += rank(max(rec.combo.cards)) / 12.0
                else:
                    answers[p] += 1
                    gap = (
                        rank(max(rec.combo.cards)) - rank(max(table.cards))
                    ) / 12.0
                    jumps[p] += max(0.0, gap)
                    if gap > 0.25:
                        over[p] += 1
                    elif gap <= 1 / 12.0 + 1e-9:
                        tight[p] += 1
                    if passed_trick[p]:
                        pass_then_play[p] += 1
            table = rec.combo
        if rec.trick_end:
            table = None
            for q in passed_trick:
                passed_trick[q] = False

    for p, row in out.items():
        a = answers[p]
        if a:
            row[0] = over[p] / a
            row[1] = jumps[p] / a
            row[2] = tight[p] / a
            row[4] = pass_then_play[p] / a
        if leads[p]:
            row[3] = high_leads[p] / leads[p]
        row[5] = min(1.0, a / 8.0)
    return out

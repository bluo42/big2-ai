"""Belief tracking: what each opponent probably holds.

Big 2 deals the whole deck, so from any player's viewpoint the hidden
state is just an assignment of the *unseen* cards (deck minus my hand
minus everything played) to the other players' known hand counts — plus,
in 2-3 player games, an undealt pile of known size.  That structure
gives two complementary layers:

**Analytic layer (exact under uniform deals, O(1) per query).**
Marginals are hypergeometric: an opponent holding n of the U unseen
cards holds any specific card with probability n/U, and holds at least
one of a target set of H cards with probability 1 - C(U-H, n)/C(U, n).
This covers the per-card probability map, "holds a 2 / an ace / the 2♠",
and "holds something that beats this single".  As the game progresses U
shrinks and these sharpen automatically — pure card elimination: when
U equals a player's count, their hand is known exactly.

**Monte-Carlo layer (any event, with play evidence).**
Sample complete worlds consistent with the counts, optionally weighted
by *pass evidence*: a player who passed on a trick is less likely to
have held cards that beat what was on the table.  Strategic passing is
legal, so this is soft evidence — a world where a passer could have
beaten the table is down-weighted by ``pass_honesty`` (0 = passes are
fully honest, 1 = passes carry no information).  Events that need whole
hands (has a pair / triple / bomb, can beat a 5-card combo) come from
these weighted samples.

These probabilities feed agent features (big2/dmc.py), belief-weighted
determinization for search (big2/ismcts.py), the Gym observation, and
the UI's belief panel.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from big2.cards import ACE_RANK, NUM_CARDS, NUM_RANKS, TWO_RANK, Card, rank, suit
from big2.combos import Combo, generate_moves
from big2.game import Big2Game


def prob_at_least_one(hits: int, unseen: int, hand_size: int) -> float:
    """P(a uniform hand of ``hand_size`` from ``unseen`` cards contains
    >= 1 of ``hits`` target cards)."""
    if hits <= 0 or hand_size <= 0 or unseen <= 0:
        return 0.0
    if hits + hand_size > unseen:
        return 1.0
    return 1.0 - math.comb(unseen - hits, hand_size) / math.comb(unseen, hand_size)


@dataclass
class _PassEvidence:
    player: int
    table: Combo
    played_after: Tuple[Card, ...]  # cards this player shed after passing


class BeliefState:
    """Beliefs about hidden hands from one player's viewpoint."""

    def __init__(self, game: Big2Game, viewpoint: int,
                 pass_honesty=0.35,
                 rng: Optional[random.Random] = None):
        """``pass_honesty`` may be a float (same weight for everyone) or a
        per-opponent dict {player: weight} — e.g. from
        OpponentModel.honesty_map, which adapts to observed pass styles."""
        self.game = game
        self.viewpoint = viewpoint
        self.pass_honesty = pass_honesty
        self.rng = rng or random.Random(0)

        seen = set(game.hands[viewpoint]) | set(game.played_cards)
        self.unseen: List[Card] = [c for c in range(NUM_CARDS) if c not in seen]
        self.opponents: List[int] = [
            p for p in range(game.num_players) if p != viewpoint
        ]
        self.counts: Dict[int, int] = {
            p: len(game.hands[p]) for p in self.opponents
        }
        self._evidence = self._collect_pass_evidence()
        # Fitted play-evidence factors (big2/playmodel.py): each single
        # an opponent played shifts the odds they hold its pair-mate,
        # damped and clamped.  Collected once; weighted into every
        # sampled world below, so search determinization, class
        # probabilities and can-beat estimates all inherit it.
        try:
            from big2.playmodel import collect_events
            self._play_events = collect_events(game, viewpoint)
        except Exception:
            self._play_events = []
        self._worlds: Optional[List[Tuple[Dict[int, List[Card]], float]]] = None

    # ------------------------------------------------------------------
    # Analytic layer
    # ------------------------------------------------------------------

    @property
    def n_unseen(self) -> int:
        return len(self.unseen)

    def card_probability_map(self) -> Dict[int, Dict[Card, float]]:
        """P(opponent p holds card c) for every unseen card: the uniform
        hypergeometric row, nudged by fitted play evidence (a player who
        led a lone single is less likely to hold its pair-mate) and
        re-fit so rows still sum to hand counts and columns to 1."""
        u = self.n_unseen
        if not u:
            return {p: {} for p in self.opponents}
        base = {
            p: {c: self.counts[p] / u for c in self.unseen}
            for p in self.opponents
        }
        if not self._play_events:
            return base
        from big2.playmodel import adjust_card_map
        return adjust_card_map(base, self._play_events, self.counts)

    def rank_probability_map(self) -> Dict[int, List[float]]:
        """P(opponent p holds >= 1 card of each rank), 13 entries each."""
        unseen_per_rank = [0] * NUM_RANKS
        for c in self.unseen:
            unseen_per_rank[rank(c)] += 1
        return {
            p: [
                prob_at_least_one(unseen_per_rank[r], self.n_unseen, self.counts[p])
                for r in range(NUM_RANKS)
            ]
            for p in self.opponents
        }

    def prob_holds_any(self, player: int, cards: Sequence[Card]) -> float:
        """P(player holds >= 1 of these specific cards)."""
        hits = sum(1 for c in cards if c in set(self.unseen))
        return prob_at_least_one(hits, self.n_unseen, self.counts[player])

    def prob_holds_rank(self, player: int, r: int) -> float:
        return self.prob_holds_any(player, [c for c in self.unseen if rank(c) == r])

    def prob_holds_two(self, player: int) -> float:
        return self.prob_holds_rank(player, TWO_RANK)

    def prob_holds_ace(self, player: int) -> float:
        return self.prob_holds_rank(player, ACE_RANK)

    def prob_beats_single(self, player: int, card: Card) -> float:
        """P(player holds >= 1 single stronger than ``card``)."""
        return self.prob_holds_any(player, [c for c in self.unseen if c > card])

    def known_hand(self, player: int) -> Optional[List[Card]]:
        """Endgame elimination: the exact hand, when it is forced.

        In a 4-player game the unseen pool is exactly the opponents'
        hands, so when only one opponent remains unaccounted (or the
        pool equals a player's count), inference is complete.
        """
        if self.counts[player] == self.n_unseen:
            return list(self.unseen)
        return None

    # ------------------------------------------------------------------
    # Monte-Carlo layer (pass-evidence weighting)
    # ------------------------------------------------------------------

    def _collect_pass_evidence(self) -> List[_PassEvidence]:
        evidence = []
        table: Optional[Combo] = None
        pending: List[Tuple[int, Combo, List[Card]]] = []
        for rec in self.game.history:
            if rec.combo is not None:
                table = rec.combo
                for entry in pending:
                    if entry[0] == rec.player:
                        entry[2].extend(rec.combo.cards)
            elif table is not None and rec.player != self.viewpoint:
                pending.append((rec.player, table, []))
        return [_PassEvidence(p, t, tuple(cs)) for p, t, cs in pending]

    def _sample_world(self) -> Dict[int, List[Card]]:
        pool = self.unseen[:]
        self.rng.shuffle(pool)
        world, i = {}, 0
        for p in self.opponents:
            world[p] = sorted(pool[i : i + self.counts[p]])
            i += self.counts[p]
        return world  # leftover pool cards = undealt (2-3 player games)

    def _honesty(self, player: int) -> float:
        if isinstance(self.pass_honesty, dict):
            return float(self.pass_honesty.get(player, 1.0))
        return float(self.pass_honesty)

    def _world_weight(self, world: Dict[int, List[Card]]) -> float:
        w = 1.0
        if self._play_events:
            from big2.playmodel import world_factor
            w = world_factor(self._play_events, world)
        if not self._evidence:
            return w
        for ev in self._evidence:
            if ev.player not in world:
                continue
            h = self._honesty(ev.player)
            if h >= 1.0:
                continue
            hand_at_pass = world[ev.player] + list(ev.played_after)
            if generate_moves(hand_at_pass, ev.table, self.game.rules):
                w *= h  # could have beaten it, yet passed
        return w

    def sample_worlds(self, k: int = 150) -> List[Tuple[Dict[int, List[Card]], float]]:
        """k weighted worlds; cached per BeliefState instance."""
        if self._worlds is None or len(self._worlds) < k:
            self._worlds = [
                (w, self._world_weight(w))
                for w in (self._sample_world() for _ in range(k))
            ]
        return self._worlds[:k]

    def event_probability(self, event, k: int = 150) -> Dict[int, float]:
        """P(event(player, hand)) per opponent from weighted worlds."""
        worlds = self.sample_worlds(k)
        out = {}
        for p in self.opponents:
            num = den = 0.0
            for world, w in worlds:
                den += w
                if event(p, world[p]):
                    num += w
            out[p] = num / den if den else 0.0
        return out

    def class_probabilities(self, k: int = 150) -> Dict[int, Dict[str, float]]:
        """P(opponent holds a pair / triple / bomb / straight / flush)."""
        worlds = self.sample_worlds(k)
        out = {}
        for p in self.opponents:
            acc = {"pair": 0.0, "triple": 0.0, "bomb": 0.0,
                   "straight": 0.0, "flush": 0.0}
            den = 0.0
            for world, w in worlds:
                den += w
                hand = world[p]
                rank_counts = [0] * NUM_RANKS
                suit_counts = [0] * 4
                for c in hand:
                    rank_counts[rank(c)] += 1
                    suit_counts[suit(c)] += 1
                if max(rank_counts, default=0) >= 2:
                    acc["pair"] += w
                if max(rank_counts, default=0) >= 3:
                    acc["triple"] += w
                if max(rank_counts, default=0) >= 4:
                    acc["bomb"] += w
                if max(suit_counts, default=0) >= 5:
                    acc["flush"] += w
                run = 0
                for r in range(ACE_RANK + 1):
                    run = run + 1 if rank_counts[r] else 0
                    if run >= 5:
                        acc["straight"] += w
                        break
            out[p] = {key: v / den if den else 0.0 for key, v in acc.items()}
        return out

    def prob_can_beat(self, combo: Combo, k: int = 150) -> Dict[int, float]:
        """P(opponent can legally beat ``combo``), from weighted worlds."""
        return self.event_probability(
            lambda p, hand: bool(generate_moves(hand, combo, self.game.rules)),
            k,
        )

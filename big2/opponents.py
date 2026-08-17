"""Within-match opponent modeling from the public action history.

Watches what each opponent has *done* this game and summarizes it as a
style vector — the statistical rung of research-ladder step 8 (the
neural opponent-embedding upgrade is documented in EXPERIMENTS.md):

- ``pass_rate``      how often they pass when following
- ``avg_rank``       mean Big-2 rank of cards they've played (aggression:
                     high = burning strong cards early)
- ``multi_frac``     fraction of plays that were multi-card (pairs/5s)
- ``twos_frac``      fraction of the four 2s they have already spent

Two consumers:

1. **Adaptive pass evidence.**  A player who passes constantly is
   passing strategically, so their passes say little about their hand; a
   player who almost never passes folds only when genuinely stuck.  The
   model maps observed pass rate to a per-opponent ``pass_honesty``
   weight for BeliefState's world weighting.
2. **NN features.**  The style vector feeds encoding v4 directly, so a
   learned agent can condition on who it is playing against — guessing
   their holdings *and* their tendencies rather than assuming a uniform
   opponent.
"""

from __future__ import annotations

from typing import Dict, List

from big2.cards import TWO_RANK, rank
from big2.game import Big2Game

STYLE_DIM = 4


class OpponentModel:
    """Cheap incremental summary; recompute per decision from history."""

    def __init__(self, game: Big2Game, viewpoint: int):
        self.viewpoint = viewpoint
        n = game.num_players
        self.n_plays = [0] * n
        self.n_passes = [0] * n
        self.rank_sum = [0.0] * n
        self.cards_played = [0] * n
        self.multi_plays = [0] * n
        self.twos_played = [0] * n
        for rec in game.history:
            p = rec.player
            if rec.combo is None:
                self.n_passes[p] += 1
                continue
            self.n_plays[p] += 1
            cards = rec.combo.cards
            self.cards_played[p] += len(cards)
            self.rank_sum[p] += sum(rank(c) for c in cards)
            if len(cards) > 1:
                self.multi_plays[p] += 1
            self.twos_played[p] += sum(1 for c in cards if rank(c) == TWO_RANK)

    def pass_rate(self, p: int) -> float:
        acts = self.n_plays[p] + self.n_passes[p]
        return self.n_passes[p] / acts if acts else 0.0

    def avg_rank(self, p: int) -> float:
        return (
            self.rank_sum[p] / self.cards_played[p] / 12.0
            if self.cards_played[p]
            else 0.5
        )

    def multi_frac(self, p: int) -> float:
        return self.multi_plays[p] / self.n_plays[p] if self.n_plays[p] else 0.0

    def twos_frac(self, p: int) -> float:
        return self.twos_played[p] / 4.0

    def style(self, p: int) -> List[float]:
        return [self.pass_rate(p), self.avg_rank(p), self.multi_frac(p),
                self.twos_frac(p)]

    def pass_honesty(self, p: int) -> float:
        """World-weight for 'could beat but passed' under this opponent.

        Frequent passers pass strategically -> their passes carry little
        information (weight near 1).  Rare passers only pass when stuck
        -> strong evidence (weight near 0.2).
        """
        return min(0.95, max(0.20, 0.20 + 1.2 * self.pass_rate(p)))

    def honesty_map(self, game: Big2Game) -> Dict[int, float]:
        return {
            p: self.pass_honesty(p)
            for p in range(game.num_players)
            if p != self.viewpoint
        }

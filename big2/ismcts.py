"""ISMCTS reference opponent: determinized Monte-Carlo search, no learning.

Research-ladder step 3.  At each decision the agent:

1. builds the pool of unseen cards (full deck minus its own hand and
   everything played — undealt cards in 2-3 player games stay hidden in
   the pool, exactly as the player would experience),
2. repeatedly *determinizes*: deals opponents random hands consistent
   with their observed card counts,
3. selects a root move by UCB1 over average rollout payoffs, plays it in
   the determinized world, and rolls the game out with a fast scripted
   policy for every seat,
4. finally plays the root move with the best mean payoff.

This is root-level ISMCTS (a.k.a. PIMC with adaptive root allocation).
Big 2 sits in the regime where determinized search is known to work
well — high leaf correlation, low disambiguation (Long et al., AAAI
2010) — and it gives a genuinely strong opponent with zero training.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from big2.cards import NUM_CARDS
from big2.combos import Combo
from big2.game import Big2Game
from big2.strategies import PlayLowest, Strategy

# Normalizes payoffs into roughly [-1, 1] for UCB (max payment is 39).
SCORE_SCALE = 39.0


class ISMCTSStrategy(Strategy):
    name = "ismcts"

    def __init__(
        self,
        n_sims: int = 200,
        exploration: float = 1.2,
        rollout: Optional[Strategy] = None,
        seed: Optional[int] = None,
    ):
        self.n_sims = n_sims
        self.exploration = exploration
        self.rollout = rollout or PlayLowest()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------

    def _determinize(self, game: Big2Game, player: int) -> Big2Game:
        """Clone the game with opponents' hands redealt from unseen cards."""
        world = game.clone(rng=random.Random(self.rng.randrange(2**31)))
        seen = set(game.hands[player]) | set(game.played_cards)
        pool = [c for c in range(NUM_CARDS) if c not in seen]
        self.rng.shuffle(pool)
        i = 0
        for p in range(game.num_players):
            if p == player:
                continue
            n = len(game.hands[p])
            world.hands[p] = sorted(pool[i : i + n])
            i += n
        return world

    def _rollout(self, world: Big2Game, player: int) -> float:
        steps = 0
        while not world.game_over:
            move = self.rollout.select(world, world.turn)
            world.step(move)
            steps += 1
            if steps > 500:  # unreachable safety valve
                return 0.0
        return world.scores[player] / SCORE_SCALE

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        if len(options) == 1:
            return options[0]

        counts = [0] * len(options)
        sums = [0.0] * len(options)
        for t in range(self.n_sims):
            if t < len(options):
                idx = t  # visit everything once before UCB kicks in
            else:
                log_t = math.log(t + 1)
                idx = max(
                    range(len(options)),
                    key=lambda i: sums[i] / counts[i]
                    + self.exploration * math.sqrt(log_t / counts[i]),
                )
            world = self._determinize(game, player)
            world.step(options[idx])
            reward = (
                world.scores[player] / SCORE_SCALE
                if world.game_over
                else self._rollout(world, player)
            )
            counts[idx] += 1
            sums[idx] += reward

        best = max(range(len(options)), key=lambda i: sums[i] / counts[i])
        return options[best]

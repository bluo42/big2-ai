"""Encoding v3: the v2 DMC state-action encoding plus engineered features.

Extra block (all O(hand) to compute — cheap enough for millions of
training games):

- hand composition: 2s, aces, extreme cards, pairable/triplable ranks,
  quad flag, longest same-suit block, longest rank run
- decomposition: exact ``min_plays`` when the hand is small (endgame,
  where it matters most and costs microseconds), ceil(n/5) proxy earlier
- game phase: fraction of the deck played, table size, my hand size,
  smallest opponent hand

Everything move-independent is computed once per decision in
``DecisionContext`` and shared across the candidate moves.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from big2.beliefs import BeliefState
from big2.cards import ACE_RANK, NUM_RANKS, TWO_RANK, rank, suit
from big2.combos import Combo
from big2.dmc import DIM as BASE_DIM
from big2.dmc import encode as encode_base
from big2.game import Big2Game

EXTRA_DIM = 14
FEAT_DIM = BASE_DIM + EXTRA_DIM

# Hands at or below this size get an exact minimum-plays computation.
EXACT_DECOMP_MAX = 8


class DecisionContext:
    """Per-decision cache shared by every candidate move's encoding."""

    def __init__(self, game: Big2Game, player: int):
        self.game = game
        self.player = player
        self.belief = BeliefState(game, player)

        hand = game.hands[player]
        n = len(hand)
        rank_counts = [0] * NUM_RANKS
        suit_counts = [0] * 4
        for c in hand:
            rank_counts[rank(c)] += 1
            suit_counts[suit(c)] += 1

        if n and n <= EXACT_DECOMP_MAX:
            from big2.decomposition import min_plays

            k, _ = min_plays(hand, game.rules, budget=4000)
        else:
            k = (n + 4) // 5  # optimistic proxy; exact signal arrives late-game
        run = best_run = 0
        for r in range(ACE_RANK + 1):
            run = run + 1 if rank_counts[r] else 0
            best_run = max(best_run, run)

        opp_counts = [
            len(game.hands[p]) for p in range(game.num_players) if p != player
        ]
        self.static = np.array(
            [
                rank_counts[TWO_RANK] / 4.0,
                rank_counts[ACE_RANK] / 4.0,
                (max(hand) / 51.0) if hand else 0.0,
                (min(hand) / 51.0) if hand else 0.0,
                sum(1 for c in rank_counts if c >= 2) / 6.0,
                sum(1 for c in rank_counts if c >= 3) / 4.0,
                1.0 if any(c == 4 for c in rank_counts) else 0.0,
                max(suit_counts) / 13.0 if hand else 0.0,
                best_run / 5.0,
                k / 8.0,
                len(game.played_cards) / 52.0,
                game.num_players / 4.0,
                n / 13.0,
                (min(opp_counts) / 13.0) if opp_counts else 0.0,
            ],
            dtype=np.float32,
        )


def encode_sa(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    ctx: Optional[DecisionContext] = None,
) -> np.ndarray:
    ctx = ctx or DecisionContext(game, player)
    x = np.empty(FEAT_DIM, dtype=np.float32)
    x[:BASE_DIM] = encode_base(game, player, move, ctx.belief)
    x[BASE_DIM:] = ctx.static
    return x


def encode_options(
    game: Big2Game, player: int
) -> Tuple[List[Optional[Combo]], np.ndarray]:
    """All legal options (with pass) and their encodings, ready for a
    batched Q evaluation."""
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    ctx = DecisionContext(game, player)
    feats = np.stack([encode_sa(game, player, m, ctx) for m in options])
    return options, feats

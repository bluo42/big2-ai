"""State-action encodings for neural scorers (versioned).

v3 = the v2 DMC encoding plus 14 engineered features (hand composition,
decomposition proxy, game phase).  Kept intact so nets trained on it
still load.

v4 (current) = v3 plus 25 more features:

- **opponent style** (3 per opponent): observed pass rate, mean rank of
  cards played, multi-card play fraction — the OpponentModel summary, so
  the net can condition on *how* each opponent plays;
- **feature-driven holding guesses** (2 per opponent): P(holds a pair),
  P(holds a triple) from a small uniform Monte-Carlo over the unseen
  pool (trump-card probabilities are already in the base encoding);
- **decomposition delta** (2): exact change in minimum-plays caused by
  this move when the hand is small, plus a validity flag — "does this
  move keep my hand's structure falling?";
- **payment-tier risk** (4): the multiplier tier (1x/2x/3x) my hand
  would sit in after this move, and how far above the double threshold
  it leaves me — the tiered scoring rule, made directly visible;
- **trick context** (2): consecutive passes since the last play, and
  whether the combo on the table is mine;
- **card memory** (2): fraction of 2s and aces already played.

Everything move-independent is computed once per decision in
``DecisionContext`` and shared across candidate moves.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from big2.beliefs import BeliefState
from big2.cards import ACE_RANK, NUM_RANKS, TWO_RANK, rank, suit
from big2.combos import Combo
from big2.dmc import DIM as BASE_DIM
from big2.dmc import encode as encode_base
from big2.game import Big2Game
from big2.opponents import OpponentModel

EXTRA_DIM_V3 = 14
EXTRA_DIM_V4 = EXTRA_DIM_V3 + 25
FEAT_DIM_V3 = BASE_DIM + EXTRA_DIM_V3
FEAT_DIM = BASE_DIM + EXTRA_DIM_V4  # current version (v4)

# Hands at or below this size get exact minimum-plays computations.
EXACT_DECOMP_MAX = 8
MINI_MC_WORLDS = 16


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

        self.decomp_memo: dict = {}
        if n and n <= EXACT_DECOMP_MAX:
            from big2.decomposition import min_plays

            self.k_current, _ = min_plays(
                hand, game.rules, budget=4000, _memo=self.decomp_memo
            )
            self.k_exact = True
        else:
            self.k_current = (n + 4) // 5
            self.k_exact = False
        run = best_run = 0
        for r in range(ACE_RANK + 1):
            run = run + 1 if rank_counts[r] else 0
            best_run = max(best_run, run)

        opp_counts = [
            len(game.hands[p]) for p in range(game.num_players) if p != player
        ]
        # ---- v3 static block (must stay byte-compatible with v3 nets)
        self.static_v3 = np.array(
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
                self.k_current / 8.0,
                len(game.played_cards) / 52.0,
                game.num_players / 4.0,
                n / 13.0,
                (min(opp_counts) / 13.0) if opp_counts else 0.0,
            ],
            dtype=np.float32,
        )

        # ---- v4 static block
        model = OpponentModel(game, player)
        others = [p for p in range(game.num_players) if p != player][:3]

        style = np.zeros(9, dtype=np.float32)
        for j, p in enumerate(others):
            style[3 * j : 3 * j + 3] = [
                model.pass_rate(p), model.avg_rank(p), model.multi_frac(p)
            ]

        guesses = np.zeros(6, dtype=np.float32)
        if self.belief.n_unseen:
            rng = random.Random(len(game.played_cards) * 977 + player)
            pair_hits = {p: 0 for p in others}
            triple_hits = {p: 0 for p in others}
            pool = self.belief.unseen
            for _ in range(MINI_MC_WORLDS):
                sample = pool[:]
                rng.shuffle(sample)
                i = 0
                for p in others:
                    cnt = len(game.hands[p])
                    counts = [0] * NUM_RANKS
                    for c in sample[i : i + cnt]:
                        counts[rank(c)] += 1
                    i += cnt
                    top = max(counts, default=0)
                    if top >= 2:
                        pair_hits[p] += 1
                    if top >= 3:
                        triple_hits[p] += 1
            for j, p in enumerate(others):
                guesses[2 * j] = pair_hits[p] / MINI_MC_WORLDS
                guesses[2 * j + 1] = triple_hits[p] / MINI_MC_WORLDS

        consec = 0
        for rec in reversed(game.history):
            if rec.combo is None and not rec.trick_end:
                consec += 1
            else:
                break
        twos_gone = sum(1 for c in game.played_cards if rank(c) == TWO_RANK)
        aces_gone = sum(1 for c in game.played_cards if rank(c) == ACE_RANK)

        self.static_v4 = np.concatenate([
            style,
            guesses,
            np.array(
                [
                    min(consec, 3) / 3.0,
                    1.0 if game.table_player == player else 0.0,
                    twos_gone / 4.0,
                    aces_gone / 4.0,
                ],
                dtype=np.float32,
            ),
        ])

    def move_extras_v4(self, move: Optional[Combo]) -> np.ndarray:
        """Move-dependent v4 features: decomposition delta + payment tier."""
        game, player = self.game, self.player
        hand = game.hands[player]
        n_after = len(hand) - (len(move) if move else 0)

        delta_norm, delta_known = 0.0, 0.0
        if move is not None and self.k_exact:
            from big2.decomposition import min_plays

            rest = list(hand)
            for c in move.cards:
                rest.remove(c)
            k_after, _ = min_plays(
                rest, game.rules, budget=4000, _memo=self.decomp_memo
            )
            # 1 + k_after - k_current: 0 = structure-neutral move.
            delta_norm = max(-1.0, min(2.0, 1 + k_after - self.k_current)) / 2.0
            delta_known = 1.0
        elif move is None and self.k_exact:
            delta_known = 1.0  # passing is structure-neutral by definition

        tier = np.zeros(3, dtype=np.float32)
        if n_after == 13:
            tier[2] = 1.0
        elif n_after >= 10:
            tier[1] = 1.0
        else:
            tier[0] = 1.0
        above_double = max(0, n_after - 9) / 4.0

        return np.array(
            [delta_norm, delta_known, *tier, above_double], dtype=np.float32
        )


def encode_sa(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    ctx: Optional[DecisionContext] = None,
    version: int = 4,
) -> np.ndarray:
    ctx = ctx or DecisionContext(game, player)
    base = encode_base(game, player, move, ctx.belief)
    if version == 3:
        x = np.empty(FEAT_DIM_V3, dtype=np.float32)
        x[:BASE_DIM] = base
        x[BASE_DIM:] = ctx.static_v3
        return x
    x = np.empty(FEAT_DIM, dtype=np.float32)
    x[:BASE_DIM] = base
    o = BASE_DIM
    x[o : o + EXTRA_DIM_V3] = ctx.static_v3
    o += EXTRA_DIM_V3
    x[o : o + 19] = ctx.static_v4
    o += 19
    x[o:] = ctx.move_extras_v4(move)
    return x


def version_for_dim(in_dim: int) -> int:
    if in_dim == FEAT_DIM_V3:
        return 3
    if in_dim == FEAT_DIM:
        return 4
    raise ValueError(f"no feature version with dim {in_dim}")


def encode_options(
    game: Big2Game, player: int, version: int = 4
) -> Tuple[List[Optional[Combo]], np.ndarray]:
    """All legal options (with pass) and their encodings, ready for a
    batched Q evaluation."""
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    ctx = DecisionContext(game, player)
    feats = np.stack(
        [encode_sa(game, player, m, ctx, version) for m in options]
    )
    return options, feats

"""A first learned strategy: linear move scoring + cross-entropy method.

The policy scores every legal option (including pass) with a linear
function over hand-crafted state/move features and plays the argmax.
Training uses the cross-entropy method (CEM): sample weight vectors from
a Gaussian, evaluate each over a batch of games against fixed opponents,
refit the Gaussian to the elite fraction, repeat.

This is deliberately simple — it is the stepping stone toward
self-play / deep RL (see README).  Run:

    python -m big2.rl --iters 15 --pop 24 --games 32
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional

import numpy as np

from big2.cards import TWO_RANK, rank
from big2.combos import Combo
from big2.game import NUM_PLAYERS, Big2Game, ScoringConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    SmartHeuristic,
    Strategy,
)

FEATURE_NAMES = [
    "bias",
    "is_pass",
    "is_pass_and_endgame",
    "is_pass_and_danger",
    "size1",
    "size2",
    "size5",
    "type_norm",          # combo type index / 6
    "top_card_norm",      # strongest card in move / 51
    "avg_rank_norm",      # mean rank of move / 12
    "frac_hand_used",
    "cards_left_after",   # / 13
    "uses_two",           # count of 2s in move
    "uses_ace",
    "breaks_unit",        # move is not exactly one greedy-partition unit
    "leading",
    "danger",             # some opponent has <= 2 cards
    "endgame",            # my hand <= 6 cards
    "top_card_and_danger",
    "goes_out",           # move empties my hand
]
NUM_FEATURES = len(FEATURE_NAMES)


def move_features(game: Big2Game, player: int, move: Optional[Combo],
                  units_keys: set) -> np.ndarray:
    f = np.zeros(NUM_FEATURES, dtype=np.float64)
    hand_n = len(game.hands[player])
    opp_min = min(
        len(game.hands[p]) for p in range(NUM_PLAYERS) if p != player
    )
    danger = 1.0 if opp_min <= 2 else 0.0
    endgame = 1.0 if hand_n <= 6 else 0.0
    leading = 1.0 if game.leading else 0.0

    f[0] = 1.0
    f[15] = leading
    f[16] = danger
    f[17] = endgame

    if move is None:
        f[1] = 1.0
        f[2] = endgame
        f[3] = danger
        f[11] = hand_n / 13.0
        return f

    n = len(move)
    f[4] = 1.0 if n == 1 else 0.0
    f[5] = 1.0 if n == 2 else 0.0
    f[6] = 1.0 if n == 5 else 0.0
    f[7] = int(move.type) / 6.0
    top = max(move.cards)
    f[8] = top / 51.0
    f[9] = float(np.mean([rank(c) for c in move.cards])) / 12.0
    f[10] = n / hand_n
    f[11] = (hand_n - n) / 13.0
    f[12] = sum(1 for c in move.cards if rank(c) == TWO_RANK)
    f[13] = sum(1 for c in move.cards if rank(c) == TWO_RANK - 1)
    f[14] = 0.0 if frozenset(move.cards) in units_keys else 1.0
    f[18] = f[8] * danger
    f[19] = 1.0 if n == hand_n else 0.0
    return f


class LinearPolicy(Strategy):
    """Greedy argmax over linear move scores."""

    name = "linear"

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else np.zeros(NUM_FEATURES)
        )

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        moves: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            moves.append(None)
        if len(moves) == 1:
            return moves[0]
        units_keys = {
            frozenset(u.cards)
            for u in SmartHeuristic._partition(game.hands[player])
        }
        feats = np.stack(
            [move_features(game, player, m, units_keys) for m in moves]
        )
        return moves[int(np.argmax(feats @ self.weights))]

    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights, features=np.array(FEATURE_NAMES))

    @classmethod
    def load(cls, path: str) -> "LinearPolicy":
        data = np.load(path, allow_pickle=True)
        return cls(data["weights"])


def evaluate(
    weights: np.ndarray,
    opponents: List[Strategy],
    n_games: int,
    scoring: ScoringConfig,
    seed: int,
) -> float:
    """Mean score of the linear policy over n_games (rotating seats)."""
    policy = LinearPolicy(weights)
    rng = random.Random(seed)
    total = 0.0
    for g in range(n_games):
        seat = g % NUM_PLAYERS
        seats: List[Strategy] = [None] * NUM_PLAYERS  # type: ignore
        opps = opponents[:]
        for p in range(NUM_PLAYERS):
            seats[p] = policy if p == seat else opps.pop()
        game = Big2Game(scoring=scoring, rng=random.Random(rng.randrange(2**31)))
        scores = game.play_out(seats)
        total += scores[seat]
    return total / n_games


def train_cem(
    iters: int = 15,
    pop: int = 24,
    elite_frac: float = 0.25,
    games_per_eval: int = 32,
    sigma_init: float = 1.0,
    scoring: Optional[ScoringConfig] = None,
    opponents: Optional[List[Strategy]] = None,
    seed: int = 0,
    verbose: bool = True,
) -> LinearPolicy:
    scoring = scoring or ScoringConfig()
    opponents = opponents or [PlayLowest(), FiveCardDumper(), SmartHeuristic()]
    rng = np.random.default_rng(seed)
    mean = np.zeros(NUM_FEATURES)
    sigma = np.full(NUM_FEATURES, sigma_init)
    n_elite = max(2, int(pop * elite_frac))

    for it in range(iters):
        samples = rng.normal(mean, sigma, size=(pop, NUM_FEATURES))
        # Common random numbers: same deals for every candidate this iter.
        eval_seed = int(rng.integers(2**31))
        scores = np.array(
            [
                evaluate(w, opponents, games_per_eval, scoring, eval_seed)
                for w in samples
            ]
        )
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.05  # noise floor keeps exploring
        if verbose:
            print(
                f"iter {it + 1:3d}/{iters}  "
                f"best {scores.max():+7.2f}  mean {scores.mean():+7.2f}  "
                f"elite-mean {scores[np.argsort(scores)[-n_elite:]].mean():+7.2f}"
            )
    return LinearPolicy(mean)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--pop", type=int, default=24)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="big2_linear_policy.npz")
    parser.add_argument(
        "--eval-games", type=int, default=400,
        help="final evaluation games vs the training opponents",
    )
    args = parser.parse_args()

    policy = train_cem(
        iters=args.iters, pop=args.pop, games_per_eval=args.games, seed=args.seed
    )
    policy.save(args.out)
    print(f"saved weights to {args.out}")
    for name, w in zip(FEATURE_NAMES, policy.weights):
        print(f"  {name:24s} {w:+.3f}")

    opponents = [PlayLowest(), FiveCardDumper(), SmartHeuristic()]
    score = evaluate(
        policy.weights, opponents, args.eval_games, ScoringConfig(), seed=123
    )
    print(
        f"final: {score:+.3f} avg score/game over {args.eval_games} games "
        f"vs lowest/dumper/smart"
    )


if __name__ == "__main__":
    main()

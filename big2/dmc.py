"""Deep-Monte-Carlo-style agent with DouZero action encoding (step 4).

DouZero's core trick, transplanted to Big 2: instead of a policy head
over an enumerated action space, encode each candidate action *as
input* — a 52-dim card mask plus move metadata concatenated with the
state — and learn Q(s, a) by regressing on final Monte-Carlo returns
from self-play.  Every legal move is scored; play the argmax
(epsilon-greedy during training).

This implementation keeps the estimator linear (numpy only, no torch)
so it trains in minutes on CPU; the encoding is the important part and
carries over unchanged to an MLP/torch upgrade.  Rewards are the actual
game scores (ADP-style, not win rate), so the agent feels the tiered
card-count multipliers directly.

    python -m big2.dmc --episodes 30000            # train + save
    python -m big2.experiments --games 1000 \
        --strategies lowest dumper smart smart --dmc-file big2/policies/dmc_linear.npz
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from big2.beliefs import BeliefState
from big2.cards import NUM_CARDS
from big2.combos import Combo, ComboType
from big2.game import CARDS_PER_PLAYER, NUM_PLAYERS, Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES, RuleConfig
from big2.strategies import Strategy

SCORE_SCALE = 39.0  # max single-loser payment under tiered scoring

N_TYPES = len(ComboType) + 1  # + "none" slot for an empty table

# Layout: hand(52) action(52) is_pass(1) table(52) table_type(N) played(52)
#         opp_counts(3) opp_passed(3) leading(1) left_after(1) size(1) bias(1)
BASE_DIM = 52 + 52 + 1 + 52 + N_TYPES + 52 + 3 + 3 + 1 + 1 + 1 + 1
# Belief block (encoding v2): per opponent [P(beats ref card), P(holds a 2)]
# for up to 3 opponents, then [frac of unseen above ref, unseen/39].
BELIEF_DIM = 3 * 2 + 2
DIM = BASE_DIM + BELIEF_DIM
ENCODING_VERSION = 2


def encode(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    belief: Optional[BeliefState] = None,
) -> np.ndarray:
    """Feature vector for Q(s, a).  Pass one BeliefState per decision —
    it depends only on the state, and building it per candidate move
    would triple the encoding cost."""
    if belief is None:
        belief = BeliefState(game, player)
    x = np.zeros(DIM, dtype=np.float32)
    o = 0
    hand = game.hands[player]
    for c in hand:
        x[o + c] = 1.0
    o += NUM_CARDS
    if move is not None:
        for c in move.cards:
            x[o + c] = 1.0
    o += NUM_CARDS
    x[o] = 1.0 if move is None else 0.0
    o += 1
    if game.table_combo is not None:
        for c in game.table_combo.cards:
            x[o + c] = 1.0
    o += NUM_CARDS
    t_idx = 0 if game.table_combo is None else int(game.table_combo.type) + 1
    x[o + t_idx] = 1.0
    o += N_TYPES
    for c in game.played_cards:
        x[o + c] = 1.0
    o += NUM_CARDS
    others = [p for p in range(game.num_players) if p != player]
    for j, p in enumerate(others[:3]):
        x[o + j] = len(game.hands[p]) / CARDS_PER_PLAYER
    o += 3
    for j, p in enumerate(others[:3]):
        x[o + j] = 1.0 if game.passed[p] else 0.0
    o += 3
    x[o] = 1.0 if game.leading else 0.0
    o += 1
    n = 0 if move is None else len(move)
    x[o] = (len(hand) - n) / CARDS_PER_PLAYER
    o += 1
    x[o] = n / 5.0
    o += 1
    x[o] = 1.0
    o += 1

    # Belief block: how contestable is this move?  Reference card is the
    # move's top card (for a pass: the card currently to beat).
    if move is not None:
        ref = max(move.cards)
    elif game.table_combo is not None:
        ref = max(game.table_combo.cards)
    else:
        ref = -1
    for j, p in enumerate(others[:3]):
        if ref >= 0:
            x[o + 2 * j] = belief.prob_beats_single(p, ref)
        x[o + 2 * j + 1] = belief.prob_holds_two(p)
    o += 6
    u = belief.n_unseen
    if ref >= 0 and u:
        x[o] = sum(1 for c in belief.unseen if c > ref) / u
    o += 1
    x[o] = u / 39.0
    return x


class DMCPolicy(Strategy):
    """Greedy argmax over learned Q(s, a)."""

    name = "dmc"

    def __init__(self, weights: Optional[np.ndarray] = None):
        self.weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else np.zeros(DIM, dtype=np.float32)
        )

    def _options(self, game: Big2Game, player: int) -> List[Optional[Combo]]:
        options: List[Optional[Combo]] = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        return options

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options = self._options(game, player)
        if len(options) == 1:
            return options[0]
        belief = BeliefState(game, player)
        feats = np.stack([encode(game, player, m, belief) for m in options])
        return options[int(np.argmax(feats @ self.weights))]

    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights, version=ENCODING_VERSION)

    @classmethod
    def load(cls, path: str) -> "DMCPolicy":
        data = np.load(path)
        weights = data["weights"]
        if weights.shape != (DIM,):
            raise ValueError(
                f"{path} has {weights.shape[0]}-dim weights but the current "
                f"encoding (v{ENCODING_VERSION}) is {DIM}-dim — retrain with "
                f"python -m big2.dmc"
            )
        return cls(weights)


def train_dmc(
    episodes: int = 30_000,
    lr: float = 0.05,
    eps_start: float = 1.0,
    eps_end: float = 0.03,
    eps_decay_frac: float = 0.4,
    scoring: Optional[ScoringConfig] = None,
    rules: Optional[RuleConfig] = None,
    seed: int = 0,
    verbose_every: int = 5000,
) -> DMCPolicy:
    """Self-play Monte-Carlo Q-learning: all four seats share weights.

    Each seat's transitions regress toward its own final score, so the
    reward is exactly the payoff structure of the configured variant.
    """
    scoring = scoring or ScoringConfig()
    rules = rules or DEFAULT_RULES
    rng = random.Random(seed)
    w = np.zeros(DIM, dtype=np.float32)
    policy = DMCPolicy(w)
    decay_steps = max(1, int(episodes * eps_decay_frac))

    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / decay_steps)
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=NUM_PLAYERS,
            rng=random.Random(rng.randrange(2**31)),
        )
        trajectories: Dict[int, List[np.ndarray]] = {p: [] for p in range(NUM_PLAYERS)}
        while not game.game_over:
            p = game.turn
            options = policy._options(game, p)
            belief = BeliefState(game, p)
            if len(options) == 1:
                choice, feat = options[0], encode(game, p, options[0], belief)
            elif rng.random() < eps:
                choice = options[rng.randrange(len(options))]
                feat = encode(game, p, choice, belief)
            else:
                feats = np.stack([encode(game, p, m, belief) for m in options])
                idx = int(np.argmax(feats @ w))
                choice, feat = options[idx], feats[idx]
            trajectories[p].append(feat)
            game.step(choice)

        for p in range(NUM_PLAYERS):
            target = game.scores[p] / SCORE_SCALE
            for feat in trajectories[p]:
                # SGD on (Q - G)^2; features are ~0/1 so scale lr by density.
                q = float(feat @ w)
                w += (lr / max(1.0, float(feat.sum()))) * (target - q) * feat

        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"episode {ep + 1}/{episodes}  eps={eps:.3f}")

    policy.weights = w
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="big2/policies/dmc_linear.npz")
    parser.add_argument("--eval-games", type=int, default=400)
    args = parser.parse_args()

    policy = train_dmc(episodes=args.episodes, lr=args.lr, seed=args.seed)
    policy.save(args.out)
    print(f"saved weights to {args.out}")

    from big2.strategies import FiveCardDumper, PlayLowest, SmartHeuristic

    opponents = [PlayLowest(), FiveCardDumper(), SmartHeuristic()]
    total = 0.0
    rng = random.Random(123)
    for g in range(args.eval_games):
        seat = g % NUM_PLAYERS
        opps = opponents[:]
        seats = [policy if p == seat else opps.pop() for p in range(NUM_PLAYERS)]
        game = Big2Game(rng=random.Random(rng.randrange(2**31)))
        scores = game.play_out(seats)
        total += scores[seat]
    print(
        f"final: {total / args.eval_games:+.3f} avg score/game over "
        f"{args.eval_games} games vs lowest/dumper/smart"
    )


if __name__ == "__main__":
    main()

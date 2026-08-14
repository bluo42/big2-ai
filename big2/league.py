"""League training: the population plays itself, not fixed benchmarks.

Research-ladder step 6 (first rung).  A single self-play run in a
4-player game converges to conventions that collapse against unfamiliar
opponents, and training against fixed scripted benchmarks overfits to
their quirks.  The league keeps a *population* — scripted anchors,
trainable agents (CEM linear and DMC), and frozen checkpoints of past
generations — and every generation:

1. **Match phase**: many games with 4 members drawn at random from the
   population, seats rotated, ratings = mean score/game vs the field.
2. **Training phase**: DMC continues Q-learning with its opponents
   sampled from the population each episode; CEM candidates are
   evaluated against sampled population lineups (common random numbers
   within an iteration).
3. **Checkpoint phase**: frozen copies of the trainables join the
   population (bounded history), so later generations must keep beating
   earlier selves as well as the anchors.

Missing vs. full PSRO (documented in docs/EXPERIMENTS.md): dedicated
exploiter agents and a meta-solver over the payoff matrix.  Ratings
here are mean-score-vs-field, which is the right first proxy.

    python -m big2.league --generations 3 --games-per-gen 300
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from big2.dmc import DIM, DMCPolicy, encode
from big2.beliefs import BeliefState
from big2.game import NUM_PLAYERS, Big2Game, ScoringConfig
from big2.rl import NUM_FEATURES, LinearPolicy
from big2.rules import DEFAULT_RULES, RuleConfig
from big2.strategies import (
    FiveCardDumper,
    PlayLowest,
    SmartHeuristic,
    Strategy,
)

SCORE_SCALE = 39.0


@dataclass
class Member:
    name: str
    strategy: Strategy
    kind: str  # scripted | cem | dmc | checkpoint
    trainable: bool = False
    games: int = 0
    total: float = 0.0

    @property
    def rating(self) -> float:
        return self.total / self.games if self.games else 0.0


class League:
    def __init__(
        self,
        scoring: Optional[ScoringConfig] = None,
        rules: Optional[RuleConfig] = None,
        seed: int = 0,
        max_checkpoints: int = 4,
    ):
        self.scoring = scoring or ScoringConfig()
        self.rules = rules or DEFAULT_RULES
        self.rng = random.Random(seed)
        self.max_checkpoints = max_checkpoints
        self.population: List[Member] = []

    def add(self, name: str, strategy: Strategy, kind: str,
            trainable: bool = False) -> Member:
        m = Member(name, strategy, kind, trainable)
        self.population.append(m)
        return m

    # ------------------------------------------------------------------

    def reset_ratings(self) -> None:
        for m in self.population:
            m.games = 0
            m.total = 0.0

    def sample_members(self, k: int, exclude: Optional[Member] = None) -> List[Member]:
        pool = [m for m in self.population if m is not exclude]
        return self.rng.sample(pool, k)

    def play_matches(self, n_games: int) -> None:
        for _ in range(n_games):
            lineup = self.sample_members(NUM_PLAYERS)
            self.rng.shuffle(lineup)
            game = Big2Game(
                scoring=self.scoring, rules=self.rules,
                rng=random.Random(self.rng.randrange(2**31)),
            )
            scores = game.play_out([m.strategy for m in lineup])
            for seat, m in enumerate(lineup):
                m.games += 1
                m.total += scores[seat]

    def table(self) -> str:
        rows = sorted(self.population, key=lambda m: -m.rating)
        lines = [f"{'member':>16s} {'kind':>10s} {'games':>6s} {'avg':>7s}"]
        for m in rows:
            lines.append(
                f"{m.name:>16s} {m.kind:>10s} {m.games:>6d} {m.rating:>+7.2f}"
            )
        return "\n".join(lines)

    def prune_checkpoints(self) -> None:
        checkpoints = [m for m in self.population if m.kind == "checkpoint"]
        if len(checkpoints) > self.max_checkpoints:
            # Drop the oldest first (list order == insertion order).
            for m in checkpoints[: len(checkpoints) - self.max_checkpoints]:
                self.population.remove(m)


# ----------------------------------------------------------------------
# Training vs the field
# ----------------------------------------------------------------------


def train_dmc_vs_field(
    weights: np.ndarray,
    league: League,
    member: Member,
    episodes: int,
    lr: float = 0.05,
    eps_start: float = 0.4,
    eps_end: float = 0.05,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Continue DMC training with opponents sampled from the population.

    Only the DMC seat's own trajectory updates the weights; the field
    plays itself out with frozen strategies.
    """
    rng = rng or random.Random(0)
    w = weights.copy()
    policy = DMCPolicy(w)
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * min(1.0, ep / max(1, episodes * 0.6))
        my_seat = rng.randrange(NUM_PLAYERS)
        others = league.sample_members(NUM_PLAYERS - 1, exclude=member)
        seats: List[Strategy] = []
        oi = 0
        for p in range(NUM_PLAYERS):
            if p == my_seat:
                seats.append(policy)
            else:
                seats.append(others[oi].strategy)
                oi += 1
        game = Big2Game(
            scoring=league.scoring, rules=league.rules,
            rng=random.Random(rng.randrange(2**31)),
        )
        trajectory: List[np.ndarray] = []
        while not game.game_over:
            p = game.turn
            if p != my_seat:
                game.step(seats[p].select(game, p))
                continue
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
            trajectory.append(feat)
            game.step(choice)

        target = game.scores[my_seat] / SCORE_SCALE
        for feat in trajectory:
            q = float(feat @ w)
            w += (lr / max(1.0, float(feat.sum()))) * (target - q) * feat
    return w


def evaluate_cem_vs_field(
    weights: np.ndarray,
    league: League,
    exclude: Member,
    n_games: int,
    seed: int,
) -> float:
    """Mean score of a CEM candidate against sampled population lineups.

    The candidate rotates seats; deals and lineups are driven by
    ``seed`` so every candidate in a CEM iteration sees the same worlds
    (common random numbers).
    """
    policy = LinearPolicy(weights)
    rng = random.Random(seed)
    total = 0.0
    for g in range(n_games):
        seat = g % NUM_PLAYERS
        pool = [m for m in league.population if m is not exclude]
        others = rng.sample(pool, NUM_PLAYERS - 1)
        seats: List[Strategy] = []
        oi = 0
        for p in range(NUM_PLAYERS):
            if p == seat:
                seats.append(policy)
            else:
                seats.append(others[oi].strategy)
                oi += 1
        game = Big2Game(
            scoring=league.scoring, rules=league.rules,
            rng=random.Random(rng.randrange(2**31)),
        )
        scores = game.play_out(seats)
        total += scores[seat]
    return total / n_games


def train_cem_vs_field(
    weights: np.ndarray,
    league: League,
    member: Member,
    iters: int,
    pop: int,
    games_per_eval: int,
    sigma_init: float = 0.5,
    elite_frac: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    mean = weights.copy()
    sigma = np.full(NUM_FEATURES, sigma_init)
    n_elite = max(2, int(pop * elite_frac))
    for _ in range(iters):
        samples = rng.normal(mean, sigma, size=(pop, NUM_FEATURES))
        eval_seed = int(rng.integers(2**31))
        scores = np.array(
            [
                evaluate_cem_vs_field(w, league, member, games_per_eval, eval_seed)
                for w in samples
            ]
        )
        elite = samples[np.argsort(scores)[-n_elite:]]
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.05
    return mean


# ----------------------------------------------------------------------
# Generation loop
# ----------------------------------------------------------------------


def run_league(
    generations: int = 3,
    games_per_gen: int = 300,
    dmc_episodes: int = 6000,
    cem_iters: int = 5,
    cem_pop: int = 16,
    cem_games: int = 24,
    scoring: Optional[ScoringConfig] = None,
    rules: Optional[RuleConfig] = None,
    seed: int = 0,
    warm_start_cem: Optional[str] = "big2/policies/linear_cem.npz",
    verbose: bool = True,
) -> League:
    league = League(scoring=scoring, rules=rules, seed=seed)
    league.add("smart", SmartHeuristic(), "scripted")
    league.add("dumper", FiveCardDumper(), "scripted")
    league.add("lowest", PlayLowest(), "scripted")
    from big2.decomposition import DecompositionStrategy

    league.add("decomp", DecompositionStrategy(), "scripted")

    cem_weights = np.zeros(NUM_FEATURES)
    if warm_start_cem:
        try:
            cem_weights = LinearPolicy.load(warm_start_cem).weights.copy()
        except Exception:
            pass
    cem_policy = LinearPolicy(cem_weights)
    cem_member = league.add("cem", cem_policy, "cem", trainable=True)

    dmc_weights = np.zeros(DIM, dtype=np.float32)
    dmc_policy = DMCPolicy(dmc_weights)
    dmc_member = league.add("dmc", dmc_policy, "dmc", trainable=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    for g in range(generations):
        if verbose:
            print(f"\n=== generation {g + 1}/{generations} ===")

        league.reset_ratings()
        league.play_matches(games_per_gen)
        if verbose:
            print(league.table())

        dmc_weights = train_dmc_vs_field(
            dmc_weights, league, dmc_member, dmc_episodes,
            rng=random.Random(rng.randrange(2**31)),
        )
        dmc_policy.weights = dmc_weights

        cem_weights = train_cem_vs_field(
            cem_weights, league, cem_member, cem_iters, cem_pop, cem_games,
            rng=np_rng,
        )
        cem_policy.weights = cem_weights

        league.add(f"dmc-g{g + 1}", DMCPolicy(dmc_weights.copy()), "checkpoint")
        league.add(f"cem-g{g + 1}", LinearPolicy(cem_weights.copy()), "checkpoint")
        league.prune_checkpoints()

    # Final standings decide the champions, so give them extra games.
    league.reset_ratings()
    league.play_matches(games_per_gen * 2)
    if verbose:
        print("\n=== final standings ===")
        print(league.table())
    return league


def best_of_lineage(league: League, prefix: str) -> Member:
    """Best-rated member of a trainable lineage (live agent or any of its
    frozen checkpoints) by final-standings rating.  Saving the *best*
    rather than the *last* matters: training against a shifting field is
    non-stationary and the live agent can end below its own checkpoint."""
    candidates = [
        m for m in league.population
        if m.name == prefix or m.name.startswith(prefix + "-g")
    ]
    return max(candidates, key=lambda m: m.rating)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--games-per-gen", type=int, default=300)
    parser.add_argument("--dmc-episodes", type=int, default=6000)
    parser.add_argument("--cem-iters", type=int, default=5)
    parser.add_argument("--cem-pop", type=int, default=16)
    parser.add_argument("--cem-games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dmc-out", default="big2/policies/dmc_linear.npz")
    parser.add_argument("--cem-out", default="big2/policies/linear_cem.npz")
    args = parser.parse_args()

    league = run_league(
        generations=args.generations,
        games_per_gen=args.games_per_gen,
        dmc_episodes=args.dmc_episodes,
        cem_iters=args.cem_iters,
        cem_pop=args.cem_pop,
        cem_games=args.cem_games,
        seed=args.seed,
        warm_start_cem=args.cem_out,
    )
    dmc_champ = best_of_lineage(league, "dmc")
    cem_champ = best_of_lineage(league, "cem")
    dmc_champ.strategy.save(args.dmc_out)
    cem_champ.strategy.save(args.cem_out)
    print(
        f"\nsaved champions: {dmc_champ.name} ({dmc_champ.rating:+.2f}) -> "
        f"{args.dmc_out}, {cem_champ.name} ({cem_champ.rating:+.2f}) -> {args.cem_out}"
    )


if __name__ == "__main__":
    main()

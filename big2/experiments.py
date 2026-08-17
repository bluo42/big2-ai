"""Strategy tournaments across scoring-rule variants.

Runs 4-seat tournaments where the given strategies rotate seats every
game (removing positional bias), for each scoring configuration:

    plain        no modifiers, pay cards-remaining
    tiered       house rule: 10-12 cards pay double, all 13 pay triple
    two          legacy: holding a 2 doubles the payment
    tiered+two   tiered multipliers plus the 2-holder rule

Examples:
    python -m big2.experiments --games 2000
    python -m big2.experiments --games 500 --strategies smart lowest highest random
    python -m big2.experiments --games 1000 --policy-file big2_linear_policy.npz
"""

from __future__ import annotations

import argparse
import random
from itertools import permutations
from typing import Dict, List, Optional

from big2.game import NUM_PLAYERS, Big2Game, ScoringConfig
from big2.strategies import (
    FiveCardDumper,
    PlayHighest,
    PlayLowest,
    RandomPolicy,
    SmartHeuristic,
    Strategy,
)

SCORING_VARIANTS: Dict[str, ScoringConfig] = {
    # no modifiers: pay cards-remaining flat
    "plain": ScoringConfig(big_hand_double=False, full_hand_triple=False),
    # house rule: 10-12 cards pay double, 13 pays triple
    "tiered": ScoringConfig(),
    # legacy 2-holder rule on its own
    "two": ScoringConfig(
        big_hand_double=False, full_hand_triple=False, two_modifier=True
    ),
    # tiered card-count multipliers plus the 2-holder rule
    "tiered+two": ScoringConfig(two_modifier=True),
}


def make_strategy(name: str, seed: Optional[int] = None) -> Strategy:
    name = name.lower()
    if name == "random":
        return RandomPolicy(seed)
    if name == "lowest":
        return PlayLowest()
    if name == "highest":
        return PlayHighest()
    if name == "dumper":
        return FiveCardDumper()
    if name == "smart":
        return SmartHeuristic()
    if name == "decomp":
        from big2.decomposition import DecompositionStrategy

        return DecompositionStrategy()
    if name == "ismcts":
        from big2.ismcts import ISMCTSStrategy

        return ISMCTSStrategy(n_sims=100, seed=seed)
    if name == "linear":
        from big2.rl import LinearPolicy

        return LinearPolicy.load("big2/policies/linear_cem.npz")
    if name == "dmc":
        from big2.dmc import DMCPolicy

        return DMCPolicy.load("big2/policies/dmc_linear.npz")
    if name == "evo":
        from big2.nn import NNPolicy

        return NNPolicy.load("big2/policies/evo_mlp.npz")
    if name == "ppo":
        from big2.neural import PPOPolicy

        return PPOPolicy.load()
    raise ValueError(f"unknown strategy {name!r}")


def run_tournament(
    strategies: List[Strategy],
    n_games: int,
    scoring: ScoringConfig,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Round-robin with seat rotation; returns per-strategy stats."""
    assert len(strategies) == NUM_PLAYERS
    rng = random.Random(seed)
    seatings = list(permutations(range(NUM_PLAYERS)))
    totals = {i: 0.0 for i in range(NUM_PLAYERS)}
    wins = {i: 0 for i in range(NUM_PLAYERS)}

    for g in range(n_games):
        seating = seatings[g % len(seatings)]  # strategy index -> seat
        seats: List[Strategy] = [None] * NUM_PLAYERS  # type: ignore
        for strat_idx, seat in enumerate(seating):
            seats[seat] = strategies[strat_idx]
        game = Big2Game(scoring=scoring, rng=random.Random(rng.randrange(2**31)))
        scores = game.play_out(seats)
        for strat_idx, seat in enumerate(seating):
            totals[strat_idx] += scores[seat]
            if game.winner == seat:
                wins[strat_idx] += 1

    return {
        strategies[i].name: {
            "avg_score": totals[i] / n_games,
            "win_rate": wins[i] / n_games,
        }
        for i in range(NUM_PLAYERS)
    }


def print_table(results_by_variant: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    names = list(next(iter(results_by_variant.values())).keys())
    header = f"{'variant':>10s} | " + " | ".join(f"{n:>18s}" for n in names)
    print(header)
    print("-" * len(header))
    for variant, res in results_by_variant.items():
        cells = [
            f"{res[n]['avg_score']:+7.2f} ({res[n]['win_rate']:4.0%})" for n in names
        ]
        print(f"{variant:>10s} | " + " | ".join(f"{c:>18s}" for c in cells))
    print("\ncells: average score per game (win rate)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000, help="games per variant")
    parser.add_argument(
        "--strategies",
        nargs=4,
        default=["lowest", "highest", "dumper", "smart"],
        help=(
            "four of: random lowest highest dumper smart decomp ismcts "
            "linear dmc (ismcts is ~100x slower than the scripted ones)"
        ),
    )
    parser.add_argument(
        "--variants", nargs="*", default=["tiered"],
        choices=list(SCORING_VARIANTS),
        help="default: tiered only (the house scoring rule)",
    )
    parser.add_argument(
        "--policy-file", type=str, default=None,
        help="replace the 4th strategy with a trained LinearPolicy (.npz)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    strategies = [make_strategy(n, seed=args.seed + i) for i, n in enumerate(args.strategies)]
    if args.policy_file:
        from big2.rl import LinearPolicy

        strategies[3] = LinearPolicy.load(args.policy_file)

    results = {}
    for variant in args.variants:
        results[variant] = run_tournament(
            strategies, args.games, SCORING_VARIANTS[variant], seed=args.seed
        )
    print(f"{args.games} games per variant, seats rotated every game\n")
    print_table(results)


if __name__ == "__main__":
    main()

"""Composition-matrix tournament of the chain finals.

Every multiset of the four models over four seats (35 compositions,
seats shuffled per game so position washes out), N games each — the
full picture of how each model fares in every company: 4-same mirrors,
3v1, 2v2, 2v1v1, and the all-different table.  Then reference rows:
each model against three copies of PPO v1 and of WangBot_v1.

Models play raw policy (the gating currency); the deployed search
wrapper only rescales everything upward.

    python run_tournament.py --games 10000 --workers 20
"""

import argparse
import multiprocessing as mp
from collections import defaultdict
from itertools import combinations_with_replacement

MODELS = {
    "A": "big2/policies/wangbot_v2.pt",
    "B": "big2/policies/sicario_v1.pt",
    "C": "big2/policies/leonidas_v1.pt",
    "D": "big2/policies/khabib_v1.pt",
}
REFS = {
    "ppo_v1": "big2/policies/ppo_attn.pt",
    "wangbot": "big2/policies/ppo_attn_v11.pt",
}


def play_block(args):
    """(composition, n_games, seed) -> {name: (total, seats_played)}."""
    import random

    comp, n_games, seed = args
    from big2.game import Big2Game, ScoringConfig
    from big2.neural import _load_snapshot_policy
    from big2.rules import DEFAULT_RULES

    pols = {name: _load_snapshot_policy(path)
            for name, path in {**MODELS, **REFS}.items()}
    rng = random.Random(seed)
    totals = defaultdict(float)
    seats_ct = defaultdict(int)
    for _ in range(n_games):
        order = list(comp)
        rng.shuffle(order)
        seats = [pols[name] for name in order]
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        scores = game.play_out(seats)
        for i, name in enumerate(order):
            totals[name] += scores[i]
            seats_ct[name] += 1
    return dict(totals), dict(seats_ct)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=1_000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=2_000)
    args = ap.parse_args()

    comps = [tuple(c) for c in
             combinations_with_replacement(sorted(MODELS), 4)]
    refs = [(m, r) for m in sorted(MODELS) for r in sorted(REFS)]

    blocks = []
    for comp in comps:
        left = args.games
        while left > 0:
            n = min(args.chunk, left)
            blocks.append((comp, n, hash((comp, left)) & 0x7FFFFFFF))
            left -= n
    for m, r in refs:
        comp = (m, r, r, r)
        left = args.games
        while left > 0:
            n = min(args.chunk, left)
            blocks.append((comp, n, hash((comp, left)) & 0x7FFFFFFF))
            left -= n

    print(f"{len(comps)} compositions + {len(refs)} reference rows, "
          f"{len(blocks)} blocks", flush=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_block, blocks)

    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(lambda: defaultdict(int))
    for (comp, _, _), (totals, seats_ct) in zip(blocks, results):
        for name, t in totals.items():
            agg[comp][name] += t
            cnt[comp][name] += seats_ct[name]

    print("\n=== composition matrix (score/game by model) ===")
    for comp in comps:
        row = "  ".join(
            f"{n}:{agg[comp][n] / max(1, cnt[comp][n]):+.2f}"
            for n in sorted(set(comp))
        )
        print(f"{''.join(comp):<6} {row}")

    print("\n=== vs 3x reference ===")
    for m, r in refs:
        comp = (m, r, r, r)
        mine = agg[comp][m] / max(1, cnt[comp][m])
        theirs = agg[comp][r] / max(1, cnt[comp][r])
        print(f"{m} vs 3x {r:<8} {mine:+.3f}/game (ref {theirs:+.3f})")


if __name__ == "__main__":
    main()

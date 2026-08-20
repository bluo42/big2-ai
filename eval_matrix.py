"""Full evaluation of the four models against each other and the field.

Every cell is 4-player, seat-rotated, and paired: each configuration
sees the SAME deals, so differences are not deal luck.

  table        all four at one table -- zero-sum, who leads
  vs WangBot   each alone against three WangBot_v1
  vs PPO v1    each alone against three PPO v1
  vs diet      each against three draws from the whole legacy field
  vs house     each against the deployed house trio (v2 + 2x Khabib)

Raw policies by default (fast, and the search wrapper is the same for
everyone); --search evaluates the deployed shape instead.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES

POLICIES = os.path.join(REPO, "big2", "policies")
MODELS = {
    "v2_patient": "v2_patient.pt",
    "v2_adversarial": "sicario_v1.pt",
    "v2_self_trained": "leonidas_v1.pt",
    "v2_human_trained": "khabib_v1.pt",
}


def load(stem, search=False, seed=0):
    from big2.neural import load_checkpoint_policy

    joint = os.path.join(POLICIES, stem.replace(".pt", "_joint.pt"))
    path = joint if os.path.exists(joint) else os.path.join(POLICIES, stem)
    pol = load_checkpoint_policy(path)
    if search:
        from big2.neural import SearchAssist
        return SearchAssist(pol, seed=seed, simulations=64,
                            time_budget=0.5, search_from=53)
    return pol


def play(seats, seed):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(seed))
    while not g.game_over:
        g.step(seats[g.turn].select(g, g.turn))
    return g


def one_vs_three(hero, pool, n, seed):
    """Hero rotates through all four seats over the same deals.

    A pool of exactly three is that fixed trio; a larger pool (the
    mixed diet) is DRAWN from per game -- taking its first three would
    silently turn "vs the diet" into "vs whatever heads the list",
    which is how this cell first came back identical to the WangBot
    one.
    """
    rng = random.Random(seed)
    tot, wins = 0.0, 0
    fixed = len(pool) <= 3
    for i in range(n):
        deal = rng.randrange(2 ** 31)
        seat = i % 4
        others = (list(pool) if fixed
                  else [rng.choice(pool) for _ in range(3)])
        seats = []
        j = 0
        for p in range(4):
            if p == seat:
                seats.append(hero)
            else:
                seats.append(others[j % len(others)])
                j += 1
        g = play(seats, deal)
        tot += g.scores[seat]
        wins += int(g.winner == seat)
    return tot / n, wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--label", default="post-stop")
    args = ap.parse_args()

    S = args.search
    nets = {n: load(f, S, seed=i) for i, (n, f) in enumerate(MODELS.items())}
    names = list(nets)
    wang = [load("ppo_attn_v11.pt", S, seed=90 + i) for i in range(3)]
    ppo1 = [load("ppo_attn.pt", S, seed=93 + i) for i in range(3)]
    house = [load("wangbot_v2.pt", S, seed=96),
             load("khabib_v1.pt", S, seed=97),
             load("khabib_v1.pt", S, seed=98)]
    from big2.strategies import SmartHeuristic
    diet = wang + ppo1 + [load("humanlike.pt", S, seed=99), SmartHeuristic()]

    shape = "deployed search (64 sims / 500ms)" if S else "raw policy"
    print(f"=== {args.games} games per cell, {shape} ===\n")

    print("all four at one table (zero-sum, seats shuffled)")
    tot, wins = defaultdict(float), defaultdict(int)
    rng = random.Random(4242)
    for _ in range(args.games):
        order = names[:]
        rng.shuffle(order)
        g = play([nets[o] for o in order], rng.randrange(2 ** 31))
        for s, o in enumerate(order):
            tot[o] += g.scores[s]
            wins[o] += int(g.winner == s)
    print(f"  {'model':<20}{'pts/game':>10}{'win %':>9}")
    table = {}
    for n in sorted(names, key=lambda k: -tot[k]):
        table[n] = round(tot[n] / args.games, 3)
        print(f"  {n:<20}{tot[n] / args.games:>+10.3f}"
              f"{100 * wins[n] / args.games:>8.1f}%")

    results = {"label": args.label, "games": args.games,
               "shape": shape, "table": table}
    for field_name, trio in (("vs 3x WangBot_v1", wang),
                             ("vs 3x PPO v1", ppo1),
                             ("vs the house trio", house),
                             ("vs the mixed diet", diet)):
        print(f"\n{field_name}")
        print(f"  {'model':<20}{'pts/game':>10}{'win %':>9}")
        cell = {}
        for n in names:
            s, w = one_vs_three(nets[n], trio, args.games, seed=7)
            cell[n] = round(s, 3)
            print(f"  {n:<20}{s:>+10.3f}{100 * w:>8.1f}%")
        results[field_name] = cell

    out = os.path.join(REPO, "bench_history.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(results) + "\n")
    print(f"\nappended to bench_history.jsonl as '{args.label}'")


if __name__ == "__main__":
    main()

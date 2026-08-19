"""Where the four models stand: against each other, and against the diet.

Two references, because they answer different questions:

* **table** -- all four at one table, shuffled seats.  Zero-sum, so the
  numbers say who is winning the arms race *relative to each other*.
* **diet** -- each model alone against three draws from the legacy
  field (WangBot_v1, PPO v1, humanlike, heuristic).  This is the
  benchmark that must not regress: getting better at each other while
  getting worse against the room is the failure mode the joint
  trainer's rollback bar exists to catch.

Raw policies, no search: this is a fast relative measure run often, and
the search is the same wrapper for everyone anyway.

    python bench_snapshot.py --games 600
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
DIET = ("ppo_attn_v11.pt", "ppo_attn.pt", "humanlike.pt")


def load(stem):
    from big2.neural import load_checkpoint_policy

    joint = os.path.join(POLICIES, stem.replace(".pt", "_joint.pt"))
    return load_checkpoint_policy(
        joint if os.path.exists(joint) else os.path.join(POLICIES, stem))


def play(seats, seed):
    g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                 num_players=4, rng=random.Random(seed))
    while not g.game_over:
        g.step(seats[g.turn].select(g, g.turn))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(REPO, "bench_history.jsonl"))
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    nets = {n: load(f) for n, f in MODELS.items()}
    names = list(nets)

    print(f"=== all four at one table ({args.games} games, shuffled seats) ===")
    tot = defaultdict(float)
    wins = defaultdict(int)
    rng = random.Random(4242)
    for _ in range(args.games):
        order = names[:]
        rng.shuffle(order)
        g = play([nets[o] for o in order], rng.randrange(2 ** 31))
        for s, o in enumerate(order):
            tot[o] += g.scores[s]
            wins[o] += int(g.winner == s)
    print(f"{'model':<20}{'pts/game':>10}{'win %':>9}")
    table = {}
    for n in sorted(names, key=lambda k: -tot[k]):
        table[n] = round(tot[n] / args.games, 3)
        print(f"{n:<20}{tot[n] / args.games:>+10.3f}"
              f"{100 * wins[n] / args.games:>8.1f}%")

    print(f"\n=== each vs three draws from the legacy diet "
          f"({args.games} games) ===")
    field = [load(f) for f in DIET]
    from big2.strategies import SmartHeuristic
    field.append(SmartHeuristic())
    diet = {}
    print(f"{'model':<20}{'pts/game':>10}{'win %':>9}")
    for n in names:
        rng = random.Random(99)          # same deals for every model
        t = w = 0
        for _ in range(args.games):
            opp = [rng.choice(field) for _ in range(3)]
            g = play([nets[n], opp[0], opp[1], opp[2]],
                     rng.randrange(2 ** 31))
            t += g.scores[0]
            w += int(g.winner == 0)
        diet[n] = round(t / args.games, 3)
        print(f"{n:<20}{t / args.games:>+10.3f}{100 * w / args.games:>8.1f}%")

    row = {"label": args.label, "games": args.games,
           "table": table, "diet": diet}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\nappended to {os.path.basename(args.out)} as '{args.label}'")


if __name__ == "__main__":
    main()

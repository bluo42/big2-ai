"""Champions-table eval: the current net vs BOTH shipped champions.

Seats a 4p table with the newest training snapshot, WangBot_v1
(ppo_attn_v11.pt) and PPO v1 (ppo_attn.pt) together in every game, the
fourth seat drawn from the lean field, seats rotated per game.  Reports
each player's mean score/game over the sample — the head-to-head the
run is actually trying to win.  The candidate plays raw (no search),
matching the confirmation methodology; deployed strength with the
IS-MCTS wrapper reads higher.

    python champ_table.py                 # newest snapshot, 400 games
    python champ_table.py --games 1200 --candidate big2/policies/ppo_v3.pt
"""

import argparse
import glob
import os
import random

from big2.game import Big2Game, ScoringConfig
from big2.neural import PPOPolicy, SearchAssist, _load_snapshot_policy
from big2.rules import DEFAULT_RULES


def newest_candidate() -> str:
    cands = glob.glob("big2/policies/ppo_snapshots/*.pt")
    cands += [p for p in ("big2/policies/ppo_v3.pt",
                          "big2/policies/ppo_v3_latest.pt")
              if os.path.exists(p)]
    if not cands:
        raise SystemExit("no candidate checkpoint found")
    return max(cands, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="candidate plays raw instead of its deployed "
                         "search-assisted shape")
    args = ap.parse_args()

    path = args.candidate or newest_candidate()
    cand = PPOPolicy.load(path)
    if not args.raw:
        cand = SearchAssist(cand, seed=args.seed)
    cand.name = "candidate" + (" (raw)" if args.raw else " (deployed)")
    wang = PPOPolicy.load("big2/policies/ppo_attn_v11.pt")
    wang.name = "WangBot_v1"
    ppo1 = PPOPolicy.load("big2/policies/ppo_attn.pt")
    ppo1.name = "PPO_v1"
    # Fourth seat: only the two hardest field members — the exploiter
    # (trained to beat this line) and the humanlike clone.
    field = [p for p in (
        _load_snapshot_policy("big2/policies/ppo_exploiter.pt"),
        _load_snapshot_policy("big2/policies/humanlike.pt"),
    ) if p is not None]

    rng = random.Random(args.seed)
    totals = {"candidate": 0.0, "WangBot_v1": 0.0, "PPO_v1": 0.0,
              "field": 0.0}
    for g in range(args.games):
        cand_seat = g % 4
        others = [p for p in range(4) if p != cand_seat]
        rng.shuffle(others)
        fourth = rng.choice(field)
        seats = [None] * 4
        seats[cand_seat] = cand
        seats[others[0]] = wang
        seats[others[1]] = ppo1
        seats[others[2]] = fourth
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        scores = game.play_out(seats)
        totals["candidate"] += scores[cand_seat]
        totals["WangBot_v1"] += scores[others[0]]
        totals["PPO_v1"] += scores[others[1]]
        totals["field"] += scores[others[2]]

    n = args.games
    print(f"champions table: {os.path.basename(path)}, {n} games "
          f"(candidate + WangBot_v1 + PPO_v1 + field draw, rotated seats)")
    for name in ("candidate", "WangBot_v1", "PPO_v1", "field"):
        print(f"  {name:<11} {totals[name] / n:+.3f}/game")


if __name__ == "__main__":
    main()

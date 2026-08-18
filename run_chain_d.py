"""Chain D: the final boss.

Warm-starts WangBot_v2 (the chain-A recipe at full validated width) and
trains it to win the exact table the other three finals define: 60% of
rollout games seat WangBot_v2, Sicario, and Leonidas together as the
opponents (shuffled seats), the rest draw the usual weighted collection
so the original diet stays respected.  Confirmation room: the trio,
seats with replacement.  Policy-only training (both search-stage gates
measured search-in-rollouts as a net negative).

    python run_chain_d.py --games 50000
"""

import argparse

from big2.neural import train_ppo
from big2.overnight import save_latest

TRIO = ("big2/policies/wangbot_v2.pt",
        "big2/policies/sicario_v1.pt",
        "big2/policies/leonidas_v1.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=50_000)
    ap.add_argument("--games-done", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--resume", default=TRIO[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    gpi = 64
    net = train_ppo(
        iters=max(1, (args.games - args.games_done) // gpi),
        games_per_iter=gpi, workers=args.workers, seed=args.seed,
        resume=args.resume, games_offset=args.games_done,
        d_model=192, layers=2, attn_blocks=2,
        state_dim=323, act_dim=330,
        num_players=4, selfplay_prob=0.0, wildcard_prob=0.07,
        champ_prob=0.0, random_draw=1.0,
        fixed_files=TRIO, fixed_prob=0.6,
        confirm_files=TRIO,
        # Per direction: chain D trains with the tree in the loop like
        # the deployed agent (64 sims / 100ms / depth 10, sub-26 4p).
        search_rollouts=True, search_depth=10,
        confirm_deployed=False, use_profiles=False,
        out="big2/policies/chain_d.pt",
        init_bar=0.0, fresh_bar=True, confirm_games=400,
        ent_coef=0.02, lr=1.5e-4, note="chain-D",
        snapshot_dir="big2/policies/ppo_snapshots_D",
        progress_path="big2/policies/evolve/ladder_D.csv",
        probe_every_iters=100, snapshot_every_iters=150,
        past_self_prob=0.3, device=args.device,
    )
    save_latest(net, 192, 4, 2,
                path="big2/policies/chain_d_latest.pt",
                meta={"chain": "D"})
    print("chain D complete", flush=True)


if __name__ == "__main__":
    main()

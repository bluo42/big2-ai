"""Chain C: the humanlike contender.

Warm-starts the humanlike clone (d128, already at full 323/326 feature
width from its distillation), grafts the second attention block
(zero-init: bit-identical resume), widens for the beat-risk block, and
trains with:

* an imitation anchor — KL toward the frozen original humanlike net on
  every visited state, so the style survives while the results improve
* a diet weighted toward the chain A and chain B finals — the models
  this chain exists to beat — over the usual full-collection floor
* search-assisted rollouts from the start (depth 10, trick-boundary
  cutoff): the tactical arsenal the human style historically lacked
* confirmation room {A_final, B_final, humanlike}

    python run_chain_c.py --games 500000
"""

import argparse

from big2.neural import ACT_DIM_BEAT, STATE_DIM, train_ppo
from big2.overnight import save_latest

# The christened lineage (2026-08-18): chain A ships as WangBot_v2,
# chain B as Sicario (the Wang killer), and this chain — Leonidas,
# the exploitative humanlike — trains to beat them both.
HUMANLIKE = "big2/policies/humanlike.pt"
A_FINAL = "big2/policies/wangbot_v2.pt"
B_FINAL = "big2/policies/sicario_v1.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=500_000)
    ap.add_argument("--games-done", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--resume", default=HUMANLIKE)
    ap.add_argument("--imitate-coef", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    gpi = 64
    net = train_ppo(
        iters=max(1, (args.games - args.games_done) // gpi),
        games_per_iter=gpi, workers=args.workers, seed=args.seed,
        resume=args.resume, games_offset=args.games_done,
        d_model=128, layers=2, attn_blocks=2,
        state_dim=STATE_DIM, act_dim=ACT_DIM_BEAT,
        num_players=4, selfplay_prob=0.0, wildcard_prob=0.07,
        champ_prob=0.5, random_draw=0.25,
        champ_files=(A_FINAL, B_FINAL),
        confirm_files=(A_FINAL, B_FINAL, HUMANLIKE),
        imitate_path=HUMANLIKE, imitate_coef=args.imitate_coef,
        # Both A and B search-stage gates measured search-assisted
        # training as a net negative (-0.29 and -0.15 paired); the tree
        # stays a deployment weapon, not a training one.
        search_rollouts=False, search_depth=10,
        confirm_deployed=False, use_profiles=False,
        out="big2/policies/leonidas_v1.pt",
        init_bar=0.0, fresh_bar=True, confirm_games=400,
        ent_coef=0.02, lr=1.5e-4, note="Leonidas",
        snapshot_dir="big2/policies/ppo_snapshots_C",
        progress_path="big2/policies/evolve/ladder_C.csv",
        probe_every_iters=100, snapshot_every_iters=150,
        past_self_prob=0.3, device=args.device,
    )
    save_latest(net, 128, 4, 2,
                path="big2/policies/chain_C_latest.pt",
                meta={"chain": "C"})
    print("chain C complete", flush=True)


if __name__ == "__main__":
    main()

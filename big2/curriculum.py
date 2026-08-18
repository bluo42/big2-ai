"""Two-stage training run for the v2 (planning + search) agent.

Stage 1 — **1v1**.  Heads-up Big 2 is the same game with the noise
turned down: one hidden hand instead of three, no question of *which*
opponent takes the trick, and a much shorter path from a decision to its
consequence.  The new planning features (is this answerable? can I run
out?) are exactly the ones that resolve cleanly there, so the curriculum
learns them cheaply before paying four-player prices for them.

Stage 2 — **4-player**, warm-started from stage 1.  Same encoders, same
weights; only the table changes.  The gate is the one that matters for
shipping: confirmed score against the live model, not against a stale
champion set.

    python -m big2.curriculum --stage1-games 120000 --stage2-games 880000
"""

from __future__ import annotations

import argparse
import os

from big2.neural import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-games", type=int, default=120_000)
    parser.add_argument("--stage2-games", type=int, default=880_000)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", default="big2/policies/ppo_attn_v11.pt")
    parser.add_argument("--stage1-out", default="big2/policies/ppo_v2_1v1.pt")
    parser.add_argument("--out", default="big2/policies/ppo_v2.pt")
    parser.add_argument("--probe-vs", default="big2/policies/ppo_attn_v11.pt")
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    if not args.skip_stage1:
        iters = max(1, args.stage1_games // args.games_per_iter)
        print(f"=== stage 1: 1v1 curriculum, {args.stage1_games} games "
              f"({iters} iters) ===", flush=True)
        train_ppo(
            iters=iters,
            games_per_iter=args.games_per_iter,
            workers=args.workers,
            resume=args.resume,
            out=args.stage1_out,
            num_players=2,
            probe_every_iters=60,
            fresh_bar=True,
            note="v2-1v1",
            selfplay_prob=0.6,
            games_offset=0,
        )

    start = args.stage1_out if (
        not args.skip_stage1 and os.path.exists(args.stage1_out)
    ) else args.resume
    iters = max(1, args.stage2_games // args.games_per_iter)
    print(f"=== stage 2: 4-player, {args.stage2_games} games ({iters} iters) "
          f"warm-started from {start} ===", flush=True)
    train_ppo(
        iters=iters,
        games_per_iter=args.games_per_iter,
        workers=args.workers,
        resume=start,
        out=args.out,
        num_players=4,
        probe_every_iters=40,
        probe_vs=args.probe_vs,
        fresh_bar=True,
        init_bar=0.0,   # must actually beat the live model to be saved
        note="v2",
        games_offset=args.stage1_games,
    )


if __name__ == "__main__":
    main()

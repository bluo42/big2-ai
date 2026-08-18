"""The v3 line: one long run to build the final model.

Everything this project has learned about training, in one recipe:

* **Fresh d384 net** (1.49M params).  The capacity is close to free at
  inference — feature encoding dominates per-decision cost — and the
  d192 line plateaued.  A wider net cannot warm-start from the d192
  checkpoints, which is fine: the run begins with random restarts, and
  both shipped lines sit in its diet instead of under its skin.
* **Beat the diet by a lot.**  Probes and confirmation are random
  3-member draws from the mixed field (train_ppo's panel default), and
  the save gate starts at +0.5/game vs that field — the checkpoint file
  only ever records a model that outscores the whole field decisively,
  not one that learned to farm a single reference.
* **Alternating 2/3/4-player games**, sampled per game in the rollout
  workers: the 2p game is the clean credit-assignment classroom, 4p is
  the deployed game.  Every head — policy, value, belief — trains
  across counts, and the value/belief heads are exactly what the
  play-time search consumes.
* **One dedicated wildcard seat per table** drawn from the whole model
  collection — scripted regulars, decomposition, CEM/MLP/DMC champions,
  PPO v1, WangBot_v1, the humanlike clone, the exploiter — with the
  remaining opponent seats on lean pool + past selves.  The diet never
  goes stale and PPO v1 (the line the humans rate stronger) is a
  constant presence.  Heads-up tables give the wildcard its 4p share
  (1 game in 3).
* **The best file answers to the diet.**  A candidate that out-probes
  the field must then confirm on 1200 fresh-deal games seated exactly
  like training — same mix, new cards — before ppo_v3.pt is written.
* **Random restarts**: several short seeds first; the one that probes
  best against the field earns the long night.

    python -m big2.overnight                     # full recipe
    python -m big2.overnight --restarts 3 --restart-games 3072
"""

from __future__ import annotations

import argparse
import os

LATEST = "big2/policies/ppo_v3_latest.pt"
BEST = "big2/policies/ppo_v3.pt"
PLAYERS = (2, 3, 4)


def save_latest(net, d_model, heads, layers, path=LATEST, meta=None):
    import torch

    payload = {"state_dict": {k: v.cpu() for k, v in
                              net.state_dict().items()},
               "d_model": d_model, "heads": heads, "layers": layers}
    if meta:
        payload["meta"] = meta
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--restart-games", type=int, default=3072)
    parser.add_argument("--games-target", type=int, default=100_000)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--wildcard-prob", type=float, default=0.07)
    parser.add_argument("--selfplay-prob", type=float, default=0.5)
    parser.add_argument("--save-bar", type=float, default=1.6,
                        help="diet-mix score a candidate must confirm above "
                             "before claiming the best file -- calibrated to "
                             "WangBot_v1's +1.65 on the same mix, so the "
                             "file only records a net at least as strong as "
                             "the best shipped model on its own diet")
    parser.add_argument("--dense-until", type=int, default=10_000,
                        help="probe every ~1k games until this many games, "
                             "then stretch the cadence for the long haul")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from big2.game import ScoringConfig
    from big2.neural import (
        PPOPolicy, confirmation_panel, panel_probe, train_ppo,
    )
    from big2.rules import DEFAULT_RULES

    gpi = args.games_per_iter
    dense_every = max(1, 1024 // gpi)          # ~ every 1k games
    common = dict(
        games_per_iter=gpi,
        workers=args.workers,
        d_model=args.d_model,
        layers=args.layers,
        num_players=PLAYERS,
        wildcard_prob=args.wildcard_prob,
        selfplay_prob=args.selfplay_prob,
        out=BEST,
        init_bar=args.save_bar,
        fresh_bar=True,
        confirm_games=1200,
        note="v3",
        progress_path="big2/policies/evolve/progress.csv",
        device=args.device,
    )

    # ---- Phase A: random restarts, judged against the field ----------
    candidates = []
    iters = max(1, args.restart_games // gpi)
    for k in range(args.restarts):
        print(f"\n===== restart {k}: {iters * gpi} games "
              f"(seed {args.seed + 101 * k}) =====", flush=True)
        net = train_ppo(
            iters=iters, seed=args.seed + 101 * k,
            probe_every_iters=dense_every, **common,
        )
        path = f"big2/policies/ppo_v3_restart{k}.pt"
        save_latest(net, args.d_model, 4, args.layers, path=path,
                    meta={"restart": k})
        candidates.append(path)

    panel = confirmation_panel()
    scores = []
    for k, path in enumerate(candidates):
        s = panel_probe(PPOPolicy.load(path), panel, 360, ScoringConfig(),
                        DEFAULT_RULES, seed=777 + k)
        scores.append(s)
        print(f"restart {k}: {s:+.3f}/game vs the field (360 games)",
              flush=True)
    winner = int(max(range(len(scores)), key=lambda i: scores[i]))
    print(f"\n===== restart {winner} wins ({scores[winner]:+.3f}) — "
          f"continuing it =====", flush=True)

    # ---- Phase B: the long night, dense probes first -----------------
    done = args.restart_games
    start = candidates[winner]
    if done < args.dense_until:
        iters_b1 = max(1, (args.dense_until - done) // gpi)
        net = train_ppo(
            iters=iters_b1, seed=args.seed + 5000, resume=start,
            games_offset=done, probe_every_iters=dense_every, **common,
        )
        save_latest(net, args.d_model, 4, args.layers,
                    meta={"games": done + iters_b1 * gpi})
        done += iters_b1 * gpi
        start = LATEST

    iters_b2 = max(1, (args.games_target - done) // gpi)
    print(f"\n===== long phase: {iters_b2 * gpi} more games, probes every "
          f"{100 * gpi} =====", flush=True)
    net = train_ppo(
        iters=iters_b2, seed=args.seed + 9000, resume=start,
        games_offset=done, probe_every_iters=100,
        snapshot_every_iters=150, past_self_prob=0.4, **common,
    )
    save_latest(net, args.d_model, 4, args.layers,
                meta={"games": done + iters_b2 * gpi})
    print("overnight run complete", flush=True)


if __name__ == "__main__":
    main()

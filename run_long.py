"""v3.1 long phase (local GPU box): resume the v3 net and keep going.

Changes from the committed v3 recipe, per Brandon's direction:
* table mix 25% 2p / 25% 3p / 50% 4p (``num_players=(2, 3, 4, 4)``)
* gate metric = recent rollout form on 4p field games, measured for
  free from the rollouts themselves (big2/neural.py)
* the learner's rollout seats consult IS-MCTS below 26 cards, with
  deliberately shallow/narrow search while beliefs are fuzzy; opponents
  stay raw policies (big2/neural.py)
* eval (gate + diet confirmation) is 4-player only
* no mirror self-play: every game seats the searched learner against
  raw policy networks (field, wildcard, past-self snapshots) — the goal
  is to beat the existing models, and it also removes the 4-searching-
  seat straggler games that were gating iteration wall-clock
* second attention block (attn_blocks=2): move-vs-move comparisons
  compose; warm-starts from the 1-block checkpoint unchanged via the
  zero-initialized output projection

    python run_long.py --games-offset 10000
"""

import argparse
import glob
import os

from big2.neural import train_ppo
from big2.overnight import BEST, LATEST, save_latest


def newest_checkpoint() -> str:
    cands = [p for p in (LATEST, BEST) if os.path.exists(p)]
    cands += glob.glob("big2/policies/ppo_snapshots/*.pt")
    if not cands:
        raise SystemExit("no checkpoint to resume from")
    return max(cands, key=os.path.getmtime)


def current_bar(default: float = 0.0) -> float:
    """The ratchet: the bar is whatever the best file last confirmed on
    the champions metric (seats drawn with replacement from WangBot_v1,
    PPO v1, humanlike), floored at 0.0 — the first confirmed edge over
    that room records, and the record must keep improving.  Bests saved
    under older metrics don't count."""
    try:
        import torch

        meta = torch.load(BEST, map_location="cpu",
                          weights_only=True).get("meta", {})
        if meta.get("metric") != "champs-replacement":
            return default
        return max(default, float(meta.get("probe", default)))
    except Exception:
        return default


def build_phases(offset: int):
    """The curriculum, rebuilt 2026-08-18 against hyper-optimization:
    500k more games of the phase-1 mix, then a randomized phase 3 —
    every opponent seat a fresh draw from the whole collection, at most
    one old self at the table, entropy high but a notch lower."""
    return (
        dict(until=offset + 200_000, num_players=4,
             selfplay_prob=0.0, hard_pool=False, champ_prob=0.5,
             ent_coef=0.03, random_draw=0.25,
             label="phase1b: 4p only, full field, champion seat 50%, "
                   "random tables 25%, search from 26"),
        dict(until=offset + 400_000, num_players=4,
             selfplay_prob=0.0, hard_pool=False, champ_prob=0.0,
             ent_coef=0.02, random_draw=1.0,
             label="phase3: 4p, random draws from the collection, "
                   "<=1 old self"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-offset", type=int, default=10_000)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume (default: newest on disk)")
    ap.add_argument("--lr", type=float, default=1.5e-4,
                    help="halved from the 3e-4 default: the phase-1 "
                         "collapse at ~200k games began right after the "
                         "policy sharpened — constant high LR is the "
                         "classic culprit")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    gpi = args.games_per_iter
    done = args.games_offset
    resume = args.resume or newest_checkpoint()
    for k, ph in enumerate(build_phases(args.games_offset)):
        if done >= ph["until"]:
            continue
        bar = current_bar()
        print(f"\n===== {ph['label']}: {done} -> {ph['until']} games "
              f"(bar {bar:+.2f}, resume {resume}) =====", flush=True)
        iters = max(1, (ph["until"] - done) // gpi)
        net = train_ppo(
            iters=iters, seed=args.seed + 31 * k, resume=resume,
            games_offset=done,
            games_per_iter=gpi, workers=args.workers,
            d_model=384, layers=2, attn_blocks=2,
            num_players=ph["num_players"],
            wildcard_prob=0.07, selfplay_prob=ph["selfplay_prob"],
            hard_pool=ph["hard_pool"], champ_prob=ph["champ_prob"],
            random_draw=ph["random_draw"],
            out=BEST, init_bar=bar, fresh_bar=True, confirm_games=400,
            ent_coef=ph["ent_coef"], lr=args.lr,
            note="v3.1",
            progress_path="big2/policies/evolve/progress.csv",
            probe_every_iters=100, snapshot_every_iters=150,
            past_self_prob=0.4, device=args.device,
        )
        save_latest(net, 384, 4, 2,
                    meta={"games": ph["until"], "note": "v3.1"})
        done = ph["until"]
        resume = LATEST
    print("v3.1 run complete", flush=True)


if __name__ == "__main__":
    main()

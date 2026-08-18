"""Feature ladder: rebuild the policy from the proven base, one block
at a time, keeping only what measurably helps.

Direction (2026-08-18): the v2-era everything-at-once feature pile is
suspect — the best models were PPO v1 and WangBot_v1.  So: start from
the v1 feature set at the original d192 width, keep the two attention
blocks (measured: the second block carries most move-interaction
traffic), train pure policy rollouts (no tree — the tree returns for
mid/endgame after the policy is rebuilt), and reintroduce feature
blocks stage by stage.  A stage's block is kept only if the champions'
room score improves by at least GATE_MARGIN; otherwise the stage is
reverted and the block recorded as not beneficial.

Two chains, run as separate processes:
  A: warm-start PPO v1        (state 275 / act 296)
  B: warm-start the exploiter (state 299 / act 296 — adversarial line)

Stages (dims grow by zero-padded warm start, so each block begins with
exactly zero influence):
  danger    +14 act   WangBot's endgame-danger block — the endgame
                      theory (thresholded counts, who is nearly out,
                      cheap-single-gifted, out-on-this-combo-size)
  profiles  +24 state cross-game opponent profiles (500-game EMA)
  plan      +16 act   exact boss detection (is my pair unbeatable given
            +24 state the unseen set -> free turn), guaranteed-win,
                      keeps-the-lead, wastes-control, race margin,
                      denial vs gift, share of unseen mass above the
                      move; plus run-out summary and opponent read

    python run_ladder.py --chain A
    python run_ladder.py --chain B
"""

import argparse
import os

from big2.game import ScoringConfig
from big2.neural import (
    ACT_DIM, ACT_DIM_BEAT, ACT_DIM_V1, ACT_DIM_V11, FEAT_DIM, STATE_DIM,
    STATE_DIM_V11, PPOPolicy, confirm_pool, replacement_probe, train_ppo,
)
from big2.overnight import save_latest
from big2.rules import DEFAULT_RULES

GATE_MARGIN = 0.10          # champions-room gain a block must show
STAGE_GAMES = 40_000
# Gate protocol (revised 2026-08-18 after a variance audit): one seed's
# 600 games swing by +-1 point in ABSOLUTE terms because a single RNG
# chain drives seats and deals together.  Paired same-seed differences
# are tight, so the gate averages the paired diff over three fixed
# seeds at 1500 games each (~+-0.06 on the diff).
GATE_GAMES = 1_500
GATE_SEEDS = (11, 22, 33)

CHAINS = {
    "A": dict(start="big2/policies/ppo_attn.pt",
              state_dim=FEAT_DIM, act_dim=ACT_DIM_V1),
    "B": dict(start="big2/policies/ppo_exploiter.pt",
              state_dim=STATE_DIM_V11, act_dim=ACT_DIM_V1),
}

# (name, state_dim, act_dim) targets per stage; a stage is skipped when
# the chain already has those dims.  Opponent profiles are OUT
# (2026-08-18): cross-game opponent inference violates the low-bias
# directive, and its columns were zero at eval anyway — training now
# zero-feeds them too (use_profiles=False below), so the plan stage's
# profile columns exist in the layout but never carry information.
STAGES = (
    ("danger", None, ACT_DIM_V11),
    ("plan", STATE_DIM, ACT_DIM),
    # Beat-risk: per-move P(opponents can answer) from uniform deals of
    # the unseen pool — combinatorics only (big2/features.py).
    ("beatprob", None, ACT_DIM_BEAT),
    # Search-assisted rollouts LAST (it slows training ~4x): IS-MCTS
    # plays the trick out (~depth 10) and the value head prices the
    # horizon; opponents in-tree play the agent's own distribution.
    ("search", None, None),
)


def measure(path: str, seed: int = 0) -> float:
    """Mean champions-room score over the fixed gate seeds (the seed
    argument is kept for call-site compatibility and ignored: pairing
    across stages and chains beats per-stage seed variety)."""
    pol = PPOPolicy.load(path)
    return sum(
        replacement_probe(pol, confirm_pool(), GATE_GAMES,
                          ScoringConfig(), DEFAULT_RULES, seed=s)
        for s in GATE_SEEDS
    ) / len(GATE_SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", choices=("A", "B"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stage-games", type=int, default=STAGE_GAMES)
    ap.add_argument("--total-games", type=int, default=1_000_000,
                    help="consolidation trains the final feature set "
                         "out to this many games")
    ap.add_argument("--consolidate", default=None,
                    help="skip the stages: long-train this checkpoint "
                         "at its native dims to --total-games")
    ap.add_argument("--games-done", type=int, default=0,
                    help="games already trained (consolidate mode)")
    ap.add_argument("--resume-from", default=None,
                    help="override the chain's starting checkpoint "
                         "(e.g. its own stage best after a crash); "
                         "dims are read off the checkpoint")
    ap.add_argument("--no-base", action="store_true",
                    help="skip the settle stage (resuming mid-ladder)")
    ap.add_argument("--skip", default="",
                    help="comma-separated stage names to skip (e.g. a "
                         "stage this chain's gate already dropped, or "
                         "one whose columns would smuggle a dropped "
                         "block back in)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.consolidate:
        import torch

        sd = torch.load(args.consolidate, map_location="cpu",
                        weights_only=True)["state_dict"]
        sdim = sd["state_mlp.0.weight"].shape[1]
        adim = sd["act_mlp.0.weight"].shape[1]
        remaining = max(1, args.total_games - args.games_done)
        out = f"big2/policies/ladder_{args.chain}_final.pt"
        print(f"===== chain {args.chain} consolidation: {remaining} games "
              f"at dims {sdim}/{adim} from {args.consolidate} =====",
              flush=True)
        net = train_ppo(
            iters=max(1, remaining // 64), games_per_iter=64,
            workers=args.workers, seed=args.seed + 999,
            resume=args.consolidate, games_offset=args.games_done,
            d_model=192, layers=2, attn_blocks=2,
            state_dim=sdim, act_dim=adim,
            num_players=4, selfplay_prob=0.0, wildcard_prob=0.07,
            champ_prob=0.5, random_draw=0.25,
            search_rollouts=False, confirm_deployed=False,
            use_profiles=False,
            out=out, init_bar=0.0, fresh_bar=True, confirm_games=400,
            ent_coef=0.02, lr=1.5e-4,
            note=f"ladder-{args.chain}-final",
            snapshot_dir=f"big2/policies/ppo_snapshots_{args.chain}",
            progress_path=f"big2/policies/evolve/ladder_{args.chain}.csv",
            probe_every_iters=100, snapshot_every_iters=150,
            past_self_prob=0.3, device=args.device,
        )
        save_latest(net, 192, 4, 2,
                    path=f"big2/policies/ladder_{args.chain}_final_latest.pt",
                    meta={"stage": "final"})
        print(f"chain {args.chain} consolidation complete", flush=True)
        return

    chain = CHAINS[args.chain]
    cur = args.resume_from or chain["start"]
    sdim, adim = chain["state_dim"], chain["act_dim"]
    if args.resume_from:
        import torch

        sd0 = torch.load(args.resume_from, map_location="cpu",
                         weights_only=True)["state_dict"]
        sdim = sd0["state_mlp.0.weight"].shape[1]
        adim = sd0["act_mlp.0.weight"].shape[1]
    kept, dropped = [], []
    gpi = 64
    games = 0

    # Stage 0 settles the warm start (2-block attention wakes up) at
    # the chain's native dims before any block is judged.  Dims then
    # accumulate stage by stage — but only when a stage's block is
    # KEPT, so a dropped block's columns never haunt later stages.
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    stage_list = ([] if args.no_base else [("base", None, None)]) \
        + [st for st in STAGES if st[0] not in skip]

    for k, (name, s_want, a_want) in enumerate(stage_list):
        s_next = max(s_want or 0, sdim)
        a_next = max(a_want or 0, adim)
        is_search = name == "search"
        if (name != "base" and not is_search
                and s_next == sdim and a_next == adim):
            continue  # chain already carries these dims
        out = f"big2/policies/ladder_{args.chain}_{name}.pt"
        print(f"\n===== chain {args.chain} stage {name}: "
              f"dims {s_next}/{a_next}, {args.stage_games} games =====",
              flush=True)
        net = train_ppo(
            iters=max(1, args.stage_games // gpi), games_per_iter=gpi,
            workers=args.workers, seed=args.seed + 17 * k,
            resume=cur, games_offset=games,
            d_model=192, layers=2, attn_blocks=2,
            state_dim=s_next, act_dim=a_next,
            num_players=4, selfplay_prob=0.0, wildcard_prob=0.07,
            champ_prob=0.5, random_draw=0.25,
            search_rollouts=is_search, search_depth=10,
            confirm_deployed=False, use_profiles=False,
            out=out, init_bar=0.0, fresh_bar=True, confirm_games=400,
            ent_coef=0.02, lr=1.5e-4,
            note=f"ladder-{args.chain}-{name}",
            snapshot_dir=f"big2/policies/ppo_snapshots_{args.chain}",
            progress_path=f"big2/policies/evolve/ladder_{args.chain}.csv",
            probe_every_iters=100, snapshot_every_iters=150,
            past_self_prob=0.3, device=args.device,
        )
        games += args.stage_games
        latest = f"big2/policies/ladder_{args.chain}_{name}_latest.pt"
        save_latest(net, 192, 4, 2, path=latest, meta={"stage": name})
        candidate = out if os.path.exists(out) else latest

        score = measure(candidate, seed=1000 + k)
        base_score = measure(cur, seed=1000 + k)  # same seed: paired deals
        gain = score - base_score
        print(f"[ladder {args.chain}] {name}: {base_score:+.3f} -> "
              f"{score:+.3f} (gain {gain:+.3f})", flush=True)
        if name == "base" or gain >= GATE_MARGIN:
            cur, sdim, adim = candidate, s_next, a_next
            if name != "base":
                kept.append(name)
        else:
            dropped.append(name)
            print(f"[ladder {args.chain}] {name} dropped (below "
                  f"{GATE_MARGIN:+.2f}); reverting", flush=True)

    print(f"\nchain {args.chain} done: kept {kept or 'none'}, "
          f"dropped {dropped or 'none'}, final {cur}", flush=True)


if __name__ == "__main__":
    main()

"""Build v2_patient: AWR on the human decisions that measurably beat v2.

Stage 1  extract -- walk replays.jsonl, find every human decision where
         the human chose differently from wangbot_v2 and the human's
         branch measurably wins the playout comparison (house-bot
         opponents, exact solver late).  Cache the dataset to disk so
         reruns skip the expensive part.
Stage 2  train  -- AWR fine-tune of a copy of wangbot_v2 with a KL +
         value anchor to the base, then numpy export.
Stage 3  gate   -- v2_patient vs the base at the house table (paired
         seeds), plus an early-hand patience census: the model should
         pass more / spend fewer boss cards early without losing to
         the room.

    python run_patient.py [--skip-extract] [--games N]
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

DATASET = os.path.join(REPO, "patient_dataset.pkl")
OUT = os.path.join(REPO, "big2", "policies", "v2_patient.pt")
BASE = os.path.join(REPO, "big2", "policies", "wangbot_v2.pt")


def extract(max_games=None):
    from big2.neural import load_checkpoint_policy
    from big2.offline import build_advantage_dataset, load_replays

    model = load_checkpoint_policy(BASE)
    opponents = [
        load_checkpoint_policy(os.path.join(REPO, "big2", "policies", f))
        for f in ("wangbot_v2.pt", "khabib_v1.pt", "sicario_v1.pt")
    ]
    rows = load_replays(os.path.join(REPO, "replays.jsonl"))
    print(f"[extract] {len(rows)} recorded games", flush=True)
    data = build_advantage_dataset(
        rows, model, opponents=opponents, seats="human",
        rollouts=16, beta=2.0, max_weight=8.0, min_advantage=0.25,
        max_games=max_games,
    )
    print(f"[extract] kept {data.get('n', 0)} measured-advantage decisions",
          flush=True)
    with open(DATASET, "wb") as f:
        pickle.dump(data, f)
    return data


def gate(n_games=500):
    """Paired: same seeds, v2_patient vs wangbot_v2 in seat 0 at the
    house table.  Also count early-hand passes with a playable single."""
    import torch

    from big2.game import Big2Game, ScoringConfig
    from big2.neural import load_checkpoint_policy
    from big2.rules import DEFAULT_RULES
    from big2.endgame import remaining_cards

    base = load_checkpoint_policy(BASE)
    patient = load_checkpoint_policy(OUT)
    opps = [load_checkpoint_policy(
        os.path.join(REPO, "big2", "policies", f))
        for f in ("wangbot_v2.pt", "khabib_v1.pt", "sicario_v1.pt")]

    def run(policy, seed):
        rng = random.Random(seed)
        total, wins, early_pass, early_dec = 0.0, 0, 0, 0
        for _ in range(n_games):
            g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                         num_players=4,
                         rng=random.Random(rng.randrange(2 ** 31)))
            seats = {0: policy, 1: opps[0], 2: opps[1], 3: opps[2]}
            while not g.game_over:
                p = g.turn
                mv = seats[p].select(g, p)
                if p == 0 and remaining_cards(g) > 34 and g.can_pass():
                    early_dec += 1
                    early_pass += int(mv is None)
                g.step(mv)
            total += g.scores[0]
            wins += int(g.winner == 0)
        return (total / n_games, wins / n_games,
                early_pass / max(1, early_dec))

    for seed in (11, 22, 33):
        b = run(base, seed)
        p = run(patient, seed)
        print(f"[gate] seed {seed}: base {b[0]:+.3f}/g win {b[1]:.1%} "
              f"earlypass {b[2]:.1%} | patient {p[0]:+.3f}/g win {p[1]:.1%} "
              f"earlypass {p[2]:.1%} | diff {p[0] - b[0]:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--anchor", type=float, default=0.3)
    ap.add_argument("--gate-games", type=int, default=500)
    args = ap.parse_args()

    if args.skip_extract and os.path.exists(DATASET):
        with open(DATASET, "rb") as f:
            data = pickle.load(f)
        print(f"[extract] cached: {data.get('n', 0)} decisions", flush=True)
    else:
        data = extract(args.games)

    if not args.skip_train:
        from big2.offline import train_awr
        train_awr(data, resume=BASE, out=OUT, epochs=args.epochs,
                  lr=5e-5, anchor=args.anchor, seed=0)
        from big2.ppo_numpy import export_numpy
        export_numpy(OUT, OUT.replace(".pt", "_np.npz"))
        print("[train] exported numpy", flush=True)

    if args.gate_games > 0:
        gate(args.gate_games)


if __name__ == "__main__":
    main()

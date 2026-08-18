"""Distill the strong testers' play into a human-like opponent.

The four testers who actually beat the shipped bots don't play like any
model in the training diet: they pass more, they hold their high cards
longer, and they answer about 1.8 ranks above the minimum when they do
act.  The diet has nothing shaped like that, which is precisely why the
leak survived 200k games of self-play — the net never trained against
the style that exploits it.

This module is *behavioral cloning*, not improvement.  The earlier
attempt to fine-tune the champion on these replays (big2/offline.py's
AWR path) failed for a sound reason: a thousand decisions cannot
out-shout deal luck when the target is "play better".  The target here
is far easier — "play *like them*" — a plain supervised problem where
every decision is a clean label, wins and losses alike, because style
is what we are copying, not outcomes.

The product is a normal policy checkpoint (loads via ``PPOPolicy.load``)
whose job is to sit in the training diet and make future candidates
earn their score against human-shaped opposition.  It is measured on
what it is for:

* held-out top-1/top-3 agreement with the humans it clones, against two
  baselines — chance and WangBot_v1's own agreement rate (the model the
  humans beat); it has to land closer to them than the bot does,
* style statistics on self-play (pass rate with a beat in hand, rank
  height of answers) compared to the humans' measured numbers.

    python -m big2.humanlike --replays replays.jsonl \\
        --players justinwang1688,brandonluo,LEX,brandontest
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.cards import rank
from big2.endgame import move_key
from big2.game import Big2Game
from big2.neural import build_net, encode_decision
from big2.offline import _replay_body, iter_decisions, load_replays

STRONG_TESTERS = ("justinwang1688", "brandonluo", "LEX", "brandontest")
DEFAULT_OUT = "big2/policies/humanlike.pt"


# ----------------------------------------------------------------------
# Dataset: one row per human decision, split by game
# ----------------------------------------------------------------------


def human_decisions(
    rows: Sequence[Dict], players: Sequence[str]
) -> List[Dict]:
    """Every multi-option decision the named humans made.

    Forced positions carry no information about preference and are
    dropped.  Each item keeps its game id so the train/validation split
    can hold out whole games — splitting a game across the two would
    leak: consecutive decisions in one hand are nearly duplicates.
    """
    want = {p.lower() for p in players}
    out = []
    for gi, row in enumerate(rows):
        if str(row.get("username", "")).lower() not in want:
            continue
        body = _replay_body(row)
        if body is None:
            continue
        seat = int(row.get("user_seat", 0))
        for game, p, cards in iter_decisions(body):
            if p != seat:
                continue
            options, state, acts = encode_decision(game, p)
            if len(options) < 2:
                continue
            key = (None if not cards
                   else tuple(sorted(int(c) for c in cards)))
            try:
                chosen = next(i for i, m in enumerate(options)
                              if move_key(m) == key)
            except StopIteration:
                continue  # desynced record: no honest label
            out.append({
                "game": gi,
                "user": row["username"],
                "state": state,
                "acts": acts,
                "chosen": chosen,
                "n_options": len(options),
            })
    return out


def split_by_game(
    items: Sequence[Dict], val_frac: float = 0.15, seed: int = 0
) -> Tuple[List[Dict], List[Dict]]:
    games = sorted({it["game"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_frac))
    val_games = set(games[:n_val])
    train = [it for it in items if it["game"] not in val_games]
    val = [it for it in items if it["game"] in val_games]
    return train, val


def collate(items: Sequence[Dict]):
    """Pad variable option sets into (state, acts, mask, chosen)."""
    import torch

    b = len(items)
    a = max(it["n_options"] for it in items)
    sdim = items[0]["state"].shape[0]
    adim = items[0]["acts"].shape[1]
    state = torch.zeros(b, sdim)
    acts = torch.zeros(b, a, adim)
    mask = torch.zeros(b, a, dtype=torch.bool)
    chosen = torch.zeros(b, dtype=torch.long)
    for i, it in enumerate(items):
        n = it["n_options"]
        state[i] = torch.from_numpy(it["state"])
        acts[i, :n] = torch.from_numpy(it["acts"])
        mask[i, :n] = True
        chosen[i] = it["chosen"]
    return state, acts, mask, chosen


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train(
    train_items: Sequence[Dict],
    val_items: Sequence[Dict],
    d_model: int = 128,
    heads: int = 4,
    epochs: int = 60,
    batch: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    patience: int = 8,
    seed: int = 0,
    verbose: bool = True,
):
    """Cross-entropy on the human's choice, early-stopped on held-out
    games.  ~2.5k decisions is tiny, so everything here leans against
    memorising: a small net, weight decay, smoothed labels, and the
    checkpoint that generalised best rather than the last one."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    sdim = train_items[0]["state"].shape[0]
    adim = train_items[0]["acts"].shape[1]
    net = build_net(d_model, heads, state_dim=sdim, act_dim=adim)
    opt = torch.optim.Adam(net.parameters(), lr=lr,
                           weight_decay=weight_decay)
    rng = random.Random(seed)
    order = list(range(len(train_items)))
    best_val, best_state, since = float("inf"), None, 0

    for ep in range(epochs):
        net.train()
        rng.shuffle(order)
        tot = n = 0.0
        for i in range(0, len(order), batch):
            items = [train_items[j] for j in order[i : i + batch]]
            state, acts, mask, chosen = collate(items)
            logits, _v, _b = net(state, acts, mask)
            # Smoothing must stay inside the *legal* options: padded
            # slots sit at -1e9, and F.cross_entropy's label_smoothing
            # would spread target mass onto them.
            logp = F.log_softmax(logits, dim=1)
            nll = -logp.gather(1, chosen.unsqueeze(1)).squeeze(1)
            smooth = -(logp * mask).sum(1) / mask.sum(1)
            loss = ((1.0 - label_smoothing) * nll
                    + label_smoothing * smooth).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(items)
            n += len(items)
        val = evaluate(net, val_items)
        if verbose:
            print(f"[humanlike] epoch {ep + 1:3d} train {tot / n:.4f} "
                  f"val {val['loss']:.4f} top1 {val['top1']:.3f} "
                  f"top3 {val['top3']:.3f}", flush=True)
        if val["loss"] < best_val - 1e-4:
            best_val, since = val["loss"], 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def evaluate(net, items: Sequence[Dict], batch: int = 256) -> Dict[str, float]:
    """Loss and top-k agreement with the humans on held-out games."""
    import torch
    import torch.nn.functional as F

    if not items:
        return {"loss": float("nan"), "top1": 0.0, "top3": 0.0}
    net.eval()
    tot = t1 = t3 = 0.0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            state, acts, mask, chosen = collate(chunk)
            logits, _v, _b = net(state, acts, mask)
            tot += float(F.cross_entropy(logits, chosen, reduction="sum"))
            top = logits.topk(min(3, logits.shape[1]), dim=1).indices
            t1 += float((top[:, 0] == chosen).sum())
            t3 += float((top == chosen.unsqueeze(1)).any(1).sum())
    n = len(items)
    return {"loss": tot / n, "top1": t1 / n, "top3": t3 / n}


def policy_agreement(policy, items: Sequence[Dict]) -> float:
    """How often an existing policy's argmax matches the human choice —
    the baseline a clone has to beat to be worth seating in the diet."""
    hits = 0
    for it in items:
        scores = policy_scores_for(policy, it)
        if int(np.argmax(scores)) == it["chosen"]:
            hits += 1
    return hits / max(1, len(items))


def policy_scores_for(policy, item: Dict) -> np.ndarray:
    """Score a stored decision with a live policy net, tolerating older
    (narrower) encodings by truncating the stored feature rows."""
    import torch

    net = policy.net
    sdim = getattr(net, "state_dim", item["state"].shape[0])
    adim = getattr(net, "act_dim", item["acts"].shape[1])
    state = torch.from_numpy(item["state"][:sdim]).unsqueeze(0)
    acts = torch.from_numpy(item["acts"][:, :adim]).unsqueeze(0)
    mask = torch.ones(1, item["n_options"], dtype=torch.bool)
    with torch.no_grad():
        logits, _v, _b = net(state, acts, mask)
    return logits[0].numpy()


# ----------------------------------------------------------------------
# Style: does it *behave* like them?
# ----------------------------------------------------------------------


def style_stats_from_replays(
    rows: Sequence[Dict], players: Sequence[str]
) -> Dict[str, float]:
    """The humans' measured tendencies, from their own decisions."""
    want = {p.lower() for p in players}
    passes = followable = 0
    jumps: List[float] = []
    for row in rows:
        if str(row.get("username", "")).lower() not in want:
            continue
        body = _replay_body(row)
        if body is None:
            continue
        seat = int(row.get("user_seat", 0))
        for game, p, cards in iter_decisions(body):
            if p != seat or game.table_combo is None:
                continue
            if not game.legal_moves(p):
                continue  # forced pass: not a choice
            followable += 1
            if not cards:
                passes += 1
            else:
                jumps.append(_rank_jump(game, cards))
    return {
        "pass_rate": passes / max(1, followable),
        "mean_rank_jump": float(np.mean(jumps)) if jumps else 0.0,
        "followable": followable,
    }


def style_stats_from_policy(
    policy, n_games: int = 60, num_players: int = 4, seed: int = 0
) -> Dict[str, float]:
    """The same tendencies, measured on the policy's own play (seat 0
    against three copies of itself)."""
    passes = followable = 0
    jumps: List[float] = []
    for k in range(n_games):
        g = Big2Game(num_players=num_players,
                     rng=random.Random(seed + k))
        while not g.game_over:
            p = g.turn
            move = policy.select(g, p)
            if p == 0 and g.table_combo is not None and g.legal_moves(p):
                followable += 1
                if move is None:
                    passes += 1
                else:
                    jumps.append(_rank_jump(g, list(move.cards)))
            g.step(move)
    return {
        "pass_rate": passes / max(1, followable),
        "mean_rank_jump": float(np.mean(jumps)) if jumps else 0.0,
        "followable": followable,
    }


def _rank_jump(game: Big2Game, cards: Sequence[int]) -> float:
    """How far above the table's top rank this answer landed."""
    table = game.table_combo
    return float(rank(max(int(c) for c in cards))
                 - rank(max(table.cards)))


# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True)
    parser.add_argument("--players", default=",".join(STRONG_TESTERS))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline",
                        default="big2/policies/ppo_attn_v11.pt",
                        help="agreement baseline (the model they beat)")
    args = parser.parse_args()

    import torch

    from big2.neural import PPOPolicy

    players = [p for p in args.players.split(",") if p]
    rows = load_replays(args.replays)
    items = human_decisions(rows, players)
    users = sorted({it["user"] for it in items})
    print(f"[humanlike] {len(items)} decisions from "
          f"{len({it['game'] for it in items})} games by {users}")
    train_items, val_items = split_by_game(items, seed=args.seed)
    print(f"[humanlike] train {len(train_items)} / val {len(val_items)} "
          f"(held-out whole games)")

    net = train(train_items, val_items, d_model=args.d_model,
                epochs=args.epochs, seed=args.seed)
    report = evaluate(net, val_items)
    chance = float(np.mean([1.0 / it["n_options"] for it in val_items]))
    print(f"\n[humanlike] held-out agreement: top1 {report['top1']:.3f} "
          f"top3 {report['top3']:.3f} (chance {chance:.3f})")
    try:
        base = PPOPolicy.load(args.baseline)
        agree = policy_agreement(base, val_items)
        print(f"[humanlike] {args.baseline} agrees with the humans on "
              f"{agree:.3f} of the same positions")
    except Exception as exc:  # baseline is a diagnostic, not a dependency
        print(f"[humanlike] baseline skipped: {exc}")

    torch.save(
        {"state_dict": net.state_dict(), "d_model": args.d_model,
         "heads": 4,
         "meta": {"kind": "humanlike", "players": players,
                  "decisions": len(items), "val_top1": report["top1"],
                  "val_top3": report["top3"]}},
        args.out,
    )
    print(f"[humanlike] saved -> {args.out}")

    human = style_stats_from_replays(rows, players)
    clone = style_stats_from_policy(PPOPolicy.load(args.out), seed=7)
    print(f"\n[humanlike] style   {'pass rate':>10} {'rank jump':>10}")
    print(f"  humans            {human['pass_rate']:>10.3f} "
          f"{human['mean_rank_jump']:>10.2f}")
    print(f"  clone (self-play) {clone['pass_rate']:>10.3f} "
          f"{clone['mean_rank_jump']:>10.2f}")
    try:
        bot = style_stats_from_policy(PPOPolicy.load(args.baseline), seed=7)
        print(f"  {os.path.basename(args.baseline):<17} "
              f"{bot['pass_rate']:>10.3f} {bot['mean_rank_jump']:>10.2f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

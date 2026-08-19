"""Learn from recorded human games.

Self-play teaches an agent to beat *itself*.  When specific humans beat
the bots by a wide margin, they are demonstrating something the self-play
distribution never produced — and those games are recorded, replayable,
and labelled with the outcome.  This module turns them into gradient.

The method is **advantage-weighted regression** (Peng et al., 2019): the
policy is regressed onto the observed action, each decision weighted by

    w = exp(advantage / beta)

so moves from games that went well pull hard and moves from games that
went badly barely pull at all.  Unlike plain behaviour cloning it will
not imitate a losing line just because a human played it, and unlike
policy gradient it needs no environment interaction — the recorded
replay *is* the data.

Whose decisions to imitate is a choice, exposed as ``seats``:

* ``human`` — the recorded human's moves in games they won big.  This is
  the "learn what beat us" pass.
* ``all`` — everyone's moves, each weighted by that seat's own outcome;
  the bots' good games count too.

Usage (replays come from ``python -m big2.store --export``):

    python -m big2.offline --replays replays.jsonl \\
        --resume big2/policies/ppo_attn_v11.pt --out big2/policies/ppo_v2.pt
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from big2.combos import classify
from big2.game import Big2Game, PlayRecord, ScoringConfig
from big2.rules import RuleConfig
from big2.webapi import rules_from_dict, scoring_from_dict

SCORE_SCALE = 39.0


def load_replays(path: str) -> List[Dict]:
    """Rows as written by ``Store.export_replays`` (JSONL)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _replay_body(row: Dict) -> Optional[Dict]:
    """Accept either an export row ({..., 'replay': {...}}) or a bare
    replay object as logged by the frontend."""
    body = row.get("replay") if "replay" in row else row
    if not body or not body.get("initial_hands") or not body.get("actions"):
        return None
    return body


def rebuild_game(body: Dict) -> Big2Game:
    """A fresh game positioned at the start of the recorded hand."""
    hands = [sorted(int(c) for c in h) for h in body["initial_hands"]]
    rules: RuleConfig = rules_from_dict(body.get("rules") or {})
    scoring: ScoringConfig = scoring_from_dict(
        {"tiered": True} if isinstance(body.get("scoring"), str)
        else (body.get("scoring") or {})
    )
    g = object.__new__(Big2Game)
    g.scoring = scoring
    g.rules = rules
    g.num_players = int(body.get("num_players", len(hands)))
    g.rng = random.Random(0)
    g.hands = hands
    g.start_card = int(
        body.get("start_card", min(min(h) for h in hands if h))
    )
    g.turn = next(p for p in range(g.num_players) if g.start_card in hands[p])
    g.first_play = True
    g.table_combo = None
    g.table_player = None
    g.passed = [False] * g.num_players
    g.history = []
    g.played_cards = []
    g.winner = None
    g.scores = None
    return g


def iter_decisions(
    body: Dict,
) -> Iterator[Tuple[Big2Game, int, Optional[Sequence[int]]]]:
    """Walk a recorded hand, yielding (position, player, chosen cards).

    The position is the live game *before* the move, so callers can
    encode it exactly as the agent would have seen it.
    """
    game = rebuild_game(body)
    for act in body["actions"]:
        if game.game_over:
            break
        p = int(act["p"])
        if p != game.turn:  # desynced recording: stop rather than guess
            return
        cards = act.get("cards")
        yield game, p, cards
        move = (
            None if not cards
            else classify([int(c) for c in cards], game.rules)
        )
        try:
            game.step(move)
        except (ValueError, RuntimeError):
            return


def replay_outcomes(body: Dict) -> Dict[int, float]:
    """Final score per seat, from the record (falling back to a replay)."""
    scores = body.get("scores") or {}
    if scores:
        return {int(k): float(v) for k, v in scores.items()}
    return {}


def build_dataset(
    rows: Sequence[Dict],
    seats: str = "human",
    min_margin: float = 1.0,
    beta: float = 6.0,
    max_weight: float = 8.0,
    include_plan: bool = True,
) -> Dict[str, np.ndarray]:
    """Encoded decisions + AWR weights from recorded games.

    ``min_margin`` keeps only games decided by at least that many points
    (the lopsided ones carry the signal); ``beta`` is the AWR
    temperature in score units.
    """
    from big2.neural import encode_decision

    states, acts, masks, chosen, weights = [], [], [], [], []
    for row in rows:
        body = _replay_body(row)
        if body is None:
            continue
        outcomes = replay_outcomes(body)
        if not outcomes:
            continue
        user_seat = int(row.get("user_seat", body.get("user_seat", 0)))
        for game, p, cards in iter_decisions(body):
            if seats == "human" and p != user_seat:
                continue
            score = outcomes.get(p, 0.0)
            if score < min_margin:
                continue
            options, state, act_rows = encode_decision(
                game, p, include_plan=include_plan
            )
            if len(options) < 2:
                continue
            key = None if not cards else tuple(sorted(int(c) for c in cards))
            idx = next(
                (
                    i
                    for i, m in enumerate(options)
                    if (None if m is None else tuple(sorted(m.cards))) == key
                ),
                None,
            )
            if idx is None:
                continue
            states.append(state)
            acts.append(act_rows)
            chosen.append(idx)
            weights.append(min(max_weight, float(np.exp(score / beta))))
    if not states:
        return {"n": 0}
    max_a = max(a.shape[0] for a in acts)
    dim = acts[0].shape[1]
    padded = np.zeros((len(acts), max_a, dim), dtype=np.float32)
    mask = np.zeros((len(acts), max_a), dtype=bool)
    for i, a in enumerate(acts):
        padded[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return {
        "n": len(states),
        "state": np.stack(states),
        "acts": padded,
        "mask": mask,
        "chosen": np.asarray(chosen, dtype=np.int64),
        "weight": np.asarray(weights, dtype=np.float32),
    }


def build_advantage_dataset(
    rows: Sequence[Dict],
    model,
    opponents: Optional[Sequence] = None,
    seats: str = "human",
    rollouts: int = 16,
    beta: float = 2.0,
    max_weight: float = 8.0,
    min_advantage: float = 0.25,
    max_games: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Learn only from moves that *measurably* beat the model's choice.

    Weighting imitation by the final score is confounded in Big 2: a
    player who won by fifteen may simply have been dealt a monster, and
    every move they made inherits that credit.  Here each decision is
    judged on its own — the recorded move and the model's preferred move
    are both played out from the same position over belief-sampled
    deals, and the gap between them is the advantage.

    Only positions where the recorded move actually won the comparison
    are kept, so the training signal is "here is a spot you get wrong,
    and here is what beats it" rather than "this player had a good day".
    """
    from big2.critique import move_ev
    from big2.endgame import move_key
    from big2.neural import encode_decision
    from big2.strategies import SmartHeuristic

    opponents = list(opponents or [model, SmartHeuristic()])
    states, acts, chosen, weights = [], [], [], []
    seen_games = 0
    for row in rows:
        body = _replay_body(row)
        if body is None:
            continue
        if max_games and seen_games >= max_games:
            break
        seen_games += 1
        user_seat = int(row.get("user_seat", body.get("user_seat", 0)))
        for game, p, cards in iter_decisions(body):
            if seats == "human" and p != user_seat:
                continue
            # Encode with the MODEL's feature set (danger/plan/beat...),
            # not the v1 defaults -- the dataset must match the net it
            # will fine-tune.
            options, state, act_rows = encode_decision(
                game, p,
                include_profiles=getattr(model, "uses_profiles", False),
                include_danger=getattr(model, "uses_danger", False),
                include_plan=getattr(model, "uses_plan", False),
                include_beat=getattr(model, "uses_beat", False),
            )
            if len(options) < 2:
                continue
            key = None if not cards else tuple(sorted(int(c) for c in cards))
            idx = next(
                (i for i, m in enumerate(options)
                 if (None if m is None else tuple(sorted(m.cards))) == key),
                None,
            )
            if idx is None:
                continue
            model_key = move_key(model.select(game, p))
            if model_key == key:
                continue  # already agrees: nothing to learn here
            played = options[idx]
            theirs, _ = move_ev(game, p, played, opponents, rollouts)
            mine, _ = move_ev(
                game, p,
                next((m for m in options if move_key(m) == model_key), None),
                opponents, rollouts,
            )
            advantage = theirs - mine
            if advantage < min_advantage:
                continue
            states.append(state)
            acts.append(act_rows)
            chosen.append(idx)
            weights.append(min(max_weight, float(np.exp(advantage / beta))))
    if not states:
        return {"n": 0}
    max_a = max(a.shape[0] for a in acts)
    dim = acts[0].shape[1]
    padded = np.zeros((len(acts), max_a, dim), dtype=np.float32)
    mask = np.zeros((len(acts), max_a), dtype=bool)
    for i, a in enumerate(acts):
        padded[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return {
        "n": len(states),
        "state": np.stack(states),
        "acts": padded,
        "mask": mask,
        "chosen": np.asarray(chosen, dtype=np.int64),
        "weight": np.asarray(weights, dtype=np.float32),
    }


def train_awr(
    data: Dict[str, np.ndarray],
    resume: str,
    out: str,
    epochs: int = 8,
    lr: float = 5e-5,
    minibatch: int = 128,
    seed: int = 0,
    anchor: float = 0.3,
    verbose: bool = True,
):
    """Advantage-weighted regression fine-tune of a trained checkpoint.

    The net is rebuilt at the checkpoint's own dimensions, so v2-era
    files (beat features, second attention block) fine-tune as
    themselves.  ``anchor`` adds a KL pull toward the frozen base
    policy plus a value-consistency term on the same states: the point
    is a model that gains the humans' choices at these positions
    without forgetting how to play everywhere else -- these are the
    only states in the batch, so unanchored gradient here is free to
    wreck the trunk for every state not in the batch.
    """
    import torch

    from big2.neural import build_net

    if not data.get("n"):
        raise ValueError("empty dataset: no usable decisions in the replays")
    torch.manual_seed(seed)
    payload = torch.load(resume, map_location="cpu", weights_only=True)
    sd = payload["state_dict"]

    def _mk_net():
        return build_net(
            payload.get("d_model", 192), payload.get("heads", 4),
            state_dim=sd["state_mlp.0.weight"].shape[1],
            act_dim=sd["act_mlp.0.weight"].shape[1],
            layers=payload.get("layers", 2),
            attn_blocks=2 if any(k.startswith("attn2.") for k in sd) else 1,
        )

    net = _mk_net()
    net.load_state_dict(sd)
    base = _mk_net()
    base.load_state_dict(sd)
    base.eval()
    for prm in base.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    S = torch.from_numpy(data["state"])
    A = torch.from_numpy(data["acts"])
    M = torch.from_numpy(data["mask"])
    C = torch.from_numpy(data["chosen"])
    W = torch.from_numpy(data["weight"])
    W = W / W.mean().clamp(min=1e-6)
    n = len(S)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            logits, value, _ = net(S[idx], A[idx], M[idx])
            with torch.no_grad():
                blogits, bvalue, _ = base(S[idx], A[idx], M[idx])
            logp = torch.log_softmax(logits, dim=-1)
            picked = logp.gather(1, C[idx].unsqueeze(1)).squeeze(1)
            loss = -(W[idx] * picked).mean()
            if anchor > 0:
                bp = torch.softmax(blogits, dim=-1)
                kl = (bp * (torch.log_softmax(blogits, dim=-1) - logp))
                loss = loss + anchor * kl.sum(-1).mean()
                loss = loss + anchor * (value - bvalue).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        if verbose:
            print(f"[awr] epoch {ep + 1}/{epochs} loss {total / n:.4f}",
                  flush=True)
    meta = dict(payload.get("meta", {}))
    meta.update({"awr_decisions": int(n), "note": "awr-human",
                 "anchor": anchor, "resume": resume})
    torch.save(
        {"state_dict": net.state_dict(),
         "d_model": payload.get("d_model", 192),
         "heads": payload.get("heads", 4),
         "layers": payload.get("layers", 2),
         "meta": meta},
        out,
    )
    if verbose:
        print(f"[awr] saved -> {out}", flush=True)
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True,
                        help="JSONL from python -m big2.store --export")
    parser.add_argument("--resume", default="big2/policies/ppo_attn_v11.pt")
    parser.add_argument("--out", default="big2/policies/ppo_awr.pt")
    parser.add_argument("--seats", choices=("human", "all"), default="human")
    parser.add_argument("--min-margin", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    rows = load_replays(args.replays)
    data = build_dataset(
        rows, seats=args.seats, min_margin=args.min_margin, beta=args.beta
    )
    print(f"[awr] {len(rows)} replays -> {data.get('n', 0)} weighted decisions")
    train_awr(data, args.resume, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()

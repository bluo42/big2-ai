"""A trained 52-card posterior over every opponent's hand.

The PPO net already carries a belief head, but only as an auxiliary
gradient — nothing ever checked whether its numbers were *true*.  This
module makes the posterior a first-class model: trained on real played
hands, scored against reality, and good enough to hand determinizations
to the endgame solver.

**What it predicts.**  For each of the three opponents, a probability
for each of the 52 cards.  Two facts are structural, not learned, and
are enforced rather than hoped for:

* cards we hold or have watched being played are impossible (masked to
  zero), and
* each opponent's probabilities must sum to the number of cards they
  are known to be holding — a constraint the plain BCE head ignored,
  and the single biggest source of its miscalibration.

**Hide some, show some.**  During training a random subset of the
opponents' real cards is *revealed* to the model as input, and it must
predict the rest.  Revealing nothing is ordinary play; revealing a lot
is the endgame, where most of the deal is already accounted for.
Training across that whole range teaches conditional structure ("given
they hold the K, what else?") instead of a fixed marginal, and matches
how the model is actually used — the solver asks for hands *given*
everything already known.

**How it is judged.**  Against the analytic baseline every previous
agent used: each unseen card equally likely, scaled by hand size.  A
learned posterior only earns its place by beating that on held-out
games — reported as Brier score, log loss, calibration error, and
top-k precision (of the 13 cards it is most sure about, how many are
really there).

    python -m big2.belief --games 4000 --epochs 6
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.cards import NUM_CARDS
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES

BELIEF_SLOTS = 3 * NUM_CARDS
DEFAULT_PATH = "big2/policies/belief_net.pt"


# ----------------------------------------------------------------------
# Sample assembly
# ----------------------------------------------------------------------


def opponent_order(game: Big2Game, player: int) -> List[int]:
    """Opponents in seat order after ``player`` — the row order used by
    every belief tensor here."""
    return [p for p in range(game.num_players) if p != player][:3]


def truth_matrix(game: Big2Game, player: int) -> np.ndarray:
    """(3, 52) ground truth: does opponent j hold card c?"""
    t = np.zeros((3, NUM_CARDS), dtype=np.float32)
    for j, p in enumerate(opponent_order(game, player)):
        for c in game.hands[p]:
            t[j, c] = 1.0
    return t


def candidate_mask(game: Big2Game, player: int) -> np.ndarray:
    """(3, 52) which (opponent, card) pairs are still possible at all."""
    m = np.ones((3, NUM_CARDS), dtype=np.float32)
    impossible = set(game.played_cards) | set(game.hands[player])
    for c in impossible:
        m[:, c] = 0.0
    for j, p in enumerate(opponent_order(game, player)):
        if not game.hands[p]:
            m[j, :] = 0.0
    if len(opponent_order(game, player)) < 3:
        for j in range(len(opponent_order(game, player)), 3):
            m[j, :] = 0.0
    return m


def hand_sizes(game: Big2Game, player: int) -> np.ndarray:
    sizes = np.zeros(3, dtype=np.float32)
    for j, p in enumerate(opponent_order(game, player)):
        sizes[j] = len(game.hands[p])
    return sizes


def reveal_split(
    truth: np.ndarray, frac: float, rng: random.Random
) -> np.ndarray:
    """Randomly reveal ``frac`` of the real cards as known input.

    Returns a (3, 52) matrix of revealed cards; the model predicts what
    is left.
    """
    revealed = np.zeros_like(truth)
    held = np.argwhere(truth > 0)
    if len(held) == 0 or frac <= 0:
        return revealed
    k = int(round(frac * len(held)))
    if k <= 0:
        return revealed
    for idx in rng.sample(range(len(held)), min(k, len(held))):
        j, c = held[idx]
        revealed[j, c] = 1.0
    return revealed


def analytic_posterior(
    mask: np.ndarray, sizes: np.ndarray, revealed: np.ndarray
) -> np.ndarray:
    """The baseline every earlier agent used: unseen cards equally
    likely, scaled to each opponent's remaining hand size."""
    out = np.zeros_like(mask)
    unknown = mask * (1.0 - revealed)
    # a card revealed in one hand is impossible in the others
    taken = revealed.sum(axis=0, keepdims=True) > 0
    unknown = unknown * (~taken)
    pool = float(unknown.max(axis=0).sum())
    for j in range(mask.shape[0]):
        need = sizes[j] - revealed[j].sum()
        if need <= 0 or pool <= 0:
            out[j] = revealed[j]
            continue
        out[j] = revealed[j] + unknown[j] * (need / pool)
    return np.clip(out, 0.0, 1.0)


def collect_samples(
    n_games: int,
    policies: Optional[Sequence] = None,
    seed: int = 0,
    reveal_range: Tuple[float, float] = (0.0, 0.6),
    stride: int = 3,
) -> Dict[str, np.ndarray]:
    """Play games and snapshot decision points with ground truth."""
    from big2.neural import encode_decision
    from big2.strategies import FiveCardDumper, SmartHeuristic

    if policies is None:
        policies = [SmartHeuristic(), FiveCardDumper(), SmartHeuristic(),
                    FiveCardDumper()]
    rng = random.Random(seed)
    states, revs, masks, sizes, truths, priors = [], [], [], [], [], []
    for g in range(n_games):
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4, rng=random.Random(rng.randrange(2**31)))
        step = 0
        while not game.game_over:
            p = game.turn
            if step % stride == 0:
                _, state, _ = encode_decision(game, p)
                truth = truth_matrix(game, p)
                frac = rng.uniform(*reveal_range)
                rev = reveal_split(truth, frac, rng)
                m, z = candidate_mask(game, p), hand_sizes(game, p)
                states.append(state)
                revs.append(rev)
                masks.append(m)
                sizes.append(z)
                priors.append(analytic_posterior(m, z, rev))
                truths.append(truth)
            game.step(policies[p].select(game, p))
            step += 1
    return {
        "state": np.stack(states),
        "revealed": np.stack(revs),
        "mask": np.stack(masks),
        "sizes": np.stack(sizes),
        "prior": np.stack(priors),
        "truth": np.stack(truths),
    }


def samples_from_replays(
    rows: Sequence[Dict],
    seed: int = 0,
    reveal_range: Tuple[float, float] = (0.0, 0.6),
    stride: int = 2,
) -> Dict[str, np.ndarray]:
    """The same snapshots, taken from recorded (human) games."""
    from big2.neural import encode_decision
    from big2.offline import _replay_body, iter_decisions

    rng = random.Random(seed)
    states, revs, masks, sizes, truths, priors = [], [], [], [], [], []
    for row in rows:
        body = _replay_body(row)
        if body is None:
            continue
        for step, (game, p, _cards) in enumerate(iter_decisions(body)):
            if step % stride:
                continue
            _, state, _ = encode_decision(game, p)
            truth = truth_matrix(game, p)
            rev = reveal_split(truth, rng.uniform(*reveal_range), rng)
            m, z = candidate_mask(game, p), hand_sizes(game, p)
            states.append(state)
            revs.append(rev)
            masks.append(m)
            sizes.append(z)
            priors.append(analytic_posterior(m, z, rev))
            truths.append(truth)
    if not states:
        return {"n": 0}
    return {
        "state": np.stack(states),
        "revealed": np.stack(revs),
        "mask": np.stack(masks),
        "sizes": np.stack(sizes),
        "prior": np.stack(priors),
        "truth": np.stack(truths),
    }


# ----------------------------------------------------------------------
# Scoring: is the posterior actually true?
# ----------------------------------------------------------------------


def calibration_report(
    pred: np.ndarray, truth: np.ndarray, mask: np.ndarray, bins: int = 10
) -> Dict[str, float]:
    """Brier, log loss, expected calibration error, top-k precision."""
    sel = mask > 0
    p = np.clip(pred[sel], 1e-6, 1 - 1e-6)
    y = truth[sel]
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    # top-k precision per sample row: of the k cards we are most sure
    # about, how many does that opponent really hold?
    hits = total = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            k = int(truth[i, j].sum())
            if k == 0:
                continue
            row = np.where(mask[i, j] > 0, pred[i, j], -1.0)
            top = np.argsort(-row)[:k]
            hits += int(truth[i, j][top].sum())
            total += k
    return {
        "brier": brier,
        "logloss": logloss,
        "ece": float(ece),
        "topk_precision": (hits / total) if total else 0.0,
    }


def build_net(state_dim: int, hidden: int = 256):
    import torch.nn as nn

    class BeliefNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_dim = state_dim
            self.trunk = nn.Sequential(
                nn.Linear(state_dim + BELIEF_SLOTS + 3, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head = nn.Linear(hidden, BELIEF_SLOTS)
            # Zero-init the head: the model *starts* as the analytic
            # baseline and can only earn its way above it.  Learning a
            # correction beats learning the whole posterior from
            # scratch, which starts far worse than uniform-by-count.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, state, revealed, sizes):
            x = self.trunk(
                __import__("torch").cat(
                    [state, revealed.flatten(1), sizes], dim=1
                )
            )
            return self.head(x).view(-1, 3, NUM_CARDS)

    return BeliefNet()


def _prior_logit(prior):
    import torch

    return torch.log(prior.clamp(1e-4, 1 - 1e-4)) - torch.log(
        (1 - prior).clamp(1e-4, 1 - 1e-4)
    )


def predict(net, state, revealed, mask, sizes, prior):
    """Masked, count-normalised probabilities (prior + learned delta)."""
    import torch

    with torch.no_grad():
        delta = net(state, revealed, sizes)
    p = torch.sigmoid(_prior_logit(prior) + delta) * mask
    # a revealed card is certain; the rest share what is left of the hand
    known = revealed * mask
    free = mask * (1.0 - revealed)
    need = (sizes - known.sum(-1)).clamp(min=0.0)
    tot = (p * free).sum(-1, keepdim=True).clamp(min=1e-6)
    return (known + free * p * (need.unsqueeze(-1) / tot)).clamp(0.0, 1.0)


def train(
    data: Dict[str, np.ndarray],
    epochs: int = 6,
    lr: float = 1e-3,
    minibatch: int = 256,
    hidden: int = 256,
    seed: int = 0,
    count_coef: float = 0.5,
    verbose: bool = True,
    net=None,
):
    """Fit the posterior: BCE on membership + a hand-size consistency term."""
    import torch

    torch.manual_seed(seed)
    S = torch.from_numpy(data["state"])
    R = torch.from_numpy(data["revealed"])
    M = torch.from_numpy(data["mask"])
    Z = torch.from_numpy(data["sizes"])
    Y = torch.from_numpy(data["truth"])
    P0 = _prior_logit(torch.from_numpy(data["prior"]))
    if net is None:
        net = build_net(S.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(S)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            logits = P0[idx] + net(S[idx], R[idx], Z[idx])
            m = M[idx]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, Y[idx], weight=m, reduction="sum"
            ) / m.sum().clamp(min=1.0)
            probs = torch.sigmoid(logits) * m
            counts = probs.sum(-1)
            count_loss = ((counts - Z[idx]) ** 2).mean() / 13.0
            loss = bce + count_coef * count_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if verbose:
            print(f"[belief] epoch {ep + 1}/{epochs} loss {tot / n:.4f}",
                  flush=True)
    return net


def evaluate(net, data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Learned posterior vs the analytic baseline, on the same states."""
    import torch

    S = torch.from_numpy(data["state"])
    R = torch.from_numpy(data["revealed"])
    M = torch.from_numpy(data["mask"])
    Z = torch.from_numpy(data["sizes"])
    P0 = torch.from_numpy(data["prior"])
    pred = predict(net, S, R, M, Z, P0).numpy()
    base = data["prior"]
    return {
        "learned": calibration_report(pred, data["truth"], data["mask"]),
        "analytic": calibration_report(base, data["truth"], data["mask"]),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--eval-games", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--replays", default=None,
                        help="JSONL of recorded games to add to training")
    parser.add_argument("--out", default=DEFAULT_PATH)
    args = parser.parse_args()

    print(f"[belief] collecting {args.games} self-play games...", flush=True)
    train_data = collect_samples(args.games, seed=1)
    if args.replays:
        from big2.offline import load_replays

        extra = samples_from_replays(load_replays(args.replays), seed=2)
        if extra.get("state") is not None:
            train_data = {
                k: np.concatenate([train_data[k], extra[k]])
                for k in train_data
            }
            print(f"[belief] + {len(extra['state'])} recorded-game samples",
                  flush=True)
    print(f"[belief] {len(train_data['state'])} training states", flush=True)
    net = train(train_data, epochs=args.epochs, hidden=args.hidden)

    held = collect_samples(args.eval_games, seed=999)
    report = evaluate(net, held)
    print("\n=== held-out calibration (lower is better except top-k) ===")
    for name, r in report.items():
        print(f"  {name:9s} brier {r['brier']:.4f}  logloss {r['logloss']:.4f}"
              f"  ece {r['ece']:.4f}  top-k {r['topk_precision']:.1%}")
    torch.save({"state_dict": net.state_dict(),
                "state_dim": net.state_dim,
                "hidden": args.hidden,
                "report": report}, args.out)
    print(f"[belief] saved -> {args.out}")


if __name__ == "__main__":
    main()

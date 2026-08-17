"""PPO with a set-attention head — research-ladder step 5.

The action space is a variable *set* of candidate moves, so the policy
is a pointer/set head: encode the state, encode every legal action,
run one self-attention block over the action set conditioned on the
state, and softmax over the per-action scores.  Alongside the policy
and value heads, a **belief head** predicts each opponent's actual hand
(a 3x52 membership map) as an auxiliary loss — the trainer sees the
full deal (perfect-information supervision), the policy at play time
does not; the auxiliary gradient shapes the trunk (Suphx-style).

Action features deliberately subsume the whole ladder:

- encoding v4 (DouZero action encoding + analytic beliefs + opponent
  style modeling + decomposition/payment features),
- the **CEM linear move-scorer's feature vector** (rl.move_features),
- the CEM champion's own scalar advice w·phi for the move — the best
  hand-crafted evaluator we have, offered to the net as one input among
  many (distillation without obligation).

Training games are 4-player under the house rules, with opponents
sampled per game: pure self-play, or the PPO seat against the current
champions (CEM linear, evolved MLP, DMC) and scripted regulars — so
the net trains explicitly to exploit the field it must beat, while the
opponent-style features let it adapt within a match.

    python -m big2.neural --iters 400            # train
    python -m big2.neural --resume ppo_attn.pt   # continue
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.cards import NUM_CARDS
from big2.combos import Combo
from big2.features import FEAT_DIM, DecisionContext, encode_sa
from big2.game import Big2Game, ScoringConfig
from big2.rl import NUM_FEATURES as CEM_DIM
from big2.rl import LinearPolicy, move_features as cem_move_features
from big2.rules import DEFAULT_RULES
from big2.strategies import FiveCardDumper, SmartHeuristic, Strategy

SCORE_SCALE = 39.0
ACT_DIM = FEAT_DIM + CEM_DIM + 1  # v4 features + CEM features + champion advice
BELIEF_SLOTS = 3 * NUM_CARDS  # per-opponent hand membership, seat order after me

DEFAULT_MODEL_PATH = "big2/policies/ppo_attn.pt"


# ----------------------------------------------------------------------
# Feature assembly
# ----------------------------------------------------------------------


class _ChampionAdvice:
    """Lazy singleton: the trained CEM linear champion as a feature."""

    _weights: Optional[np.ndarray] = None

    @classmethod
    def weights(cls) -> np.ndarray:
        if cls._weights is None:
            try:
                cls._weights = LinearPolicy.load(
                    "big2/policies/linear_cem.npz"
                ).weights.astype(np.float64)
            except Exception:
                cls._weights = np.zeros(CEM_DIM)
        return cls._weights


def encode_decision(
    game: Big2Game, player: int
) -> Tuple[List[Optional[Combo]], np.ndarray, np.ndarray]:
    """(options, state_vec, action_matrix) for one decision."""
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    ctx = DecisionContext(game, player)
    state = encode_sa(game, player, None, ctx)  # pass-encoding == state view
    units_keys = {
        frozenset(u.cards)
        for u in SmartHeuristic._partition(game.hands[player])
    }
    champ_w = _ChampionAdvice.weights()
    rows = []
    for m in options:
        v4 = encode_sa(game, player, m, ctx)
        cem = cem_move_features(game, player, m, units_keys)
        advice = float(cem @ champ_w) / 5.0
        rows.append(
            np.concatenate([v4, cem.astype(np.float32), [advice]]).astype(
                np.float32
            )
        )
    return options, state, np.stack(rows)


def belief_target(game: Big2Game, player: int) -> np.ndarray:
    """Ground-truth opponent hands (training-time perfect information)."""
    t = np.zeros(BELIEF_SLOTS, dtype=np.float32)
    others = [p for p in range(game.num_players) if p != player][:3]
    for j, p in enumerate(others):
        for c in game.hands[p]:
            t[j * NUM_CARDS + c] = 1.0
    return t


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


def build_net(d_model: int = 192, heads: int = 4):
    import torch
    import torch.nn as nn

    class Big2Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d_model
            self.state_mlp = nn.Sequential(
                nn.Linear(FEAT_DIM, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
            )
            self.act_mlp = nn.Sequential(
                nn.Linear(ACT_DIM, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.policy_head = nn.Linear(d_model, 1)
            self.value_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1),
            )
            self.belief_head = nn.Linear(d_model, BELIEF_SLOTS)

        def forward(self, state, acts, mask):
            """state (B,F) acts (B,A,D) mask (B,A) True=valid."""
            s = self.state_mlp(state)                     # (B,d)
            h = self.act_mlp(acts) + s.unsqueeze(1)       # (B,A,d)
            attn_out, _ = self.attn(
                h, h, h, key_padding_mask=~mask, need_weights=False
            )
            h = self.norm(h + attn_out)
            logits = self.policy_head(h).squeeze(-1)      # (B,A)
            logits = logits.masked_fill(~mask, -1e9)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (
                mask.sum(1, keepdim=True).clamp(min=1)
            )
            value = self.value_head(
                torch.cat([s, pooled], dim=-1)
            ).squeeze(-1)                                  # (B,)
            belief = self.belief_head(s)                   # (B,156)
            return logits, value, belief

    return Big2Net()


class PPOPolicy(Strategy):
    """Greedy inference wrapper for a trained set-attention net."""

    name = "ppo"

    def __init__(self, net, d_model: int = 192):
        self.net = net
        self.net.eval()

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        import torch

        options, state, acts = encode_decision(game, player)
        if len(options) == 1:
            return options[0]
        with torch.no_grad():
            logits, _, _ = self.net(
                torch.from_numpy(state).unsqueeze(0),
                torch.from_numpy(acts).unsqueeze(0),
                torch.ones(1, len(options), dtype=torch.bool),
            )
        return options[int(logits[0].argmax())]

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "PPOPolicy":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        net = build_net(payload.get("d_model", 192), payload.get("heads", 4))
        net.load_state_dict(payload["state_dict"])
        return cls(net)


# ----------------------------------------------------------------------
# Rollouts (worker processes)
# ----------------------------------------------------------------------


def _opponent_pool() -> List[Strategy]:
    """Deliberately lean: the strongest scripted agent (dumper, per the
    current baseline table) plus the CEM linear champion.  Weak opponents
    dilute the best-response gradient; past-self snapshots and self-play
    supply the diversity instead."""
    pool: List[Strategy] = [FiveCardDumper()]
    try:
        pool.append(LinearPolicy.load("big2/policies/linear_cem.npz"))
    except Exception:
        pool.append(SmartHeuristic())
    return pool


# Worker-side cache of frozen past-self policies, keyed by (path, mtime)
# so a refreshed snapshot file is reloaded exactly once per worker.
_SNAP_CACHE: Dict[Tuple[str, float], "PPOPolicy"] = {}


def _load_snapshot_policy(path: str) -> Optional["PPOPolicy"]:
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _SNAP_CACHE:
        if len(_SNAP_CACHE) > 8:
            _SNAP_CACHE.clear()
        try:
            _SNAP_CACHE[key] = PPOPolicy.load(path)
        except Exception:
            return None
    return _SNAP_CACHE[key]


def rollout_games(args) -> bytes:
    """Worker: play games with the given weights, return episode batch."""
    state_bytes, n_games, seed, selfplay_prob, snapshot_paths, past_self_prob = args
    import torch

    torch.set_num_threads(1)
    payload = pickle.loads(state_bytes)
    net = build_net(payload["d_model"], payload["heads"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    rng = random.Random(seed)
    pool = _opponent_pool()
    past_selves = [
        p for p in (_load_snapshot_policy(pth) for pth in snapshot_paths)
        if p is not None
    ]
    episodes = []

    for _ in range(n_games):
        selfplay = rng.random() < selfplay_prob
        if selfplay:
            ppo_seats = set(range(4))
        else:
            ppo_seats = {rng.randrange(4)}
        opponents = {}
        for p in range(4):
            if p in ppo_seats:
                continue
            # League-style: previous generations sit at the table too.
            if past_selves and rng.random() < past_self_prob:
                opponents[p] = rng.choice(past_selves)
            else:
                opponents[p] = rng.choice(pool)
        game = Big2Game(
            scoring=ScoringConfig(), rules=DEFAULT_RULES, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        trajs: Dict[int, Dict[str, list]] = {
            p: {"state": [], "acts": [], "chosen": [], "logp": [],
                "value": [], "belief": []}
            for p in ppo_seats
        }
        while not game.game_over:
            p = game.turn
            if p not in ppo_seats:
                game.step(opponents[p].select(game, p))
                continue
            options, state, acts = encode_decision(game, p)
            if len(options) == 1:
                game.step(options[0])
                continue
            with torch.no_grad():
                logits, value, _ = net(
                    torch.from_numpy(state).unsqueeze(0),
                    torch.from_numpy(acts).unsqueeze(0),
                    torch.ones(1, len(options), dtype=torch.bool),
                )
                dist = torch.distributions.Categorical(logits=logits[0])
                idx = int(dist.sample())
                logp = float(dist.log_prob(torch.tensor(idx)))
            t = trajs[p]
            t["state"].append(state)
            t["acts"].append(acts)
            t["chosen"].append(idx)
            t["logp"].append(logp)
            t["value"].append(float(value[0]))
            t["belief"].append(belief_target(game, p))
            game.step(options[idx])

        for p, t in trajs.items():
            if t["state"]:
                episodes.append(
                    {**{k: v for k, v in t.items()},
                     "score": game.scores[p] / SCORE_SCALE}
                )
    return pickle.dumps(episodes)


# ----------------------------------------------------------------------
# PPO learner
# ----------------------------------------------------------------------


def _gae(values: List[float], final_return: float, lam: float = 0.95):
    """Terminal-reward GAE with gamma=1."""
    T = len(values)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = final_return if t == T - 1 else values[t + 1]
        # reward is 0 everywhere; the terminal value IS the game score
        delta = next_v - values[t]
        gae = delta + lam * gae
        adv[t] = gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def train_ppo(
    iters: int = 400,
    games_per_iter: int = 64,
    workers: int = 4,
    selfplay_prob: float = 0.5,
    lr: float = 3e-4,
    clip: float = 0.2,
    epochs: int = 3,
    minibatch: int = 512,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    belief_coef: float = 0.5,
    d_model: int = 192,
    heads: int = 4,
    seed: int = 0,
    out: str = DEFAULT_MODEL_PATH,
    resume: Optional[str] = None,
    probe_every_iters: int = 30,
    probe_games: int = 120,
    confirm_games: int = 480,  # independent re-test before a new "best"
    progress_path: str = "big2/policies/evolve/progress.csv",
    games_offset: int = 0,  # games trained in prior runs (resume bookkeeping)
    snapshot_every_iters: int = 100,  # freeze a past-self opponent this often
    max_snapshots: int = 3,
    past_self_prob: float = 0.5,  # opponent-seat draw: past selves vs field
    verbose: bool = True,
):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = build_net(d_model, heads)
    best_probe = -1e9
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        net.load_state_dict(payload["state_dict"])
        # Don't let a resumed run overwrite a better checkpoint with its
        # first mediocre probe: inherit the saved best.
        best_probe = float(payload.get("meta", {}).get("probe", -1e9))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    pool = mp.Pool(workers)
    total_games = games_offset
    t0 = time.time()

    from big2.decomposition import DecompositionStrategy
    from big2.evolve import _probe

    probe_baselines = [SmartHeuristic(), DecompositionStrategy(), FiveCardDumper()]
    champs: List[Strategy] = []
    for loader in (
        lambda: LinearPolicy.load("big2/policies/linear_cem.npz"),
        lambda: __import__("big2.nn", fromlist=["NNPolicy"]).NNPolicy.load(
            "big2/policies/evo_mlp.npz"
        ),
        lambda: __import__("big2.dmc", fromlist=["DMCPolicy"]).DMCPolicy.load(
            "big2/policies/dmc_linear.npz"
        ),
    ):
        try:
            champs.append(loader())
        except Exception:
            pass

    snap_dir = os.path.join(os.path.dirname(out) or ".", "ppo_snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    def _atomic_save(payload, path):
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _snapshot_paths() -> List[str]:
        paths = sorted(
            (os.path.join(snap_dir, f) for f in os.listdir(snap_dir)
             if f.endswith(".pt")),
            key=os.path.getmtime,
        )
        if os.path.exists(out):
            paths.append(out)  # the best-so-far checkpoint plays too
        return paths

    for it in range(1, iters + 1):
        blob = pickle.dumps(
            {"state_dict": net.state_dict(), "d_model": d_model, "heads": heads}
        )
        per = games_per_iter // workers
        snaps = _snapshot_paths()
        results = pool.map(
            rollout_games,
            [(blob, per, rng.randrange(2**31), selfplay_prob, snaps,
              past_self_prob)
             for _ in range(workers)],
        )
        episodes = [e for r in results for e in pickle.loads(r)]
        total_games += per * workers

        flat_state, flat_acts, flat_mask = [], [], []
        flat_chosen, flat_logp, flat_adv, flat_ret, flat_belief = [], [], [], [], []
        max_a = max(len(a) for e in episodes for a in e["acts"])
        for e in episodes:
            adv, ret = _gae(e["value"], e["score"])
            for i, acts in enumerate(e["acts"]):
                A = len(acts)
                padded = np.zeros((max_a, ACT_DIM), dtype=np.float32)
                padded[:A] = acts
                flat_state.append(e["state"][i])
                flat_acts.append(padded)
                m = np.zeros(max_a, dtype=bool)
                m[:A] = True
                flat_mask.append(m)
                flat_chosen.append(e["chosen"][i])
                flat_logp.append(e["logp"][i])
                flat_adv.append(adv[i])
                flat_ret.append(ret[i])
                flat_belief.append(e["belief"][i])

        S = torch.from_numpy(np.stack(flat_state))
        A_ = torch.from_numpy(np.stack(flat_acts))
        M = torch.from_numpy(np.stack(flat_mask))
        C = torch.tensor(flat_chosen, dtype=torch.long)
        LP = torch.tensor(flat_logp, dtype=torch.float32)
        ADV = torch.tensor(np.asarray(flat_adv))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        RET = torch.tensor(np.asarray(flat_ret))
        B = torch.from_numpy(np.stack(flat_belief))
        n = len(S)

        pol_loss = val_loss = ent = bel_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = perm[start : start + minibatch]
                logits, value, belief = net(S[idx], A_[idx], M[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(C[idx])
                ratio = torch.exp(logp - LP[idx])
                surr = torch.min(
                    ratio * ADV[idx],
                    torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx],
                )
                p_loss = -surr.mean()
                v_loss = ((value - RET[idx]) ** 2).mean()
                e_loss = dist.entropy().mean()
                b_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    belief, B[idx]
                )
                loss = (
                    p_loss + vf_coef * v_loss - ent_coef * e_loss
                    + belief_coef * b_loss
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                pol_loss = float(p_loss.detach())
                val_loss = float(v_loss.detach())
                ent = float(e_loss.detach())
                bel_loss = float(b_loss.detach())

        if verbose and it % 5 == 0:
            mean_score = float(np.mean([e["score"] for e in episodes])) * SCORE_SCALE
            rate = (total_games - games_offset) / (time.time() - t0)
            print(
                f"iter {it}/{iters} games {total_games} ({rate:.0f} g/s) "
                f"score/ep {mean_score:+.2f} pi {pol_loss:.3f} v {val_loss:.3f} "
                f"H {ent:.2f} belief {bel_loss:.3f}",
                flush=True,
            )

        if snapshot_every_iters and it % snapshot_every_iters == 0:
            slot = os.path.join(
                snap_dir,
                f"snap_{(it // snapshot_every_iters - 1) % max_snapshots}.pt",
            )
            _atomic_save(
                {"state_dict": net.state_dict(), "d_model": d_model,
                 "heads": heads},
                slot,
            )
            if verbose:
                print(f"[ppo] froze past-self snapshot -> {slot}", flush=True)

        if probe_every_iters and it % probe_every_iters == 0:
            policy = PPOPolicy(net)
            vs_base = _probe(
                policy, probe_baselines, probe_games, ScoringConfig(),
                DEFAULT_RULES, seed=it,
            )
            vs_champ = (
                _probe(policy, champs[:3], probe_games, ScoringConfig(),
                       DEFAULT_RULES, seed=it + 1)
                if len(champs) == 3
                else float("nan")
            )
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "a") as f:
                f.write(
                    f"5,{total_games},4,ppo,0,{lr:.5f},"
                    f"{vs_base:.3f},{vs_champ:.3f}\n"
                )
            print(
                f"[ppo] probe @{total_games}: vs-baselines {vs_base:+.2f}  "
                f"vs-champions {vs_champ:+.2f}",
                flush=True,
            )
            net.train()
            score = vs_champ if vs_champ == vs_champ else vs_base
            if score > best_probe:
                # Two-stage gate: a 120-game probe is +-1 noisy, so a
                # would-be best must confirm on an independent larger
                # re-test; the confirmed number is what gets recorded.
                opponents = champs[:3] if len(champs) == 3 else probe_baselines
                confirmed = _probe(
                    policy, opponents, confirm_games, ScoringConfig(),
                    DEFAULT_RULES, seed=it + 7919,
                )
                net.train()
                print(
                    f"[ppo] candidate {score:+.2f} -> confirmation "
                    f"({confirm_games} games): {confirmed:+.2f}",
                    flush=True,
                )
                if confirmed > best_probe:
                    best_probe = confirmed
                    _atomic_save(
                        {"state_dict": net.state_dict(), "d_model": d_model,
                         "heads": heads,
                         "meta": {"iter": it, "games": total_games,
                                  "probe": confirmed,
                                  "confirm_games": confirm_games}},
                        out,
                    )
                    print(
                        f"[ppo] saved best (confirmed {confirmed:+.2f}) -> {out}",
                        flush=True,
                    )

    pool.close()
    pool.join()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--games-per-iter", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selfplay-prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--probe-every-iters", type=int, default=30)
    parser.add_argument("--games-offset", type=int, default=0)
    parser.add_argument("--snapshot-every-iters", type=int, default=100)
    parser.add_argument("--past-self-prob", type=float, default=0.5)
    parser.add_argument("--confirm-games", type=int, default=480)
    args = parser.parse_args()
    train_ppo(
        iters=args.iters, games_per_iter=args.games_per_iter,
        workers=args.workers, selfplay_prob=args.selfplay_prob, lr=args.lr,
        d_model=args.d_model, seed=args.seed, out=args.out,
        resume=args.resume, probe_every_iters=args.probe_every_iters,
        games_offset=args.games_offset,
        snapshot_every_iters=args.snapshot_every_iters,
        past_self_prob=args.past_self_prob,
        confirm_games=args.confirm_games,
    )


if __name__ == "__main__":
    main()

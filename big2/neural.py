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
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.cards import NUM_CARDS
from big2.combos import Combo
from big2.features import (
    BEAT_DIM, FEAT_DIM, DecisionContext, beat_features, encode_sa,
)
from big2.game import Big2Game, ScoringConfig
from big2.inference import OVERPLAY_DIM, overplay_features
from big2.planning import (
    PLAN_DIM,
    PLAN_STATE_DIM,
    PlanContext,
    plan_features,
    plan_state_features,
)
from big2.profiles import PROFILE_DIM, OpponentProfileBook
from big2.rl import NUM_FEATURES as CEM_DIM
from big2.rl import LinearPolicy, move_features as cem_move_features
from big2.rules import DEFAULT_RULES
from big2.strategies import FiveCardDumper, SmartHeuristic, Strategy

SCORE_SCALE = 39.0
# Total cards left below which the learner's rollout seats consult the
# IS-MCTS agent (with the exact solver inside it taking over <= 14).
# 4p searches from 26 down — the band that trained stably for hours.
# Deeper bands were tried 2026-08-18 (40, then 35): the deviation
# analysis put the human edge early in the hand, but searching there
# doubled the wall clock and the first probes after the change went
# sharply negative (raw-policy drift under a search-dominated executed
# distribution is the suspected mechanism).  2p/3p are retired from
# training entirely.
SEARCH_CARDS_4P = 26
SEARCH_CARDS_23P = 18
# v1.1: sharp endgame-danger block appended to every action encoding.
# Counts existed as soft /13 scalars before; these are the thresholded
# versions plus the interactions that matter ("the next actor is nearly
# out AND this move is a cheap gift").
DANGER_DIM = 14
ACT_DIM_V1 = FEAT_DIM + CEM_DIM + 1  # v1.0 nets: v4 + CEM + champion advice
ACT_DIM_V11 = ACT_DIM_V1 + DANGER_DIM
# v2: planning block per move (is it answerable? what happens after?)
# and, on the state side, the hand's run-out summary plus what each
# opponent's own play has revealed about how they answer.
ACT_DIM = ACT_DIM_V11 + PLAN_DIM
BELIEF_SLOTS = 3 * NUM_CARDS  # per-opponent hand membership, seat order after me
PROFILE_SLOTS = 3 * PROFILE_DIM  # cross-game opponent profiles, same order
OVERPLAY_SLOTS = 3 * OVERPLAY_DIM
STATE_DIM_V11 = FEAT_DIM + PROFILE_SLOTS
STATE_DIM = STATE_DIM_V11 + PLAN_STATE_DIM + OVERPLAY_SLOTS
# Beat-risk block appended after the plan block (see big2/features.py).
ACT_DIM_BEAT = ACT_DIM + BEAT_DIM


def danger_features(game: Big2Game, player: int,
                    move: Optional[Combo]) -> np.ndarray:
    """Per-move endgame awareness (v1.1).

    Layout: next-actor cards [==1, ==2, <=3, /13], field minimum
    [==1, ==2, <=3], relative seat holding the minimum (3), cheapness of
    this move as a single, cheap-single x next-actor-nearly-out,
    someone/next-actor could go out on exactly this class size.
    """
    f = np.zeros(DANGER_DIM, dtype=np.float32)
    n = game.num_players
    nxt = (player + 1) % n
    cn = len(game.hands[nxt])
    others = [p for p in range(n) if p != player]
    counts = [len(game.hands[p]) for p in others]
    mn = min(counts)
    f[0] = 1.0 if cn == 1 else 0.0
    f[1] = 1.0 if cn == 2 else 0.0
    f[2] = 1.0 if cn <= 3 else 0.0
    f[3] = cn / 13.0
    f[4] = 1.0 if mn == 1 else 0.0
    f[5] = 1.0 if mn == 2 else 0.0
    f[6] = 1.0 if mn <= 3 else 0.0
    f[7 + min(counts.index(mn), 2)] = 1.0  # 7,8,9: who is nearly out
    if move is not None and len(move) == 1:
        f[10] = 1.0 - max(move.cards) / 51.0  # cheap single = easy to beat
        f[11] = f[10] if cn <= 2 else 0.0     # ...gifted to a near-winner
    size = len(move) if move is not None else (
        len(game.table_combo) if game.table_combo else 0
    )
    if size:
        f[12] = 1.0 if any(c == size for c in counts) else 0.0
        f[13] = 1.0 if cn == size else 0.0
    return f

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
    game: Big2Game,
    player: int,
    book: Optional[OpponentProfileBook] = None,
    seat_keys: Optional[Dict[int, object]] = None,
    include_profiles: bool = True,
    include_danger: bool = True,
    include_plan: bool = True,
    include_beat: bool = False,
) -> Tuple[List[Optional[Combo]], np.ndarray, np.ndarray]:
    """(options, state_vec, action_matrix) for one decision.

    With ``include_profiles`` the state carries cross-game opponent
    profiles (zeros when no ``book``/``seat_keys`` are supplied — e.g.
    at deploy time or in probes).  ``include_danger`` appends the v1.1
    endgame-danger block to every action row; v1.0 nets have a narrower
    action input and set it False.  ``include_plan`` adds the v2
    planning block (per move) and the run-out/opponent-read summary (on
    the state), which older nets likewise skip."""
    options: List[Optional[Combo]] = list(game.legal_moves(player))
    if game.can_pass():
        options.append(None)
    ctx = DecisionContext(game, player)
    state = encode_sa(game, player, None, ctx)  # pass-encoding == state view
    if include_profiles:
        prof = np.zeros(PROFILE_SLOTS, dtype=np.float32)
        if book is not None and seat_keys:
            others = [p for p in range(game.num_players) if p != player][:3]
            for j, p in enumerate(others):
                key = seat_keys.get(p)
                if key is not None:
                    prof[j * PROFILE_DIM : (j + 1) * PROFILE_DIM] = (
                        book.features(key)
                    )
        state = np.concatenate([state, prof])
    pctx: Optional[PlanContext] = None
    if include_plan:
        pctx = PlanContext(game, player)
        reads = overplay_features(game, player)
        others = [p for p in range(game.num_players) if p != player][:3]
        opp = np.zeros(OVERPLAY_SLOTS, dtype=np.float32)
        for j, p in enumerate(others):
            opp[j * OVERPLAY_DIM : (j + 1) * OVERPLAY_DIM] = reads[p]
        state = np.concatenate([state, plan_state_features(pctx), opp])
    # The plan context already partitioned the hand; reuse it rather than
    # paying for a second identical partition per decision.
    units = (
        pctx.units if pctx is not None
        else SmartHeuristic._partition(game.hands[player])
    )
    units_keys = {frozenset(u.cards) for u in units}
    champ_w = _ChampionAdvice.weights()
    rows = []
    for m in options:
        v4 = encode_sa(game, player, m, ctx)
        cem = cem_move_features(game, player, m, units_keys)
        advice = float(cem @ champ_w) / 5.0
        parts = [v4, cem.astype(np.float32),
                 np.array([advice], dtype=np.float32)]
        if include_danger:
            parts.append(danger_features(game, player, m))
        if pctx is not None:
            parts.append(plan_features(pctx, m))
        if include_beat:
            parts.append(beat_features(ctx, m))
        rows.append(np.concatenate(parts).astype(np.float32))
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


def build_net(d_model: int = 192, heads: int = 4, state_dim: int = STATE_DIM,
              act_dim: int = ACT_DIM, layers: int = 2, attn_blocks: int = 1):
    """``layers`` counts Linear layers in each input MLP (2 = the
    original shape every shipped checkpoint has; 3 adds one d->d block
    to both towers).  Depth and width (``d_model``) are the two axes of
    the capacity experiments.

    ``attn_blocks`` is the move-interaction depth: with 2, a second
    residual attention block lets move-vs-move comparisons compose
    (raise-over-this-then-keep-control patterns).  Its output projection
    is zero-initialized, so a 1-block checkpoint warm-starts into a
    2-block net functionally unchanged and gradients open the new block
    up from there.  Loaders detect the block from the state_dict itself
    (``attn2.*`` keys), never from metadata."""
    import torch
    import torch.nn as nn

    def _tower(in_dim: int, final_relu: bool) -> nn.Sequential:
        mods = [nn.Linear(in_dim, d_model), nn.ReLU()]
        for i in range(max(2, layers) - 2):
            mods += [nn.Linear(d_model, d_model), nn.ReLU()]
        mods.append(nn.Linear(d_model, d_model))
        if final_relu:
            mods.append(nn.ReLU())
        return nn.Sequential(*mods)

    class Big2Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d_model
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.layers = layers
            self.attn_blocks = attn_blocks
            self.state_mlp = _tower(state_dim, final_relu=True)
            self.act_mlp = _tower(act_dim, final_relu=False)
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            if attn_blocks >= 2:
                self.attn2 = nn.MultiheadAttention(
                    d_model, heads, batch_first=True
                )
                self.norm2 = nn.LayerNorm(d_model)
                nn.init.zeros_(self.attn2.out_proj.weight)
                nn.init.zeros_(self.attn2.out_proj.bias)
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
            if self.attn_blocks >= 2:
                # Pre-norm residual: with the zero-initialized output
                # projection this is exactly h at warm-start, so a
                # 1-block checkpoint resumes bit-identical.
                n2h = self.norm2(h)
                a2, _ = self.attn2(
                    n2h, n2h, n2h, key_padding_mask=~mask,
                    need_weights=False,
                )
                h = h + a2
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
    """Greedy inference wrapper for a trained set-attention net.

    Attach ``book`` (an OpponentProfileBook) and ``seat_keys`` to feed
    cross-game opponent profiles; without them, profile inputs are zero
    (nets trained before profiles existed simply have a narrower input
    and skip them entirely)."""

    name = "ppo"

    def __init__(self, net, d_model: int = 192):
        self.net = net
        self.net.eval()
        sdim = getattr(net, "state_dim", STATE_DIM)
        adim = getattr(net, "act_dim", ACT_DIM)
        self.uses_profiles = sdim != FEAT_DIM
        self.uses_danger = adim >= ACT_DIM_V11
        self.uses_plan = adim >= ACT_DIM and sdim >= STATE_DIM
        self.uses_beat = adim >= ACT_DIM_BEAT
        self.book: Optional[OpponentProfileBook] = None
        self.seat_keys: Optional[Dict[int, object]] = None

    def option_scores(self, game: Big2Game, player: int):
        """(options, logits) — the policy's preference over every legal
        option, for callers that blend it with search (big2/search.py)."""
        import torch

        options, state, acts = encode_decision(
            game, player, book=self.book, seat_keys=self.seat_keys,
            include_profiles=self.uses_profiles,
            include_danger=self.uses_danger,
            include_plan=self.uses_plan,
            include_beat=self.uses_beat,
        )
        with torch.no_grad():
            logits, _, _ = self.net(
                torch.from_numpy(state).unsqueeze(0),
                torch.from_numpy(acts).unsqueeze(0),
                torch.ones(1, len(options), dtype=torch.bool),
            )
        return options, logits[0].numpy()

    def select(self, game: Big2Game, player: int) -> Optional[Combo]:
        options = list(game.legal_moves(player))
        if game.can_pass():
            options.append(None)
        if len(options) == 1:
            return options[0]
        options, logits = self.option_scores(game, player)
        return options[int(np.argmax(logits))]

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "PPOPolicy":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        sd = payload["state_dict"]
        in_dim = sd["state_mlp.0.weight"].shape[1]
        a_dim = sd["act_mlp.0.weight"].shape[1]
        # Depth is read off the weights themselves, so a checkpoint never
        # depends on its metadata being complete.
        depth = sum(1 for k in sd
                    if k.startswith("state_mlp.") and k.endswith(".weight"))
        blocks = 2 if any(k.startswith("attn2.") for k in sd) else 1
        net = build_net(
            payload.get("d_model", 192), payload.get("heads", 4),
            state_dim=in_dim, act_dim=a_dim, layers=depth,
            attn_blocks=blocks,
        )
        net.load_state_dict(payload["state_dict"])
        return cls(net)


def widen_state_dict(sd: Dict, new_dim: int = STATE_DIM,
                     new_act_dim: int = ACT_DIM) -> Dict:
    """Grow a checkpoint's input layers with zero columns so an older net
    can warm-start a wider run (new features begin with exactly zero
    influence).  Handles both the state input (opponent profiles) and the
    action input (v1.1 endgame-danger block, appended at the end)."""
    import torch

    sd = dict(sd)
    for key, dim in (("state_mlp.0.weight", new_dim),
                     ("act_mlp.0.weight", new_act_dim)):
        w = sd[key]
        if w.shape[1] >= dim:
            continue
        wide = torch.zeros(w.shape[0], dim)
        wide[:, : w.shape[1]] = w
        sd[key] = wide
    return sd


# ----------------------------------------------------------------------
# Rollouts (worker processes)
# ----------------------------------------------------------------------


def probe_2v2(policy: Strategy, ref: Strategy, n_games: int,
              scoring, rules, seed: int) -> float:
    """Mean score per ``policy`` seat over games seating 2 copies of it
    against 2 copies of ``ref``, cycling the seat pattern."""
    rng = random.Random(seed)
    patterns = ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0])
    total = 0.0
    for g in range(n_games):
        pat = patterns[g % 3]
        seats = [policy if side == 0 else ref for side in pat]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        scores = game.play_out(seats)
        total += sum(scores[i] for i in range(4) if pat[i] == 0) / 2
    return total / n_games


def confirmation_panel(
    reference: Optional[str] = None,
    snapshot_dir: str = "big2/policies/ppo_snapshots",
) -> List[Strategy]:
    """The field a candidate must actually beat.

    Confirming against three copies of one model measures a single
    matchup, not strength: a candidate can clear it by learning that
    opponent's quirks while losing ground to everything else.  The panel
    is the whole field instead — the live reference, the previous
    champion lines, the scripted regulars from the training diet, the
    CEM linear champion, the dedicated exploiter, and the most recent
    frozen past-selves — and confirmation seats three *different*
    members per game.
    """
    panel: List[Strategy] = []
    seen = set()
    for path in (reference, "big2/policies/ppo_attn_v11.pt",
                 "big2/policies/ppo_attn.pt",
                 "big2/policies/ppo_exploiter.pt"):
        if not path or path in seen:
            continue
        seen.add(path)
        p = _load_snapshot_policy(path)
        if p is not None:
            panel.append(p)
    try:
        panel.append(LinearPolicy.load("big2/policies/linear_cem.npz"))
    except Exception:
        pass
    human = _load_snapshot_policy("big2/policies/humanlike.pt")
    if human is not None:
        panel.append(human)  # the strong testers' style, distilled
    panel.append(FiveCardDumper())
    panel.append(SmartHeuristic())
    try:
        snaps = sorted(
            (os.path.join(snapshot_dir, f) for f in os.listdir(snapshot_dir)
             if f.endswith(".pt")),
            key=os.path.getmtime,
        )[-3:]
    except OSError:
        snaps = []
    for path in snaps:
        p = _load_snapshot_policy(path)
        if p is not None:
            panel.append(p)
    return panel


CONFIRM_FILES = ("big2/policies/ppo_attn_v11.pt",
                 "big2/policies/ppo_attn.pt",
                 "big2/policies/humanlike.pt")


def confirm_pool(files: Optional[Sequence[str]] = None) -> List[Strategy]:
    """The stationary confirmation room (default: WangBot_v1, PPO v1,
    humanlike).  Seats are drawn from this with replacement by
    replacement_probe.  Later chains pass their own room — e.g. chain C
    confirms against the A/B finals."""
    return [p for p in (_load_snapshot_policy(f)
                        for f in (files or CONFIRM_FILES))
            if p is not None]


def replacement_probe(policy: Strategy, members: Sequence[Strategy],
                      n_games: int, scoring, rules, seed: int) -> float:
    """Mean score over 4p games whose three opponent seats are drawn
    from ``members`` independently, with replacement, per game — so
    tables of 3x one model, 2+1, and 1+1+1 all occur."""
    rng = random.Random(seed)
    total = 0.0
    for g in range(n_games):
        seat = g % 4
        draws = [rng.choice(list(members)) for _ in range(3)]
        seats = [policy if p == seat else draws.pop() for p in range(4)]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        total += game.play_out(seats)[seat]
    return total / n_games


class SearchAssist(Strategy):
    """The learner in its deployed shape: raw policy while the table is
    wide, IS-MCTS below SEARCH_CARDS_4P with the rollout configuration
    (64 sims, depth 10, 100ms), exact solver inside the agent.  Used by the
    confirmation so the gate, the confirmation, and deployment all
    measure the same player; opponents stay raw policies."""

    def __init__(self, policy: Strategy, seed: int = 0,
                 simulations: int = 64, time_budget: float = 0.10):
        from big2.agent import IntegratedAgent

        self.policy = policy
        self.agent = IntegratedAgent(
            policy, simulations=simulations, depth=10, breadth=6,
            top_p=0.9, time_budget=time_budget, seed=seed,
        )
        self.name = f"assist({getattr(policy, 'name', 'policy')})"

    def select(self, game, player):
        from big2.endgame import remaining_cards

        if game.num_players == 4 and remaining_cards(game) < SEARCH_CARDS_4P:
            return self.agent.select(game, player)
        return self.policy.select(game, player)


def diet_confirm(
    policy: Strategy,
    n_games: int,
    scoring,
    rules,
    seed: int,
    snapshot_paths: Sequence[str] = (),
    past_self_prob: float = 0.5,
) -> float:
    """Large-sample confirmation on the *training* distribution.

    Same seat mix the rollouts use -- one wildcard from the whole
    collection, the rest lean pool + past selves -- with fresh deals
    every game.  A candidate that only shines against exotic panel
    seatings or lucky cards does not survive this; one that genuinely
    beats its own diet does.  This is the number the best-file gate
    trusts, so it is run at a much larger sample than the probes.
    """
    rng = random.Random(seed)
    pool = _opponent_pool()
    past = [p for p in (_load_snapshot_policy(pth)
                        for pth in snapshot_paths) if p is not None]
    total = 0.0
    for g in range(n_games):
        seat = g % 4
        opponents = {}
        others = [p for p in range(4) if p != seat]
        rng.shuffle(others)
        for j, p in enumerate(others):
            if j == 0:
                opponents[p] = rng.choice(_wide_collection())
            elif past and rng.random() < past_self_prob:
                opponents[p] = rng.choice(past)
            else:
                opponents[p] = rng.choice(pool)
        seats = [policy if p == seat else opponents[p] for p in range(4)]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        total += game.play_out(seats)[seat]
    return total / n_games


def panel_probe(
    policy: Strategy,
    panel: Sequence[Strategy],
    n_games: int,
    scoring,
    rules,
    seed: int,
) -> float:
    """Mean score over games against three *different* panel members.

    Seats rotate and the opponent triple is redrawn every game, so the
    number measures strength against the field rather than against
    whoever happened to be sitting there.
    """
    if len(panel) < 3:
        raise ValueError("panel needs at least 3 members")
    rng = random.Random(seed)
    total = 0.0
    for g in range(n_games):
        seat = g % 4
        opps = rng.sample(list(panel), 3)
        seats = [policy if p == seat else opps.pop() for p in range(4)]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        total += game.play_out(seats)[seat]
    return total / n_games


def _opponent_pool() -> List[Strategy]:
    """Deliberately lean: the strongest scripted agent (dumper, per the
    current baseline table), the CEM linear champion, and — closing the
    PSRO loop — the trained exploiter, so the next generation faces its
    own best adversary.  Weak opponents dilute the best-response
    gradient; past-self snapshots and self-play supply the diversity."""
    pool: List[Strategy] = [FiveCardDumper()]
    try:
        pool.append(LinearPolicy.load("big2/policies/linear_cem.npz"))
    except Exception:
        pool.append(SmartHeuristic())
    exploiter = _load_snapshot_policy("big2/policies/ppo_exploiter.pt")
    if exploiter is not None:
        pool.append(exploiter)
    # The strong testers, distilled (big2/humanlike.py): pass-heavy,
    # card-conserving play no scripted or self-play opponent supplies —
    # the exact style that has been beating the shipped models.
    human = _load_snapshot_policy("big2/policies/humanlike.pt")
    if human is not None:
        pool.append(human)
    return pool


# Worker-side cache of frozen past-self policies, keyed by (path, mtime)
# so a refreshed snapshot file is reloaded exactly once per worker.
_SNAP_CACHE: Dict[Tuple[str, float], "PPOPolicy"] = {}

# Worker-side wildcard collection: every playable model in the project,
# for the occasional random seat that keeps the diet from going stale.
_WIDE_POOL: Optional[List[Strategy]] = None

# Worker-side weighted collection: the whole pool with the preferred
# models (default: shipped champions + humanlike) at double weight —
# stay good against everything, preferentially pressure-test against
# the models to beat.  Cached as (preferred-key, pool).
_WEIGHTED_POOL: Optional[Tuple[Tuple[str, ...], List[Strategy]]] = None


_PREFERRED_DEFAULT = ("big2/policies/ppo_attn_v11.pt",
                      "big2/policies/ppo_attn.pt",
                      "big2/policies/humanlike.pt")


def _weighted_collection(
    preferred: Sequence[str] = _PREFERRED_DEFAULT,
) -> List[Strategy]:
    global _WEIGHTED_POOL
    key = tuple(preferred)
    if _WEIGHTED_POOL is None or _WEIGHTED_POOL[0] != key:
        coll = list(_wide_collection())
        boost = [p for p in (_load_snapshot_policy(f) for f in key)
                 if p is not None]
        _WEIGHTED_POOL = (key, coll + boost)
    return _WEIGHTED_POOL[1]


def _wide_collection() -> List[Strategy]:
    """The whole zoo, loaded lazily and defensively.

    A seat drawn from here (at ``wildcard_prob``) confronts the learner
    with a style the lean pool does not carry: the scripted regulars,
    the decomposition baseline, the evolved/DMC/CEM champions, both
    shipped PPO lines, the humanlike clone, and the exploiter.
    """
    global _WIDE_POOL
    if _WIDE_POOL is not None:
        return _WIDE_POOL
    from big2.strategies import PlayLowest

    pool: List[Strategy] = [FiveCardDumper(), SmartHeuristic(), PlayLowest()]
    try:
        from big2.decomposition import DecompositionStrategy

        pool.append(DecompositionStrategy())
    except Exception:
        pass
    for loader in (
        lambda: LinearPolicy.load("big2/policies/linear_cem.npz"),
        lambda: __import__("big2.nn", fromlist=["NNPolicy"]).NNPolicy.load(
            "big2/policies/evo_mlp.npz"),
        lambda: __import__("big2.dmc", fromlist=["DMCPolicy"]).DMCPolicy.load(
            "big2/policies/dmc_linear.npz"),
    ):
        try:
            pool.append(loader())
        except Exception:
            pass
    for path in ("big2/policies/ppo_attn.pt",       # PPO v1
                 "big2/policies/ppo_attn_v11.pt",   # WangBot_v1
                 "big2/policies/humanlike.pt",
                 "big2/policies/ppo_exploiter.pt"):
        p = _load_snapshot_policy(path)
        if p is not None:
            pool.append(p)
    _WIDE_POOL = pool
    return pool

# Worker-side cross-game opponent profiles: persists across games within
# the worker process; recency-weighted EMA with a ~500-game half-life.
_PROFILE_BOOK = OpponentProfileBook(half_life_games=500)


def _load_snapshot_policy(path: str) -> Optional["PPOPolicy"]:
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _SNAP_CACHE:
        if len(_SNAP_CACHE) > 8:
            _SNAP_CACHE.clear()
        try:
            policy = PPOPolicy.load(path)
            policy.name = os.path.basename(path)  # stable profile identity
            _SNAP_CACHE[key] = policy
        except Exception:
            return None
    return _SNAP_CACHE[key]


def rollout_games(args) -> bytes:
    """Worker: play games with the given weights, return episode batch."""
    (state_bytes, n_games, seed, selfplay_prob, snapshot_paths,
     past_self_prob, exploit_path, num_players, wildcard_prob,
     hard_pool, champ_prob, random_draw, search_rollouts,
     use_profiles, search_depth, champ_files, fixed_files,
     fixed_prob) = args
    preferred = tuple(champ_files) if champ_files else _PREFERRED_DEFAULT
    fixed_seats: List[Strategy] = []
    if fixed_files:
        # The final-boss table: with probability fixed_prob a game seats
        # exactly these models (shuffled) as the three opponents.
        fixed_seats = [p for p in (_load_snapshot_policy(f)
                                   for f in fixed_files) if p is not None]
    import torch

    torch.set_num_threads(1)
    payload = pickle.loads(state_bytes)
    net = build_net(payload["d_model"], payload["heads"],
                    state_dim=payload.get("state_dim", STATE_DIM),
                    act_dim=payload.get("act_dim", ACT_DIM),
                    layers=payload.get("layers", 2),
                    attn_blocks=2 if any(k.startswith("attn2.")
                                         for k in payload["state_dict"])
                    else 1)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    rng = random.Random(seed)
    # v3.1: the learner's own seats get policy-guided IS-MCTS once the
    # hand has narrowed (< SEARCH_CARDS cards left on the table).  Depth
    # and breadth stay small — beliefs are still fuzzy there, and the
    # search only needs to catch tactical blunders, not re-plan the
    # hand; the exact solver inside the agent takes over <= 14 as ever.
    # Opponent seats stay raw policies: they are the field to beat.
    searcher = None
    if not exploit_path and search_rollouts:
        try:
            from big2.agent import IntegratedAgent
            searcher = IntegratedAgent(
                PPOPolicy(net), simulations=64, depth=search_depth,
                breadth=6, top_p=0.9, time_budget=0.10, seed=seed,
            )
        except Exception:
            searcher = None
    if exploit_path:
        # Pure best response: every opponent seat is the frozen target.
        target = _load_snapshot_policy(exploit_path)
        pool = [target] if target else _opponent_pool()
        past_selves: List[PPOPolicy] = []
        selfplay_prob = 0.0
    else:
        pool = _opponent_pool()
        if hard_pool:
            # Crush-phase diet: only the hardest members — the exploiter
            # and the humanlike clone.  The soft scripted/linear seats
            # stop diluting the best-response gradient.
            hard = [p for p in (
                _load_snapshot_policy("big2/policies/ppo_exploiter.pt"),
                _load_snapshot_policy("big2/policies/humanlike.pt"),
            ) if p is not None]
            pool = hard or pool
        past_selves = [
            p for p in (_load_snapshot_policy(pth) for pth in snapshot_paths)
            if p is not None
        ]
    champ_seats: List[Strategy] = []
    if not exploit_path and champ_prob > 0:
        # The models the run is explicitly trying to dethrone get a
        # dedicated seat (below), not just wildcard visits.
        files = (champ_files if champ_files else
                 ("big2/policies/ppo_attn_v11.pt",
                  "big2/policies/ppo_attn.pt"))
        champ_seats = [p for p in (_load_snapshot_policy(f)
                                   for f in files) if p is not None]
    episodes = []

    for _ in range(n_games):
        # Alternating table sizes trains every head across counts: the
        # 2p game is the clean credit-assignment classroom, 4p is the
        # deployed game, 3p sits between.
        n_seats = (rng.choice(num_players)
                   if isinstance(num_players, (tuple, list))
                   else num_players)
        selfplay = rng.random() < selfplay_prob
        # Diversity injection: this table (with probability
        # ``random_draw``) seats every opponent by fresh draw from the
        # whole collection instead of the structured seating.
        rd_table = rng.random() < random_draw
        if selfplay:
            ppo_seats = set(range(n_seats))
        else:
            ppo_seats = {rng.randrange(n_seats)}
        opponents = {}
        others = [p for p in range(n_seats) if p not in ppo_seats]
        rng.shuffle(others)
        for j, p in enumerate(others):
            if rd_table:
                # Anti-overfit mix: every opponent seat is a fresh draw
                # from the whole collection (champions and humanlike at
                # double weight), so no specific table composition is
                # common enough to hyper-optimize against; at most one
                # seat (j == 0) may instead be a recent old self.
                if j == 0 and past_selves and rng.random() < past_self_prob:
                    opponents[p] = rng.choice(past_selves)
                else:
                    opponents[p] = rng.choice(_weighted_collection(preferred))
                continue
            # One seat per table is DEDICATED to a wildcard from the
            # whole collection; the rest draw lean pool + past selves.
            # Heads-up tables have only one opponent seat, so there the
            # wildcard takes it 1 game in 3 -- its 4p share.
            wildcard = wildcard_prob > 0 and (
                j == 0 if len(others) >= 2 else rng.random() < 1 / 3
            )
            if wildcard:
                opponents[p] = rng.choice(_wide_collection())
            # Dedicated champion seat: WangBot_v1 or PPO v1 directly.
            elif (champ_seats and j == 1 and len(others) >= 2
                  and rng.random() < champ_prob):
                opponents[p] = rng.choice(champ_seats)
            # League-style: previous generations sit at the table too.
            elif past_selves and rng.random() < past_self_prob:
                opponents[p] = rng.choice(past_selves)
            else:
                opponents[p] = rng.choice(pool)
        if (len(fixed_seats) >= 3 and n_seats == 4 and not selfplay
                and rng.random() < fixed_prob):
            trio = fixed_seats[:3]
            rng.shuffle(trio)
            for j, p in enumerate(others):
                opponents[p] = trio[j]
        seat_keys = {p: o.name for p, o in opponents.items()}
        game = Big2Game(
            scoring=ScoringConfig(), rules=DEFAULT_RULES,
            num_players=n_seats,
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
            options, state, acts = encode_decision(
                game, p,
                book=_PROFILE_BOOK if use_profiles else None,
                seat_keys=seat_keys,
                include_profiles=net.state_dim != FEAT_DIM,
                include_danger=net.act_dim >= ACT_DIM_V11,
                include_plan=(net.act_dim >= ACT_DIM
                              and net.state_dim >= STATE_DIM),
                include_beat=net.act_dim >= ACT_DIM_BEAT,
            )
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
            if searcher is not None:
                from big2.endgame import move_key as _mk
                from big2.endgame import remaining_cards as _rc
                limit = (SEARCH_CARDS_4P if n_seats == 4
                         else SEARCH_CARDS_23P)
                if _rc(game) < limit:
                    # Search proposes; the executed move is logged with
                    # the net's own logp so PPO's clipped ratio stays
                    # anchored to the policy that produced the batch.
                    try:
                        dec = searcher.explain(game, p)
                        for j, m in enumerate(options):
                            if _mk(m) == dec.move:
                                idx = j
                                break
                    except Exception:
                        pass
            logp = float(dist.log_prob(torch.tensor(idx)))
            t = trajs[p]
            t["state"].append(state)
            t["acts"].append(acts)
            t["chosen"].append(idx)
            t["logp"].append(logp)
            t["value"].append(float(value[0]))
            t["belief"].append(belief_target(game, p))
            game.step(options[idx])

        # Fold the finished game into the cross-game opponent profiles.
        _PROFILE_BOOK.observe_game(game, seat_keys)

        for p, t in trajs.items():
            if t["state"]:
                episodes.append(
                    {**{k: v for k, v in t.items()},
                     "score": game.scores[p] / SCORE_SCALE,
                     # Field games (real opponents at the table) feed the
                     # learner's recent-form gate; pure self-play mirrors
                     # score ~0 by symmetry and would only dilute it.
                     "field": not selfplay,
                     "n_seats": n_seats}
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
    layers: int = 2,  # input-MLP depth; 2 is every shipped checkpoint
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
    exploit_target: Optional[str] = None,  # train a pure best response to
    #   this frozen checkpoint; probes then measure points extracted from it
    fresh_bar: bool = False,  # don't inherit the resumed checkpoint's bar:
    #   `out` records this run's own best (for new-version files where the
    #   old champion file stays untouched)
    probe_vs: Optional[str] = None,  # measure candidates against 3 copies
    #   of this checkpoint instead of the champions trio (training still
    #   uses the normal diet — unlike exploit_target, only the metric
    #   changes); best-saves then mean "best against this reference"
    init_bar: Optional[float] = None,  # explicit starting bar for the
    #   best-save gate (e.g. the current out-file's known score under a
    #   new metric, so nothing worse ever overwrites it)
    wildcard_prob: float = 0.0,  # per-seat chance of a wildcard opponent
    #   drawn from the whole collection (_wide_collection)
    num_players=4,  # 2 runs the 1v1 curriculum: the same encoders
    #   and net, on the smaller game where credit assignment is cleanest;
    #   a tuple like (2, 3, 4) alternates table sizes per game
    #   and net, on the smaller game where credit assignment is cleanest
    confirm_panel: bool = True,  # gate on the training distribution: the
    #   probe metric is the rolling mean of recent rollout scores on 4p
    #   field games (the exact mix diet_confirm retests, measured for
    #   free), and confirmation is diet_confirm on the same mix.  Forced
    #   off for exploiter runs, where points extracted from the single
    #   target is the entire metric.
    recent_window: int = 1000,  # 4p field games in the recent-form gate;
    #   ~1/6 of rollout games qualify, so 1000 spans roughly the last
    #   6k training games — recent enough to track the current net,
    #   wide enough that its noise matches the old 120-game probe.
    note: Optional[str] = None,  # version label stored in saved meta
    attn_blocks: int = 1,  # move-interaction depth (see build_net); a
    #   1-block checkpoint warm-starts into 2 blocks unchanged thanks to
    #   the zero-initialized second block
    hard_pool: bool = False,  # rollout opponents: lean pool trimmed to
    #   the exploiter + humanlike clone only (crush phase); wildcard and
    #   past-self seats unaffected, confirmation mix unaffected
    champ_prob: float = 0.0,  # per-table chance (tables with >= 2
    #   opponent seats) that one seat is a shipped champion drawn from
    #   {WangBot_v1, PPO v1} — the models this run is out to beat
    random_draw: float = 0.0,  # per-table probability of anti-overfit
    #   seating: every opponent seat drawn fresh from the whole
    #   collection, at most one seat a recent old self (overrides
    #   wildcard/champ/hard_pool for that table).  1.0 = every table
    state_dim: int = STATE_DIM,  # input widths: the feature ladder
    act_dim: int = ACT_DIM,  # trains nets at historical widths (v1
    #   275/296, +danger 310, +profiles 299, full 323/326); encoding
    #   gates itself off these, and widen_state_dict zero-pads a
    #   narrower checkpoint into the next stage
    search_rollouts: bool = True,  # False = pure policy rollouts (the
    #   ladder trains the policy net first; the tree returns later)
    snapshot_dir: Optional[str] = None,  # past-self/breakout snapshot
    #   directory; concurrent runs must each get their own (shared dirs
    #   race on Windows file locks and cross-feed past selves)
    use_profiles: bool = True,  # False zero-feeds the opponent-profile
    #   columns during training, matching eval/deployment (which never
    #   supply a profile book) — the low-bias directive: no cross-game
    #   opponent inference in the inputs
    search_depth: int = 4,  # tree depth for search-assisted rollouts:
    #   ~a trick plus three rotations (10) plays the exchange out, then
    #   the value head prices the resulting position; opponents inside
    #   the tree play the agent's own distribution
    champ_files: Optional[Sequence[str]] = None,  # dedicated-seat and
    #   weighted-pool-boost opponents (default: the shipped champions);
    #   chain C passes the A/B finals here
    fixed_files: Optional[Sequence[str]] = None,  # chain D's table: with
    #   probability fixed_prob a rollout game seats exactly these three
    #   models (shuffled) as the opponents
    fixed_prob: float = 0.0,
    confirm_files: Optional[Sequence[str]] = None,  # confirmation-room
    #   members (default CONFIRM_FILES)
    imitate_path: Optional[str] = None,  # frozen teacher net: adds a
    #   KL(student || teacher) term on visited states — chain C anchors
    #   to the humanlike clone while learning to win
    imitate_coef: float = 0.0,
    confirm_deployed: bool = True,  # False = confirmation plays the raw
    #   policy instead of the search-assisted deployed shape
    device: str = "auto",  # learner device: cuda when available, else
    #   cpu.  Rollout workers always run cpu -- batch-1 inference in a
    #   Python game loop loses more to transfer than a GPU gives back;
    #   what a GPU accelerates is the minibatch learner below.
    verbose: bool = True,
):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    net = build_net(d_model, heads, state_dim=state_dim, act_dim=act_dim,
                    layers=layers, attn_blocks=attn_blocks)
    net.to(dev)
    best_probe = -1e9
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        # Pre-profile checkpoints warm-start via zero-column widening;
        # pre-attn2 checkpoints warm-start via the zero-initialized
        # second block (the only keys allowed to be missing).
        res = net.load_state_dict(
            widen_state_dict(payload["state_dict"], state_dim, act_dim),
            strict=False,
        )
        bad = [k for k in res.missing_keys
               if not k.startswith(("attn2.", "norm2."))]
        if bad or res.unexpected_keys:
            raise RuntimeError(
                f"resume mismatch: missing {bad}, "
                f"unexpected {list(res.unexpected_keys)}"
            )
        # Don't let a resumed run overwrite a better checkpoint with its
        # first mediocre probe: inherit the saved best — unless this is an
        # exploiter warm start (metric starts fresh) or the run writes to
        # a new version file that should record its own best.
        if not exploit_target and not fresh_bar:
            best_probe = float(payload.get("meta", {}).get("probe", -1e9))
    if init_bar is not None:
        best_probe = float(init_bar)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    pool = mp.Pool(workers)
    total_games = games_offset
    t0 = time.time()

    from big2.decomposition import DecompositionStrategy
    from big2.evolve import _probe

    probe_baselines = [SmartHeuristic(), DecompositionStrategy(), FiveCardDumper()]
    # Confirmation pool: seats drawn with replacement per game —
    # stationary (no past selves), exactly the room to beat.
    confirm_champs: List[Strategy] = (
        confirm_pool(confirm_files) if confirm_panel else []
    )

    imitator = None
    if imitate_path and imitate_coef > 0:
        ipay = torch.load(imitate_path, map_location="cpu",
                          weights_only=True)
        isd = ipay["state_dict"]
        imitator = build_net(
            ipay.get("d_model", 192), ipay.get("heads", 4),
            state_dim=isd["state_mlp.0.weight"].shape[1],
            act_dim=isd["act_mlp.0.weight"].shape[1],
            layers=sum(1 for k in isd
                       if k.startswith("state_mlp.") and
                       k.endswith(".weight")),
            attn_blocks=2 if any(k.startswith("attn2.") for k in isd)
            else 1,
        )
        imitator.load_state_dict(isd)
        imitator.to(dev).eval()
        if imitator.state_dim > state_dim or imitator.act_dim > act_dim:
            raise ValueError("imitation teacher wider than the student")
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

    snap_dir = snapshot_dir or os.path.join(
        os.path.dirname(out) or ".", "ppo_snapshots"
    )
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

    if probe_vs:
        # With the panel on, this only seeds the field with the named
        # reference; with the panel off it is the whole (fixed) metric.
        champs = [PPOPolicy.load(probe_vs) for _ in range(3)]
    if exploit_target:
        # The exploitability probe: how much a dedicated adversary
        # extracts from ONE target — a fixed reference is the point here,
        # so the mixed-field panel does not apply.
        champs = [PPOPolicy.load(exploit_target) for _ in range(3)]
        confirm_panel = False

    # Recent form on the training distribution: per-game scores from 4p
    # field games, streamed out of the rollouts themselves.  This is the
    # gate metric — the same mix diet_confirm retests, at zero probe cost.
    recent_field: deque = deque(maxlen=recent_window)
    diet_best = -1e9  # high-water mark for the recent-form series

    for it in range(1, iters + 1):
        blob = pickle.dumps(
            {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
             "d_model": d_model, "heads": heads, "layers": layers,
             "state_dim": state_dim, "act_dim": act_dim}
        )
        per = games_per_iter // workers
        snaps = [] if exploit_target else _snapshot_paths()
        results = pool.map(
            rollout_games,
            [(blob, per, rng.randrange(2**31), selfplay_prob, snaps,
              past_self_prob, exploit_target, num_players, wildcard_prob,
              hard_pool, champ_prob, random_draw, search_rollouts,
              use_profiles, search_depth,
              tuple(champ_files) if champ_files else None,
              tuple(fixed_files) if fixed_files else None, fixed_prob)
             for _ in range(workers)],
        )
        episodes = [e for r in results for e in pickle.loads(r)]
        total_games += per * workers
        recent_field.extend(
            e["score"] * SCORE_SCALE for e in episodes
            if e.get("field") and e.get("n_seats") == 4
        )

        flat_state, flat_acts, flat_mask = [], [], []
        flat_chosen, flat_logp, flat_adv, flat_ret, flat_belief = [], [], [], [], []
        max_a = max(len(a) for e in episodes for a in e["acts"])
        for e in episodes:
            adv, ret = _gae(e["value"], e["score"])
            for i, acts in enumerate(e["acts"]):
                A = len(acts)
                padded = np.zeros((max_a, act_dim), dtype=np.float32)
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
        if dev.type != "cpu":
            S, A_, M, C = S.to(dev), A_.to(dev), M.to(dev), C.to(dev)
            LP, ADV, RET, B = (LP.to(dev), ADV.to(dev), RET.to(dev),
                               B.to(dev))
        n = len(S)

        pol_loss = val_loss = ent = bel_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = perm[start : start + minibatch]
                logits, value, belief = net(S[idx], A_[idx], M[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(C[idx])
                # Search-executed moves can sit far out in the policy's
                # tail (large negative stored logp), and once entropy is
                # low the multi-epoch update can push the log-ratio past
                # float range — exp overflows to inf and one bad
                # minibatch NaNs the net (this killed a run at iter 90).
                # Past the PPO clip band the surrogate is flat anyway,
                # so clamping the log-ratio changes nothing legitimate.
                ratio = torch.exp((logp - LP[idx]).clamp(-20.0, 4.0))
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
                if imitator is not None:
                    with torch.no_grad():
                        t_logits, _, _ = imitator(
                            S[idx][:, :imitator.state_dim],
                            A_[idx][:, :, :imitator.act_dim],
                            M[idx],
                        )
                    i_loss = torch.nn.functional.kl_div(
                        torch.log_softmax(logits, dim=-1),
                        torch.softmax(t_logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + imitate_coef * i_loss
                if not torch.isfinite(loss):
                    # One poisoned minibatch must not NaN the whole
                    # net; skip it and let the next one proceed.
                    print("[ppo] non-finite loss, minibatch skipped",
                          flush=True)
                    continue
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
                {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                 "d_model": d_model,
                 "heads": heads, "layers": layers},
                slot,
            )
            if verbose:
                print(f"[ppo] froze past-self snapshot -> {slot}", flush=True)

        if probe_every_iters and it % probe_every_iters == 0:
            if dev.type == "cpu":
                policy = PPOPolicy(net)
            else:
                twin = build_net(d_model, heads, state_dim=state_dim,
                                 act_dim=act_dim, layers=layers,
                                 attn_blocks=attn_blocks)
                twin.load_state_dict(
                    {k: v.cpu() for k, v in net.state_dict().items()}
                )
                policy = PPOPolicy(twin)
            vs_base = _probe(
                policy, probe_baselines, probe_games, ScoringConfig(),
                DEFAULT_RULES, seed=it,
            )
            if confirm_panel:
                # The gate metric: the learner's own recent rollout
                # scores on 4p field games — the training distribution
                # itself, measured for free.  A separate probe field
                # would gate on a different metric than the one the
                # confirmation retests.
                vs_champ = (
                    sum(recent_field) / len(recent_field)
                    if len(recent_field) >= 120 else float("nan")
                )
            else:
                vs_champ = (
                    _probe(policy, champs[:3], probe_games, ScoringConfig(),
                           DEFAULT_RULES, seed=it + 1)
                    if len(champs) == 3
                    else float("nan")
                )
            vs_diet = vs_champ
            tag = 6 if exploit_target else 5
            label = ("vs-target" if exploit_target
                     else "recent-field" if confirm_panel
                     else f"vs-{os.path.basename(probe_vs).split('.')[0]}"
                     if probe_vs else "vs-champions")
            os.makedirs(os.path.dirname(progress_path), exist_ok=True)
            with open(progress_path, "a") as f:
                f.write(
                    f"{tag},{total_games},4,"
                    f"{'exploiter' if exploit_target else 'ppo'},0,{lr:.5f},"
                    f"{vs_base:.3f},{vs_champ:.3f}\n"
                )
                # Diet series charts as its own island tag (7).
                f.write(
                    f"7,{total_games},4,diet,0,{lr:.5f},{vs_diet:.3f},nan\n"
                )
            print(
                f"[ppo] probe @{total_games}: vs-baselines {vs_base:+.2f}  "
                f"{label} {vs_champ:+.2f}  vs-diet {vs_diet:+.2f}",
                flush=True,
            )
            net.train()
            score = vs_champ if vs_champ == vs_champ else vs_base
            # Dual trigger: beating the shipping bar OR breaking the diet
            # high-water mark earns a confirmation retest (a strong diet
            # result can reveal improvement the noisy champions probe
            # missed).  Shipping still requires confirming vs champions.
            diet_breakout = vs_diet > diet_best + 0.25
            diet_best = max(diet_best, vs_diet)
            if diet_breakout and score <= best_probe:
                print(
                    f"[ppo] diet breakout ({vs_diet:+.2f}) -> retest",
                    flush=True,
                )
            if score > best_probe or diet_breakout:
                # Merit snapshot: a breakout candidate is a new play style
                # worth beating — freeze it into the training diet now,
                # whatever the confirmation decides about shipping it.
                bpath = os.path.join(snap_dir, f"breakout_{it}.pt")
                _atomic_save(
                    {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                 "d_model": d_model,
                     "heads": heads, "layers": layers},
                    bpath,
                )
                breakouts = sorted(
                    (os.path.join(snap_dir, f) for f in os.listdir(snap_dir)
                     if f.startswith("breakout_")),
                    key=os.path.getmtime,
                )
                for stale in breakouts[:-2]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass  # locked by a reader; retire it next round

                # Two-stage gate: a 120-game probe is +-1 noisy, so a
                # would-be best must confirm on an independent larger
                # re-test; the confirmed number is what gets recorded.
                if confirm_panel and confirm_champs:
                    # Probe said "beats the field"; confirmation asks
                    # the question the file actually stands for -- does
                    # the candidate beat the champions' room: seats
                    # drawn with replacement from {WangBot_v1, PPO v1,
                    # humanlike}, fresh deals, rotated seats.  The
                    # candidate plays deployed (search assist) or raw
                    # per confirm_deployed — the ladder gates on the
                    # raw policy, the main line on the deployed shape.
                    cand = (SearchAssist(policy, seed=it)
                            if confirm_deployed else policy)
                    confirmed = replacement_probe(
                        cand, confirm_champs,
                        confirm_games, ScoringConfig(), DEFAULT_RULES,
                        seed=it + 7919,
                    )
                elif confirm_panel:
                    confirmed = diet_confirm(
                        SearchAssist(policy, seed=it), confirm_games,
                        ScoringConfig(), DEFAULT_RULES, seed=it + 7919,
                        snapshot_paths=_snapshot_paths(),
                        past_self_prob=past_self_prob,
                    )
                else:
                    opponents = (
                        champs[:3] if len(champs) == 3 else probe_baselines
                    )
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
                    meta = {"iter": it, "games": total_games,
                            "probe": confirmed,
                            "confirm_games": confirm_games,
                            "metric": ("champs-replacement"
                                       if confirm_champs else "diet")}
                    if note:
                        meta["note"] = note
                    _atomic_save(
                        {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                 "d_model": d_model,
                         "heads": heads, "layers": layers, "meta": meta},
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
    parser.add_argument("--layers", type=int, default=2,
                        help="input-MLP depth (Linear layers per tower)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--probe-every-iters", type=int, default=30)
    parser.add_argument("--games-offset", type=int, default=0)
    parser.add_argument("--snapshot-every-iters", type=int, default=100)
    parser.add_argument("--past-self-prob", type=float, default=0.5)
    parser.add_argument("--confirm-games", type=int, default=480)
    parser.add_argument("--exploit-target", default=None,
                        help="freeze this checkpoint as the only opponent "
                             "and train a pure best response to it")
    parser.add_argument("--fresh-bar", action="store_true",
                        help="don't inherit the resumed checkpoint's best "
                             "bar; --out records this run's own best")
    parser.add_argument("--probe-vs", default=None,
                        help="measure candidates against 3 copies of this "
                             "checkpoint instead of the champions trio")
    parser.add_argument("--init-bar", type=float, default=None,
                        help="explicit starting bar for the best-save gate")
    parser.add_argument("--num-players", type=int, default=4,
                        help="2 for the 1v1 curriculum, 4 for the real game")
    parser.add_argument("--confirm-panel", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="probe and confirm against random draws from "
                             "the mixed field of diet, peers and past "
                             "selves (--no-confirm-panel restores the "
                             "fixed-reference metric)")
    parser.add_argument("--device", default="auto",
                        help="learner device: auto (cuda when available), cuda, or cpu")
    parser.add_argument("--note", default=None,
                        help="version label stored in the saved meta")
    args = parser.parse_args()
    train_ppo(
        iters=args.iters, games_per_iter=args.games_per_iter,
        workers=args.workers, selfplay_prob=args.selfplay_prob, lr=args.lr,
        d_model=args.d_model, layers=args.layers, seed=args.seed,
        out=args.out,
        resume=args.resume, probe_every_iters=args.probe_every_iters,
        games_offset=args.games_offset,
        snapshot_every_iters=args.snapshot_every_iters,
        past_self_prob=args.past_self_prob,
        confirm_games=args.confirm_games,
        exploit_target=args.exploit_target,
        fresh_bar=args.fresh_bar,
        probe_vs=args.probe_vs,
        init_bar=args.init_bar,
        num_players=args.num_players,
        confirm_panel=args.confirm_panel,
        note=args.note,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""Joint training: all four models at one table, thinking deep.

The chain era trained one learner against frozen fields.  This trains
the whole final generation together -- v2_patient, v2_adversarial,
v2_self_trained, v2_human_trained -- every seat search-assisted at
tournament depth, every model updated after every batch.  Depth over
volume: the credit study (docs/DEPTH_PROGRAM.md) showed early-game
gradient is mostly noise at terminal-reward distance, so each decision
here carries trick-level shaping, and search conclusions distill into
the priors instead of being faked as policy samples.

Per decision, each seat runs the production decision stack (exact
solver, shortlisted IS-MCTS at high budget, policy) and:

* searched decisions train the policy toward the **visit
  distribution** (cross-entropy; excluded from the PPO ratio);
* solver decisions distill as one-hot targets -- provable moves teach;
* policy decisions are sampled and train by clipped PPO as usual;
* above MIX_ARGMAX_BELOW cards the executed move is **sampled** from
  the visits: every model trains as -- and against -- a mixed
  strategy, the unexploitability requirement;
* every decision records the trick-level potential for shaped GAE.

After each batch every model probes against the legacy diet
(WangBot_v1, PPO v1, humanlike, heuristic); a model that regresses
more than ``rollback_bar`` points/game against the field rolls back to
its last passing weights -- beating each other must not cost the room.

    python -m big2.joint --pilot            # calibrate throughput
    python -m big2.joint --games 5000       # one real batch
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from typing import Dict, List, Optional

import numpy as np

from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = os.path.join(HERE, "policies")

# Canonical depth-program names -> lineage checkpoints (first batch
# starts from these; later batches resume from the joint outputs).
LINEAGE = {
    "v2_patient": "v2_patient.pt",        # AWR child of wangbot_v2
    "v2_adversarial": "sicario_v1.pt",
    "v2_self_trained": "leonidas_v1.pt",
    "v2_human_trained": "khabib_v1.pt",
}
DIET = ("ppo_attn_v11.pt", "ppo_attn.pt", "humanlike.pt")

SIMULATIONS = 1024
TIME_BUDGET = 5.0
DEPTH = 12


def _out_path(name: str) -> str:
    return os.path.join(POLICIES, f"{name}_joint.pt")


def _load_payload(name: str) -> Dict:
    import torch

    path = _out_path(name)
    if not os.path.exists(path):
        path = os.path.join(POLICIES, LINEAGE[name])
    if not os.path.exists(path):
        # v2_patient before its AWR pass lands: start from its parent.
        path = os.path.join(POLICIES, "wangbot_v2.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    sd = dict(payload["state_dict"])
    # The privileged critic rides in its own slot so the shipped
    # state_dict stays exactly what deployment loaders expect.
    sd.update(payload.get("oracle_state", {}))
    return {
        "state_dict": sd,
        "d_model": payload.get("d_model", 192),
        "heads": payload.get("heads", 4),
        "layers": payload.get("layers", 2),
        "state_dim": sd["state_mlp.0.weight"].shape[1],
        "act_dim": sd["act_mlp.0.weight"].shape[1],
    }


def _build(payload: Dict, oracle: bool = False):
    from big2.neural import BELIEF_SLOTS, build_net

    net = build_net(
        payload["d_model"], payload["heads"],
        state_dim=payload["state_dim"], act_dim=payload["act_dim"],
        layers=payload["layers"],
        attn_blocks=2 if any(k.startswith("attn2.")
                             for k in payload["state_dict"]) else 1,
        oracle_dim=BELIEF_SLOTS if oracle else 0,
    )
    # Oracle-head keys are absent from every existing checkpoint and
    # start fresh; nothing else may be missing.
    res = net.load_state_dict(payload["state_dict"], strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("oracle_")]
    if bad or res.unexpected_keys:
        raise RuntimeError(f"load mismatch: missing {bad}, "
                           f"unexpected {list(res.unexpected_keys)}")
    net.eval()
    return net


# ----------------------------------------------------------------------
# Rollout worker (top-level: Windows spawn must pickle it)
# ----------------------------------------------------------------------


def _joint_worker(args):
    (blobs, games, seed, sims, budget, depth, mix_below,
     hb_path, hb_every) = args
    import torch

    from big2.agent import IntegratedAgent
    from big2.endgame import move_key as _mk
    from big2.endgame import remaining_cards as _rc
    from big2.neural import (
        ACT_DIM, ACT_DIM_BEAT, ACT_DIM_V11, FEAT_DIM, SCORE_SCALE,
        STATE_DIM, PPOPolicy, belief_target, encode_decision,
    )
    from big2.shaping import potential

    torch.set_num_threads(1)
    rng = random.Random(seed)
    nets, agents = {}, {}
    for name, blob in blobs.items():
        payload = pickle.loads(blob)
        net = _build(payload)
        nets[name] = net
        agents[name] = IntegratedAgent(
            PPOPolicy(net), simulations=sims, depth=depth,
            breadth=6, top_p=0.9, time_budget=budget,
            seed=rng.randrange(2 ** 31),
        )

    episodes: Dict[str, List[Dict]] = {name: [] for name in blobs}
    names = list(blobs)
    # Heartbeat: pool.map is a barrier, so without this the run is a
    # 12-hour silence in which "working" and "hung" look identical.
    # Each worker appends a small JSON line; the reader sums across
    # workers.  Appends of a short line are atomic enough on Windows
    # for a progress file, and a failed write must never kill a run.
    hb_scores: Dict[str, float] = {n: 0.0 for n in names}
    hb_wins: Dict[str, int] = {n: 0 for n in names}
    hb_searched = hb_decisions = 0
    t_start = time.time()

    def beat(done: int) -> None:
        if not hb_path:
            return
        try:
            with open(hb_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pid": os.getpid(), "games": done,
                    "elapsed": round(time.time() - t_start, 1),
                    "score": {n: round(v / max(1, done), 3)
                              for n, v in hb_scores.items()},
                    "wins": dict(hb_wins),
                    "searched_frac": round(
                        hb_searched / max(1, hb_decisions), 3),
                }) + "\n")
        except OSError:
            pass

    for g_i in range(games):
        seating = names[:]
        rng.shuffle(seating)
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        trajs = {s: {"state": [], "acts": [], "chosen": [], "logp": [],
                     "value": [], "belief": [], "phi": [], "visits": []}
                 for s in range(4)}
        while not game.game_over:
            p = game.turn
            name = seating[p]
            net = nets[name]
            options, state, acts = encode_decision(
                game, p,
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

            visit_target = None
            idx: Optional[int] = None
            try:
                dec = agents[name].explain(game, p)
                vis = getattr(dec, "visits", None) or {}
                counts = np.array(
                    [float(vis.get(_mk(m), 0.0)) for m in options],
                    dtype=np.float64,
                )
                if getattr(dec, "exact", False):
                    # Solver truth distills as a one-hot target.
                    visit_target = np.zeros(len(options), dtype=np.float32)
                    for j, m in enumerate(options):
                        if _mk(m) == dec.move:
                            visit_target[j] = 1.0
                            idx = j
                            break
                elif counts.sum() > 0:
                    visit_target = (counts / counts.sum()).astype(np.float32)
                    if _rc(game) > mix_below:
                        idx = int(np.random.default_rng(
                            rng.randrange(2 ** 31)).choice(
                                len(options), p=visit_target))
                    else:
                        idx = int(np.argmax(counts))
            except Exception:
                visit_target = None
            if idx is None:
                idx = int(dist.sample())
                visit_target = None          # pure policy sample: PPO path
            logp = float(dist.log_prob(torch.tensor(idx)))
            t = trajs[p]
            t["state"].append(state)
            t["acts"].append(acts)
            t["chosen"].append(idx)
            t["logp"].append(logp)
            t["value"].append(float(value[0]))
            t["belief"].append(belief_target(game, p))
            t["phi"].append(potential(game, p))
            t["visits"].append(visit_target)
            hb_decisions += 1
            hb_searched += int(visit_target is not None)
            game.step(options[idx])

        for s, t in trajs.items():
            hb_scores[seating[s]] += game.scores[s]
            hb_wins[seating[s]] += int(game.winner == s)
            if t["state"]:
                episodes[seating[s]].append(
                    {**t, "score": game.scores[s] / SCORE_SCALE})
        if hb_every and (g_i + 1) % hb_every == 0:
            beat(g_i + 1)
    beat(games)
    return pickle.dumps(episodes)


# ----------------------------------------------------------------------
# Per-model PPO + distillation update
# ----------------------------------------------------------------------


def _update(net, episodes: List[Dict], lr: float, epochs: int,
            minibatch: int, clip: float = 0.2, vf_coef: float = 0.5,
            ent_coef: float = 0.005, belief_coef: float = 0.3,
            device: str = "auto") -> Dict[str, float]:
    import torch

    from big2.neural import DISTILL_COEF, _gae

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    net.to(dev)
    net.train()

    act_dim = net.act_dim
    fs, fa, fm, fc, flp, fadv, fret, fb, fv, fsr = ([] for _ in range(10))
    max_a = max(len(a) for e in episodes for a in e["acts"])

    # Privileged baseline: recompute each state's value with the true
    # opponent hands visible (the rollout already recorded them as the
    # belief target).  A baseline that does not depend on the action
    # leaves the policy gradient unbiased, and seeing the hands is
    # worth +0.05 to +0.10 correlation with the outcome in the early
    # and middle game -- exactly where the ordinary critic is blind.
    oracle_values = {}
    if getattr(net, "oracle_dim", 0):
        with torch.no_grad():
            for ei, e in enumerate(episodes):
                A = max(len(a) for a in e["acts"])
                st = torch.from_numpy(np.stack(e["state"])).to(dev)
                ac = torch.zeros(len(e["acts"]), A, act_dim, device=dev)
                mk = torch.zeros(len(e["acts"]), A, dtype=torch.bool,
                                 device=dev)
                for i, a in enumerate(e["acts"]):
                    ac[i, : len(a)] = torch.from_numpy(a).to(dev)
                    mk[i, : len(a)] = True
                ob = torch.from_numpy(np.stack(e["belief"])).to(dev)
                oracle_values[ei] = [
                    float(v) for v in net.oracle_value(st, ac, mk, ob)
                ]

    for ei, e in enumerate(episodes):
        base = oracle_values.get(ei, e["value"])
        adv, ret = _gae(base, e["score"], phi=e.get("phi"))
        for i, acts in enumerate(e["acts"]):
            A = len(acts)
            padded = np.zeros((max_a, act_dim), dtype=np.float32)
            padded[:A] = acts
            fs.append(e["state"][i]); fa.append(padded)
            m = np.zeros(max_a, dtype=bool); m[:A] = True
            fm.append(m)
            fc.append(e["chosen"][i]); flp.append(e["logp"][i])
            fadv.append(adv[i]); fret.append(ret[i])
            fb.append(e["belief"][i])
            v = np.zeros(max_a, dtype=np.float32)
            tgt = e["visits"][i]
            if tgt is not None:
                v[: len(tgt)] = tgt
                fsr.append(True)
            else:
                fsr.append(False)
            fv.append(v)

    S = torch.from_numpy(np.stack(fs)).to(dev)
    A_ = torch.from_numpy(np.stack(fa)).to(dev)
    M = torch.from_numpy(np.stack(fm)).to(dev)
    C = torch.tensor(fc, dtype=torch.long, device=dev)
    LP = torch.tensor(flp, dtype=torch.float32, device=dev)
    ADV = torch.tensor(np.asarray(fadv), device=dev)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
    RET = torch.tensor(np.asarray(fret), device=dev)
    B = torch.from_numpy(np.stack(fb)).to(dev)
    VT = torch.from_numpy(np.stack(fv)).to(dev)
    SM = torch.tensor(fsr, dtype=torch.bool, device=dev)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(S)
    stats = {}
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for start in range(0, n, minibatch):
            idx = perm[start: start + minibatch]
            logits, value, belief = net(S[idx], A_[idx], M[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(C[idx])
            ratio = torch.exp((logp - LP[idx]).clamp(-20.0, 4.0))
            surr = torch.min(
                ratio * ADV[idx],
                torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx],
            )
            searched = SM[idx]
            free = ~searched
            p_loss = -(surr * free).sum() / free.sum().clamp(min=1)
            if searched.any():
                lsm = torch.log_softmax(logits, dim=-1)
                ce = -(VT[idx] * lsm).masked_fill(~M[idx], 0.0).sum(-1)
                p_loss = p_loss + DISTILL_COEF * (
                    (ce * searched).sum() / searched.sum())
            v_loss = ((value - RET[idx]) ** 2).mean()
            e_loss = dist.entropy().mean()
            b_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                belief, B[idx])
            loss = (p_loss + vf_coef * v_loss - ent_coef * e_loss
                    + belief_coef * b_loss)
            if getattr(net, "oracle_dim", 0):
                # The privileged critic is fit to the same returns; it
                # is only ever read to form advantages, never at play.
                ov = net.oracle_value(S[idx], A_[idx], M[idx], B[idx])
                loss = loss + vf_coef * ((ov - RET[idx]) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats = {"pol": float(p_loss.detach()),
                     "val": float(v_loss.detach()),
                     "ent": float(e_loss.detach())}
    net.to("cpu")
    net.eval()
    return stats


# ----------------------------------------------------------------------
# Diet probe: has progress against each other cost the field?
# ----------------------------------------------------------------------


def probe_vs_diet(net, n_games: int = 300, seed: int = 11) -> float:
    from big2.neural import PPOPolicy, load_checkpoint_policy

    me = PPOPolicy(net)
    field = [load_checkpoint_policy(os.path.join(POLICIES, f))
             for f in DIET]
    from big2.strategies import SmartHeuristic
    field.append(SmartHeuristic())
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_games):
        opps = [rng.choice(field) for _ in range(3)]
        seats = {0: me, 1: opps[0], 2: opps[1], 3: opps[2]}
        g = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                     num_players=4,
                     rng=random.Random(rng.randrange(2 ** 31)))
        while not g.game_over:
            g.step(seats[g.turn].select(g, g.turn))
        total += g.scores[0]
    return total / n_games


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def train_joint(
    games: int = 5000,
    batches: int = 1,
    workers: Optional[int] = None,
    simulations: int = SIMULATIONS,
    time_budget: float = TIME_BUDGET,
    depth: int = DEPTH,
    mix_below: int = 20,
    lr: float = 1e-4,
    epochs: int = 2,
    minibatch: int = 512,
    probe_games: int = 300,
    rollback_bar: float = 1.0,
    oracle: bool = True,
    heartbeat: int = 250,
    heartbeat_path: str = os.path.join(HERE, "..", "joint_heartbeat.jsonl"),
    progress_path: str = os.path.join(HERE, "..", "joint_progress.jsonl"),
) -> None:
    import torch

    workers = workers or max(2, min(14, (os.cpu_count() or 8) - 2))
    payloads = {name: _load_payload(name) for name in LINEAGE}
    nets = {name: _build(p, oracle=oracle) for name, p in payloads.items()}
    baseline = {name: probe_vs_diet(net, probe_games, seed=7)
                for name, net in nets.items()}
    print("[joint] diet baseline: "
          + "  ".join(f"{n} {v:+.2f}" for n, v in baseline.items()),
          flush=True)
    last_good = {name: {k: v.clone() for k, v in net.state_dict().items()}
                 for name, net in nets.items()}

    ctx = mp.get_context("spawn")
    for batch in range(batches):
        t0 = time.time()
        if heartbeat_path:
            # Header first: the reader needs the pool size to scale the
            # rate off the workers that have reported so far.
            with open(heartbeat_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "header": True, "workers": workers,
                    "total": per * workers, "batch": batch + 1,
                    "sims": simulations, "budget": time_budget,
                }) + "\n")
        blobs = {}
        for name, net in nets.items():
            p = payloads[name]
            # Workers play, they do not learn: ship them the ordinary
            # net only -- the privileged critic never leaves training.
            blobs[name] = pickle.dumps({
                "state_dict": {k: v for k, v in net.state_dict().items()
                               if not k.startswith("oracle_")},
                "d_model": p["d_model"], "heads": p["heads"],
                "layers": p["layers"], "state_dim": p["state_dim"],
                "act_dim": p["act_dim"],
            })
        per = max(1, games // workers)
        rng = random.Random(batch * 7919 + 13)
        # Aim the per-worker cadence so the AGGREGATE beat lands about
        # every `heartbeat` games across the pool.
        hb_every = max(1, heartbeat // max(1, workers))
        with ctx.Pool(workers) as pool:
            results = pool.map(_joint_worker, [
                (blobs, per, rng.randrange(2 ** 31), simulations,
                 time_budget, depth, mix_below, heartbeat_path, hb_every)
                for _ in range(workers)
            ])
        episodes: Dict[str, List[Dict]] = {name: [] for name in nets}
        for r in results:
            for name, eps in pickle.loads(r).items():
                episodes[name].extend(eps)
        roll_dt = time.time() - t0
        played = per * workers
        print(f"[joint] batch {batch + 1}/{batches}: {played} games "
              f"in {roll_dt / 60:.1f} min "
              f"({roll_dt / played:.1f}s/game)", flush=True)

        row = {"batch": batch + 1, "games": played,
               "rollout_min": round(roll_dt / 60, 1), "models": {}}
        for name, net in nets.items():
            stats = _update(net, episodes[name], lr=lr, epochs=epochs,
                            minibatch=minibatch)
            score = probe_vs_diet(net, probe_games, seed=7)
            mean = float(np.mean([e["score"] for e in episodes[name]])) * 39.0
            drop = baseline[name] - score
            status = "ok"
            if drop > rollback_bar:
                net.load_state_dict(last_good[name])
                status = f"ROLLBACK (diet {score:+.2f}, "\
                         f"bar {baseline[name]:+.2f}-{rollback_bar})"
            else:
                last_good[name] = {k: v.clone()
                                   for k, v in net.state_dict().items()}
                baseline[name] = max(baseline[name], score)
                p = payloads[name]
                full = net.state_dict()
                torch.save({"state_dict": {k: v for k, v in full.items()
                                           if not k.startswith("oracle_")},
                            "oracle_state": {k: v for k, v in full.items()
                                             if k.startswith("oracle_")},
                            "d_model": p["d_model"], "heads": p["heads"],
                            "layers": p["layers"],
                            "meta": {"joint_batch": batch + 1,
                                     "diet": round(score, 3)}},
                           _out_path(name))
            print(f"[joint]   {name:<18} table {mean:+.2f}/g  "
                  f"diet {score:+.2f}/g  {status}", flush=True)
            row["models"][name] = {"table": round(mean, 3),
                                   "diet": round(score, 3),
                                   "status": status, **stats}
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def read_heartbeat(path: str = os.path.join(HERE, "..",
                                            "joint_heartbeat.jsonl"),
                   total: Optional[int] = None) -> Dict:
    """Aggregate the workers' latest beats into one progress view."""
    latest: Dict[int, Dict] = {}
    header: Dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue            # a torn append: skip, never crash
                if row.get("header"):
                    header = row
                else:
                    latest[row["pid"]] = row
    except OSError:
        return {"workers": 0, "games": 0}
    if not latest:
        return {"workers": 0, "games": 0}
    games = sum(r["games"] for r in latest.values())
    score: Dict[str, float] = {}
    wins: Dict[str, int] = {}
    for r in latest.values():
        for n, v in r["score"].items():
            score[n] = score.get(n, 0.0) + v * r["games"]
        for n, v in r.get("wins", {}).items():
            wins[n] = wins.get(n, 0) + v
    # Workers beat on their own schedule, so early on only a few have
    # reported.  Rate must come from each worker's OWN games/elapsed and
    # then scale to the pool -- dividing the reported games by the
    # longest elapsed counts silent workers as doing nothing, which
    # once produced a 93-hour ETA for a 10-hour run.
    pool = int(header.get("workers") or len(latest))
    per_worker = sum(r["games"] / max(1e-9, r["elapsed"])
                     for r in latest.values()) / len(latest)
    rate = per_worker * pool
    done = int(round(games * pool / len(latest)))    # scale up the silent
    out = {
        "workers": f"{len(latest)}/{pool}", "games": games,
        "games_est": done,
        "games_per_min": round(rate * 60, 2),
        "sec_per_game_per_worker": round(1.0 / max(1e-9, per_worker), 1),
        "score": {n: round(v / max(1, games), 3) for n, v in score.items()},
        "wins": {n: round(100.0 * v / max(1, games), 1)
                 for n, v in wins.items()},
        "searched_frac": round(
            sum(r.get("searched_frac", 0.0) for r in latest.values())
            / len(latest), 3),
    }
    if total:
        out["pct"] = round(100.0 * done / total, 1)
        remain = max(0, total - done) / max(1e-9, rate)
        out["eta_hours"] = round(remain / 3600, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sims", type=int, default=SIMULATIONS)
    ap.add_argument("--budget", type=float, default=TIME_BUDGET)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--probe-games", type=int, default=300)
    ap.add_argument("--pilot", action="store_true",
                    help="tiny calibration batch: measure s/game, no save")
    ap.add_argument("--heartbeat", type=int, default=250,
                    help="aggregate games between progress beats")
    ap.add_argument("--status", type=int, default=0, metavar="TOTAL",
                    help="print the live heartbeat for a run of TOTAL "
                         "games and exit")
    args = ap.parse_args()
    if args.status:
        st = read_heartbeat(total=args.status)
        print(json.dumps(st, indent=1))
        return
    if args.pilot:
        train_joint(games=24, batches=1, workers=args.workers,
                    simulations=args.sims, time_budget=args.budget,
                    depth=args.depth, probe_games=60, rollback_bar=99.0)
    else:
        train_joint(games=args.games, batches=args.batches,
                    workers=args.workers, simulations=args.sims,
                    time_budget=args.budget, depth=args.depth,
                    probe_games=args.probe_games)


if __name__ == "__main__":
    main()

"""Evolutionary population training at scale (PBT islands).

Agents ONLY play other agents: every game's seats are drawn at random
from trainable MLPs, frozen checkpoints of earlier selves, champions
migrated from other islands, and a thin scripted floor (dumper, decomp,
lowest).  There is no benchmark evaluation inside the training loop —
ratings come from the training games themselves.

Structure (population-based training, island model):

- **Islands**: independent worker processes, one per core.  Each holds a
  small population of trainable agents with *sampled hyperparameters* —
  hidden-layer architecture (1-3 layers, up to 256 wide), learning rate,
  exploration epsilon.
- **Rounds**: play G games (seats drawn randomly; several trainables can
  learn from the same game — including true self-play when one agent
  draws multiple seats), then evolve: the worst-rated trainee adopts the
  best's network and architecture and mutates its learning rate and
  epsilon (exploit + explore).  The round's best agent is frozen into
  the checkpoint pool, so the population must keep beating its past.
- **Migration**: each round an island publishes its champion to a shared
  directory and pulls the other islands' champions in as frozen
  opponents — cross-pollination without locking.
- **Curriculum**: the first phase plays 1v1 (2-player) games — roughly
  3x cheaper, so early learning sees far more deals — then switches to
  full 4-player games.
- **Playoff**: after all islands finish, their final champions meet in a
  seat-rotated 4-player tournament; the winner becomes
  ``big2/policies/evo_mlp.npz`` (the UI/tournament agent named "evo").

    python -m big2.evolve --games 1000000 --islands 4
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from big2.features import FEAT_DIM, encode_options
from big2.game import Big2Game, ScoringConfig
from big2.nn import MLPQ, NNPolicy
from big2.rules import DEFAULT_RULES, RuleConfig
from big2.strategies import FiveCardDumper, PlayLowest, Strategy

SCORE_SCALE = 39.0

ARCHITECTURES: List[Tuple[int, ...]] = [
    (64,),
    (128, 64),
    (256, 128, 64),
    (256, 192, 128, 64),  # deepest lineage
]


@dataclass(frozen=True)
class Genome:
    hidden: Tuple[int, ...]
    lr: float
    eps: float

    def mutate(self, rng: random.Random) -> "Genome":
        lr = min(5e-2, max(1e-4, self.lr * rng.choice([0.5, 0.8, 1.25, 2.0])))
        eps = min(0.30, max(0.02, self.eps + rng.uniform(-0.04, 0.04)))
        return replace(self, lr=lr, eps=eps)


def sample_genome(rng: random.Random) -> Genome:
    return Genome(
        hidden=rng.choice(ARCHITECTURES),
        lr=10 ** rng.uniform(-3.0, -1.7),  # ~1e-3 .. 2e-2
        eps=rng.uniform(0.05, 0.20),
    )


class Trainee:
    def __init__(self, tid: str, genome: Genome, seed: int):
        self.tid = tid
        self.genome = genome
        self.net = MLPQ(FEAT_DIM, genome.hidden, seed=seed)
        self.window_games = 0
        self.window_total = 0.0
        self.lifetime_games = 0
        self.last_rating = 0.0  # rating of the most recent completed round

    @property
    def rating(self) -> float:
        return self.window_total / self.window_games if self.window_games else 0.0

    def reset_window(self) -> None:
        self.window_games = 0
        self.window_total = 0.0


@dataclass
class Frozen:
    name: str
    policy: Strategy


def _play_training_game(
    seats: List[Tuple[str, object]],
    num_players: int,
    scoring: ScoringConfig,
    rules: RuleConfig,
    rng: random.Random,
) -> Dict[int, int]:
    """One game; every trainee seat learns from its own trajectory."""
    game = Big2Game(
        scoring=scoring, rules=rules, num_players=num_players,
        rng=random.Random(rng.randrange(2**31)),
    )
    trajs: Dict[int, List[np.ndarray]] = {
        i: [] for i, (kind, _) in enumerate(seats) if kind == "trainee"
    }
    while not game.game_over:
        p = game.turn
        kind, agent = seats[p]
        if kind == "trainee":
            options, feats = encode_options(game, p)
            if len(options) == 1:
                idx = 0
            elif rng.random() < agent.genome.eps:
                idx = rng.randrange(len(options))
            else:
                idx = int(np.argmax(agent.net.predict(feats)))
            trajs[p].append(feats[idx])
            game.step(options[idx])
        else:
            game.step(agent.policy.select(game, p))

    for seat_idx, rows in trajs.items():
        agent = seats[seat_idx][1]
        score = game.scores[seat_idx]
        if rows:
            X = np.stack(rows)
            y = np.full(len(rows), score / SCORE_SCALE)
            agent.net.train_batch(X, y, lr=agent.genome.lr)
        agent.window_games += 1
        agent.window_total += score
        agent.lifetime_games += 1
    return game.scores


def _load_anchors() -> List[Frozen]:
    """Current best agents as permanent frozen opponents — the field the
    new population must learn to beat."""
    anchors: List[Frozen] = []
    from big2.nn import NNPolicy

    try:
        anchors.append(Frozen("anchor-evo", NNPolicy.load("big2/policies/evo_mlp.npz")))
    except Exception:
        pass
    try:
        from big2.rl import LinearPolicy

        anchors.append(
            Frozen("anchor-linear", LinearPolicy.load("big2/policies/linear_cem.npz"))
        )
    except Exception:
        pass
    try:
        from big2.dmc import DMCPolicy

        anchors.append(
            Frozen("anchor-dmc", DMCPolicy.load("big2/policies/dmc_linear.npz"))
        )
    except Exception:
        pass
    return anchors


def _probe(policy: Strategy, opponents: List[Strategy], n_games: int,
           scoring: ScoringConfig, rules: RuleConfig, seed: int) -> float:
    """Mean score of ``policy`` over seat-rotated 4p games vs 3 opponents."""
    rng = random.Random(seed)
    total = 0.0
    for g in range(n_games):
        seat = g % 4
        opps = opponents[:]
        seats = [policy if p == seat else opps.pop() for p in range(4)]
        game = Big2Game(
            scoring=scoring, rules=rules, num_players=4,
            rng=random.Random(rng.randrange(2**31)),
        )
        total += game.play_out(seats)[seat]
    return total / n_games


def run_island(island_id: int, cfg: Dict, shared_dir: str) -> None:
    rng = random.Random(cfg["seed"] * 1000 + island_id)
    scoring = ScoringConfig()  # tiered house scoring only
    rules = DEFAULT_RULES

    trainees = [
        Trainee(f"i{island_id}t{k}", sample_genome(rng), seed=rng.randrange(2**31))
        for k in range(cfg["pop_size"])
    ]
    from big2.decomposition import DecompositionStrategy
    from big2.strategies import SmartHeuristic

    scripted = [
        Frozen("dumper", FiveCardDumper()),
        Frozen("decomp", DecompositionStrategy()),
        Frozen("lowest", PlayLowest()),
    ]
    anchors = _load_anchors() if cfg.get("use_anchors", True) else []
    probe_baselines = [SmartHeuristic(), DecompositionStrategy(), FiveCardDumper()]
    probe_anchors = [a.policy for a in anchors[:3]]
    progress_path = os.path.join(shared_dir, "progress.csv")
    next_probe = cfg.get("probe_every", 0) or None

    checkpoints: List[Frozen] = []
    migrants: Dict[int, Frozen] = {}
    migrant_mtimes: Dict[int, float] = {}

    total = 0
    round_no = 0
    t0 = time.time()
    while total < cfg["games_per_island"]:
        round_no += 1
        two_player = total < cfg["phase_2p_games"]
        num_players = 2 if two_player else 4
        games = min(cfg["games_per_round"], cfg["games_per_island"] - total)

        for _ in range(games):
            frozen_pool = checkpoints + list(migrants.values())
            seats: List[Tuple[str, object]] = [("trainee", rng.choice(trainees))]
            for _ in range(num_players - 1):
                r = rng.random()
                if r < 0.55:
                    seats.append(("trainee", rng.choice(trainees)))
                elif r < 0.75 and frozen_pool:
                    seats.append(("frozen", rng.choice(frozen_pool)))
                elif r < 0.90 and anchors:
                    seats.append(("frozen", rng.choice(anchors)))
                else:
                    seats.append(("frozen", rng.choice(scripted)))
            rng.shuffle(seats)
            _play_training_game(seats, num_players, scoring, rules, rng)
        total += games

        # --- plateau probe: fixed benchmarks OUTSIDE the training field
        if next_probe is not None and total >= next_probe:
            next_probe += cfg["probe_every"]
            best_now = max(trainees, key=lambda t: t.rating)
            probe_policy = NNPolicy(best_now.net.clone())
            vs_base = _probe(
                probe_policy, probe_baselines, cfg["probe_games"],
                scoring, rules, seed=total,
            )
            vs_anchor = (
                _probe(probe_policy, probe_anchors, cfg["probe_games"],
                       scoring, rules, seed=total + 1)
                if len(probe_anchors) == 3
                else float("nan")
            )
            with open(progress_path, "a") as f:
                f.write(
                    f"{island_id},{total},{2 if two_player else 4},"
                    f"{best_now.tid},{len(best_now.genome.hidden)},"
                    f"{best_now.genome.lr:.5f},{vs_base:.3f},{vs_anchor:.3f}\n"
                )
            print(
                f"[island {island_id}] probe @{total}: vs-baselines "
                f"{vs_base:+.2f}  vs-anchors {vs_anchor:+.2f}",
                flush=True,
            )

        # --- evolve: worst adopts best, then mutates (exploit + explore)
        rated = [t for t in trainees if t.window_games >= cfg["min_window_games"]]
        if len(rated) >= 2:
            rated.sort(key=lambda t: t.rating)
            worst, best = rated[0], rated[-1]
            if worst is not best and best.rating - worst.rating > cfg["evolve_margin"]:
                worst.net = best.net.clone()
                worst.genome = replace(
                    best.genome.mutate(rng), hidden=best.genome.hidden
                )

        # --- freeze the round's best as an opponent checkpoint
        best = max(trainees, key=lambda t: t.rating)
        checkpoints.append(
            Frozen(f"{best.tid}-r{round_no}", NNPolicy(best.net.clone()))
        )
        if len(checkpoints) > cfg["max_checkpoints"]:
            checkpoints.pop(0)

        # --- migration: publish champion, pull other islands' champions
        champ_path = os.path.join(shared_dir, f"island{island_id}_champ.npz")
        best.net.save(champ_path, meta={
            "tid": best.tid, "rating": best.rating, "round": round_no,
            "genome": {"hidden": list(best.genome.hidden),
                       "lr": best.genome.lr, "eps": best.genome.eps},
        })
        for other in range(cfg["islands"]):
            if other == island_id:
                continue
            path = os.path.join(shared_dir, f"island{other}_champ.npz")
            try:
                mtime = os.path.getmtime(path)
                if migrant_mtimes.get(other) != mtime:
                    net, _ = MLPQ.load(path)
                    migrants[other] = Frozen(f"migrant{other}", NNPolicy(net))
                    migrant_mtimes[other] = mtime
            except (FileNotFoundError, ValueError, KeyError, OSError):
                pass  # other island mid-write or not started; retry next round

        rate = total / max(1e-9, time.time() - t0)
        summary = " ".join(
            f"{t.tid}:{t.rating:+.2f}(lr={t.genome.lr:.4f},{len(t.genome.hidden)}L)"
            for t in sorted(trainees, key=lambda t: -t.rating)
        )
        print(
            f"[island {island_id}] round {round_no} games {total}/{cfg['games_per_island']} "
            f"{num_players}p {rate:.0f} g/s | {summary}",
            flush=True,
        )
        for t in trainees:
            t.last_rating = t.rating
            t.reset_window()

    best = max(trainees, key=lambda t: t.last_rating)
    final_path = os.path.join(shared_dir, f"island{island_id}_final.npz")
    best.net.save(final_path, meta={
        "tid": best.tid, "island": island_id,
        "genome": {"hidden": list(best.genome.hidden),
                   "lr": best.genome.lr, "eps": best.genome.eps},
        "lifetime_games": best.lifetime_games,
    })
    print(f"[island {island_id}] done: saved {final_path}", flush=True)


def playoff(shared_dir: str, n_games: int, seed: int) -> Tuple[str, Dict]:
    """Island champions meet in a seat-rotated 4p tournament."""
    from big2.experiments import run_tournament

    finals = sorted(glob.glob(os.path.join(shared_dir, "island*_final.npz")))
    if len(finals) < 2:
        raise RuntimeError(f"need >= 2 island finals in {shared_dir}")
    entrants: List[Strategy] = []
    metas = []
    for path in finals[:4]:
        net, meta = MLPQ.load(path)
        policy = NNPolicy(net)
        policy.name = f"isl{len(entrants)}"
        entrants.append(policy)
        metas.append((path, meta))
    while len(entrants) < 4:  # fewer islands: pad with clones
        clone = NNPolicy(entrants[0].net.clone())
        clone.name = f"pad{len(entrants)}"
        entrants.append(clone)
        metas.append(metas[0])

    results = run_tournament(entrants, n_games, ScoringConfig(), seed=seed)
    print("\n=== champion playoff ===")
    for name, stats in sorted(results.items(), key=lambda kv: -kv[1]["avg_score"]):
        print(f"{name:>8s}  {stats['avg_score']:+6.2f}  ({stats['win_rate']:.0%})")
    winner_idx = max(
        range(len(entrants)),
        key=lambda i: results[entrants[i].name]["avg_score"],
    )
    return metas[winner_idx][0], metas[winner_idx][1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1_000_000,
                        help="total games across all islands")
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=4)
    parser.add_argument("--phase-2p-frac", type=float, default=0.6,
                        help="fraction of each island's games played 1v1")
    parser.add_argument("--games-per-round", type=int, default=4000)
    parser.add_argument("--min-window-games", type=int, default=300)
    parser.add_argument("--evolve-margin", type=float, default=0.5)
    parser.add_argument("--max-checkpoints", type=int, default=4)
    parser.add_argument("--playoff-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-dir", default="big2/policies/evolve")
    parser.add_argument("--out", default="big2/policies/evo_mlp.npz")
    parser.add_argument("--probe-every", type=int, default=25_000,
                        help="games between plateau probes (0 = off)")
    parser.add_argument("--probe-games", type=int, default=120)
    parser.add_argument("--no-anchors", action="store_true",
                        help="drop the current-best anchor opponents")
    args = parser.parse_args()

    os.makedirs(args.shared_dir, exist_ok=True)
    per_island = args.games // args.islands
    cfg = {
        "seed": args.seed,
        "islands": args.islands,
        "pop_size": args.pop_size,
        "games_per_island": per_island,
        "phase_2p_games": int(per_island * args.phase_2p_frac),
        "games_per_round": args.games_per_round,
        "min_window_games": args.min_window_games,
        "evolve_margin": args.evolve_margin,
        "max_checkpoints": args.max_checkpoints,
        "probe_every": args.probe_every,
        "probe_games": args.probe_games,
        "use_anchors": not args.no_anchors,
    }
    print(
        f"evolving {args.games} games on {args.islands} islands "
        f"({per_island} each, {cfg['phase_2p_games']} of them 1v1)",
        flush=True,
    )
    procs = [
        mp.Process(target=run_island, args=(i, cfg, args.shared_dir))
        for i in range(args.islands)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise RuntimeError(f"island exit codes: {[p.exitcode for p in procs]}")

    winner_path, meta = playoff(args.shared_dir, args.playoff_games, args.seed)
    net, _ = MLPQ.load(winner_path)
    net.save(args.out, meta=meta)
    print(f"\nchampion: {winner_path} {meta.get('genome')} -> {args.out}")


if __name__ == "__main__":
    main()

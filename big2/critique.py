"""Find the spots that actually cost the game, and settle them.

A losing game is forty decisions long and almost all of them are
irrelevant.  What matters is the handful of nodes where the outcome
turned — and, when comparing two models, the far smaller handful where
they would have played *differently* and it mattered.  This module
narrows a pile of replays to those nodes and then decides them by
playing the position out rather than by opinion.

Three stages, each cutting the search space hard:

1. **Bad games** — replays where a seat lost by a wide margin.  Nothing
   is learned from a close game.
2. **Divergences** — positions inside those games where two models
   disagree about the move.  Agreement is not interesting: if both the
   old and new model play the same card, that decision explains nothing
   about the difference between them.
3. **Critical nodes** — divergences where the disagreement is *worth*
   something.  Each candidate move is played to the end many times
   (exactly, by the endgame solver, once the position is small enough;
   by Monte Carlo rollout before that), and the gap between the two
   moves' expected values is the node's criticality.  A divergence worth
   0.1 points is noise; one worth 4 points is where the game was lost.

The output is a ranked list of positions with both moves and their
measured EVs — the raw material for "was the new model actually right
here", and for building targeted training sets from real losses.

    python -m big2.critique --replays replays.jsonl \\
        --old big2/policies/ppo_attn_v11.pt --new big2/policies/ppo_v2.pt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from big2.cards import hand_to_str
from big2.combos import Combo
from big2.endgame import (
    MoveKey,
    move_key,
    remaining_cards,
    solve_move_values,
)
from big2.game import Big2Game, ScoringConfig
from big2.rules import DEFAULT_RULES
from big2.strategies import Strategy

# Positions at or below this many cards are settled exactly instead of
# by rollout (matches the solver's affordable range).
EXACT_BELOW = 16


@dataclass
class Node:
    """One position where two models disagree, with the verdict."""

    game_index: int
    ply: int
    player: int
    hand: Tuple[int, ...]
    cards_left: Tuple[int, ...]
    old_move: MoveKey
    new_move: MoveKey
    old_ev: float
    new_ev: float
    exact: bool
    played_move: MoveKey = None

    @property
    def criticality(self) -> float:
        return abs(self.new_ev - self.old_ev)

    @property
    def verdict(self) -> str:
        if self.new_ev > self.old_ev:
            return "new better"
        if self.new_ev < self.old_ev:
            return "old better"
        return "tie"

    def describe(self) -> str:
        fmt = lambda m: "pass" if m is None else hand_to_str(list(m))
        kind = "exact" if self.exact else "rollout"
        return (
            f"game {self.game_index} ply {self.ply} seat {self.player} "
            f"[{hand_to_str(list(self.hand))}] left={list(self.cards_left)}\n"
            f"    old: {fmt(self.old_move):<18} EV {self.old_ev:+.2f}\n"
            f"    new: {fmt(self.new_move):<18} EV {self.new_ev:+.2f}"
            f"   ({kind}, {self.verdict} by {self.criticality:.2f})"
        )


def losing_games(
    rows: Sequence[Dict], seat: Optional[int] = None, margin: float = 5.0
) -> List[Tuple[int, Dict, int]]:
    """(index, replay body, seat) for games a seat lost badly.

    With ``seat=None`` every seat is considered, so the worst-beaten
    player in each game is the one examined — which is what "the model
    lost badly here" means when the model held several chairs.
    """
    from big2.offline import _replay_body, replay_outcomes

    out = []
    for i, row in enumerate(rows):
        body = _replay_body(row)
        if body is None:
            continue
        scores = replay_outcomes(body)
        if not scores:
            continue
        if seat is None:
            worst = min(scores, key=lambda p: scores[p])
        else:
            worst = seat
        if scores.get(worst, 0.0) <= -abs(margin):
            out.append((i, body, worst))
    return out


def rollout_ev(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    opponents: Sequence[Strategy],
    n: int = 40,
    seed: int = 0,
) -> float:
    """Average score for ``player`` after playing ``move``, by playing
    the rest of the hand out ``n`` times."""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        sim = game.clone(rng=random.Random(rng.randrange(2**31)))
        sim.step(move)
        if sim.game_over:
            total += float(sim.scores[player])
            continue
        seats = list(opponents)
        pol = {p: seats[i % len(seats)]
               for i, p in enumerate(
                   q for q in range(sim.num_players) if q != player)}
        pol[player] = seats[0]
        while not sim.game_over:
            p = sim.turn
            sim.step(pol[p].select(sim, p))
        total += float(sim.scores[player])
    return total / n


def move_ev(
    game: Big2Game,
    player: int,
    move: Optional[Combo],
    opponents: Sequence[Strategy],
    rollouts: int = 40,
    seed: int = 0,
) -> Tuple[float, bool]:
    """(expected value, was_it_exact) for one move in this position."""
    if remaining_cards(game) <= EXACT_BELOW:
        values, exact = solve_move_values(game, player)
        if exact and move_key(move) in values:
            return values[move_key(move)], True
    return rollout_ev(game, player, move, opponents, n=rollouts, seed=seed), False


def _find(options: Sequence[Optional[Combo]], key: MoveKey):
    return next((m for m in options if move_key(m) == key), None)


def critical_nodes(
    rows: Sequence[Dict],
    old: Strategy,
    new: Strategy,
    opponents: Optional[Sequence[Strategy]] = None,
    seat: Optional[int] = None,
    margin: float = 5.0,
    rollouts: int = 40,
    min_criticality: float = 0.0,
    max_games: Optional[int] = None,
    seed: int = 0,
) -> List[Node]:
    """Rank the positions where the two models disagree and it matters."""
    from big2.offline import iter_decisions
    from big2.strategies import SmartHeuristic

    opponents = list(opponents or [SmartHeuristic()])
    nodes: List[Node] = []
    games = losing_games(rows, seat=seat, margin=margin)
    if max_games:
        games = games[:max_games]
    for gi, body, bad_seat in games:
        for ply, (game, p, cards) in enumerate(iter_decisions(body)):
            if p != bad_seat:
                continue
            options = list(game.legal_moves(p))
            if game.can_pass():
                options.append(None)
            if len(options) < 2:
                continue
            old_key = move_key(old.select(game, p))
            new_key = move_key(new.select(game, p))
            if old_key == new_key:
                continue  # agreement explains nothing
            old_ev, e1 = move_ev(game, p, _find(options, old_key), opponents,
                                 rollouts, seed=seed + ply)
            new_ev, e2 = move_ev(game, p, _find(options, new_key), opponents,
                                 rollouts, seed=seed + ply)
            node = Node(
                game_index=gi,
                ply=ply,
                player=p,
                hand=tuple(game.hands[p]),
                cards_left=tuple(len(h) for h in game.hands),
                old_move=old_key,
                new_move=new_key,
                old_ev=old_ev,
                new_ev=new_ev,
                exact=bool(e1 and e2),
                played_move=(
                    None if not cards else tuple(sorted(int(c) for c in cards))
                ),
            )
            if node.criticality >= min_criticality:
                nodes.append(node)
    nodes.sort(key=lambda n: -n.criticality)
    return nodes


def summarize(nodes: Sequence[Node]) -> Dict[str, float]:
    """Does the new model actually decide these spots better?"""
    if not nodes:
        return {"nodes": 0}
    better = sum(1 for n in nodes if n.new_ev > n.old_ev)
    worse = sum(1 for n in nodes if n.new_ev < n.old_ev)
    gain = sum(n.new_ev - n.old_ev for n in nodes) / len(nodes)
    exact = sum(1 for n in nodes if n.exact)
    return {
        "nodes": len(nodes),
        "new_better": better,
        "old_better": worse,
        "mean_ev_gain": gain,
        "exactly_solved": exact,
        "max_criticality": max(n.criticality for n in nodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True)
    parser.add_argument("--old", default="big2/policies/ppo_attn_v11.pt")
    parser.add_argument("--new", default="big2/policies/ppo_v2.pt")
    parser.add_argument("--seat", type=int, default=None,
                        help="seat to examine (default: whoever lost worst)")
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--rollouts", type=int, default=40)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    from big2.neural import PPOPolicy
    from big2.offline import load_replays

    rows = load_replays(args.replays)
    old, new = PPOPolicy.load(args.old), PPOPolicy.load(args.new)
    nodes = critical_nodes(
        rows, old, new, seat=args.seat, margin=args.margin,
        rollouts=args.rollouts, max_games=args.max_games,
    )
    stats = summarize(nodes)
    print(f"\n=== {stats.get('nodes', 0)} divergent decisions in games lost "
          f"by {args.margin:+.0f} or worse ===")
    if stats.get("nodes"):
        print(f"new better at {stats['new_better']}, old better at "
              f"{stats['old_better']}, mean EV gain "
              f"{stats['mean_ev_gain']:+.3f}/decision "
              f"({stats['exactly_solved']} settled exactly)\n")
        for n in nodes[: args.top]:
            print(n.describe())
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([n.__dict__ for n in nodes], f, indent=2, default=str)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()

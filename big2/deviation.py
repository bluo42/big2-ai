"""Where good humans disagree with the bot — and are right.

Aggregate scores say a model is losing to humans.  They do not say
*how*.  This walks recorded games, finds every decision where a named
player chose differently from the model, plays both choices out over
belief-sampled deals, and keeps the ones where the human's move
measurably won.  Those survivors are then described rather than
counted: what class of move did they play instead, how did its rank
compare, and at what stage of the hand.

That last axis is the one that changes what you build.  A leak
concentrated in the last few tricks argues for search and endgame
features; a leak spread through the middle of the hand argues for
something else entirely, and the two are easy to confuse from scores
alone.

    python -m big2.deviation --replays replays.jsonl \\
        --players justinwang1688,LEX --model big2/policies/ppo_attn_v11.pt
"""

from __future__ import annotations

import argparse
import collections
from typing import Dict, List, Optional, Sequence

import numpy as np

from big2.cards import hand_to_str, rank
from big2.combos import Combo
from big2.endgame import move_key
from big2.strategies import Strategy


def move_class(move: Optional[Combo]) -> str:
    if move is None:
        return "pass"
    return {1: "single", 2: "pair", 5: "five"}.get(len(move), str(len(move)))


def phase_of(game) -> str:
    left = sum(len(h) for h in game.hands)
    if left > 36:
        return "early (>36 left)"
    if left > 20:
        return "mid (20-36)"
    return "late (<=20)"


def winning_deviations(
    rows: Sequence[Dict],
    model: Strategy,
    opponents: Optional[Sequence[Strategy]] = None,
    seat: int = 0,
    rollouts: int = 10,
    min_gain: float = 0.25,
    max_games: Optional[int] = None,
) -> Dict[str, object]:
    """Find, verify and characterise the human's better choices."""
    from big2.critique import move_ev
    from big2.offline import _replay_body, iter_decisions
    from big2.strategies import SmartHeuristic

    opponents = list(opponents or [model, SmartHeuristic()])
    diverged = 0
    swaps: collections.Counter = collections.Counter()
    phases: collections.Counter = collections.Counter()
    rank_deltas: List[int] = []
    found: List[Dict] = []

    for row in (rows[:max_games] if max_games else rows):
        body = _replay_body(row)
        if body is None:
            continue
        for ply, (game, p, cards) in enumerate(iter_decisions(body)):
            if p != seat:
                continue
            options: List[Optional[Combo]] = list(game.legal_moves(p))
            if game.can_pass():
                options.append(None)
            if len(options) < 2:
                continue
            key = None if not cards else tuple(sorted(int(c) for c in cards))
            human = next((m for m in options if move_key(m) == key), None)
            if key is not None and human is None:
                continue
            bot_key = move_key(model.select(game, p))
            if bot_key == key:
                continue
            diverged += 1
            bot = next((m for m in options if move_key(m) == bot_key), None)
            h_ev, _ = move_ev(game, p, human, opponents, rollouts, seed=ply)
            b_ev, _ = move_ev(game, p, bot, opponents, rollouts, seed=ply)
            if h_ev - b_ev < min_gain:
                continue
            swaps[f"{move_class(human)} <- {move_class(bot)}"] += 1
            phases[phase_of(game)] += 1
            if human is not None and bot is not None:
                rank_deltas.append(
                    rank(max(human.cards)) - rank(max(bot.cards))
                )
            found.append({
                "ply": ply,
                "hand": hand_to_str(game.hands[p]),
                "human": "pass" if human is None
                         else hand_to_str(list(human.cards)),
                "bot": "pass" if bot is None else hand_to_str(list(bot.cards)),
                "gain": h_ev - b_ev,
                "left": tuple(len(h) for h in game.hands),
            })
    found.sort(key=lambda d: -d["gain"])
    return {
        "diverged": diverged,
        "better": len(found),
        "swaps": swaps,
        "phases": phases,
        "rank_delta": (float(np.mean(rank_deltas)) if rank_deltas else 0.0),
        "rank_higher": sum(1 for d in rank_deltas if d > 0),
        "rank_lower": sum(1 for d in rank_deltas if d < 0),
        "cases": found,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", required=True)
    parser.add_argument("--players", default="",
                        help="comma-separated usernames (default: everyone)")
    parser.add_argument("--model", default="big2/policies/ppo_attn_v11.pt")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--max-games", type=int, default=70)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    from big2.neural import PPOPolicy
    from big2.offline import load_replays

    rows = load_replays(args.replays)
    if args.players:
        wanted = {p.strip() for p in args.players.split(",") if p.strip()}
        rows = [r for r in rows if r.get("username") in wanted]
    model = PPOPolicy.load(args.model)
    out = winning_deviations(rows, model, rollouts=args.rollouts,
                             max_games=args.max_games)
    print(f"diverged from the model: {out['diverged']}")
    share = out["better"] / max(1, out["diverged"])
    print(f"human move measurably better: {out['better']} ({share:.0%})\n")
    print("what they played instead:")
    for k, v in out["swaps"].most_common(8):
        print(f"  {k:<22}{v:>4}")
    print("\nwhen:")
    for k, v in out["phases"].most_common():
        print(f"  {k:<20}{v:>4}")
    print(f"\nrank vs the model's card: {out['rank_delta']:+.2f} "
          f"({out['rank_higher']} higher / {out['rank_lower']} lower)")
    print("\nbiggest gains:")
    for c in out["cases"][: args.top]:
        print(f"  ply {c['ply']:>2} left={list(c['left'])} [{c['hand']}]")
        print(f"     human {c['human']:<16} model {c['bot']:<16} "
              f"+{c['gain']:.2f}")


if __name__ == "__main__":
    main()

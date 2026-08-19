"""Play-evidence belief factors: what an action says about a hand.

The analytic beliefs (big2/beliefs.py) are exact under uniform play and
blind to evidence: after a player leads a lone 7, they still assign the
same probability that the 7's pair-mate sits in that hand.  But actions
leak.  A policy that holds a pair rarely breaks it to lead a single, so
the single *lowers* the odds of the pair-mate in the leader's hand and
raises it everywhere else.

This module fits that leak from data and applies it with a damper:

**Fit.**  Walk games where all hands are known (recorded human replays,
AI self-play).  Every time a player plays a lone single of rank r, ask:
did they still hold another card of rank r?  Compare the empirical rate
to the hypergeometric baseline a public observer would compute at that
moment.  The ratio of odds is the Bayes factor for the event, estimated
per (stage of hand x rank bucket) so the table stays dense.

**Apply.**  From a viewpoint, collect each opponent's single-plays; for
each, the sampled worlds where that opponent holds the pair-mate are
reweighted by factor**lambda (damped, clamped).  Under importance
weighting this shifts the posterior odds by exactly the damped factor
and renormalization spreads the freed mass to the other hands -- the
"others are now more likely to hold it" half falls out for free.  The
same events adjust the analytic per-card map with a few rounds of
proportional fitting so rows (hand counts) and columns (each card lives
somewhere) stay consistent.

Damping is deliberate (default lambda 0.25, factors clamped to
[0.6, 1.6]): small, statistically sound adjustments now; the table is
keyed by corpus name so per-player-type tables ("this player plays like
X") can slot in later without changing the plumbing.

    python -m big2.playmodel --fit            # refit from replays + self-play
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from big2.cards import NUM_CARDS, Card, rank
from big2.game import Big2Game

FACTORS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "policies",
    "belief_factors.json",
)

LAMBDA = 0.25            # damping exponent on every applied factor
CLAMP = (0.6, 1.6)       # a fitted factor may not swing odds further

_STAGES = ((35, "early"), (21, "mid"), (0, "late"))
_BUCKETS = ((0, 4, "low"), (5, 8, "mid"), (9, 12, "high"))


def stage_of(cards_left: int) -> str:
    for floor, name in _STAGES:
        if cards_left >= floor:
            return name
    return "late"


def bucket_of(r: int) -> str:
    for lo, hi, name in _BUCKETS:
        if lo <= r <= hi:
            return name
    return "high"


def _hyper_at_least_one(hits: int, pool: int, hand: int) -> float:
    if hits <= 0 or hand <= 0 or pool <= 0:
        return 0.0
    if hits + hand > pool:
        return 1.0
    return 1.0 - math.comb(pool - hits, hand) / math.comb(pool, hand)


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return p / (1.0 - p)


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def _observe_game(counts: Dict, game_stream) -> None:
    """Accumulate (event, condition, baseline) triples from one game.

    ``game_stream`` yields (game_before_move, player, move_cards) with
    full hands known -- offline.iter_decisions provides exactly this.
    """
    for game, p, cards in game_stream:
        if not cards or len(cards) != 1:
            continue
        if game.first_play:
            continue                      # forced opening: no choice, no leak
        card = int(cards[0])
        r = rank(card)
        left = sum(len(h) for h in game.hands)
        played = set(game.played_cards) | {card}
        pool = NUM_CARDS - len(played)
        hand_after = [c for c in game.hands[p] if c != card]
        hits = sum(1 for c in range(NUM_CARDS)
                   if c not in played and rank(c) == r)
        base = _hyper_at_least_one(hits, pool, len(hand_after))
        holds = any(rank(c) == r for c in hand_after)
        key = f"single|{stage_of(left)}|{bucket_of(r)}"
        row = counts.setdefault(key, {"n": 0, "held": 0, "base": 0.0})
        row["n"] += 1
        row["held"] += int(holds)
        row["base"] += base


def fit_factors(streams, corpus: str = "all") -> Dict:
    """Fit the factor table from an iterable of game streams."""
    counts: Dict = {}
    games = 0
    for stream in streams:
        _observe_game(counts, stream)
        games += 1
    table = {}
    for key, row in counts.items():
        if row["n"] < 30:
            continue                      # too sparse to trust
        emp = row["held"] / row["n"]
        base = row["base"] / row["n"]
        factor = _odds(emp) / _odds(base)
        table[key] = {
            "factor": round(min(max(factor, CLAMP[0]), CLAMP[1]), 4),
            "raw": round(factor, 4),
            "n": row["n"],
            "empirical": round(emp, 4),
            "baseline": round(base, 4),
        }
    return {"corpus": corpus, "games": games, "lambda": LAMBDA,
            "events": table}


_CACHED: Optional[Dict] = None


def load_factors() -> Dict:
    global _CACHED
    if _CACHED is None:
        try:
            with open(FACTORS_PATH, encoding="utf-8") as f:
                _CACHED = json.load(f)
        except (OSError, ValueError):
            _CACHED = {"events": {}, "lambda": LAMBDA}
    return _CACHED


# ----------------------------------------------------------------------
# Applying
# ----------------------------------------------------------------------


class PlayEvidence:
    """One opponent's single-play: reweights worlds by the pair-mate odds."""

    __slots__ = ("player", "rank", "factor")

    def __init__(self, player: int, r: int, factor: float):
        self.player = player
        self.rank = r
        self.factor = factor


def collect_events(game: Big2Game, viewpoint: int) -> List[PlayEvidence]:
    """Damped Bayes factors for every informative single an opponent
    played, restricted to ranks that still have unseen cards."""
    table = load_factors().get("events", {})
    if not table:
        return []
    lam = float(load_factors().get("lambda", LAMBDA))
    seen = set(game.hands[viewpoint]) | set(game.played_cards)
    unseen_ranks = {rank(c) for c in range(NUM_CARDS) if c not in seen}
    events: List[PlayEvidence] = []
    left = 52
    first = True
    for rec in game.history:
        if rec.combo is None:
            continue
        n = len(rec.combo.cards)
        if (n == 1 and not first and rec.player != viewpoint):
            r = rank(rec.combo.cards[0])
            if r in unseen_ranks:
                key = f"single|{stage_of(left)}|{bucket_of(r)}"
                row = table.get(key)
                if row:
                    f = min(max(float(row["factor"]), CLAMP[0]), CLAMP[1])
                    events.append(PlayEvidence(rec.player, r, f ** lam))
        left -= n
        first = False
    return events


def world_factor(events: Sequence[PlayEvidence],
                 world: Dict[int, List[Card]]) -> float:
    """Importance weight for one sampled world: worlds where the player
    holds the pair-mate carry the (damped) factor, others carry 1 --
    after normalization the posterior odds shift by exactly the factor."""
    w = 1.0
    for ev in events:
        hand = world.get(ev.player)
        if hand and any(rank(c) == ev.rank for c in hand):
            w *= ev.factor
    return w


def adjust_card_map(
    card_map: Dict[int, Dict[Card, float]],
    events: Sequence[PlayEvidence],
    counts: Dict[int, int],
    iters: int = 3,
) -> Dict[int, Dict[Card, float]]:
    """Damped analytic adjustment with proportional fitting.

    Scale the event player's probability on the event rank's remaining
    cards, then alternate row (hand size) and column (each card is in
    exactly one hand) renormalizations so the map stays a coherent
    distribution rather than a heat map with leaks.
    """
    if not events:
        return card_map
    m = {p: dict(row) for p, row in card_map.items()}
    for ev in events:
        row = m.get(ev.player)
        if not row:
            continue
        for c in row:
            if rank(c) == ev.rank:
                row[c] = min(1.0, row[c] * ev.factor)
    cards = set()
    for row in m.values():
        cards.update(row)
    for _ in range(iters):
        for p, row in m.items():           # rows: sum = hand count
            s = sum(row.values())
            if s > 0:
                k = counts.get(p, s) / s
                for c in row:
                    row[c] = min(1.0, row[c] * k)
        for c in cards:                     # columns: sum = 1
            s = sum(row.get(c, 0.0) for row in m.values())
            if s > 0:
                for row in m.values():
                    if c in row:
                        row[c] /= s
    return m


# ----------------------------------------------------------------------
# Fit entry point
# ----------------------------------------------------------------------


def _replay_streams(path: str):
    from big2.offline import _replay_body, iter_decisions, load_replays

    for row in load_replays(path):
        body = _replay_body(row)
        if body is not None:
            yield iter_decisions(body)


def _selfplay_streams(n_games: int, seed: int = 0):
    import random

    from big2.game import ScoringConfig
    from big2.neural import load_checkpoint_policy
    from big2.rules import DEFAULT_RULES

    here = os.path.dirname(os.path.abspath(__file__))
    models = [load_checkpoint_policy(os.path.join(here, "policies", f))
              for f in ("wangbot_v2.pt", "khabib_v1.pt",
                        "sicario_v1.pt", "leonidas_v1.pt")]
    rng = random.Random(seed)

    def one_game():
        seats = models[:]
        rng.shuffle(seats)
        game = Big2Game(scoring=ScoringConfig(), rules=DEFAULT_RULES,
                        num_players=4,
                        rng=random.Random(rng.randrange(2 ** 31)))
        while not game.game_over:
            p = game.turn
            mv = seats[p].select(game, p)
            cards = None if mv is None else list(mv.cards)
            yield game, p, cards
            game.step(mv)

    for _ in range(n_games):
        yield one_game()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays", default="replays.jsonl")
    ap.add_argument("--selfplay", type=int, default=1200)
    ap.add_argument("--out", default=FACTORS_PATH)
    args = ap.parse_args()

    def streams():
        if os.path.exists(args.replays):
            yield from _replay_streams(args.replays)
        yield from _selfplay_streams(args.selfplay)

    table = fit_factors(streams(), corpus="humans+selfplay")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    print(f"fit {len(table['events'])} event keys "
          f"from {table['games']} games -> {args.out}")
    for k, v in sorted(table["events"].items()):
        print(f"  {k:<24} factor {v['factor']:>6.3f} (raw {v['raw']:>6.3f}) "
              f"emp {v['empirical']:.3f} vs base {v['baseline']:.3f} "
              f"n={v['n']}")


if __name__ == "__main__":
    main()
